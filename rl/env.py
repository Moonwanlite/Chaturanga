"""
Single-agent Gymnasium environment for training an RL policy to play
Chaturanga against a configurable opponent.

Design note -- read this before assuming it's "self-play":
    True self-play (one policy controlling both sides, learning purely from
    its own games, AlphaZero-style) needs a custom training loop -- PPO/DQN
    via stable-baselines3 assume a single fixed-perspective agent acting in
    an "environment," not two agents co-training. Building that is a
    meaningfully bigger project than this file.

    What THIS environment does instead: the RL agent always plays one fixed
    color (`agent_color`), and `opponent_move_fn` plays the other side
    automatically inside step() -- no separate env.step() call happens for
    the opponent's turns. This is a standard, much simpler pattern ("train
    against a scripted/frozen opponent") and reuses agents you already have
    (search.search_agent.minimax_agent_move, or a random baseline).

    Easy upgrade path to poor-man's self-play later: periodically replace
    `opponent_move_fn` with a frozen copy of the policy currently being
    trained (e.g. every N training iterations, save a checkpoint and load
    it as the new opponent). That's a training-script change, not an env
    change -- nothing here needs to be rewritten for it.

Action space: Discrete(4096), from_sq * 64 + to_sq (see models/cnn_eval.py's
move_to_action / action_to_squares -- this env reuses that exact encoding so
a CNN-based policy can be plugged in later without a re-mapping layer).
Promotion is not part of the action encoding: the rules engine auto-resolves
pawn promotion to Mantri, so each (from, to) pair already fully determines
the move.

Action masking: action_masks() returns a length-4096 boolean array, True
for the agent's currently-legal moves. Use sb3_contrib.MaskablePPO (not
vanilla PPO) so illegal actions are structurally excluded from sampling --
without masking, a random policy would pick a legal move on maybe 0.5% of
attempts out of 4096 actions, making learning impractically slow.

Reward: by default a *shaped* signal, not purely sparse:
    - Sparse component: +1 if the agent's side wins, -1 if it loses, at
      true game termination (0 for a draw or for a truncated episode).
    - Dense component (reward_shaping=True, the default): a small reward
      proportional to material gained/lost on every ply -- both the
      agent's own moves and the opponent's auto-played moves -- using
      search.minimax.material_evaluator, scaled down by `shaping_scale`.

Why shaping is on by default: in early smoke-testing, random-vs-random and
weak-vs-weak Chaturanga games routinely ran past a 200-ply cap without
reaching checkmate (consistent with what board/state.py's own self-play
tests showed). With reward only at true termination, most episodes would
truncate at 0 reward and PPO would get almost no learning signal. Material
shaping gives it something to learn from immediately, at the cost of
biasing the learned policy toward material-greedy play rather than
whatever a from-scratch sparse-reward policy might have discovered on its
own. Set reward_shaping=False if you want the "purer" sparse-reward
version once you have the compute budget for much longer training runs.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from board.state import ChaturangaState, BOARD_SIZE, OTHER
from board.encode import to_tensor, PIECE_PLANES
from models.cnn_eval import move_to_action, N_MOVES
from search.minimax import material_evaluator


def random_opponent(state):
    import random
    return random.choice(state.legal_moves())


class ChaturangaEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent_move_fn=None, agent_color: str = "w", max_plies: int = 200,
                 reward_shaping: bool = True, shaping_scale: float = 0.001):
        super().__init__()
        self.agent_color = agent_color
        self.opponent_move_fn = opponent_move_fn or random_opponent
        self.max_plies = max_plies
        self.reward_shaping = reward_shaping
        self.shaping_scale = shaping_scale

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(PIECE_PLANES), BOARD_SIZE, BOARD_SIZE), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_MOVES)

        self.state = ChaturangaState()
        self._ply_count = 0

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = ChaturangaState()
        self._ply_count = 0

        # If the agent plays Black, let White (the opponent) move first.
        if self.state.turn != self.agent_color:
            self._play_opponent_moves()

        return to_tensor(self.state), {}

    def step(self, action: int):
        if self.state.turn != self.agent_color:
            raise RuntimeError(
                "step() called on the opponent's turn -- this shouldn't happen; "
                "the environment auto-plays the opponent inside step()/reset()."
            )

        move = self._action_to_legal_move(action)
        if move is None:
            # Should not happen when using MaskablePPO with action_masks()
            # respected. Handled defensively so the env never silently
            # corrupts state if called without masking (e.g. plain PPO,
            # random-action testing, or a bug).
            return to_tensor(self.state), -1.0, True, False, {"illegal_action": True}

        material_before = self._signed_material() if self.reward_shaping else 0.0
        self.state.push(move)
        self._ply_count += 1
        agent_shaped = 0.0
        if self.reward_shaping:
            agent_shaped = (self._signed_material() - material_before) * self.shaping_scale

        if self.state.is_terminal():
            return to_tensor(self.state), self._terminal_reward() + agent_shaped, True, False, {}

        if self._ply_count >= self.max_plies:
            return to_tensor(self.state), agent_shaped, False, True, {}

        # Opponent's turn(s) -- auto-played inside this step.
        terminated, opponent_reward = self._play_opponent_moves()
        total_reward = agent_shaped + opponent_reward
        truncated = (not terminated) and self._ply_count >= self.max_plies
        return to_tensor(self.state), total_reward, terminated, truncated, {}

    def action_masks(self) -> np.ndarray:
        """Required by sb3_contrib.MaskablePPO. True = legal action index."""
        mask = np.zeros(N_MOVES, dtype=bool)
        if self.state.turn == self.agent_color:
            for move in self.state.legal_moves():
                mask[move_to_action(move)] = True
        return mask

    def render(self):
        print(self.state)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _action_to_legal_move(self, action: int):
        for move in self.state.legal_moves():
            if move_to_action(move) == action:
                return move
        return None

    def _play_opponent_moves(self):
        """Plays opponent moves until it's the agent's turn again or the
        game ends. Returns (terminated: bool, reward: float), where reward
        already folds in both the sparse terminal component (if any) and
        the dense material-shaping component (if enabled) for these moves."""
        shaped_total = 0.0
        while not self.state.is_terminal() and self.state.turn != self.agent_color:
            material_before = self._signed_material() if self.reward_shaping else 0.0
            move = self.opponent_move_fn(self.state)
            self.state.push(move)
            self._ply_count += 1
            if self.reward_shaping:
                shaped_total += (self._signed_material() - material_before) * self.shaping_scale
            if self._ply_count >= self.max_plies:
                return False, shaped_total

        if self.state.is_terminal():
            return True, self._terminal_reward() + shaped_total
        return False, shaped_total

    def _signed_material(self) -> float:
        """material_evaluator(state), sign-flipped so positive always means
        'good for the agent', regardless of which color it's playing."""
        score = material_evaluator(self.state)
        return score if self.agent_color == "w" else -score

    def _terminal_reward(self) -> float:
        """Call only when self.state.is_terminal() is True."""
        result = self.state.result()  # +1 White win / -1 Black win / 0 draw
        if result == 0.0:
            return 0.0
        agent_won = (result > 0) == (self.agent_color == "w")
        return 1.0 if agent_won else -1.0
