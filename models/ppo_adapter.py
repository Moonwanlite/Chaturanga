"""
Adapter that lets a trained PPO policy (rl/train_rl.py's MaskablePPO
checkpoint) serve as the evaluator/policy source for search/mcts.py and
search/minimax.py.

This realizes the "PPO as the last layer feeding search" architecture:
PPO trains first (via rl/train_rl.py, using the same CNN-shaped conv
architecture as models/cnn_eval.py through rl/features_extractor.py), and
its trained network -- not a separately-trained bootstrap CNN -- becomes
what MCTS's rollouts and Minimax's leaf evaluation actually consult.

Perspective convention: PPO's critic was trained against a reward defined
relative to `agent_color` (see rl/env.py's _terminal_reward). If it was
trained with the default agent_color="w", its value output already means
"how good is this for White" -- the same convention used everywhere else
in this project (material_evaluator, CNNEval's value head). Pass
agent_color="b" here only if you trained the checkpoint with agent_color="b".
"""

import numpy as np
import torch

from board.encode import to_tensor
from models.cnn_eval import move_to_action, N_MOVES


def load_ppo_model(model_path: str):
    from sb3_contrib import MaskablePPO
    return MaskablePPO.load(model_path)


def ppo_evaluator_fn(model, agent_color: str = "w"):
    """
    Returns evaluator(state) -> float, White's-perspective value -- the
    same signature search/minimax.py and search/mcts.py already expect
    from models.cnn_eval.evaluator_fn. Values come from PPO's critic
    (value) head instead of a separately-trained CNN's value head.
    """
    def evaluate(state) -> float:
        if state.is_terminal():
            return state.result()  # exact outcome, already White's-perspective +1/-1/0

        obs = to_tensor(state)
        obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.policy.device)
        with torch.no_grad():
            value = model.policy.predict_values(obs_tensor).item()
        return value if agent_color == "w" else -value

    return evaluate


def ppo_policy_fn(model, agent_color: str = "w"):
    """
    Returns policy(state) -> {move: prior_probability}, restricted to and
    renormalized over legal moves -- the same signature
    models.cnn_eval.policy_fn provides. Priors come from PPO's actor
    (policy) head, with action masking applied so illegal moves get zero
    probability before normalization.

    Note on agent_color: PPO's actor was only ever trained to choose moves
    while playing `agent_color` (see rl/env.py -- the environment only
    calls step() on the agent's own turns). If you use this policy for the
    side PPO was NOT trained to play, its move preferences are an
    untested extrapolation, not something it actually learned.
    """
    def policy(state):
        obs = to_tensor(state)
        obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.policy.device)

        legal_moves = state.legal_moves()
        mask = np.zeros(N_MOVES, dtype=bool)
        move_by_action = {}
        for move in legal_moves:
            action = move_to_action(move)
            mask[action] = True
            move_by_action[action] = move

        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_tensor, action_masks=mask)
            probs = distribution.distribution.probs.squeeze(0).cpu().numpy()

        return {move: float(probs[action]) for action, move in move_by_action.items()}

    return policy
