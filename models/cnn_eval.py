"""
CNN board evaluator: policy + value heads over the 12-plane board encoding
from board/encode.py.

Drop-in replacement for search.minimax.material_evaluator -- the search
code (alphabeta/get_best_move/etc.) takes any evaluator(state) -> float,
so nothing in search/minimax.py needs to change to use this instead.

Two heads:
    value_head:  scalar in [-1, 1], tanh-squashed, positive = good for White.
                 This is what evaluator_fn() exposes to the classical search.
    policy_head: a distribution over a fixed 4096-way (from_square * 64 +
                 to_square) move space, ignoring promotion piece (promotion
                 is auto-resolved to Mantri by the rules engine, so it's not
                 part of the action encoding). Used later by rl/env.py and
                 by MCTS as a move-priors source -- not used yet by minimax.
"""

import torch
import torch.nn as nn

from board.encode import to_tensor, PIECE_PLANES
from board.state import BOARD_SIZE

N_PLANES = len(PIECE_PLANES)   # 12
N_SQUARES = BOARD_SIZE * BOARD_SIZE  # 64
N_MOVES = N_SQUARES * N_SQUARES      # 4096, from_sq * 64 + to_sq


def build_conv_backbone(in_planes: int = N_PLANES, channels: int = 64) -> nn.Sequential:
    """
    The convolutional board-representation stack, factored out so it can be
    reused verbatim by rl/features_extractor.py -- the RL policy should
    learn board features through the same kind of architecture as this
    evaluator, not SB3's default flatten-to-MLP (which discards all spatial
    structure: adjacency, diagonals, lines of attack).

    Kept as a plain nn.Sequential with this exact layer list (not wrapped in
    a further submodule) so CNNEval's state_dict keys are unchanged --
    existing trained checkpoints (models/checkpoints/cnn_bootstrap.pt) still
    load correctly after this refactor.
    """
    return nn.Sequential(
        nn.Conv2d(in_planes, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
        nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
        nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
    )


class CNNEval(nn.Module):
    def __init__(self, in_planes: int = N_PLANES, channels: int = 64, n_moves: int = N_MOVES):
        super().__init__()
        self.conv = build_conv_backbone(in_planes, channels)
        flat = channels * BOARD_SIZE * BOARD_SIZE

        self.policy_head = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(),
            nn.Linear(256, n_moves),
        )
        self.value_head = nn.Sequential(
            nn.Linear(flat, 128), nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        """x: (batch, 12, 8, 8) -> (policy_logits (batch, 4096), value (batch, 1))"""
        features = self.conv(x)
        features = features.flatten(1)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value


def move_to_action(move) -> int:
    """(fr, fc, tr, tc, promo) -> flat index in [0, 4096). Promotion is not
    encoded -- the rules engine resolves it automatically to Mantri."""
    fr, fc, tr, tc, _promo = move
    from_sq = fr * BOARD_SIZE + fc
    to_sq = tr * BOARD_SIZE + tc
    return from_sq * N_SQUARES + to_sq


def action_to_squares(action: int):
    """flat index -> ((fr, fc), (tr, tc)). Doesn't know about promotion --
    callers should match against actual legal moves to fill that in."""
    from_sq, to_sq = divmod(action, N_SQUARES)
    fr, fc = divmod(from_sq, BOARD_SIZE)
    tr, tc = divmod(to_sq, BOARD_SIZE)
    return (fr, fc), (tr, tc)


def evaluator_fn(model: CNNEval, device: str = "cpu", checkmate_score: float = 100_000.0, scale: float = 1000.0):
    """
    Returns a function(state) -> float suitable for search/minimax.py's
    `evaluator` parameter. Terminal positions are still scored via
    state.result() (scaled up) rather than the network, since a trained
    value head is a heuristic estimate and the network was never trained to
    see truly terminal boards as anything special -- the exact outcome is
    already known at that point, so there's no reason to guess.

    `scale` and `checkmate_score` control the output magnitude:
        - Default (scale=1000, checkmate_score=100000): matches
          material_evaluator's typical magnitude (hundreds), for use with
          search/minimax.py's alpha-beta, which only cares about relative
          ordering between branches, not absolute scale.
        - For search/mcts.py: pass scale=1.0, checkmate_score=1.0 instead.
          PUCT selection compares this value against priors in [0,1] with
          a small constant (c_puct ~1-2) -- feeding it hundreds-scale
          values (the default) makes a single leaf evaluation completely
          dominate the exploration term, and the tree stops exploring
          anything else. This is not a hypothetical concern: it's exactly
          what happened in initial testing before this parameter existed.
    """
    model.eval()

    def evaluate(state) -> float:
        if state.is_terminal():
            return state.result() * checkmate_score

        tensor = torch.from_numpy(to_tensor(state)).unsqueeze(0).to(device)
        with torch.no_grad():
            _policy_logits, value = model(tensor)
        return value.item() * scale

    return evaluate


def policy_fn(model: CNNEval, device: str = "cpu"):
    """
    Returns a function(state) -> dict[move, prior_probability], restricted
    to and renormalized over the state's actual legal moves. Used by MCTS
    (search/mcts.py) as move priors instead of uniform random expansion --
    not used by minimax, which doesn't consult a policy.
    """
    model.eval()

    def policy(state):
        tensor = torch.from_numpy(to_tensor(state)).unsqueeze(0).to(device)
        with torch.no_grad():
            policy_logits, _value = model(tensor)
        probs = torch.softmax(policy_logits, dim=-1).squeeze(0)

        legal_moves = state.legal_moves()
        legal_probs = {move: probs[move_to_action(move)].item() for move in legal_moves}
        total = sum(legal_probs.values()) or 1e-8
        return {move: p / total for move, p in legal_probs.items()}

    return policy


def save_checkpoint(model: CNNEval, path: str):
    torch.save(model.state_dict(), path)


def load_checkpoint(path: str, device: str = "cpu") -> CNNEval:
    model = CNNEval()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model
