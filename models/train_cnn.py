"""
Bootstrap training for models/cnn_eval.py via distillation.

There's no existing Chaturanga game database, so we generate training data
by having the minimax + material_evaluator agent (search/minimax.py) play
itself, and train the CNN to reproduce its judgments:
    value target  = material_evaluator(state) at each position, squashed to [-1, 1]
    policy target = the move minimax actually chose from that position

This is a *bootstrap*, not the end goal -- it teaches the CNN to imitate a
weak, purely-material search. Once that's working end-to-end, replace this
data source with real self-play against the CNN itself (see rl/train_rl.py)
so the network can eventually exceed what generated its own training data.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim

from board.state import ChaturangaState
from board.encode import to_tensor
from search.minimax import get_best_move, material_evaluator
from models.cnn_eval import CNNEval, move_to_action, save_checkpoint

VALUE_SCALE = 1000.0  # roughly matches material_evaluator's typical magnitude


def generate_self_play_data(n_games: int, search_depth: int = 2, max_plies: int = 60):
    """
    Returns a list of (board_tensor, value_target, policy_target_action).
    Plays n_games of minimax-vs-minimax, recording every position reached.
    """
    samples = []
    for game_idx in range(n_games):
        state = ChaturangaState()
        for _ply in range(max_plies):
            if state.is_terminal():
                break

            move = get_best_move(state, depth=search_depth)
            value_target = math.tanh(material_evaluator(state) / VALUE_SCALE)
            samples.append((to_tensor(state), value_target, move_to_action(move)))
            state.push(move)

        print(f"  game {game_idx + 1}/{n_games}: {len(state.move_history)} plies played")
    return samples


def train(model: CNNEval, samples, epochs: int = 10, batch_size: int = 32, lr: float = 1e-3, device: str = "cpu"):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    value_loss_fn = nn.MSELoss()
    policy_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        random.shuffle(samples)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            if not batch:
                continue

            boards = torch.stack([torch.from_numpy(b) for b, _v, _a in batch]).to(device)
            values = torch.tensor([v for _b, v, _a in batch], dtype=torch.float32).unsqueeze(1).to(device)
            actions = torch.tensor([a for _b, _v, a in batch], dtype=torch.long).to(device)

            optimizer.zero_grad()
            policy_logits, pred_values = model(boards)

            v_loss = value_loss_fn(pred_values, values)
            p_loss = policy_loss_fn(policy_logits, actions)
            loss = v_loss + p_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"epoch {epoch + 1}/{epochs}  avg_loss={avg_loss:.4f}")

    return model


def main():
    print("Generating self-play data from minimax + material_evaluator...")
    samples = generate_self_play_data(n_games=20, search_depth=2)
    print(f"Collected {len(samples)} training positions.")

    model = CNNEval()
    print("Training CNN to imitate the minimax agent...")
    train(model, samples, epochs=15)

    checkpoint_path = "models/checkpoints/cnn_bootstrap.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    save_checkpoint(model, checkpoint_path)
    print(f"Saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
