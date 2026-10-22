"""
Custom feature extractor for the RL policy (rl/train_rl.py), so PPO/DQN
actually represents the board through the same kind of CNN architecture as
models/cnn_eval.py -- not stable-baselines3's default FlattenExtractor,
which reduces the 12x8x8 board to a flat 768-length vector and discards all
spatial structure (adjacency, diagonals, lines of attack) before the policy
ever sees it. That default is why, without this file, the "CNN-based board
representation" from the project's design was only true for the classical
search path (cnn_agent_move) and not for the RL agent at all.

Usage in rl/train_rl.py:
    policy_kwargs = dict(
        features_extractor_class=ChaturangaCNNExtractor,
        features_extractor_kwargs=dict(channels=64, pretrained_cnn_path=<path or None>),
    )
    model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs)

("MlpPolicy" here is just SB3's name for the generic ActorCriticPolicy
class -- passing features_extractor_class overrides how it actually reads
the observation, so this is the standard way to plug in a custom CNN
backbone in SB3/sb3-contrib, not a contradiction.)
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from models.cnn_eval import build_conv_backbone


class ChaturangaCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, channels: int = 64, pretrained_cnn_path: str = None):
        in_planes, height, width = observation_space.shape
        features_dim = channels * height * width
        super().__init__(observation_space, features_dim=features_dim)

        self.conv = build_conv_backbone(in_planes=in_planes, channels=channels)

        if pretrained_cnn_path:
            self._load_pretrained_conv_weights(pretrained_cnn_path)

    def _load_pretrained_conv_weights(self, path: str):
        """
        Warm-starts this extractor's conv layers from a trained CNNEval
        checkpoint (models/train_cnn.py's output) -- so RL training builds
        on top of whatever board representation the bootstrap CNN already
        learned, instead of starting from random weights. Only the conv.*
        keys are used; CNNEval's policy_head/value_head are irrelevant here
        since SB3 attaches its own actor/critic heads on top of this
        extractor's output.
        """
        full_state = torch.load(path, map_location="cpu")
        conv_state = {
            key[len("conv."):]: value
            for key, value in full_state.items()
            if key.startswith("conv.")
        }
        missing, unexpected = self.conv.load_state_dict(conv_state, strict=True)
        print(f"[ChaturangaCNNExtractor] loaded pretrained conv weights from {path}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.conv(observations)
        return features.flatten(1)
