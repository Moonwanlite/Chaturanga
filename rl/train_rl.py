"""
Train an RL policy to play Chaturanga using MaskablePPO (sb3_contrib),
against a configurable opponent (default: random; swap in the minimax
agent for a stronger sparring partner once random games are learned from).

Uses MaskablePPO rather than plain PPO because the action space is a fixed
Discrete(4096) but only ~15-30 actions are legal in any given position --
without masking, the policy would need to stumble onto a legal move by
chance to ever get useful reward signal.

Run (either works):
    python rl/train_rl.py
    python -m rl.train_rl
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from rl.env import ChaturangaEnv, random_opponent
from rl.features_extractor import ChaturangaCNNExtractor
from search.search_agent import minimax_agent_move, DEFAULT_CHECKPOINT_PATH as DEFAULT_CNN_EVAL_CHECKPOINT


def mask_fn(env):
    return env.action_masks()


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SAVE_PATH = os.path.join(_PROJECT_ROOT, "models", "checkpoints", "ppo_chaturanga")


def make_env(opponent_move_fn=None, agent_color: str = "w", max_plies: int = 200):
    env = ChaturangaEnv(opponent_move_fn=opponent_move_fn, agent_color=agent_color, max_plies=max_plies)
    return ActionMasker(env, mask_fn)


def train(total_timesteps: int = 100_000, opponent_move_fn=None, save_path: str = None,
          max_plies: int = 200, load_path: str = None,
          use_cnn_features: bool = True, cnn_channels: int = 64, pretrained_cnn_path: str = None):
    """
    If `load_path` is given and exists, continues training an existing
    checkpoint (warm start) instead of creating a fresh model -- so runs
    accumulate progress across multiple invocations rather than starting
    over from scratch every time. `total_timesteps` is how many ADDITIONAL
    steps to train, either way.

    use_cnn_features=True (the default) makes the policy represent the
    board through the same conv architecture as models/cnn_eval.py
    (see rl/features_extractor.py), instead of SB3's default flatten+MLP.
    IMPORTANT: this is a different network architecture than earlier
    checkpoints trained before this option existed -- you cannot warm-start
    (--continue-from) across that boundary; see the error raised below if
    you try. pretrained_cnn_path optionally transfers the learned conv
    weights from a trained CNNEval checkpoint (e.g. models/train_cnn.py's
    output) into the RL policy's board representation at initialization,
    rather than starting those layers from random weights.
    """
    save_path = save_path or DEFAULT_SAVE_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    env = make_env(opponent_move_fn=opponent_move_fn, max_plies=max_plies)

    policy_kwargs = {}
    if use_cnn_features:
        policy_kwargs = dict(
            features_extractor_class=ChaturangaCNNExtractor,
            features_extractor_kwargs=dict(channels=cnn_channels, pretrained_cnn_path=pretrained_cnn_path),
        )

    if load_path and os.path.exists(load_path if load_path.endswith(".zip") else load_path + ".zip"):
        print(f"Continuing training from existing checkpoint: {load_path}")
        model = MaskablePPO.load(load_path, env=env)
        # IMPORTANT: SB3's .load() always reconstructs the EXACT architecture the
        # checkpoint was originally saved with -- it ignores policy_kwargs/
        # use_cnn_features entirely on this path (those only apply when building a
        # fresh model, below). Silently continuing under a different architecture
        # than requested would be confusing, so check and say so explicitly.
        actually_uses_cnn = isinstance(model.policy.features_extractor, ChaturangaCNNExtractor)
        if actually_uses_cnn != use_cnn_features:
            print(f"[Note] use_cnn_features={use_cnn_features} was requested, but this checkpoint was "
                  f"originally trained with {'CNN' if actually_uses_cnn else 'flatten+MLP'} features -- "
                  f"continuing training PRESERVES that original architecture regardless of this flag. "
                  f"use_cnn_features/pretrained_cnn_path only take effect when starting a fresh model "
                  f"(no --continue-from, or a --save-path with no existing checkpoint).")
    else:
        if load_path:
            print(f"No checkpoint found at {load_path} -- starting fresh instead.")
        if use_cnn_features:
            print(f"Using CNN-based board representation (ChaturangaCNNExtractor, channels={cnn_channels})"
                  + (f", initialized from {pretrained_cnn_path}" if pretrained_cnn_path else ", randomly initialized"))
        else:
            print("Using SB3's default flatten+MLP board representation (use_cnn_features=False)")
        model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=(load_path is None))
    model.save(save_path)
    print(f"Saved to {save_path}.zip")
    return model


def evaluate(model, n_episodes: int = 20, opponent_move_fn=None, max_plies: int = 200):
    """
    Quick win/loss/draw tally against a fixed opponent, no training.

    Only episodes that reach a true terminal state (terminated=True --
    checkmate, stalemate, or an actual draw) are classified as win/loss/draw.
    Episodes that hit the ply cap (truncated=True) are counted separately as
    "unfinished" -- their final reward is just a small leftover material
    shaping value from the last move, NOT a real outcome, and treating its
    sign as a win/loss/draw would silently misrepresent how often games are
    actually being decided versus just running out the clock.
    """
    env = make_env(opponent_move_fn=opponent_move_fn, max_plies=max_plies)
    wins = losses = draws = unfinished = 0

    for _ in range(n_episodes):
        obs, _info = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action_masks = env.action_masks()
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(int(action))

        if truncated and not terminated:
            unfinished += 1
        elif reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    print(f"Results over {n_episodes} games: {wins} wins, {losses} losses, "
          f"{draws} draws, {unfinished} unfinished (hit the ply cap)")
    return wins, losses, draws, unfinished


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the Chaturanga RL agent with MaskablePPO.")
    parser.add_argument("--timesteps", type=int, default=100_000,
                         help="how many ADDITIONAL timesteps to train this run (default: 100000)")
    parser.add_argument("--max-plies", type=int, default=200,
                         help="episode length cap in plies (default: 200) -- lower this for faster, "
                              "more frequent feedback while iterating, at the cost of not seeing longer games")
    parser.add_argument("--continue-from", type=str, default=None,
                         help="path to an existing checkpoint to continue training from (warm start). "
                              "If omitted, starts a fresh model.")
    parser.add_argument("--save-path", type=str, default=None,
                         help=f"where to save the result (default: {DEFAULT_SAVE_PATH})")
    parser.add_argument("--opponent", choices=["random", "minimax"], default="random",
                         help="sparring partner for training (default: random). minimax is much slower per "
                              "step (roughly 4x at depth 1, 10x at depth 2) but punishes bad moves instead of "
                              "letting them slide, giving a stronger learning signal once random is no longer useful.")
    parser.add_argument("--opponent-depth", type=int, default=1,
                         help="minimax search depth for the opponent, only used with --opponent minimax "
                              "(default: 1 -- catches free captures without making training too slow)")
    parser.add_argument("--eval-episodes", type=int, default=20, help="episodes to evaluate after training")
    parser.add_argument("--use-cnn-features", action=argparse.BooleanOptionalAction, default=True,
                         help="represent the board with the same CNN architecture as models/cnn_eval.py "
                              "(default: on). Use --no-use-cnn-features to fall back to SB3's default "
                              "flatten+MLP -- e.g. to continue an older checkpoint trained before this existed.")
    parser.add_argument("--cnn-channels", type=int, default=64, help="conv channel width, only used with --use-cnn-features")
    parser.add_argument("--pretrained-cnn-path", type=str,
                         default=(DEFAULT_CNN_EVAL_CHECKPOINT if os.path.exists(DEFAULT_CNN_EVAL_CHECKPOINT) else None),
                         help="warm-start the RL policy's board representation from a trained CNNEval "
                              f"checkpoint (default: {DEFAULT_CNN_EVAL_CHECKPOINT} if it exists, else random init). "
                              "Pass an empty string '' to force random init even if that default exists.")
    args = parser.parse_args()
    if args.pretrained_cnn_path == "":
        args.pretrained_cnn_path = None

    if args.opponent == "minimax":
        opponent_move_fn = lambda s: minimax_agent_move(s, depth=args.opponent_depth)
        opponent_label = f"minimax (depth={args.opponent_depth})"
    else:
        opponent_move_fn = random_opponent
        opponent_label = "random"

    save_path = args.save_path or DEFAULT_SAVE_PATH
    # Continuing in-place by default: if you don't specify --continue-from
    # explicitly but a checkpoint already exists at the save path, ask
    # rather than silently overwriting hours of prior training.
    load_path = args.continue_from
    if load_path is None and os.path.exists(save_path + ".zip"):
        answer = input(
            f"Found an existing checkpoint at {save_path}.zip -- "
            f"continue training it? [y/N]: "
        ).strip().lower()
        if answer == "y":
            load_path = save_path

    print(f"Training for {args.timesteps} additional timesteps against {opponent_label} opponent "
          f"({'continuing from ' + load_path if load_path else 'fresh model'}), max_plies={args.max_plies}...")
    model = train(
        total_timesteps=args.timesteps,
        opponent_move_fn=opponent_move_fn,
        save_path=save_path,
        max_plies=args.max_plies,
        load_path=load_path,
        use_cnn_features=args.use_cnn_features,
        cnn_channels=args.cnn_channels,
        pretrained_cnn_path=args.pretrained_cnn_path,
    )

    print(f"\nEvaluating against {opponent_label} opponent ({args.eval_episodes} episodes)...")
    evaluate(model, n_episodes=args.eval_episodes, opponent_move_fn=opponent_move_fn, max_plies=args.max_plies)

    # Natural next step once the policy reliably beats one opponent: retrain
    # or fine-tune against a tougher one, e.g.:
    #   python rl/train_rl.py --continue-from models/checkpoints/ppo_chaturanga \
    #       --opponent minimax --opponent-depth 1 --timesteps 200000
