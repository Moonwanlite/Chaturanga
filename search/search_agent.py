import os
from search.minimax import get_best_move, print_root_moves

DEFAULT_DEPTH = 3
DEFAULT_CNN_DEPTH = 2   
DEFAULT_PPO_EVAL_DEPTH = 2  
DEFAULT_MCTS_SIMULATIONS = 100

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINT_PATH = os.path.join(_PROJECT_ROOT, "models", "checkpoints", "cnn_bootstrap.pt")
DEFAULT_PPO_PATH = os.path.join(_PROJECT_ROOT, "models", "checkpoints", "ppo_chaturanga")

_cnn_model_cache = {}    # checkpoint path -> loaded CNNEval
_ppo_model_cache = {}    # model path -> loaded MaskablePPO
_mcts_evaluator_cache = {}   # (model path, agent_color) -> evaluator for MCTS


def minimax_agent_move(state, depth: int = DEFAULT_DEPTH, evaluator=None, verbose: bool = False):
    if verbose:
        print_root_moves(state, depth=depth, evaluator=evaluator)
    return get_best_move(state, depth=depth, evaluator=evaluator, verbose=verbose)

def _get_cnn_model(checkpoint_path: str):
    if checkpoint_path not in _cnn_model_cache:
        from models.cnn_eval import load_checkpoint
        _cnn_model_cache[checkpoint_path] = load_checkpoint(checkpoint_path)
    return _cnn_model_cache[checkpoint_path]


def _get_cnn_evaluator(checkpoint_path: str):
    from models.cnn_eval import evaluator_fn
    return evaluator_fn(_get_cnn_model(checkpoint_path))


def cnn_agent_move(state, depth: int = DEFAULT_CNN_DEPTH,
                    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH, verbose: bool = False):
    
    evaluator = _get_cnn_evaluator(checkpoint_path)
    if verbose:
        print_root_moves(state, depth=depth, evaluator=evaluator)
    return get_best_move(state, depth=depth, evaluator=evaluator, verbose=verbose)

def _get_ppo_model(model_path: str):
    if model_path not in _ppo_model_cache:
        from models.ppo_adapter import load_ppo_model
        _ppo_model_cache[model_path] = load_ppo_model(model_path)
    return _ppo_model_cache[model_path]


def _get_ppo_evaluator(model_path: str, agent_color: str = "w"):
    from models.ppo_adapter import ppo_evaluator_fn
    return ppo_evaluator_fn(_get_ppo_model(model_path), agent_color=agent_color)


def _get_ppo_evaluator_for_mcts(model_path: str, agent_color: str = "w"):
    
    key = (model_path, agent_color)
    if key not in _mcts_evaluator_cache:
        _mcts_evaluator_cache[key] = _get_ppo_evaluator(model_path, agent_color=agent_color)
    return _mcts_evaluator_cache[key]


def _get_ppo_policy(model_path: str, agent_color: str = "w"):
    from models.ppo_adapter import ppo_policy_fn
    return ppo_policy_fn(_get_ppo_model(model_path), agent_color=agent_color)


def mcts_agent_move(state, n_simulations: int = DEFAULT_MCTS_SIMULATIONS,
                     model_path: str = DEFAULT_PPO_PATH, agent_color: str = "w", verbose: bool = False):
    from search.mcts import get_best_move as mcts_get_best_move
    evaluator = _get_ppo_evaluator_for_mcts(model_path, agent_color=agent_color)
    policy = _get_ppo_policy(model_path, agent_color=agent_color)
    return mcts_get_best_move(state, n_simulations=n_simulations, evaluator=evaluator, policy=policy, verbose=verbose)


def minimax_ppo_agent_move(state, depth: int = DEFAULT_PPO_EVAL_DEPTH,
                            model_path: str = DEFAULT_PPO_PATH, agent_color: str = "w", verbose: bool = False):
    evaluator = _get_ppo_evaluator(model_path, agent_color=agent_color)
    if verbose:
        print_root_moves(state, depth=depth, evaluator=evaluator)
    return get_best_move(state, depth=depth, evaluator=evaluator, verbose=verbose)


def rl_agent_move(state, model_path: str = DEFAULT_PPO_PATH, verbose: bool = False):
    import numpy as np
    from models.cnn_eval import move_to_action, N_MOVES
    from board.encode import to_tensor

    model = _get_ppo_model(model_path)

    obs = to_tensor(state)
    legal_moves = state.legal_moves()
    move_by_action = {move_to_action(move): move for move in legal_moves}
    mask = np.zeros(N_MOVES, dtype=bool)
    for action in move_by_action:
        mask[action] = True

    action, _states = model.predict(obs, action_masks=mask, deterministic=True)
    action = int(action)
    move = move_by_action.get(action, legal_moves[0])  # fallback should never trigger with correct masking

    if verbose:
        _print_rl_move_probs(model, obs, mask, move_by_action, chosen_action=action)

    return move


def _print_rl_move_probs(model, obs, mask, move_by_action, chosen_action, top_n: int = 5):
    import torch
    import numpy as np
    from stable_baselines3.common.utils import obs_as_tensor
    from search.minimax import move_str

    obs_tensor = obs_as_tensor(np.expand_dims(obs, axis=0), model.policy.device)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor, action_masks=mask)
        probs = distribution.distribution.probs.squeeze(0).cpu().numpy()
        value = model.policy.predict_values(obs_tensor).item()

    ranked = sorted(move_by_action.items(), key=lambda item: probs[item[0]], reverse=True)
    print(f"[rl agent] value estimate: {value:+.3f} (learned scale, roughly -1..+1 like training reward)")
    print(f"[rl agent] top {min(top_n, len(ranked))} move probabilities:")
    for action, move in ranked[:top_n]:
        marker = " <-- chosen" if action == chosen_action else ""
        print(f"    {move_str(move):<8} p={probs[action]:.3f}{marker}")
