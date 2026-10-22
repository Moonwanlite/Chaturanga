"""
Monte Carlo Tree Search guided by the CNN (models/cnn_eval.py)'s policy
and value heads. This is the project's unified engine: ONE planning
method using ONE learned board representation, rather than three
separate, disconnected agent modes (material-minimax / CNN-minimax /
bare RL policy).

How the pieces the project asked for actually combine here:
    - Minimax/Alpha-Beta's contribution: tree search itself -- looking
      ahead by simulating moves, not just reacting to the current position.
      MCTS is the tree-search method used; classical alpha-beta
      (search/minimax.py) remains available as a comparison baseline and
      as the source of self-play training data for the CNN (models/train_cnn.py).
    - CNN's contribution: BOTH roles at once, replacing the two things
      classic MCTS relies on but this variant doesn't have -- exhaustive
      rollouts and uniform random expansion:
        * value head  -> leaf evaluation (instead of a random rollout to
          the end of the game)
        * policy head -> move priors (instead of expanding every legal
          move with equal, uninformed weight)
    - RL's contribution: PPO (rl/train_rl.py) is what actually trains the
      CNN's weights through self-play, so the evaluation and priors this
      engine relies on keep improving over time rather than staying fixed
      at whatever the bootstrap distillation produced.
    - RAG sits outside this loop entirely (by design): it explains
      rules/strategy in natural language, and never influences move choice.

This is standard AlphaZero-style PUCT search, adapted to a variant with no
existing MCTS engine to imitate and a hand-rolled (not library) tree.
"""

import math
import time

from search.minimax import move_str


class MCTSNode:
    __slots__ = ("state", "parent", "prior", "move", "children", "visit_count", "value_sum")

    def __init__(self, state, parent=None, prior: float = 0.0, move=None):
        self.state = state
        self.parent = parent
        self.prior = prior
        self.move = move            # the move that led from parent into this node
        self.children = {}          # move -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        """Average value from the perspective of the side TO MOVE AT THIS
        NODE (not its parent) -- see the sign-handling note in
        _select_child and run_mcts below; this convention is what makes
        both of those simple to get right."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


def _value_to_move(evaluator, state) -> float:
    """evaluator(state) (e.g. models.cnn_eval.evaluator_fn(...)) returns a
    White's-perspective value; convert it to the perspective of whichever
    side is actually to move at `state`."""
    v_white = evaluator(state)
    return v_white if state.turn == "w" else -v_white


def _terminal_value_to_move(state) -> float:
    """Exact outcome (no evaluator needed) at a genuinely finished game,
    from the perspective of the side to move -- who, under this rule set's
    checkmate/stalemate convention (board/state.py), has necessarily just
    lost (or drawn, if STALEMATE_IS_DRAW is enabled)."""
    result = state.result()  # +1 white win / -1 black win / 0.0 draw
    if result == 0.0:
        return 0.0
    white_won = result > 0
    to_move_won = white_won == (state.turn == "w")
    return 1.0 if to_move_won else -1.0


def _expand(node: MCTSNode, policy_fn):
    """Create a child for every legal move, each seeded with the policy
    network's prior probability (policy_fn already restricts and
    renormalizes this over legal moves only -- see models/cnn_eval.py)."""
    priors = policy_fn(node.state)
    for move, prior in priors.items():
        child_state = node.state.copy()
        child_state.push(move)
        node.children[move] = MCTSNode(child_state, parent=node, prior=prior, move=move)


def _select_child(node: MCTSNode, c_puct: float):
    """
    PUCT selection (as in AlphaZero): balances exploiting moves with good
    average value against exploring moves the policy network favors but
    that haven't been visited much yet.

    Sign handling: child.value() is defined as the average value from the
    perspective of the side to move AT THE CHILD, which is `node`'s
    mover's opponent. So from `node`'s mover's own perspective, a child's
    desirability is the NEGATION of the child's stored value -- exactly
    the same perspective-flip used in minimax/negamax.
    """
    best_score = float("-inf")
    best_move, best_child = None, None
    sqrt_parent_visits = math.sqrt(max(node.visit_count, 1))

    for move, child in node.children.items():
        exploit = -child.value()
        explore = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
        score = exploit + explore
        if score > best_score:
            best_score, best_move, best_child = score, move, child

    return best_move, best_child


def run_mcts(root_state, evaluator, policy_fn, n_simulations: int = 200, c_puct: float = 1.5) -> MCTSNode:
    """
    Runs MCTS from root_state and returns the root MCTSNode (read off
    child.visit_count to pick a move -- see get_best_move below).

    `evaluator`: callable(state) -> float, White's-perspective value.
    `policy_fn`: callable(state) -> {move: prior_probability}, restricted
                 to legal moves and renormalized (see models/cnn_eval.py).
    """
    root = MCTSNode(root_state)
    if root_state.is_terminal():
        return root  # nothing to search from an already-finished game

    _expand(root, policy_fn)

    for _ in range(n_simulations):
        node = root
        path = [node]

        # Selection: walk down existing tree until an unexpanded node or a
        # genuinely terminal position is reached.
        while node.is_expanded and not node.state.is_terminal():
            _move, node = _select_child(node, c_puct)
            path.append(node)

        # Evaluation: exact outcome if terminal, else expand this leaf and
        # use the CNN's value head instead of a random rollout.
        if node.state.is_terminal():
            value = _terminal_value_to_move(node.state)
        else:
            _expand(node, policy_fn)
            value = _value_to_move(evaluator, node.state)

        # Backpropagation: `value` is from the perspective of the side to
        # move at `node`. Each ancestor's mover is the opponent of the
        # level below it, so flip sign at every step up the path.
        for ancestor in reversed(path):
            ancestor.visit_count += 1
            ancestor.value_sum += value
            value = -value

    return root


def get_best_move(state, n_simulations: int = 200, evaluator=None, policy=None,
                   c_puct: float = 1.5, verbose: bool = False):
    """
    Top-level call. If `evaluator`/`policy` aren't given explicitly, both
    default to the trained CNN checkpoint (models/checkpoints/cnn_bootstrap.pt
    via search/search_agent.py's caching, so the model loads once, not
    once per move).
    """
    if evaluator is None or policy is None:
        from search.search_agent import _get_cnn_evaluator_for_mcts, _get_cnn_policy, DEFAULT_CHECKPOINT_PATH
        if evaluator is None:
            evaluator = _get_cnn_evaluator_for_mcts(DEFAULT_CHECKPOINT_PATH)
        if policy is None:
            policy = _get_cnn_policy(DEFAULT_CHECKPOINT_PATH)

    start = time.time()
    root = run_mcts(state, evaluator, policy, n_simulations=n_simulations, c_puct=c_puct)
    elapsed = time.time() - start

    if not root.children:
        return None  # state was already terminal -- no move to make

    best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]

    if verbose:
        print(f"[mcts] simulations={n_simulations}  time={elapsed:.3f}s")
        ranked = sorted(root.children.items(), key=lambda item: item[1].visit_count, reverse=True)
        print(f"[mcts] top {min(5, len(ranked))} moves by visit count:")
        for move, child in ranked[:5]:
            marker = " <-- chosen" if move == best_move else ""
            print(f"    {move_str(move):<8} visits={child.visit_count:<4} "
                  f"avg_value={child.value():+.3f}  prior={child.prior:.3f}{marker}")

    return best_move
