"""
Alpha-beta minimax search with a simple material evaluator.

This is the "dumb but genuinely competent" tier: no CNN, no learned policy --
just classical search over material value. It exists so you have a real
opponent to test against before models/cnn_eval.py exists. Once the CNN is
trained, swap `material_evaluator` for models.cnn_eval.evaluator_fn(...);
the search code itself (alphabeta / get_best_move) doesn't need to change,
since it only calls `evaluator(state) -> float`.

Score convention: positive = good for White, negative = good for Black.
This matches state.result() (+1 White win, -1 Black win), just scaled up.
"""

import time

from board.state import ChaturangaState, BOARD_SIZE

# Rough relative values. Chaturanga has no established theory the way
# chess does, so these are reasoned estimates based on mobility:
#   Padati (pawn): baseline 100
#   Ashva (knight-like, jumps, 8 destinations): strong minor piece
#   Gaja (jumps exactly 2 diag, only 4 destinations, one color complex):
#       weaker than Ashva -- comparable to the historical Alfil
#   Ratha (rook-like, unblocked line control): strongest piece after King
#   Mantri (1 square diagonal only, like a Ferz): weakest non-pawn piece
MATERIAL_VALUES = {
    "P": 100,
    "N": 300,
    "G": 180,
    "R": 500,
    "M": 150,
    "K": 0,  # King's safety is handled by legality + terminal detection, not material
}

CHECKMATE_SCORE = 100_000


def material_evaluator(state: ChaturangaState) -> float:
    """Static evaluation from White's perspective. Only used at leaf nodes
    or on terminal positions -- doesn't itself search deeper."""
    if state.is_terminal():
        # state.result() is +1/-1; scale so it dominates any material score
        # and push it slightly further from 0 the fewer moves were played,
        # so the search prefers faster mates over slower ones.
        return state.result() * CHECKMATE_SCORE

    score = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = state.board[r][c]
            if piece is None:
                continue
            value = MATERIAL_VALUES[piece[1]]
            score += value if piece[0] == "w" else -value
    return score


def _order_moves(state: ChaturangaState, moves):
    """Cheap move ordering: try captures first, to help alpha-beta prune
    earlier. Not essential for correctness, just speed."""
    def is_capture(move):
        _fr, _fc, tr, tc, _promo = move
        return state.board[tr][tc] is not None

    return sorted(moves, key=is_capture, reverse=True)


def _square(row: int, col: int) -> str:
    return f"{'abcdefgh'[col]}{row + 1}"


def move_str(move) -> str:
    fr, fc, tr, tc, promo = move
    s = f"{_square(fr, fc)}{_square(tr, tc)}"
    return s + ("=M" if promo else "")


def alphabeta(state: ChaturangaState, depth: int, alpha: float, beta: float,
              evaluator=None, stats: dict = None):
    """
    Returns (score, best_move, pv) where pv is the list of moves making up
    the predicted best line from this position onward (principal variation).
    `best_move` is None at depth 0 or on terminal nodes.

    `evaluator` is any callable(state) -> float from White's perspective,
    used only at leaf nodes / terminal positions. Defaults to material_evaluator.
    Swap in models.cnn_eval.evaluator_fn(...) here without changing anything else.

    `stats`, if given, is a dict that gets mutated in place with running
    counts: {'nodes': int, 'prunes': int}. Pass the same dict through
    recursive calls to accumulate totals for a single top-level search.
    """
    if evaluator is None:
        evaluator = material_evaluator
    if stats is not None:
        stats["nodes"] += 1

    if depth == 0 or state.is_terminal():
        return evaluator(state), None, []

    maximizing = state.turn == "w"
    moves = _order_moves(state, state.legal_moves())
    best_move = moves[0]
    best_pv = []

    if maximizing:
        value = float("-inf")
        for move in moves:
            child = state.copy()
            child.push(move)
            score, _mv, child_pv = alphabeta(child, depth - 1, alpha, beta, evaluator, stats)
            if score > value:
                value, best_move, best_pv = score, move, child_pv
            alpha = max(alpha, value)
            if alpha >= beta:
                if stats is not None:
                    stats["prunes"] += 1
                break
        return value, best_move, [best_move] + best_pv
    else:
        value = float("inf")
        for move in moves:
            child = state.copy()
            child.push(move)
            score, _mv, child_pv = alphabeta(child, depth - 1, alpha, beta, evaluator, stats)
            if score < value:
                value, best_move, best_pv = score, move, child_pv
            beta = min(beta, value)
            if alpha >= beta:
                if stats is not None:
                    stats["prunes"] += 1
                break
        return value, best_move, [best_move] + best_pv


def get_best_move(state: ChaturangaState, depth: int = 3, evaluator=None, verbose: bool = False):
    """Top-level call: returns the best move found at the given search depth.
    `evaluator` defaults to material_evaluator; pass models.cnn_eval's
    evaluator_fn(...) here to search using the CNN instead.
    With verbose=True, also prints the evaluation, node/prune counts, and
    the predicted line (principal variation) to stdout."""
    stats = {"nodes": 0, "prunes": 0}
    start = time.time()
    score, move, pv = alphabeta(state, depth, float("-inf"), float("inf"), evaluator, stats)
    elapsed = time.time() - start

    if verbose:
        pv_str = " ".join(move_str(m) for m in pv)
        print(f"[minimax] depth={depth}  nodes={stats['nodes']}  prunes={stats['prunes']}  "
              f"time={elapsed:.3f}s")
        print(f"[minimax] eval={score:+.0f} (positive favors White)  chosen={move_str(move)}")
        print(f"[minimax] predicted line: {pv_str}")

    return move


def explain_root_moves(state: ChaturangaState, depth: int = 3, evaluator=None):
    """
    Evaluate every legal move independently to `depth` plies of lookahead,
    each with its own full (-inf, +inf) window, and return
    [(move, score, pv), ...] sorted best-first for the side to move.

    Unlike get_best_move(), scores here are NOT affected by cross-sibling
    alpha-beta pruning, so every candidate move gets an exact, directly
    comparable score. This is purely for inspection -- it does strictly
    more work than get_best_move() and shouldn't be used as the actual
    move-picker in a game loop.
    """
    maximizing = state.turn == "w"
    results = []
    for move in state.legal_moves():
        child = state.copy()
        child.push(move)
        score, _mv, child_pv = alphabeta(child, depth - 1, float("-inf"), float("inf"), evaluator)
        results.append((move, score, [move] + child_pv))
    results.sort(key=lambda item: item[1], reverse=maximizing)
    return results


def print_root_moves(state: ChaturangaState, depth: int = 3, top_n: int = None, evaluator=None):
    """Pretty-print explain_root_moves() output, best move first."""
    results = explain_root_moves(state, depth, evaluator)
    if top_n:
        results = results[:top_n]

    turn_name = "White" if state.turn == "w" else "Black"
    print(f"{turn_name} to move -- ranking {len(results)} candidate move(s) at depth {depth}:")
    for rank, (move, score, pv) in enumerate(results, start=1):
        pv_str = " ".join(move_str(m) for m in pv)
        print(f"  {rank:>2}. {move_str(move):<8} eval={score:+7.0f}   line: {pv_str}")
