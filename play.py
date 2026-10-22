"""
Minimal playable CLI loop: human vs a placeholder agent, rendered with the
external visualiser. Swap `random_agent_move` for search/search_agent.py
once minimax/MCTS + CNN eval are wired in.
"""

import argparse
import random
import re

from board.state import ChaturangaState, BOARD_SIZE
from board.viz_adapter import render
from search.search_agent import (
    minimax_agent_move, cnn_agent_move, rl_agent_move, mcts_agent_move,
    DEFAULT_DEPTH, DEFAULT_CNN_DEPTH, DEFAULT_CHECKPOINT_PATH, DEFAULT_PPO_PATH,
    DEFAULT_MCTS_SIMULATIONS,
)

FILES = "abcdefgh"
PIECE_LETTERS = {"K", "M", "R", "G", "N"}  # no letter prefix = Padati (pawn)

# Optional piece letter, optional disambiguating file/rank, optional 'x' for
# capture (ignored), destination square. e.g. 'e3', 'Nf3', 'Nbd7', 'Rxe5'
_SAN_RE = re.compile(
    r"^\s*([KMRGN])?([a-h])?([1-8])?x?([a-h])([1-8])\s*$", re.IGNORECASE
)

_COORD_MOVE_RE = re.compile(r"^\s*([a-h])([1-8])\s*-?\s*([a-h])([1-8])\s*$", re.IGNORECASE)


def square_to_coords(file_char: str, rank_char: str):
    """'a' '1' -> (row, col). Row 0 = rank 1 = White's back rank."""
    col = FILES.index(file_char.lower())
    row = int(rank_char) - 1
    return row, col


def coords_to_square(row: int, col: int) -> str:
    return f"{FILES[col]}{row + 1}"


def print_labeled_board(state: ChaturangaState):
    """Reference board with file/rank labels, alongside the pretty Unicode one."""
    print(render(state))
    print()
    for r in range(BOARD_SIZE - 1, -1, -1):
        row_str = " ".join(
            (state.board[r][c] or "--") for c in range(BOARD_SIZE)
        )
        print(f"{r + 1}  {row_str}")
    print("   " + "  ".join(FILES))


def random_agent_move(state: ChaturangaState):
    return random.choice(state.legal_moves())


def parse_human_move(state: ChaturangaState, text: str):
    """
    Accepts SAN-lite notation:
        'e3'      -> Padati (pawn) moves to e3, no letter needed
        'Nf3'     -> Ashva moves to f3
        'Rd4'     -> Ratha moves to d4
        'Gc4'     -> Gaja moves to c4
        'Mb5'     -> Mantri moves to b5
        'Ke2'     -> Raja moves to e2
        'Nbd7'    -> disambiguated: the Ashva starting on the b-file
        'Rxe5'    -> 'x' for captures is accepted and ignored
    Also still accepts full coordinate form 'e2e4' / 'e2-e4' for convenience
    (a pawn on e2 moving to e4 is just the disambiguated SAN form anyway).

    Returns a legal Move tuple, a list of ambiguous candidate Move tuples
    (if more than one piece could make the described move), or None if
    unparseable/illegal.
    """
    match = _SAN_RE.match(text)
    if not match:
        return None

    piece_letter, dis_file, dis_rank, to_file, to_rank = match.groups()
    piece_code = piece_letter.upper() if piece_letter else "P"
    tr, tc = square_to_coords(to_file, to_rank)

    candidates = []
    for move in state.legal_moves():
        fr, fc, mtr, mtc, _promo = move
        if (mtr, mtc) != (tr, tc):
            continue
        moving_piece = state.board[fr][fc]
        if moving_piece[1] != piece_code:
            continue
        if dis_file and FILES[fc] != dis_file.lower():
            continue
        if dis_rank and str(fr + 1) != dis_rank:
            continue
        candidates.append(move)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return candidates  # ambiguous -- caller should ask for disambiguation
    return None


def main():
    parser = argparse.ArgumentParser(description="Play Chaturanga against an agent.")
    parser.add_argument("--agent", choices=["mcts", "material", "cnn", "rl"], default="mcts",
                         help="mcts (default): the unified engine -- MCTS guided by the trained CNN's "
                              "policy/value heads. material/cnn/rl: comparison baselines -- see README.md.")
    parser.add_argument("--simulations", type=int, default=DEFAULT_MCTS_SIMULATIONS,
                         help=f"MCTS simulation count (default: {DEFAULT_MCTS_SIMULATIONS}), only used with --agent mcts")
    parser.add_argument("--depth", type=int, default=None,
                         help=f"search depth (default: {DEFAULT_DEPTH} for material, {DEFAULT_CNN_DEPTH} for cnn). "
                              "Only used with --agent material/cnn.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH,
                         help="path to the trained CNN checkpoint, used with --agent mcts or cnn")
    parser.add_argument("--model-path", type=str, default=DEFAULT_PPO_PATH,
                         help="path to the trained PPO checkpoint, only used with --agent rl")
    parser.add_argument("--verbose", action="store_true",
                         help="show the agent's reasoning before each move")
    args = parser.parse_args()

    if args.agent == "mcts":
        agent_move_fn = lambda state, verbose=False: mcts_agent_move(
            state, n_simulations=args.simulations, checkpoint_path=args.checkpoint, verbose=verbose
        )
    elif args.agent == "cnn":
        depth = args.depth if args.depth is not None else DEFAULT_CNN_DEPTH
        agent_move_fn = lambda state, verbose=False: cnn_agent_move(
            state, depth=depth, checkpoint_path=args.checkpoint, verbose=verbose
        )
    elif args.agent == "rl":
        agent_move_fn = lambda state, verbose=False: rl_agent_move(state, model_path=args.model_path, verbose=verbose)
    else:
        depth = args.depth if args.depth is not None else DEFAULT_DEPTH
        agent_move_fn = lambda state, verbose=False: minimax_agent_move(state, depth=depth, verbose=verbose)

    state = ChaturangaState()
    human_color = "w"

    print("Rank 1 = White's back rank, rank 8 = Black's.")
    print("Enter moves in SAN-lite: pawn 'e3', or piece letter + dest, e.g. Nf3, Rd4, Gc4, Mb5, Ke2.")
    print("Use K/M/R/G/N for Raja/Mantri/Ratha/Gaja/Ashva. No letter = Padati (pawn).")
    print()

    while not state.is_terminal():
        print_labeled_board(state)
        print()

        if state.turn == human_color:
            move = None
            while move is None:
                text = input("Your move: ")
                result = parse_human_move(state, text)
                if result is None:
                    print("Illegal move, try again.")
                elif isinstance(result, list):
                    options = ", ".join(
                        f"{coords_to_square(m[0], m[1])}{coords_to_square(m[2], m[3])}"
                        for m in result
                    )
                    print(f"Ambiguous -- could be: {options}. Add a file/rank hint (e.g. Nbd7) or use full coords.")
                else:
                    move = result
        else:
            move = agent_move_fn(state, verbose=args.verbose)
            print(f"Agent plays: {coords_to_square(move[0], move[1])}{coords_to_square(move[2], move[3])}")

        state.push(move)
        print()

    print_labeled_board(state)
    reason = state.end_reason()
    result = state.result()
    if result == 0.0:
        print(f"\nGame over. Draw by {reason}.")
    else:
        winner = "White" if result > 0 else "Black"
        print(f"\nGame over. {winner} wins by {reason}.")


if __name__ == "__main__":
    main()