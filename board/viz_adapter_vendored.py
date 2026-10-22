
from board.state import ChaturangaState, BOARD_SIZE

# Copied verbatim from chaturanga.chessboard.Chessboard.PRETTY_SYMBOLS,
# keyed by our own piece letters instead of theirs.
_PRETTY_SYMBOLS = {
    "K": "\u2654", "M": "\u2655", "R": "\u2656", "G": "\u2657", "N": "\u2658", "P": "\u2659",
    "k": "\u265A", "m": "\u265B", "r": "\u265C", "g": "\u265D", "n": "\u265E", "p": "\u265F",
}


def render(state: ChaturangaState) -> str:
    """Return a Unicode pretty-printed board string for the given state."""
    rows = []
    # Row 7 (Black's back rank) printed first, matching a normal board view.
    for r in range(BOARD_SIZE - 1, -1, -1):
        row_chars = []
        for c in range(BOARD_SIZE):
            square = state.board[r][c]
            if square is None:
                row_chars.append(" ")
                continue
            color, piece = square[0], square[1]
            key = piece if color == "w" else piece.lower()
            row_chars.append(_PRETTY_SYMBOLS[key])
        rows.append("".join(row_chars))
    return "\n".join(rows)


def print_board(state: ChaturangaState) -> None:
    print(render(state))
