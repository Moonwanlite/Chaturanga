
from chaturanga.chessboard import Chessboard
from board.state import ChaturangaState, BOARD_SIZE

# Our piece letter -> standard chess piece letter, for rendering only.
_RENDER_PROXY = {
    "K": "K",  # Raja   -> King
    "M": "Q",  # Mantri -> Queen glyph
    "R": "R",  # Ratha  -> Rook
    "G": "B",  # Gaja   -> Bishop glyph
    "N": "N",  # Ashva  -> Knight
    "P": "P",  # Padati -> Pawn
}


def _to_fen_placement(state: ChaturangaState) -> str:
   
    ranks = []
    for r in range(BOARD_SIZE - 1, -1, -1):
        rank_str = ""
        empty_run = 0
        for c in range(BOARD_SIZE):
            square = state.board[r][c]
            if square is None:
                empty_run += 1
                continue
            if empty_run:
                rank_str += str(empty_run)
                empty_run = 0
            color, piece = square[0], square[1]
            letter = _RENDER_PROXY[piece]
            rank_str += letter if color == "w" else letter.lower()
        if empty_run:
            rank_str += str(empty_run)
        ranks.append(rank_str)
    return "/".join(ranks)


def _to_fen(state: ChaturangaState) -> str:
    placement = _to_fen_placement(state)
    active_color = state.turn  # 'w' or 'b'
    # Castling / en-passant are meaningless in our rule set; keep placeholders
    # so the external class's FEN parser doesn't choke.
    return f"{placement} {active_color} - - 0 1"


def render(state: ChaturangaState) -> str:
    """Return a Unicode pretty-printed board string for the given state."""
    fen = _to_fen(state)
    # pretty_print=True is what triggers the Unicode glyph rendering in __str__.
    # We only ever read str(cb) -- no rules/engine methods are called.
    cb = Chessboard(fen=fen, pretty_print=True)
    return str(cb)


def print_board(state: ChaturangaState) -> None:
    print(render(state))
