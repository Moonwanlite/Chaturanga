"""
Board encoding utilities.
    to_tensor: 12-plane 8x8 tensor for the CNN evaluator (models/cnn_eval.py)
    to_text:   plain-text board for RAG prompts (knowledge/rag.py)
"""

import numpy as np
from board.state import ChaturangaState, BOARD_SIZE

# One plane per (color, piece) combination.
PIECE_PLANES = [
    "wK", "wM", "wR", "wG", "wN", "wP",
    "bK", "bM", "bR", "bG", "bN", "bP",
]
PLANE_INDEX = {code: i for i, code in enumerate(PIECE_PLANES)}


def to_tensor(state: ChaturangaState) -> np.ndarray:
    """(12, 8, 8) float32 tensor, one binary plane per piece type/color."""
    tensor = np.zeros((len(PIECE_PLANES), BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = state.board[r][c]
            if piece is not None:
                tensor[PLANE_INDEX[piece], r, c] = 1.0
    return tensor


def to_text(state: ChaturangaState) -> str:
    """Human-readable board for RAG prompts. Row 7 (Black side) printed first."""
    rows = []
    for r in range(BOARD_SIZE - 1, -1, -1):
        row_str = " ".join(state.board[r][c] or "--" for c in range(BOARD_SIZE))
        rows.append(f"{r} {row_str}")
    rows.append("  " + " ".join(str(c) for c in range(BOARD_SIZE)))
    rows.append(f"turn to move: {'White' if state.turn == 'w' else 'Black'}")
    return "\n".join(rows)
