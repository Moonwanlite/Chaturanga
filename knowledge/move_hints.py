"""
Engine-grounded move suggestions for the RAG assistant.

The language model NEVER chooses moves here. When a question looks like
"what should I play?", the move (or short line of moves) is computed by the
game's own agent (MCTS+PPO, Minimax+PPO, RL policy, Minimax, ...) on a COPY
of the live board, formatted as plain text, and handed to the RAG layer
which (a) shows it verbatim and (b) lets the LLM explain it. Move
correctness therefore never depends on the LLM.
"""

import re

FILES = "abcdefgh"
PIECE_NAMES = {"K": "Raja", "M": "Mantri", "R": "Ratha", "G": "Gaja", "N": "Ashva", "P": "Padati"}
COLOR_NAMES = {"w": "White", "b": "Black"}
MAX_USER_MOVES = 3

# Specific phrases only -- a bare "move" must NOT trigger the engine, so that
# "how does the elephant move?" stays a rules question.
_MOVE_REQUEST_PATTERNS = [
    r"\b(next|best|good|top|opening|first|strongest|winning)\s+(moves?|plays?)\b",
    r"\b(what|which)\s+(move|moves|play)\b",
    r"\bwhat\s+(should|shall|can|do|could)\s+(i|we)\s+(play|do|move)\b",
    r"\bwhat\s+to\s+(play|do)\b",
    r"\bwhat\s+now\b",
    r"\b(suggest|recommend)\w*\b",
    r"\bhints?\b",
    r"\bplay\s+next\b",
    r"\b(set|sequence|series|line)\s+of\s+moves\b",
    r"\bnext\s+(\d+|few|couple|two|three)\s+moves?\b",
]
_MOVE_REQUEST_RE = re.compile("|".join(_MOVE_REQUEST_PATTERNS), re.IGNORECASE)

_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "a couple": 2, "couple": 2, "few": 2}
_COUNT_RE = re.compile(r"\b(\d+|one|two|three|a couple|couple|few)\s+(?:of\s+)?(?:the\s+)?(?:next\s+)?moves?\b", re.I)
_SEQUENCE_RE = re.compile(r"\b(moves|sequence|series|plan|line|set of)\b", re.I)


def is_move_request(query: str) -> bool:
    return bool(_MOVE_REQUEST_RE.search(query or ""))


def requested_plies(query: str) -> int:
    """How many plies (half-moves) of the engine's line to show."""
    user_moves = 1
    m = _COUNT_RE.search(query or "")
    if m:
        tok = m.group(1).lower()
        n = int(tok) if tok.isdigit() else _NUM_WORDS.get(tok, 1)
        user_moves = max(1, min(n, MAX_USER_MOVES))
    elif _SEQUENCE_RE.search(query or ""):
        user_moves = 2
    return 2 * user_moves - 1


def _sq(r, c):
    return f"{FILES[c]}{r + 1}"


def describe_move(state, move) -> str:
    fr, fc, tr, tc, promo = move
    piece = state.board[fr][fc]
    name = PIECE_NAMES[piece[1]]
    text = f"{name} {_sq(fr, fc)} to {_sq(tr, tc)}"
    target = state.board[tr][tc]
    if target is not None:
        text += f", capturing the opponent's {PIECE_NAMES[target[1]]}"
    if promo:
        text += ", promoting to Mantri"
    return text


def compute_line(state, move_fn, plies: int):
    """Simulate `plies` moves on a copy using move_fn(sim_state).
    Returns [(color, description), ...]. The live state is never touched."""
    sim = state.copy()
    line = []
    for _ in range(plies):
        if sim.is_terminal():
            break
        try:
            move = move_fn(sim)
            if move is None:
                break
            color = sim.turn
            desc = describe_move(sim, move)
            sim.push(move)
        except ValueError:
            break
        line.append((color, desc))
    return line


def build_engine_block(state, move_fn, query: str, engine_label: str) -> str:
    header = f"Engine recommendation (computed by {engine_label}, not by the language model):"
    if state is None or move_fn is None:
        return ""
    if state.is_terminal():
        return f"{header}\nThe game is over ({state.end_reason()}); there is no next move."
    line = compute_line(state, move_fn, requested_plies(query))
    if not line:
        return f"{header}\nThe engine could not find a move in this position."
    labels = ["Recommended move", "Expected reply", "Then"]
    rows = []
    for i, (color, desc) in enumerate(line):
        label = labels[min(i, 2)] if i < 3 else "Then"
        rows.append(f"{i + 1}. {COLOR_NAMES[color]} - {label}: {desc}")
    out = header + "\n" + "\n".join(rows)
    if len(line) > 1:
        out += "\n(This line assumes both sides keep following the engine.)"
    return out