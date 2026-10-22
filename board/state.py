"""
Chaturanga board state and rules engine.

Board representation:
    8x8 grid, indexed [row][col], row 0 = White's back rank, row 7 = Black's back rank.
    Each square is either None or a 2-char string: color + piece code.
        color: 'w' or 'b'
        piece: 'K' Raja, 'M' Mantri, 'R' Ratha, 'G' Gaja, 'N' Ashva, 'P' Padati

Rules implemented (see README for sources / disputed points):
    - Raja:   1 square, any direction
    - Mantri: 1 square, diagonal only
    - Ratha:  any distance, orthogonal, blocked by pieces
    - Gaja:   exactly 2 squares diagonal, JUMPS the intervening square
    - Ashva:  knight move, jumps
    - Padati: 1 square forward only, captures 1 square diagonal-forward,
              no double step, no en passant
    - No castling
    - Pawn promotes only to Mantri, on reaching the far rank
    - A player with NO legal moves LOSES (covers checkmate and historical
      stalemate rule -- stalemate is a loss here, not a draw)

A `Move` is (from_row, from_col, to_row, to_col, promotion or None).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import copy

Move = Tuple[int, int, int, int, Optional[str]]

BOARD_SIZE = 8
OTHER = {"w": "b", "b": "w"}

# Historical Chaturanga: a player with NO legal moves simply loses, whether
# or not they're in check (i.e. no draws at all in this variant). Set this
# to True if you'd rather use the modern chess convention instead, where
# stalemate (no legal moves, not in check) is a draw and only checkmate
# is a loss.
STALEMATE_IS_DRAW = False


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


class ChaturangaState:
    def __init__(self):
        self.board = self._initial_board()
        self.turn = "w"
        self.move_history: List[Move] = []
        self.halfmove_clock = 0  # for optional draw-by-inactivity rules if you add them later

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _initial_board(self):
        board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]

        back_rank = ["R", "N", "G", "M", "K", "G", "N", "R"]
        for col, piece in enumerate(back_rank):
            board[0][col] = "w" + piece
            board[7][col] = "b" + piece

        for col in range(BOARD_SIZE):
            board[1][col] = "wP"
            board[6][col] = "bP"

        return board

    def copy(self) -> "ChaturangaState":
        new = ChaturangaState.__new__(ChaturangaState)
        new.board = [row[:] for row in self.board]
        new.turn = self.turn
        new.move_history = self.move_history[:]
        new.halfmove_clock = self.halfmove_clock
        return new

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def legal_moves(self) -> List[Move]:
        """All moves for the side to move that do not leave own Raja in check."""
        pseudo = self._pseudo_legal_moves(self.turn)
        legal = []
        for move in pseudo:
            trial = self.copy()
            trial._apply(move)
            if not trial._in_check(self.turn):
                legal.append(move)
        return legal

    def push(self, move: Move):
        if move not in self.legal_moves():
            raise ValueError(f"Illegal move: {move}")
        self._apply(move)
        self.turn = OTHER[self.turn]

    def is_terminal(self) -> bool:
        return len(self.legal_moves()) == 0

    def result(self) -> float:
        """
        Call only when is_terminal() is True.
        Returns +1 if White has won, -1 if Black has won, 0.0 for a draw
        (only possible if STALEMATE_IS_DRAW is set to True at the top of
        this file -- by default this variant has no draws, matching the
        historical rule that a player with no legal moves simply loses,
        whether or not they were in check).
        """
        assert self.is_terminal()
        if STALEMATE_IS_DRAW and not self.is_check():
            return 0.0
        # side to move has no legal moves -> side to move loses
        return -1.0 if self.turn == "w" else 1.0

    def end_reason(self) -> str:
        """
        Call only when is_terminal() is True. Returns 'checkmate' if the
        side to move has no legal moves AND is in check, otherwise
        'stalemate' -- useful for display, since result() alone doesn't
        tell you which one happened (under the default rule, both are a
        loss for the side to move; see STALEMATE_IS_DRAW above).
        """
        assert self.is_terminal()
        return "checkmate" if self.is_check() else "stalemate"

    def is_check(self) -> bool:
        return self._in_check(self.turn)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _apply(self, move: Move):
        fr, fc, tr, tc, promo = move
        piece = self.board[fr][fc]
        self.board[fr][fc] = None
        if promo:
            self.board[tr][tc] = piece[0] + promo
        else:
            self.board[tr][tc] = piece
        self.move_history.append(move)

    def _piece_at(self, r, c):
        return self.board[r][c]

    def _in_check(self, color: str) -> bool:
        king_pos = None
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = self.board[r][c]
                if p == color + "K":
                    king_pos = (r, c)
                    break
            if king_pos:
                break
        if king_pos is None:
            # King already captured -- treat as being in check (game should have ended)
            return True

        enemy_moves = self._pseudo_legal_moves(OTHER[color], for_attack_check=True)
        return any((m[2], m[3]) == king_pos for m in enemy_moves)

    def _pseudo_legal_moves(self, color: str, for_attack_check: bool = False) -> List[Move]:
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = self.board[r][c]
                if p is None or p[0] != color:
                    continue
                kind = p[1]
                if kind == "K":
                    moves += self._king_moves(r, c, color)
                elif kind == "M":
                    moves += self._mantri_moves(r, c, color)
                elif kind == "R":
                    moves += self._ratha_moves(r, c, color)
                elif kind == "G":
                    moves += self._gaja_moves(r, c, color)
                elif kind == "N":
                    moves += self._ashva_moves(r, c, color)
                elif kind == "P":
                    moves += self._padati_moves(r, c, color, for_attack_check)
        return moves

    def _add_if_valid(self, moves, r, c, tr, tc, color):
        if not in_bounds(tr, tc):
            return
        target = self.board[tr][tc]
        if target is None or target[0] != color:
            moves.append((r, c, tr, tc, None))

    def _king_moves(self, r, c, color):
        moves = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                self._add_if_valid(moves, r, c, r + dr, c + dc, color)
        return moves

    def _mantri_moves(self, r, c, color):
        moves = []
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            self._add_if_valid(moves, r, c, r + dr, c + dc, color)
        return moves

    def _ratha_moves(self, r, c, color):
        moves = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tr, tc = r + dr, c + dc
            while in_bounds(tr, tc):
                target = self.board[tr][tc]
                if target is None:
                    moves.append((r, c, tr, tc, None))
                else:
                    if target[0] != color:
                        moves.append((r, c, tr, tc, None))
                    break
                tr += dr
                tc += dc
        return moves

    def _gaja_moves(self, r, c, color):
        """
        Disputed piece. Implemented here as: exactly 2 squares diagonally,
        jumping the intervening square (ignores what's on it, like the
        Alfil in shatranj). Swap this function out if you want to try
        an alternate historical reconstruction.
        """
        moves = []
        for dr, dc in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
            self._add_if_valid(moves, r, c, r + dr, c + dc, color)
        return moves

    def _ashva_moves(self, r, c, color):
        moves = []
        deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for dr, dc in deltas:
            self._add_if_valid(moves, r, c, r + dr, c + dc, color)
        return moves

    def _padati_moves(self, r, c, color, for_attack_check=False):
        moves = []
        direction = 1 if color == "w" else -1
        promo_rank = BOARD_SIZE - 1 if color == "w" else 0

        tr = r + direction
        if in_bounds(tr, c) and not for_attack_check:
            if self.board[tr][c] is None:
                promo = "M" if tr == promo_rank else None
                moves.append((r, c, tr, c, promo))

        for dc in (-1, 1):
            tc = c + dc
            if not in_bounds(tr, tc):
                continue
            target = self.board[tr][tc]
            if for_attack_check:
                # for check-detection we only care about squares attacked,
                # regardless of whether a piece is actually there
                moves.append((r, c, tr, tc, None))
            elif target is not None and target[0] != color:
                promo = "M" if tr == promo_rank else None
                moves.append((r, c, tr, tc, promo))

        return moves

    # ------------------------------------------------------------------ #
    # Debug / display helpers
    # ------------------------------------------------------------------ #
    def __repr__(self):
        rows = []
        for r in range(BOARD_SIZE - 1, -1, -1):
            row_str = " ".join(self.board[r][c] or "--" for c in range(BOARD_SIZE))
            rows.append(f"{r} {row_str}")
        rows.append("  " + " ".join(str(c) for c in range(BOARD_SIZE)))
        rows.append(f"turn: {self.turn}")
        return "\n".join(rows)
