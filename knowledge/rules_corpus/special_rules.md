# Special Rules: Check, Checkmate, and Stalemate

## Check
A Raja (King) is "in check" when it is under attack by an opposing piece
-- that is, if the opponent had the move right now, they could legally
capture the Raja. A player may never make a move that leaves their own
Raja in check, even if that move captures material or otherwise looks
strong.

## Checkmate
If it is a player's turn to move, their Raja is in check, and they have no
legal move that escapes check, that player has been checkmated and loses
the game immediately.

## Stalemate -- important difference from modern chess
If it is a player's turn to move, their Raja is NOT in check, but they
have no legal move available anywhere on the board, this is a stalemate.

In modern chess, stalemate is a DRAW. In this implementation of
Chaturanga, following one common historical reconstruction, stalemate is
treated the SAME as checkmate: the player with no legal moves simply
LOSES. There is no draw by stalemate in the default rule set used here.

This is a deliberate design choice, not a universal historical fact --
other reconstructions of Chaturanga, and the game's later Persian and
European descendants, use different stalemate conventions. If you prefer
the modern chess convention (stalemate = draw), this engine supports
switching to it via a configuration flag (STALEMATE_IS_DRAW in
board/state.py).

## No castling, no en passant, no double pawn step
This variant does not include several rules that were added much later in
chess's evolution:
    - No castling (a special king-and-rook move) exists in this rule set.
    - No en passant pawn capture exists.
    - Padati (pawns) may only ever move one square forward per move; there
      is no initial two-square advance option.

## Promotion
When a Padati reaches the far rank (the rank furthest from where it
started), it is immediately promoted. In this implementation, promotion is
always to a Mantri -- there is no choice of promotion piece, unlike modern
chess where a pawn may promote to Queen, Rook, Bishop, or Knight.
