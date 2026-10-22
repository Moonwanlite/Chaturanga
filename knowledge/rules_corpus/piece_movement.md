# Piece Movement Rules (as implemented in this engine)

This describes the exact rule set used by board/state.py in this project.
Other historical reconstructions of Chaturanga differ on some details
(especially the Gaja/elephant's move) -- where that's the case, it's noted
below.

## Raja (King) -- letter K
Moves exactly one square in any direction: horizontally, vertically, or
diagonally. The Raja cannot move into check. A player whose Raja has no
legal moves, and who also has no other legal moves anywhere on the board,
loses the game -- see the section on Checkmate and Stalemate below.

## Mantri (Counselor/Minister) -- letter M
Moves exactly one square diagonally, in any of the four diagonal
directions. This is the weakest non-pawn piece on the board, comparable to
the "Ferz" in Persian Shatranj. Historically, the Mantri is the direct
ancestor of the modern chess Queen, though the Queen's move is vastly more
powerful.

## Ratha (Chariot) -- letter R
Moves any number of squares horizontally or vertically, like a Rook in
modern chess. It cannot jump over other pieces -- its path is blocked by
the first piece encountered in that direction, which it may capture if
that piece belongs to the opponent.

## Gaja (Elephant) -- letter G
As implemented here: moves exactly two squares diagonally, jumping over
the intervening square regardless of what occupies it (similar to the
"Alfil" in Persian Shatranj). This means each Gaja is permanently
restricted to squares of one color complex, and can only reach 4 possible
destination squares from any given position.
Note: the Gaja's exact historical move is genuinely disputed among
historians, and other reconstructions describe different movement
patterns. This implementation picked one common reconstruction; treat it
as a design choice rather than settled historical fact.

## Ashva (Horse) -- letter N
Moves in an "L" shape: two squares in one direction (horizontal or
vertical) followed by one square perpendicular to that, exactly like a
Knight in modern chess. Like the Knight, it jumps over any pieces in
between and is not blocked by them.

## Padati (Foot Soldier) -- letter P
Moves exactly one square straight forward (toward the opponent's side),
and cannot move backward or sideways. Captures diagonally forward, one
square, exactly like a modern chess pawn's capture -- but unlike modern
chess, the Padati has no initial two-square advance option and there is no
en passant capture. Upon reaching the far rank (the opponent's back rank),
a Padati promotes -- in this implementation, always to a Mantri (some
other reconstructions allow promotion to other pieces, but this engine
keeps it simple with Mantri-only promotion).
