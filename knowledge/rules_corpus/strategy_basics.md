# Basic Strategic Concepts for This Variant

Chaturanga has no surviving body of established opening/strategic theory
the way modern chess does -- most strategic guidance for this variant is
reasoned from first principles (piece mobility and value) rather than
centuries of recorded master games. Treat the following as engineering
heuristics, not historical doctrine.

## Relative piece values (as used by this engine's material evaluator)
    Padati (pawn):    100  (baseline unit)
    Mantri:           150  (weak -- only reaches 4 squares from any position)
    Gaja:             180  (restricted to one color complex, only 4 destinations)
    Ashva:            300  (strong -- 8 possible destinations, jumps over pieces)
    Ratha:            500  (strongest non-royal piece -- full line control)
These values are estimates, not established theory, and may be revised
once self-play data (from search/minimax.py or RL training in rl/) is
available to check whether any piece is significantly over- or
under-valued in practice.

## Why the Gaja and Mantri are weak
Both the Gaja (exactly 2 squares diagonally) and the Mantri (exactly 1
square diagonally) have very limited mobility compared to the Ashva
(knight-like) or Ratha (rook-like). A Gaja is permanently confined to
squares of a single color, meaning it can never contest or defend squares
of the other color -- a significant long-term structural weakness,
directly analogous to the historical Alfil in Shatranj, which was
considered one of the weakest pieces on the board.

## Early-game priorities
Since there is no castling, king safety in the early game depends entirely
on keeping the Raja shielded by its own Padati and not advancing it
prematurely -- there's no way to "tuck it away" to the side of the board
the way castling allows in modern chess. Developing the Ashva pieces early
is generally favorable since they have the most mobility of any minor
piece and are not blocked by other pieces.

## Endgame notes
Because Padati promotion always produces a Mantri (a weak piece) rather
than a Ratha or Ashva, promoting a pawn in this variant is a meaningfully
smaller material swing than promoting to a Queen in modern chess. Racing a
pawn to promotion is still valuable, but less overwhelmingly decisive than
in standard chess endgames.

Because stalemate is a loss (not a draw) under this engine's default
rules, a losing side should NOT seek to simplify into a position with no
legal moves as a drawing resource -- unlike in modern chess, where
"stalemate tricks" are a legitimate defensive technique for a losing
player, here that same maneuver only speeds up the loss.
