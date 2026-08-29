"""The agent roster, as bare names.

Its own module, importing nothing, because two layers need it and they cannot
import each other: `config` validates `screening.consensus_min` against the
number of strategists, and `agents.roster` builds the debate from the same
list. `agents` imports `config`, so the shared fact has to sit below both.

Keeping it in one place is not tidiness. The validator's message used to say
"there are three strategists" in prose while the list said something else was
possible, and the judge's prompt asserted a consensus threshold of 2-of-3 long
after the real one became 1. Every count downstream now derives from here.
"""

from __future__ import annotations

#: The strategists. One, now: whether it proposes a buy is the decision.
#:
#: Kept as a list rather than collapsed to a single name because every count
#: downstream derives from it, and because the debate protocol is worth being
#: able to restore without hunting for the places that assumed one voice.
STRATEGISTS = ["strategy"]

#: The exit review, on an open position. Not a strategist: it never opens
#: anything, and the only moves it can make are tightening ones.
EXIT_AGENT = "exit"
