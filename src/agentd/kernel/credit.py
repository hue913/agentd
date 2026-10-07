"""Credit accounting: did a recalled experience actually help?

Why this exists
---------------
JitRL's advantage estimate is only worth something if the experiences it
recalls are worth recalling. Retrieval ranked purely on lexical similarity, so a
step that kept being recalled and then ignored outranked a step that was
followed and worked, and nothing in the system could tell those apart.

This adds the smallest thing that can: count the recall, then count whether the
decision that followed adopted that step's action. The smoothed adoption rate
nudges ranking.

It is deliberately not a second learning algorithm. It is an observable, so the
existing bench can answer "does memory ON beat memory OFF" with a number per
memory rather than only a score.

Smoothing is add-one on both sides, so a step with no outcome yet scores exactly
1.0 (no effect) and one bad outcome cannot erase a step that was right often.
"""

from __future__ import annotations

import sqlite3

# A migration, not part of CREATE TABLE: these columns land on a database that
# already exists, and CREATE TABLE IF NOT EXISTS would silently skip them.
MIGRATION = [
    ("recalls", "INTEGER NOT NULL DEFAULT 0"),
    ("adopted", "INTEGER NOT NULL DEFAULT 0"),
    ("rejected", "INTEGER NOT NULL DEFAULT 0"),
]


def migrate(db: sqlite3.Connection) -> list[str]:
    """Add credit columns if absent. Safe to run on every start."""
    added: list[str] = []
    existing = {row[1] for row in db.execute("PRAGMA table_info(steps)")}
    for column, decl in MIGRATION:
        if column in existing:
            continue
        db.execute(f"ALTER TABLE steps ADD COLUMN {column} {decl}")
        added.append(column)
    if added:
        db.commit()
    return added


def credit_factor(adopted: int, rejected: int) -> float:
    """Add-one smoothed adoption rate, normalised so "no signal" means "no effect".

    Plain Laplace smoothing gives (a+1)/(a+r+2), which is 0.5 when nothing has
    happened -- that silently penalises every never-decided memory as if it had
    already been rejected once, and the penalty is largest exactly where we
    know the least. Scaling by the no-signal value (2) puts the neutral point
    back at 1.0:

        (0,0) -> 1.000   unknown, unchanged
        (1,0) -> 1.333   one agreement
        (0,1) -> 0.667   one disagreement
        (9,1) -> 1.667   mostly right

    The attainable range is [2/3, 2], a deliberately gentle nudge: similarity
    still dominates and credit mostly breaks ties.
    """
    return 2.0 * (adopted + 1.0) / (adopted + rejected + 2.0)
