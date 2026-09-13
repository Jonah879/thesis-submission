"""
vote_filters.py — Scope filters applied to the vote population.

Currently one filter: nomination / confirmation votes.

Nominations are Senate cloture votes on confirming a person to an office
("… Nancy L. Moritz, to be U.S. Circuit Judge"). They are excluded from the
tweet-augmented analysis for two reasons:

  1. There is no bill, so no bill summary exists — the topic-matching method
     has nothing on the vote side to match tweets against. Measured coverage:
     1,182 of the 1,196 Senate votes lacking a summary are nominations.
  2. They are marginally more party-line than bill votes and score higher
     (91.8% vs 82.6% accuracy in the 18-07 run), so including them inflates
     headline numbers on a task the method cannot actually address.

Dropping them takes the Senate from 2,128 → ~947 votes, i.e. from 46% to
~21% of the vote population. Stratum sizes should be re-checked against
config.MIN_PER_STRATUM after filtering.

Public API:
    is_nomination(bill_title, chamber=None) -> bool
    filter_nominations(votes_df)            -> pd.DataFrame
"""

import logging
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# The reliable signal is the "to be <Office>" construction, plus explicit
# nomination/confirmation wording. Validated against the full vote set:
# Senate 1,182 true positives / 1 false positive; House 0 false positives
# (the House does not vote on nominations, so it is excluded by chamber).
_NOMINATION_PATTERN = re.compile(
    r"\bnominations?\b"
    r"|\bconfirmation of\b"
    r"|\bto be\b"
    r"|\bfor appointment\b"
    r"|,\s*of\s+[A-Z]\w+,\s*to\b",   # "…, of Texas, to U.S. Circuit Judge"
    re.IGNORECASE,
)


def is_nomination(bill_title, chamber: Optional[str] = None) -> bool:
    """
    True if this vote is a nomination / confirmation rather than a vote on
    legislation.

    Only the Senate confirms nominations, so House votes always return False.
    Without that guard, 9 genuine House bills match on incidental "to be"
    ("Proclaiming Casimir Pulaski to be an honorary citizen…").

    Treaty ratification votes are deliberately NOT flagged — they are
    substantive votes, they just happen to lack a congress.gov summary.
    """
    if chamber is not None and str(chamber).strip().lower() == "house":
        return False
    if bill_title is None or (isinstance(bill_title, float) and pd.isna(bill_title)):
        return False
    return bool(_NOMINATION_PATTERN.search(str(bill_title)))


def filter_nominations(votes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop nomination/confirmation rows from a votes DataFrame, logging how
    many unique votes and member-vote rows were removed per chamber.
    """
    if votes_df.empty:
        return votes_df

    chambers = votes_df["chamber"] if "chamber" in votes_df.columns else None
    if chambers is None:
        mask = votes_df["bill_title"].map(is_nomination)
    else:
        mask = [
            is_nomination(t, c)
            for t, c in zip(votes_df["bill_title"], chambers)
        ]
        mask = pd.Series(mask, index=votes_df.index)

    n_rows = int(mask.sum())
    if n_rows == 0:
        logger.info("Nomination filter: no nomination votes found.")
        return votes_df

    n_votes = votes_df.loc[mask, "vote_id"].nunique()
    filtered = votes_df.loc[~mask].copy()

    logger.info(
        "Nomination filter: dropped %d unique votes (%d member-vote rows). "
        "%d → %d rows remaining.",
        n_votes, n_rows, len(votes_df), len(filtered),
    )
    if chambers is not None:
        for ch, grp in votes_df.loc[mask].groupby("chamber"):
            logger.info("  %s: %d votes dropped", ch, grp["vote_id"].nunique())

    return filtered
