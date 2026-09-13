#!/usr/bin/env python3
"""
build_rollcall_crosswalk.py — map our vote_ids onto Voteview roll-call numbers.

Why this exists
---------------
Our `vote_id`s carry the roll-call number as the *Clerk* numbers it, which
restarts at 1 every session (`preprocess_votes.py:115,164` ->
`house_{congress}_{session}_{rollcall}`). Voteview's `rollnumber` is instead
continuous across the whole Congress. House 110, for example, is Clerk 2-1186
then 2-690, but Voteview 1-1865.

Joining our roll-call number straight onto `rollnumber` therefore lands on the
wrong bill for every vote outside the one block where the two numbering schemes
happen to coincide (Senate, session 1). That was the bug in
`run_dw_nominate.py`: it dropped the session from the key entirely and matched
~80% of rows to some other roll call.

Voteview's own roll-call metadata file carries both numbers side by side, so the
crosswalk is an exact four-column join rather than anything inferred:

    (congress, chamber, session, clerk_rollnumber)  ->  rollnumber

Source file: https://voteview.com/static/data/out/rollcalls/HSall_rollcalls.csv

Validation
----------
The join is checked against the actual votes, not just assumed: for every
matched roll call we compare Voteview's recorded Yea/Nay per member against
ours. Both sources describe the same event, so agreement must be ~1.0. Anything
below `--min-agreement` is reported and (unless `--keep-suspect`) dropped.

Usage
-----
python build_rollcall_crosswalk.py
python build_rollcall_crosswalk.py --report      # per-block agreement detail
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_rollcall_crosswalk")

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR / "data"

HSALL_ROLLCALLS_CSV = DATA_DIR / "hsall_rollcalls.csv"
DW_VOTES_CSV        = DATA_DIR / "dw_nominate_votes_with_bioguide.csv"
MEMBER_VOTE_CSVS    = {
    "house":  DATA_DIR / "house_votes_members.csv",
    "senate": DATA_DIR / "senate_votes_members.csv",
}
OUTPUT_CSV = DATA_DIR / "rollcall_crosswalk.csv"

# Congresses the thesis covers.
CONGRESS_MIN, CONGRESS_MAX = 110, 119

# Poole-Rosenthal cast codes we treat as a recorded Yea / Nay.
_CAST_CODE_LABEL = {1: "Yea", 6: "Nay"}

# vote_id -> (chamber, congress, session, clerk_rollnumber).
# The session group is \w+ because the two chambers spell it differently:
# the House writes "1st"/"2nd", the Senate writes "1"/"2".
_VOTE_ID_RE = r"^(house|senate)_(\d+)_(\w+)_(\d+)$"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def parse_vote_ids(vote_ids: pd.Series) -> pd.DataFrame:
    """Split vote_id strings into their four components.

    Returns a frame indexed like `vote_ids` with `chamber`, `congress`,
    `session` and `clerk_rollnumber`. Unparseable ids yield NaN rather than
    raising, so the caller can report them.
    """
    parts = vote_ids.str.extract(_VOTE_ID_RE)
    out = pd.DataFrame(index=vote_ids.index)
    out["vote_id"] = vote_ids.values
    out["chamber"] = parts[0]
    out["congress"] = pd.to_numeric(parts[1], errors="coerce")
    # "1st" -> 1, "2nd" -> 2, "1" -> 1. Leading digits are the session number.
    out["session"] = pd.to_numeric(
        parts[2].str.extract(r"^(\d+)")[0], errors="coerce"
    )
    out["clerk_rollnumber"] = pd.to_numeric(parts[3], errors="coerce")
    return out


def load_our_rollcalls() -> pd.DataFrame:
    """One row per distinct vote_id across both chambers."""
    frames = []
    for chamber, path in MEMBER_VOTE_CSVS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Member-vote CSV not found: {path}\n"
                "Run preprocess_votes.py first."
            )
        frames.append(pd.read_csv(path, usecols=["vote_id"]))
    vote_ids = pd.concat(frames, ignore_index=True)["vote_id"].drop_duplicates()
    vote_ids = vote_ids.reset_index(drop=True)

    parsed = parse_vote_ids(vote_ids)
    bad = parsed["clerk_rollnumber"].isna()
    if bad.any():
        logger.warning("%d vote_id(s) could not be parsed, e.g. %s",
                       int(bad.sum()), parsed.loc[bad, "vote_id"].head(3).tolist())
        parsed = parsed[~bad]

    parsed["congress"] = parsed["congress"].astype(int)
    parsed["session"] = parsed["session"].astype(int)
    parsed["clerk_rollnumber"] = parsed["clerk_rollnumber"].astype(int)
    logger.info("Our roll calls: %d distinct vote_ids.", len(parsed))
    return parsed


def load_voteview_rollcalls(path: Path) -> pd.DataFrame:
    """Voteview roll-call metadata, restricted to the congresses we cover."""
    if not path.exists():
        raise FileNotFoundError(
            f"Voteview roll-call metadata not found: {path}\n"
            "Download it once with:\n"
            "  curl -o data/hsall_rollcalls.csv \\\n"
            "    https://voteview.com/static/data/out/rollcalls/"
            "HSall_rollcalls.csv"
        )
    df = pd.read_csv(
        path, low_memory=False,
        usecols=["congress", "chamber", "rollnumber", "session",
                 "clerk_rollnumber", "date", "bill_number"],
    )
    df = df[df["chamber"].isin(["House", "Senate"])].copy()
    df["chamber"] = df["chamber"].str.lower()
    df = df[df["congress"].between(CONGRESS_MIN, CONGRESS_MAX)]

    missing = df["clerk_rollnumber"].isna() | df["session"].isna()
    if missing.any():
        logger.info("  -> dropping %d Voteview rows with no clerk_rollnumber/"
                    "session.", int(missing.sum()))
        df = df[~missing]

    df["session"] = df["session"].astype(int)
    df["clerk_rollnumber"] = df["clerk_rollnumber"].astype(int)
    logger.info("Voteview roll calls (congresses %d-%d): %d.",
                CONGRESS_MIN, CONGRESS_MAX, len(df))
    return df


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------

def build_crosswalk(ours: pd.DataFrame, vv: pd.DataFrame) -> pd.DataFrame:
    """Exact join of our roll calls onto Voteview rollnumbers."""
    key = ["chamber", "congress", "session", "clerk_rollnumber"]

    dup = vv.duplicated(subset=key).sum()
    if dup:
        raise ValueError(
            f"Voteview metadata has {dup} duplicate {key} rows; the crosswalk "
            "key is not unique on the source side."
        )

    merged = ours.merge(
        vv[key + ["rollnumber", "date", "bill_number"]], on=key, how="left"
    )
    if len(merged) != len(ours):
        raise AssertionError(
            f"Join changed row count: {len(ours)} -> {len(merged)}."
        )

    unmatched = merged["rollnumber"].isna()
    logger.info("Matched %d / %d roll calls (%.2f%%).",
                int((~unmatched).sum()), len(merged),
                100 * (~unmatched).mean())
    if unmatched.any():
        logger.warning("Unmatched roll calls by chamber/congress:\n%s",
                       merged[unmatched].groupby(["chamber", "congress"])
                       .size().to_string())

    matched = merged[~unmatched].copy()
    matched["rollnumber"] = matched["rollnumber"].astype(int)

    # A Voteview rollnumber must not be claimed by two of our roll calls.
    collisions = matched.duplicated(
        subset=["chamber", "congress", "rollnumber"], keep=False
    )
    if collisions.any():
        raise AssertionError(
            f"Crosswalk is not 1-to-1: {int(collisions.sum())} colliding rows, "
            f"e.g.\n{matched[collisions].head(6).to_string()}"
        )
    logger.info("Crosswalk is 1-to-1 within (chamber, congress).")
    return matched


# ---------------------------------------------------------------------------
# Validation against the actual votes
# ---------------------------------------------------------------------------

def validate_against_votes(crosswalk: pd.DataFrame,
                           min_agreement: float,
                           min_overlap: int) -> pd.DataFrame:
    """Compare Voteview's recorded vote to ours for every matched roll call.

    Both sources record the same event, so per-roll-call agreement must be ~1.0.
    Returns the crosswalk with `n_compared` and `agreement` columns added.
    """
    logger.info("Validating crosswalk against recorded member votes ...")

    ours = pd.concat(
        [pd.read_csv(p, usecols=["vote_id", "bioguide", "vote_cast"])
         for p in MEMBER_VOTE_CSVS.values()],
        ignore_index=True,
    )
    ours = ours[ours["vote_cast"].isin(["Yea", "Nay"])]
    ours = ours.merge(
        crosswalk[["vote_id", "chamber", "congress", "rollnumber"]],
        on="vote_id", how="inner",
    )

    vv = pd.read_csv(
        DW_VOTES_CSV, low_memory=False,
        usecols=["congress", "chamber", "rollnumber", "cast_code", "bioguide_id"],
    )
    vv["chamber"] = vv["chamber"].str.lower()
    vv = vv[vv["cast_code"].isin(_CAST_CODE_LABEL) & vv["bioguide_id"].notna()]
    vv["vv_label"] = vv["cast_code"].map(_CAST_CODE_LABEL)

    joined = ours.merge(
        vv[["chamber", "congress", "rollnumber", "bioguide_id", "vv_label"]],
        left_on=["chamber", "congress", "rollnumber", "bioguide"],
        right_on=["chamber", "congress", "rollnumber", "bioguide_id"],
        how="inner",
    )
    joined["agree"] = joined["vote_cast"] == joined["vv_label"]

    per_vote = joined.groupby("vote_id")["agree"].agg(
        n_compared="size", agreement="mean"
    )
    logger.info("  -> compared %d member-votes across %d roll calls; "
                "overall agreement %.6f.",
                len(joined), len(per_vote), joined["agree"].mean())

    out = crosswalk.merge(per_vote, on="vote_id", how="left")
    out["n_compared"] = out["n_compared"].fillna(0).astype(int)

    suspect = (out["n_compared"] >= min_overlap) & (out["agreement"] < min_agreement)
    unchecked = out["n_compared"] < min_overlap
    if suspect.any():
        logger.warning(
            "%d roll call(s) below the %.2f agreement floor:\n%s",
            int(suspect.sum()), min_agreement,
            out.loc[suspect, ["vote_id", "rollnumber", "n_compared", "agreement"]]
            .head(10).to_string(index=False),
        )
    else:
        logger.info("  -> every checked roll call clears the %.2f floor.",
                    min_agreement)
    if unchecked.any():
        logger.info("  -> %d roll call(s) had fewer than %d comparable votes "
                    "and were not checked.", int(unchecked.sum()), min_overlap)

    out["suspect"] = suspect
    return out


def report_identity_invariant(cw: pd.DataFrame) -> None:
    """Senate session 1 is the one block whose two numbering schemes coincide.

    It joined correctly even under the old broken key, so it is a free
    regression check: those rows must still map to themselves.
    """
    sen1 = cw[(cw["chamber"] == "senate") & (cw["session"] == 1)]
    if sen1.empty:
        logger.warning("No Senate session-1 rows to check the identity "
                       "invariant against.")
        return
    identical = (sen1["rollnumber"] == sen1["clerk_rollnumber"]).mean()
    logger.info("Senate session 1 maps to identity for %.2f%% of %d rows.",
                100 * identical, len(sen1))
    if identical < 1.0:
        raise AssertionError(
            "Senate session 1 must map to identity (it is the block already "
            f"known correct), but only {identical:.4f} of rows do."
        )

    shift = cw["rollnumber"] - cw["clerk_rollnumber"]
    logger.info("Roll-number shift applied, by chamber/session:\n%s",
                cw.assign(shift=shift)
                  .groupby(["chamber", "session"])["shift"]
                  .agg(["min", "max", "mean"]).round(1).to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the vote_id -> Voteview rollnumber crosswalk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--rollcalls", type=Path, default=HSALL_ROLLCALLS_CSV,
                    help="Voteview HSall_rollcalls.csv")
    ap.add_argument("--output", type=Path, default=OUTPUT_CSV,
                    help="Where to write the crosswalk")
    # The two outcomes are far apart, so this floor does not need to be tight:
    # a correct join scores 1.0 (a handful score ~0.989, where a single member's
    # recorded vote differs between the sources — Senate vote corrections), while
    # a wrong join scores ~0.60, the level you get pairing roll calls at random
    # given the ~73% Yea base rate. Nothing lands in between. 0.95 separates them
    # cleanly without discarding correctly-matched roll calls over one member.
    ap.add_argument("--min-agreement", type=float, default=0.95,
                    help="Per-roll-call recorded-vote agreement floor")
    ap.add_argument("--min-overlap", type=int, default=20,
                    help="Members needed before agreement is checked")
    ap.add_argument("--keep-suspect", action="store_true",
                    help="Write rows that fail the agreement floor instead of "
                         "dropping them")
    ap.add_argument("--skip-validation", action="store_true",
                    help="Write the join without checking it against votes")
    args = ap.parse_args()

    ours = load_our_rollcalls()
    vv = load_voteview_rollcalls(args.rollcalls)
    cw = build_crosswalk(ours, vv)
    report_identity_invariant(cw)

    if args.skip_validation:
        cw["n_compared"] = -1
        cw["agreement"] = np.nan
        cw["suspect"] = False
    else:
        cw = validate_against_votes(cw, args.min_agreement, args.min_overlap)
        if cw["suspect"].any() and not args.keep_suspect:
            logger.warning("Dropping %d suspect roll call(s); pass "
                           "--keep-suspect to retain them.",
                           int(cw["suspect"].sum()))
            cw = cw[~cw["suspect"]]

    cols = ["vote_id", "chamber", "congress", "session", "clerk_rollnumber",
            "rollnumber", "date", "bill_number", "n_compared", "agreement"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cw[cols].to_csv(args.output, index=False)
    logger.info("Wrote %d crosswalk rows -> %s", len(cw), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
