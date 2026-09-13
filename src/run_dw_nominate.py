#!/usr/bin/env python3
"""
run_dw_nominate.py — DW-NOMINATE baseline for roll-call vote prediction.

Two rules, and only one of them is a predictor.

--rule geometric (default) — THE BASELINE
-----------------------------------------
Classifies each member against the roll call's cutting line: the member is
predicted to vote for whichever outcome point lies on their side of it, from
the member's ideal point and the roll call's midpoint/spread. It reads nothing
about how the vote actually went, so its accuracy is a genuine classification
result.

--rule prob — NOT A PREDICTOR
-----------------------------
The legacy rule. Voteview's `prob` is the fitted probability of *the choice the
member actually made*, so recovering a prediction from it requires reading the
recorded vote. The rule reduces to "predict the observed vote when NOMINATE
fits it, the opposite when it doesn't", which makes its accuracy identically
equal to NOMINATE's in-sample hit rate — 1.000 wherever prob >= 50 and 0.000
wherever prob < 50, by construction. That is a goodness-of-fit statistic
wearing an accuracy label. Kept for diagnostics; never table it against the LLM.

Both rules share the standing caveat that DW-NOMINATE's ideal points and cutting
lines are estimated from these very roll calls, so even the geometric rule is an
in-sample fit rather than a forecast in the sense the LLM pipeline is.

Requires the roll-call crosswalk
--------------------------------
Our `vote_id`s number roll calls the way the Clerk does — restarting at 1 every
session — while Voteview numbers them continuously across the Congress. Joining
one onto the other directly matches most rows to the wrong bill, so this script
resolves the roll-call number through `data/rollcall_crosswalk.csv`. Build it
once before running:

    python build_rollcall_crosswalk.py

Requires the roll-call crosswalk
--------------------------------
Our `vote_id`s number roll calls the way the Clerk does — restarting at 1 every
session — while Voteview numbers them continuously across the Congress. Joining
one onto the other directly matches most rows to the wrong bill, so this script
resolves the roll-call number through `data/rollcall_crosswalk.csv`. Build it
once before running:

    python build_rollcall_crosswalk.py

Usage
-----
# Write baseline and evaluate
python run_dw_nominate.py --evaluate

# Custom threshold (default 50.0)
python run_dw_nominate.py --threshold 60 --evaluate

# Custom mirror / output paths
python run_dw_nominate.py --mirror results/predictions.csv \
    --output results/baseline_dw_nominate.csv --evaluate

# Evaluate only (baseline already written)
python run_dw_nominate.py --evaluate-only
"""

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_dw_nominate")

# ---------------------------------------------------------------------------
# Config — edit these defaults or override via CLI arguments
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent

DW_NOMINATE_CSV = SRC_DIR / "data" / "dw_nominate_votes_with_bioguide.csv"
CROSSWALK_CSV = SRC_DIR / "data" / "rollcall_crosswalk.csv"
HSALL_ROLLCALLS_CSV = SRC_DIR / "data" / "hsall_rollcalls.csv"
HSALL_MEMBERS_CSV = SRC_DIR / "data" / "hsall_members.csv"
MIRROR_CSV = SRC_DIR / "results" / "predictions_3_7_yae_nay.csv"
OUTPUT_CSV = SRC_DIR / "results" / "baseline_dw_nominate.csv"
PROB_THRESHOLD = 50.0

# cast_code -> human-readable vote label (Poole-Rosenthal coding)
_CAST_CODE_LABEL = {1: "Yea", 6: "Nay"}

# Metadata columns copied verbatim from the mirrored predictions CSV.
_MIRROR_META_COLS = [
    "vote_id", "bioguide", "member_name", "party", "state",
    "chamber", "congress", "bill_number", "bill_title",
    "vote_date", "true_label",
]

# ---------------------------------------------------------------------------
# Make src/ importable regardless of CWD
# ---------------------------------------------------------------------------
sys.path.insert(0, str(SRC_DIR))

from pipeline.evaluator import evaluate, roc_auc_analysis     # noqa: E402
from pipeline.pipeline import RESULT_COLUMNS                  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VOTE_ID_RE = re.compile(r"^(house|senate)_(\d+)_\w+_(\d+)$")


def _parse_vote_id(vid: str):
    """Extract (chamber, congress, clerk_rollnumber) from a vote_id string.

    The third element is the *Clerk's* roll-call number, which restarts at 1
    each session. It is NOT Voteview's `rollnumber`, which runs continuously
    across the whole Congress — use the crosswalk for that (see
    `load_crosswalk`). Joining on this number directly is the bug that made
    this baseline score 0.675: outside Senate session 1, where the two schemes
    coincide, it matched rows to entirely different roll calls.
    """
    m = _VOTE_ID_RE.match(vid)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None, None, None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mirror_set(mirror_csv: Path) -> pd.DataFrame:
    """Return one row per (bioguide, vote_id) pair present in mirror_csv."""
    if not mirror_csv.exists():
        raise FileNotFoundError(
            f"Mirror predictions file not found: {mirror_csv}\n"
            "Run the LLM pipeline first, or use --mirror to point at an "
            "existing results CSV."
        )
    df = pd.read_csv(mirror_csv)
    missing = [c for c in _MIRROR_META_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Mirror CSV is missing required columns: {missing}"
        )
    before = len(df)
    df = df.drop_duplicates(subset=["bioguide", "vote_id"], keep="first")
    if len(df) < before:
        logger.info("Deduplicated mirror set: %d -> %d rows.", before, len(df))
    logger.info("Loaded %d candidate pairs from %s", len(df), mirror_csv.name)
    return df[_MIRROR_META_COLS].copy()


def load_crosswalk(path: Path) -> pd.DataFrame:
    """Load the vote_id -> Voteview rollnumber crosswalk.

    Built by `build_rollcall_crosswalk.py` from Voteview's own roll-call
    metadata, which carries the Clerk number and the continuous rollnumber side
    by side. Validated there against the recorded member votes.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Roll-call crosswalk not found: {path}\n"
            "Build it first:\n"
            "  python build_rollcall_crosswalk.py"
        )
    cw = pd.read_csv(path, usecols=["vote_id", "chamber", "congress",
                                    "rollnumber"])
    cw["rollnumber"] = cw["rollnumber"].astype(int)
    logger.info("Loaded crosswalk: %d roll calls from %s",
                len(cw), path.name)
    return cw


def load_dw_nominate(dw_csv: Path) -> pd.DataFrame:
    """Load DW-NOMINATE votes, keep only Yea/Nay in congresses 110-119."""
    if not dw_csv.exists():
        raise FileNotFoundError(f"DW-NOMINATE CSV not found: {dw_csv}")

    logger.info("Loading DW-NOMINATE votes from %s ...", dw_csv.name)
    df = pd.read_csv(dw_csv, low_memory=False)
    logger.info("  -> %d rows loaded.", len(df))

    # Keep only Yea (1) and Nay (6) cast codes
    df = df[df["cast_code"].isin([1, 6])].copy()
    logger.info("  -> %d rows after filtering cast_code to {1, 6}.", len(df))

    # Keep only congresses 110-119
    df = df[df["congress"].between(110, 119)].copy()
    logger.info("  -> %d rows in congresses 110-119.", len(df))

    # Convert prob to numeric
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")

    # Map cast_code to vote label
    df["vote_label"] = df["cast_code"].map(_CAST_CODE_LABEL)

    # Normalise chamber to lowercase for merging
    df["chamber"] = df["chamber"].str.lower()

    logger.info(
        "DW-NOMINATE ready: %d rows, %d unique members, %d unique votes.",
        len(df), df["bioguide_id"].nunique(),
        df.groupby(["congress", "chamber", "rollnumber"]).ngroups,
    )
    return df


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def run_dw_nominate_baseline(
    candidates: pd.DataFrame,
    dw_df: pd.DataFrame,
    crosswalk: pd.DataFrame,
    out_csv: Path,
    threshold: float,
    tau: float = None,
) -> None:
    """Write the DW-NOMINATE baseline CSV."""
    # Chamber and congress come straight from the vote_id; the roll-call number
    # must come from the crosswalk, because ours restarts each session while
    # Voteview's runs continuously across the Congress.
    parsed = candidates["vote_id"].apply(lambda x: pd.Series(_parse_vote_id(x)))
    candidates = candidates.copy()
    candidates["_chamber"] = parsed[0]
    candidates["_congress"] = parsed[1]

    before = len(candidates)
    candidates = candidates.merge(
        crosswalk[["vote_id", "rollnumber"]].rename(
            columns={"rollnumber": "_rollnumber"}),
        on="vote_id", how="left",
    )
    if len(candidates) != before:
        raise AssertionError(
            f"Crosswalk join changed the candidate count: {before} -> "
            f"{len(candidates)}; the crosswalk is not unique on vote_id."
        )

    no_xw = candidates["_rollnumber"].isna()
    if no_xw.any():
        logger.warning(
            "%d / %d candidate rows (%.1f%%) have no crosswalk entry and "
            "cannot be scored; %d distinct vote_id(s), e.g. %s",
            int(no_xw.sum()), len(candidates), 100 * no_xw.mean(),
            candidates.loc[no_xw, "vote_id"].nunique(),
            candidates.loc[no_xw, "vote_id"].drop_duplicates().head(3).tolist(),
        )
    else:
        logger.info("Every candidate row resolved to a Voteview rollnumber.")

    # Merge with DW-NOMINATE — drop overlapping columns from DW side
    dw_merge = dw_df[["chamber", "congress", "rollnumber", "bioguide_id",
                       "cast_code", "prob", "vote_label"]].copy()
    dw_merge = dw_merge.rename(columns={"bioguide_id": "_dw_bioguide"})
    merged = candidates.merge(
        dw_merge,
        left_on=["_chamber", "_congress", "_rollnumber", "bioguide"],
        right_on=["chamber", "congress", "rollnumber", "_dw_bioguide"],
        how="left",
        suffixes=("", "_dw"),
    )

    matched = merged["cast_code"].notna().sum()
    total = len(merged)
    logger.info("Matched %d / %d rows to DW-NOMINATE (%.1f%%).",
                matched, total, matched / total * 100 if total else 0)

    # Vectorised prediction
    is_matched = merged["cast_code"].notna()
    prob_raw = merged["prob"]  # NaN for unmatched rows AND for matched rows
                                 # where Voteview simply has no prob estimate
    has_prob = prob_raw.notna()
    # Rows we can actually score: joined to DW-NOMINATE *and* have a prob.
    # A matched-but-missing-prob row must NOT be silently treated as
    # "prob=0" (that forces a near-guaranteed-wrong prediction) — it's
    # unscorable, same as an unmatched row.
    can_predict = is_matched & has_prob
    prob = prob_raw.fillna(0)
    vote_label = merged["vote_label"]  # actual recorded direction

    # p_yea_hat: unfold prob (P of whichever choice was actually recorded)
    # into a true P(Yea), independent of which direction was observed.
    p_yea_hat = pd.Series(np.nan, index=merged.index)
    p_yea_hat[can_predict] = np.where(
        vote_label[can_predict] == "Yea",
        prob_raw[can_predict] / 100.0,
        1.0 - prob_raw[can_predict] / 100.0,
    )

    if tau is not None:
        # Symmetric rule on the reconstructed P(Yea).
        predicted = pd.Series("ERROR", index=merged.index)
        predicted[can_predict] = np.where(p_yea_hat[can_predict] >= tau, "Yea", "Nay")
    else:
        # Predict: recorded direction if prob >= threshold, opposite otherwise.
        # Equivalent to thresholding p_yea_hat at 0.5 only when threshold == 50.
        predicted = pd.Series("ERROR", index=merged.index)
        predicted[can_predict & (prob >= threshold)] = \
            vote_label[can_predict & (prob >= threshold)]
        predicted[can_predict & (prob < threshold)] = \
            vote_label[can_predict & (prob < threshold)].map(
                {"Yea": "Nay", "Nay": "Yea"}
            )

    # Build reasoning strings
    reasoning = pd.Series("", index=merged.index)
    matched_mask = can_predict
    if tau is not None:
        reasoning[matched_mask] = (
            "DW-NOMINATE baseline: p_yea_hat="
            + p_yea_hat[matched_mask].map(lambda p: f"{p:.4f}")
            + ", cast_code="
            + merged.loc[matched_mask, "cast_code"].astype(int).astype(str)
            + " (" + vote_label[matched_mask] + ")"
            + f", tau={tau:.4f}"
            + " -> " + predicted[matched_mask]
        )
    else:
        reasoning[matched_mask & (prob >= threshold)] = (
            "DW-NOMINATE baseline: prob="
            + prob[matched_mask & (prob >= threshold)].map(lambda p: f"{p:.1f}")
            + ", cast_code="
            + merged.loc[matched_mask & (prob >= threshold), "cast_code"]
                .astype(int).astype(str)
            + " ("
            + vote_label[matched_mask & (prob >= threshold)]
            + "), threshold="
            + f"{threshold:.1f}"
            + " -> "
            + predicted[matched_mask & (prob >= threshold)]
        )
        reasoning[matched_mask & (prob < threshold)] = (
            "DW-NOMINATE baseline: prob="
            + prob[matched_mask & (prob < threshold)].map(lambda p: f"{p:.1f}")
            + ", cast_code="
            + merged.loc[matched_mask & (prob < threshold), "cast_code"]
                .astype(int).astype(str)
            + " ("
            + vote_label[matched_mask & (prob < threshold)]
            + "), threshold="
            + f"{threshold:.1f}"
            + " -> "
            + predicted[matched_mask & (prob < threshold)]
            + " (reversed, prob < threshold)"
        )
    reasoning[~is_matched] = (
        "DW-NOMINATE baseline: no DW-NOMINATE data for bioguide "
        + merged.loc[~is_matched, "bioguide"]
    )
    no_prob = is_matched & ~has_prob
    reasoning[no_prob] = (
        "DW-NOMINATE baseline: matched cast_code but no prob estimate for bioguide "
        + merged.loc[no_prob, "bioguide"]
    )

    # Build output DataFrame
    out_df = pd.DataFrame({
        "vote_id":         merged["vote_id"],
        "bioguide":        merged["bioguide"],
        "member_name":     merged["member_name"],
        "party":           merged["party"],
        "state":           merged["state"],
        "chamber":         merged["chamber"],
        "congress":        merged["congress"],
        "bill_number":     merged["bill_number"],
        "bill_title":      merged["bill_title"],
        "vote_date":       merged["vote_date"],
        "true_label":      merged["true_label"],
        "predicted_label": predicted,
        "reasoning":       reasoning,
        "prompt_tokens":   0,
        "completion_tokens": 0,
        "n_tweets_raw":    0,
        "n_tweets_used":   0,
        "competitiveness": 0.0,
        "had_bill_summary": None,
        "retrieval_method": "n/a",
        "mean_similarity":  np.nan,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "prob":            prob_raw,
        "p_yea_hat":       p_yea_hat,
    })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df[RESULT_COLUMNS + ["prob", "p_yea_hat"]].to_csv(out_csv, index=False)

    n_error = int((predicted == "ERROR").sum())
    if tau is not None:
        n_same = int((can_predict & (predicted == vote_label)).sum())
        n_rev = int((can_predict & (predicted != vote_label)).sum())
    else:
        n_same = int((can_predict & (prob >= threshold)).sum())
        n_rev = int((can_predict & (prob < threshold)).sum())
    logger.info(
        "Wrote DW-NOMINATE baseline -> %s  (%d rows: %d matched, %d errors, "
        "%d same direction, %d reversed)",
        out_csv, total, int(matched), n_error, n_same, n_rev,
    )


# ---------------------------------------------------------------------------
# Geometric baseline — the one that actually predicts
# ---------------------------------------------------------------------------

def load_spatial_geometry(rollcalls_csv: Path,
                          members_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load roll-call cutting lines and member ideal points from Voteview."""
    for path, what in ((rollcalls_csv, "roll-call"), (members_csv, "member")):
        if not path.exists():
            raise FileNotFoundError(
                f"Voteview {what} file not found: {path}\n"
                "Download it once with:\n"
                f"  curl -o {path} https://voteview.com/static/data/out/"
                f"{'rollcalls/HSall_rollcalls' if what == 'roll-call' else 'members/HSall_members'}.csv"
            )

    rc = pd.read_csv(
        rollcalls_csv, low_memory=False,
        usecols=["congress", "chamber", "rollnumber", "nominate_mid_1",
                 "nominate_mid_2", "nominate_spread_1", "nominate_spread_2"],
    )
    rc["chamber"] = rc["chamber"].str.lower()

    mem = pd.read_csv(
        members_csv, low_memory=False,
        usecols=["congress", "chamber", "bioguide_id", "nominate_dim1",
                 "nominate_dim2"],
    )
    mem["chamber"] = mem["chamber"].str.lower()
    mem = mem.dropna(subset=["bioguide_id"])
    mem = mem.drop_duplicates(subset=["congress", "chamber", "bioguide_id"])

    logger.info("Spatial geometry: %d roll calls, %d member-congress rows.",
                len(rc), len(mem))
    return rc, mem


def run_geometric_baseline(
    candidates: pd.DataFrame,
    crosswalk: pd.DataFrame,
    rollcalls: pd.DataFrame,
    members: pd.DataFrame,
    out_csv: Path,
) -> None:
    """DW-NOMINATE as a spatial classifier, using no information about the vote.

    Each roll call j has a cutting line through midpoint m_j with spread vector
    s_j; each member i has an ideal point x_i. The member is predicted to vote
    for whichever outcome point lies on their side of the cutting line, which
    reduces to the sign of

        proj = (x_i - m_j) . s_j

    Voteview's `spread` runs from the Yea point to the Nay point, so **proj < 0
    predicts Yea**. That orientation is a single global convention, not a fitted
    parameter: applied one way it scores 0.823 on the scorable rows, the other
    0.177, so the data identifies it unambiguously.

    Roll calls where NOMINATE never estimated a cutting line (both spread
    components exactly 0) have no geometry to classify against and are written
    as ERROR rather than being collapsed onto a default label.

    Unlike the `prob` rule this reads neither `cast_code` nor the recorded
    direction, so its accuracy is a real classification result. It remains an
    *in-sample* one: the ideal points and cutting lines were fitted on these
    very roll calls, so it is not a forecast in the sense the LLM pipeline is.
    """
    parsed = candidates["vote_id"].apply(lambda x: pd.Series(_parse_vote_id(x)))
    cand = candidates.copy()
    cand["_chamber"] = parsed[0]
    cand["_congress"] = parsed[1]

    before = len(cand)
    cand = cand.merge(crosswalk[["vote_id", "rollnumber"]], on="vote_id",
                      how="left")
    cand = cand.merge(
        rollcalls,
        left_on=["_congress", "_chamber", "rollnumber"],
        right_on=["congress", "chamber", "rollnumber"],
        how="left", suffixes=("", "_rc"),
    )
    cand = cand.merge(
        members,
        left_on=["_congress", "_chamber", "bioguide"],
        right_on=["congress", "chamber", "bioguide_id"],
        how="left", suffixes=("", "_mem"),
    )
    if len(cand) != before:
        raise AssertionError(
            f"Geometry join changed the candidate count: {before} -> {len(cand)}."
        )

    mid = ["nominate_mid_1", "nominate_mid_2"]
    spr = ["nominate_spread_1", "nominate_spread_2"]
    dim = ["nominate_dim1", "nominate_dim2"]

    has_rc = cand[mid + spr].notna().all(axis=1)
    has_mem = cand[dim].notna().all(axis=1)
    # A roll call with a zero spread vector has no cutting line to project onto.
    has_line = has_rc & ~((cand["nominate_spread_1"] == 0)
                          & (cand["nominate_spread_2"] == 0))
    scorable = has_line & has_mem

    proj = ((cand["nominate_dim1"] - cand["nominate_mid_1"])
            * cand["nominate_spread_1"]
            + (cand["nominate_dim2"] - cand["nominate_mid_2"])
            * cand["nominate_spread_2"])

    predicted = pd.Series("ERROR", index=cand.index)
    predicted[scorable] = np.where(proj[scorable] < 0, "Yea", "Nay")

    logger.info(
        "Geometry resolved for %d / %d rows (%.1f%%): %d missing a cutting "
        "line, %d missing an ideal point.",
        int(scorable.sum()), len(cand), 100 * scorable.mean(),
        int((~has_line).sum()), int((has_line & ~has_mem).sum()),
    )

    reasoning = pd.Series("", index=cand.index)
    reasoning[scorable] = (
        "DW-NOMINATE spatial rule: proj="
        + proj[scorable].map(lambda v: f"{v:+.4f}")
        + ", ideal=(" + cand.loc[scorable, "nominate_dim1"].map("{:.3f}".format)
        + ", " + cand.loc[scorable, "nominate_dim2"].map("{:.3f}".format) + ")"
        + ", midpoint=(" + cand.loc[scorable, "nominate_mid_1"].map("{:.3f}".format)
        + ", " + cand.loc[scorable, "nominate_mid_2"].map("{:.3f}".format) + ")"
        + " -> " + predicted[scorable]
    )
    reasoning[~has_line] = (
        "DW-NOMINATE spatial rule: no cutting line estimated for this roll call"
    )
    reasoning[has_line & ~has_mem] = (
        "DW-NOMINATE spatial rule: no ideal point for bioguide "
        + cand.loc[has_line & ~has_mem, "bioguide"]
    )

    out_df = pd.DataFrame({
        "vote_id":         cand["vote_id"],
        "bioguide":        cand["bioguide"],
        "member_name":     cand["member_name"],
        "party":           cand["party"],
        "state":           cand["state"],
        "chamber":         cand["_chamber"],
        "congress":        cand["_congress"],
        "bill_number":     cand["bill_number"],
        "bill_title":      cand["bill_title"],
        "vote_date":       cand["vote_date"],
        "true_label":      cand["true_label"],
        "predicted_label": predicted,
        "reasoning":       reasoning,
        "prompt_tokens":   0,
        "completion_tokens": 0,
        "n_tweets_raw":    0,
        "n_tweets_used":   0,
        "competitiveness": 0.0,
        "had_bill_summary": None,
        "retrieval_method": "n/a",
        "mean_similarity":  np.nan,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "proj":            proj,
    })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df[RESULT_COLUMNS + ["proj"]].to_csv(out_csv, index=False)
    logger.info("Wrote DW-NOMINATE spatial baseline -> %s  (%d rows, %d ERROR)",
                out_csv, len(out_df), int((predicted == "ERROR").sum()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DW-NOMINATE baseline for roll-call vote prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dw-nominate", type=Path, default=DW_NOMINATE_CSV,
        help="Path to the DW-NOMINATE votes CSV (with bioguide IDs).",
    )
    p.add_argument(
        "--crosswalk", type=Path, default=CROSSWALK_CSV,
        help="vote_id -> Voteview rollnumber crosswalk, from "
             "build_rollcall_crosswalk.py.",
    )
    p.add_argument(
        "--rule", choices=["geometric", "prob"], default="geometric",
        help="'geometric' classifies each member against the roll call's "
             "cutting line and is the only rule that actually predicts. "
             "'prob' is the legacy rule and reads the recorded vote — its "
             "accuracy is a NOMINATE goodness-of-fit statistic, not a "
             "prediction. See the module docstring.",
    )
    p.add_argument(
        "--rollcalls", type=Path, default=HSALL_ROLLCALLS_CSV,
        help="Voteview HSall_rollcalls.csv (cutting lines).",
    )
    p.add_argument(
        "--members", type=Path, default=HSALL_MEMBERS_CSV,
        help="Voteview HSall_members.csv (member ideal points).",
    )
    p.add_argument(
        "--mirror", type=Path, default=MIRROR_CSV,
        help="Predictions CSV to mirror for the candidate set.",
    )
    p.add_argument(
        "--output", type=Path, default=OUTPUT_CSV,
        help="Output path for the DW-NOMINATE baseline.",
    )
    p.add_argument(
        "--threshold", type=float, default=PROB_THRESHOLD,
        help="Probability threshold for flipping prediction direction.",
    )
    p.add_argument(
        "--tau", type=float, default=None,
        help="Apply the symmetric rule 'predict Yea iff p_yea_hat >= tau' "
             "instead of the prob/threshold branch-and-flip rule.",
    )
    p.add_argument(
        "--evaluate", action="store_true",
        help="Run evaluator.evaluate() after writing the baseline.",
    )
    p.add_argument(
        "--evaluate-only", action="store_true", dest="evaluate_only",
        help="Skip baseline computation; only evaluate an existing CSV.",
    )
    p.add_argument(
        "--restrict-to", type=Path, default=None, dest="restrict_to",
        help="Restrict evaluation to (vote_id, bioguide) pairs with a "
             "non-ERROR prediction in this reference results CSV.",
    )
    p.add_argument(
        "--roc-analysis", action="store_true", dest="roc_analysis",
        help="Run ROC/AUC diagnostic on p_yea_hat vs true_label and report "
             "the Youden's-J-optimal threshold, instead of writing/evaluating "
             "a baseline.",
    )
    p.add_argument(
        "--roc-plot", type=Path, default=None, dest="roc_plot",
        help="Save a ROC-curve PNG to this path (used with --roc-analysis).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.evaluate_only:
        if args.output.exists():
            print("\n" + "=" * 70)
            print(f"EVALUATING {args.output.name}")
            print("=" * 70)
            evaluate(args.output, restrict_to=args.restrict_to)
        else:
            logger.warning("Skipping %s (not found).", args.output)
        return

    if args.roc_analysis:
        has_score_col = (
            args.output.exists()
            and "p_yea_hat" in pd.read_csv(args.output, nrows=0).columns
        )
        if not has_score_col:
            dw_df = load_dw_nominate(args.dw_nominate)
            crosswalk = load_crosswalk(args.crosswalk)
            candidates = load_mirror_set(args.mirror)
            run_dw_nominate_baseline(candidates, dw_df, crosswalk, args.output,
                                     args.threshold)
        print("\n" + "=" * 70)
        print(f"ROC ANALYSIS {args.output.name}")
        print("=" * 70)
        roc_auc_analysis(args.output, plot_path=args.roc_plot)
        return

    # Roll-call number crosswalk (ours is per-session, Voteview's is per-Congress)
    crosswalk = load_crosswalk(args.crosswalk)

    # Mirror candidate set
    candidates = load_mirror_set(args.mirror)

    if args.rule == "geometric":
        rollcalls, members = load_spatial_geometry(args.rollcalls, args.members)
        run_geometric_baseline(candidates, crosswalk, rollcalls, members,
                               args.output)
    else:
        logger.warning(
            "--rule prob reads the recorded vote: its accuracy is the rate at "
            "which NOMINATE fits the observed choice, not predictive accuracy. "
            "Do not table it against the LLM."
        )
        dw_df = load_dw_nominate(args.dw_nominate)
        run_dw_nominate_baseline(candidates, dw_df, crosswalk, args.output,
                                 args.threshold, tau=args.tau)

    if args.evaluate:
        print("\n" + "=" * 70)
        print(f"EVALUATING {args.output.name}")
        print("=" * 70)
        evaluate(args.output, restrict_to=args.restrict_to)


if __name__ == "__main__":
    main()
