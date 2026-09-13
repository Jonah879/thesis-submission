#!/usr/bin/env python3
"""
run_baseline.py — Baselines for roll-call vote prediction.

All variants mirror the candidate set of an existing predictions CSV, so row
counts line up with the LLM run.

CAUSAL BASELINES — comparable to the LLM
----------------------------------------
These use only information available strictly before the vote's date, which is
the same constraint the LLM pipeline runs under (feature_builder filters both
tweets and past votes to `< vote_date`). Report these against the LLM.

  always_yea        — the most frequent class among all prior votes.
  member_prior      — the member's own Yea-rate over their prior votes.
  party_prior       — the member's party's Yea-rate over prior votes in the
                      same chamber and congress.
  chamber_majority  — Yea if the member's party holds the most seats in that
                      chamber/congress, else Nay. Chamber composition is fixed
                      at the start of a Congress, so this is known in advance.

LEGACY VARIANTS — not comparable to the LLM
-------------------------------------------
  majority          — the roll-call's overall majority label.
  party             — the majority label among the member's own party on that
                      roll call (falls back to overall majority for tiny
                      parties / independents).

Both read the tally of the *same* roll call being predicted — the outcome of
the event itself, unavailable at prediction time. They also count the member's
own vote in their own tally. Kept for reference; do not table them against the
LLM as if they were forecasters.

Usage
-----
# All causal baselines, mirroring config.PREDICTIONS_CSV, then evaluate
python run_baseline.py --variant causal --evaluate

# One baseline, custom mirror file
python run_baseline.py --variant party_prior \\
    --mirror results/predictions_sample500.csv --evaluate

# Evaluate only (baselines already written)
python run_baseline.py --variant causal --evaluate-only
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup — must precede pipeline imports so module loggers inherit.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_baseline")

# ---------------------------------------------------------------------------
# Make src/ importable regardless of CWD
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from pipeline import config                                   # noqa: E402
from pipeline import data_loader                              # noqa: E402
from pipeline.evaluator import evaluate                       # noqa: E402
from pipeline.pipeline import RESULT_COLUMNS                  # noqa: E402

# Metadata columns copied verbatim from the mirrored predictions CSV.
# Anything not in this list is recomputed from votes_df.
_MIRROR_META_COLS = [
    "vote_id", "bioguide", "member_name", "party", "state",
    "chamber", "congress", "bill_number", "bill_title",
    "vote_date", "true_label",
]


# ---------------------------------------------------------------------------
# Mirror candidate set from an existing predictions CSV
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
    # Deduplicate on (bioguide, vote_id): keep first occurrence.
    before = len(df)
    df = df.drop_duplicates(subset=["bioguide", "vote_id"], keep="first")
    if len(df) < before:
        logger.info("Deduplicated mirror set: %d -> %d rows.", before, len(df))
    logger.info("Loaded %d candidate pairs from %s", len(df), mirror_csv.name)
    return df[_MIRROR_META_COLS].copy()


# ---------------------------------------------------------------------------
# Baseline computations
# ---------------------------------------------------------------------------

def _overall_majority_map(votes_df: pd.DataFrame) -> dict:
    """Return {vote_id -> "Yea" | "Nay"} based on full roll-call outcome."""
    outcomes = (
        votes_df.groupby("vote_id")["vote_cast"]
        .value_counts()
        .unstack(fill_value=0)
    )
    yea = outcomes.get("Yea", pd.Series(0, index=outcomes.index))
    nay = outcomes.get("Nay", pd.Series(0, index=outcomes.index))
    # Ties default to "Yea" (rare after Yea/Nay filter; most roll calls
    # have odd counts in the House and VP tiebreaks in the Senate).
    majority = np.where(yea >= nay, "Yea", "Nay")
    return dict(zip(outcomes.index, majority))


def _party_majority_map(votes_df: pd.DataFrame) -> dict:
    """Return {(vote_id, party) -> "Yea" | "Nay"} based on party-line vote.

    Groups with fewer than 3 voting members fall back to None, signalling
    the caller to use the overall majority instead.
    """
    grouped = (
        votes_df.groupby(["vote_id", "party"])["vote_cast"]
        .value_counts()
        .unstack(fill_value=0)
    )
    yea = grouped.get("Yea", pd.Series(0, index=grouped.index))
    nay = grouped.get("Nay", pd.Series(0, index=grouped.index))
    total = yea + nay

    # Default to "Yea" on ties; mark small groups with None for fallback.
    majority = np.where(yea >= nay, "Yea", "Nay").astype(object)
    majority[total.values < 3] = None
    return dict(zip(grouped.index, majority))


# ---------------------------------------------------------------------------
# Causal baselines — strictly-prior information only
# ---------------------------------------------------------------------------

# Prediction for candidates with no prior history at all (the first votes in
# the corpus, or a member's very first vote). Matches the corpus-wide skew.
_NO_PRIOR_DEFAULT = "Yea"


def _prepare_votes(votes_df: pd.DataFrame) -> pd.DataFrame:
    """Add the helper columns the causal baselines aggregate over."""
    work = votes_df.copy()
    work["_is_yea"] = (work["vote_cast"] == "Yea").astype(int)
    work["_all"] = 0            # constant key, for the corpus-wide aggregate
    return work


def _candidate_vote_rows(candidates: pd.DataFrame, work: pd.DataFrame) -> pd.DataFrame:
    """Return the votes_df row backing each candidate (bioguide, vote_id) pair.

    Causal baselines key off the *vote row* rather than the mirror CSV, whose
    ``party`` column comes from the member's most recent record corpus-wide
    (feature_builder.get_member_profile) — the wrong affiliation for members
    who switched parties mid-career.
    """
    pairs = candidates[["bioguide", "vote_id"]].drop_duplicates()
    cols = ["bioguide", "vote_id", "vote_date", "party", "chamber", "congress", "_all"]
    rows = (
        work[cols]
        .merge(pairs, on=["bioguide", "vote_id"], how="inner")
        .drop_duplicates(subset=["bioguide", "vote_id"], keep="first")
    )
    if len(rows) < len(pairs):
        logger.warning(
            "%d/%d candidate pairs have no matching row in the vote corpus; "
            "they will fall back to %r.",
            len(pairs) - len(rows), len(pairs), _NO_PRIOR_DEFAULT,
        )
    return rows


def _prior_yea_rate(work: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Yea-rate over votes strictly before each vote_date, per key group.

    Aggregates the corpus to (keys…, vote_date) totals, cumulatively sums
    within each key group, then shifts by one step — so the totals attached to
    a given vote_date cover only votes cast *before* it. This is what keeps the
    baseline causal: no candidate can see its own vote.

    Note that `vote_date` is a full timestamp for Senate rows but date-only for
    House rows, so the effective resolution differs by chamber: a Senate
    candidate sees earlier votes from the same day, a House candidate does not.
    That is deliberate — it is exactly the comparison feature_builder makes
    when assembling the LLM's past-votes block (`votes_df["vote_date"] <
    vote_date`, feature_builder.py:58), so both sides of the comparison draw on
    the same information.

    Returns a DataFrame indexed by (keys…, vote_date) with columns
    n_yea / n_total / predicted.
    """
    grouped = (
        work.groupby(keys + ["vote_date"])["_is_yea"]
        .agg(n_yea="sum", n_total="count")
        .sort_index()
    )
    levels = list(range(len(keys)))
    prior = grouped.groupby(level=levels).cumsum().groupby(level=levels).shift(1)

    rate = prior["n_yea"] / prior["n_total"]
    prior["predicted"] = np.where(rate >= 0.5, "Yea", "Nay")
    prior.loc[rate.isna(), "predicted"] = _NO_PRIOR_DEFAULT
    return prior


def _lookup_prior(cand_rows: pd.DataFrame, prior: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Align a _prior_yea_rate table onto the candidate rows."""
    idx = pd.MultiIndex.from_frame(cand_rows[keys + ["vote_date"]])
    aligned = prior.reindex(idx)
    aligned.index = pd.MultiIndex.from_frame(cand_rows[["bioguide", "vote_id"]])
    return aligned


def _chamber_majority_party(work: pd.DataFrame) -> dict:
    """Return {(congress, chamber) -> party holding the most seats}."""
    seats = work.groupby(["congress", "chamber", "party"])["bioguide"].nunique()
    winners = seats.groupby(level=[0, 1]).idxmax()
    return {key: full_key[2] for key, full_key in winners.items()}


def _causal_predictions(
    candidates: pd.DataFrame,
    work: pd.DataFrame,
    variant: str,
) -> tuple:
    """Return (predictions, reasonings) as Series aligned to `candidates`."""
    cand_rows = _candidate_vote_rows(candidates, work)
    cand_idx = pd.MultiIndex.from_frame(candidates[["bioguide", "vote_id"]])

    if variant == "chamber_majority":
        majority_map = _chamber_majority_party(work)
        keyed = cand_rows.set_index(["bioguide", "vote_id"])
        maj = [majority_map.get((c, ch)) for c, ch in zip(keyed["congress"], keyed["chamber"])]
        keyed["_maj"] = maj
        pred = np.where(
            keyed["_maj"].isna(), _NO_PRIOR_DEFAULT,
            np.where(keyed["party"] == keyed["_maj"], "Yea", "Nay"),
        )
        reason = [
            f"chamber_majority baseline: {ch} in congress {c} was held by "
            f"{m or 'unknown'}; member is {p}"
            for c, ch, p, m in zip(
                keyed["congress"], keyed["chamber"], keyed["party"], keyed["_maj"]
            )
        ]
        pred_s = pd.Series(pred, index=keyed.index).reindex(cand_idx)
        reason_s = pd.Series(reason, index=keyed.index).reindex(cand_idx)
    else:
        keys = {
            "always_yea":   ["_all"],
            "member_prior": ["bioguide"],
            "party_prior":  ["party", "chamber", "congress"],
        }[variant]
        prior = _prior_yea_rate(work, keys)
        aligned = _lookup_prior(cand_rows, prior, keys)

        scope = {
            "always_yea":   "all prior votes",
            "member_prior": "this member's prior votes",
            "party_prior":  "prior votes by this party in this chamber/congress",
        }[variant]
        reason = []
        for n_yea, n_total in zip(aligned["n_yea"], aligned["n_total"]):
            if pd.isna(n_total) or n_total == 0:
                reason.append(f"{variant} baseline: no prior history; defaulted to {_NO_PRIOR_DEFAULT}")
            else:
                reason.append(
                    f"{variant} baseline: {int(n_yea)}/{int(n_total)} Yea "
                    f"({n_yea / n_total:.1%}) across {scope}"
                )
        aligned["_reason"] = reason
        pred_s = aligned["predicted"].reindex(cand_idx)
        reason_s = aligned["_reason"].reindex(cand_idx)

    pred_s = pred_s.fillna(_NO_PRIOR_DEFAULT)
    reason_s = reason_s.fillna(f"{variant} baseline: candidate absent from vote corpus")
    return pred_s.to_numpy(), reason_s.to_numpy()


def run_causal_baseline(
    candidates: pd.DataFrame,
    work: pd.DataFrame,
    out_csv: Path,
    variant: str,
) -> None:
    """Write one causal baseline CSV."""
    preds, reasons = _causal_predictions(candidates, work, variant)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        _row(meta, pred, reason)
        for (_, meta), pred, reason in zip(candidates.iterrows(), preds, reasons)
    ]
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(out_csv, index=False)
    logger.info("Wrote %s baseline -> %s  (%d rows)", variant, out_csv, len(rows))


# ---------------------------------------------------------------------------
# Row writer (matches RESULT_COLUMNS so evaluator.evaluate() works as-is)
# ---------------------------------------------------------------------------

def _row(meta: pd.Series, predicted: str, reasoning: str) -> dict:
    return {
        "vote_id":           meta["vote_id"],
        "bioguide":          meta["bioguide"],
        "member_name":       meta["member_name"],
        "party":             meta["party"],
        "state":             meta["state"],
        "chamber":           meta["chamber"],
        "congress":          meta["congress"],
        "bill_number":      meta["bill_number"],
        "bill_title":        meta["bill_title"],
        "vote_date":         meta["vote_date"],
        "true_label":        meta["true_label"],
        "predicted_label":   predicted,
        "reasoning":         reasoning,
        "prompt_tokens":     0,
        "completion_tokens": 0,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    }


def _counts_str(votes_df: pd.DataFrame, vote_id: str, party: str | None = None) -> str:
    """Human-readable vote tally, e.g. '230 Yea / 180 Nay'."""
    if party is None:
        subset = votes_df[votes_df["vote_id"] == vote_id]
    else:
        subset = votes_df[(votes_df["vote_id"] == vote_id) & (votes_df["party"] == party)]
    vc = subset["vote_cast"].value_counts()
    y = int(vc.get("Yea", 0))
    n = int(vc.get("Nay", 0))
    return f"{y} Yea / {n} Nay"


# ---------------------------------------------------------------------------
# Baseline runners
# ---------------------------------------------------------------------------

def run_majority_baseline(
    candidates: pd.DataFrame,
    votes_df: pd.DataFrame,
    out_csv: Path,
) -> None:
    """Write the overall-majority baseline CSV."""
    maj_map = _overall_majority_map(votes_df)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, meta in candidates.iterrows():
        vote_id = meta["vote_id"]
        pred = maj_map.get(vote_id, "Yea")
        counts = _counts_str(votes_df, vote_id)
        rows.append(_row(meta, pred, f"Majority baseline: {vote_id} had {counts}"))

    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(out_csv, index=False)
    logger.info("Wrote overall-majority baseline -> %s  (%d rows)", out_csv, len(rows))


def run_party_baseline(
    candidates: pd.DataFrame,
    votes_df: pd.DataFrame,
    out_csv: Path,
) -> None:
    """Write the party-line majority baseline CSV.

    Independents and tiny party groups (fewer than 3 voters) fall back to
    the overall majority for that vote.
    """
    party_map = _party_majority_map(votes_df)
    overall_map = _overall_majority_map(votes_df)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n_fallback = 0
    for _, meta in candidates.iterrows():
        vote_id = meta["vote_id"]
        party = meta["party"]
        pred = party_map.get((vote_id, party))
        if pred is None:
            pred = overall_map.get(vote_id, "Yea")
            n_fallback += 1
            reason = (
                f"Party-majority fallback (party={party} had <3 voters on "
                f"{vote_id}); overall majority: {_counts_str(votes_df, vote_id)}"
            )
        else:
            reason = (
                f"Party-majority baseline: {party} on {vote_id} had "
                f"{_counts_str(votes_df, vote_id, party)}"
            )
        rows.append(_row(meta, pred, reason))

    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(out_csv, index=False)
    logger.info(
        "Wrote party-majority baseline -> %s  (%d rows, %d fell back to overall)",
        out_csv, len(rows), n_fallback,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CAUSAL_VARIANTS = ["always_yea", "member_prior", "party_prior", "chamber_majority"]
LEGACY_VARIANTS = ["majority", "party"]


def _resolve_variants(variant: str) -> list:
    """Expand a --variant group name into the concrete variants to run."""
    if variant == "causal":
        return list(CAUSAL_VARIANTS)
    if variant == "both":
        return list(LEGACY_VARIANTS)
    if variant == "all":
        return CAUSAL_VARIANTS + LEGACY_VARIANTS
    return [variant]


def _output_paths(args: argparse.Namespace) -> dict:
    return {
        "majority":         args.output_majority,
        "party":            args.output_party,
        "always_yea":       args.output_always_yea,
        "member_prior":     args.output_member_prior,
        "party_prior":      args.output_party_prior,
        "chamber_majority": args.output_chamber_majority,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baselines for roll-call vote prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mirror", type=Path, default=config.PREDICTIONS_CSV,
        help="Predictions CSV to mirror for the candidate set.",
    )
    p.add_argument(
        "--variant",
        choices=CAUSAL_VARIANTS + LEGACY_VARIANTS + ["causal", "both", "all"],
        default="causal",
        help="Which baseline(s) to compute. 'causal' = the four baselines "
             "comparable to the LLM; 'both' = the two legacy same-roll-call "
             "variants; 'all' = everything.",
    )
    p.add_argument(
        "--output-majority", type=Path, default=config.BASELINE_MAJORITY_CSV,
        help="Output path for the overall-majority baseline.",
    )
    p.add_argument(
        "--output-party", type=Path, default=config.BASELINE_PARTY_CSV,
        help="Output path for the party-majority baseline.",
    )
    p.add_argument(
        "--output-always-yea", type=Path, default=config.BASELINE_ALWAYS_YEA_CSV,
        help="Output path for the prior-majority-class baseline.",
    )
    p.add_argument(
        "--output-member-prior", type=Path, default=config.BASELINE_MEMBER_PRIOR_CSV,
        help="Output path for the member prior Yea-rate baseline.",
    )
    p.add_argument(
        "--output-party-prior", type=Path, default=config.BASELINE_PARTY_PRIOR_CSV,
        help="Output path for the party prior Yea-rate baseline.",
    )
    p.add_argument(
        "--output-chamber-majority", type=Path, default=config.BASELINE_CHAMBER_MAJORITY_CSV,
        help="Output path for the chamber-majority-party baseline.",
    )
    p.add_argument(
        "--evaluate", action="store_true",
        help="Run evaluator.evaluate() on each baseline after writing.",
    )
    p.add_argument(
        "--evaluate-only", action="store_true", dest="evaluate_only",
        help="Skip baseline computation; only evaluate existing baseline CSVs.",
    )
    p.add_argument(
        "--restrict-to", type=Path, default=None, dest="restrict_to",
        help="Restrict evaluation to (vote_id, bioguide) pairs with a "
             "non-ERROR prediction in this reference results CSV.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    variants = _resolve_variants(args.variant)
    paths = _output_paths(args)

    # ---- Evaluate-only mode ----------------------------------------------
    if args.evaluate_only:
        for name in variants:
            path = paths[name]
            if path.exists():
                print("\n" + "=" * 70)
                print(f"EVALUATING {path.name}")
                print("=" * 70)
                evaluate(path, restrict_to=args.restrict_to)
            else:
                logger.warning("Skipping %s (not found).", path)
        return

    # ---- Load votes (applies the Yea/Nay filter) -------------------------
    logger.info("Loading vote data …")
    votes_df = data_loader.load_votes(chamber="both")

    # ---- Mirror candidate set -------------------------------------------
    candidates = load_mirror_set(args.mirror)

    # ---- Run baselines ---------------------------------------------------
    causal = [v for v in variants if v in CAUSAL_VARIANTS]
    if causal:
        work = _prepare_votes(votes_df)
        for name in causal:
            run_causal_baseline(candidates, work, paths[name], name)

    if "majority" in variants:
        run_majority_baseline(candidates, votes_df, paths["majority"])

    if "party" in variants:
        run_party_baseline(candidates, votes_df, paths["party"])

    # ---- Optional evaluation --------------------------------------------
    if args.evaluate:
        for name in variants:
            path = paths[name]
            if not path.exists():
                continue
            print("\n" + "=" * 70)
            print(f"EVALUATING {path.name}")
            print("=" * 70)
            evaluate(path, restrict_to=args.restrict_to)


if __name__ == "__main__":
    main()
