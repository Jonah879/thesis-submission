#!/usr/bin/env python3
"""
build_sample_cache.py — One-time script to build a high-quality sample cache.

Iteratively retrieves tweets for candidates until we have enough with
at least --min-tweets relevant tweets. Pre-fers by raw tweets (cheap)
to eliminate most candidates without LLM cost.

Usage:
    python build_sample_cache.py --target-sample 10000 --min-tweets 1
    python build_sample_cache.py --target-sample 10000 --min-tweets 5 --batch-size 20000

    # Proportional stratified sampling (congress_era x chamber x party x
    # vote_cast x competitiveness), re-drawing per stratum each batch to
    # compensate for uneven post-retrieval attrition:
    python build_sample_cache.py --stratify --target-sample 10000 --min-tweets 1
"""

import argparse
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_sample_cache")

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from pipeline import config
from pipeline import data_loader
from pipeline import feature_builder as fb
from pipeline import sample_cache
from pipeline import tweet_retriever
from pipeline import vote_filters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sample cache with iterative retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-sample", type=int, default=10000, help="Target number of candidates with tweets.")
    parser.add_argument("--batch-size", type=int, default=15000, help="Candidates per retrieval batch.")
    parser.add_argument("--min-tweets", type=int, default=1, help="Minimum relevant tweets per candidate.")
    parser.add_argument("--max-tweets-raw", type=int, default=config.MAX_TWEETS_RAW, help="Max raw tweets per candidate.")
    parser.add_argument("--max-tweets-prompt", type=int, default=config.MAX_TWEETS_PROMPT, help="Max tweets in prompt.")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="Model for retrieval.")
    parser.add_argument(
        "--retrieval-method",
        choices=["llm", "topic"],
        default=config.RETRIEVAL_METHOD,
        dest="retrieval_method",
        help="Retrieval strategy used to populate filtered_tweets in the cache.",
    )
    parser.add_argument("--chamber", choices=["house", "senate", "both"], default="both")
    parser.add_argument("--n-days", type=int, default=config.N_DAYS_TWEETS, help="Tweet lookback window (days).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=Path, default=config.DATA_DIR / "sample_cache.pkl", help="Output cache file.")
    parser.add_argument(
        "--stratify", action="store_true",
        help="Draw target-sample using proportional stratified sampling "
             "(congress_era x chamber x party x vote_cast x competitiveness), "
             "re-drawing per stratum each batch to compensate for uneven "
             "min-tweets attrition. Default is flat random sampling.",
    )
    parser.add_argument(
        "--min-per-stratum", type=int, default=config.MIN_PER_STRATUM,
        dest="min_per_stratum",
        help="Per-stratum quota floor when --stratify is set.",
    )
    return parser.parse_args()


def _has_tweets_in_window(bioguide, vote_date, tweet_index, n_days):
    """Quick check: does this member have ANY tweets in the n_days window? (no LLM)"""
    from datetime import timedelta
    member_tweets = tweet_index.get(bioguide)
    if member_tweets is None or member_tweets.empty:
        return False
    cutoff_start = vote_date - timedelta(days=n_days)
    mask = (member_tweets["created_at"] >= cutoff_start) & (member_tweets["created_at"] < vote_date)
    return mask.any()


# ---------------------------------------------------------------------------
# Stratification helpers — adapted from pipeline.VotePredictionPipeline
# (pipeline.py: _normalize_party, _congress_to_era, _add_competitiveness,
# _build_stratified_sample's allocation math). Duplicated locally rather than
# imported so this script stays self-contained; pipeline.py is not touched.
# ---------------------------------------------------------------------------

def _normalize_party(party) -> str:
    """Normalize raw party codes to D / R / I."""
    if pd.isna(party):
        return "I"
    p = str(party).strip().upper()
    if p.startswith("D"):
        return "D"
    if p.startswith("R"):
        return "R"
    return "I"


def _congress_to_era(congress: int) -> str:
    """Group congresses into political eras for stratified sampling."""
    if congress <= 111:
        return "1_Bush_Obama_start"
    elif congress <= 114:
        return "2_Obama_polarization"
    elif congress <= 116:
        return "3_Trump1"
    elif congress <= 118:
        return "4_Biden"
    else:
        return "5_Trump2"


def _add_competitiveness(votes_df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'competitiveness' column to votes_df (unanimous/moderate/contested)."""
    thresholds = config.COMPETITIVENESS_THRESHOLDS
    if thresholds is None:
        votes_df["competitiveness"] = "all"
        return votes_df

    contested_upper, moderate_upper = thresholds

    vote_counts = votes_df.groupby("vote_id")["vote_cast"].value_counts().unstack(fill_value=0)
    total = vote_counts.sum(axis=1).replace(0, 1)
    yea_frac = (vote_counts.get("Yea", 0) / total) if "Yea" in vote_counts.columns else 0.0
    nay_frac = (vote_counts.get("Nay", 0) / total) if "Nay" in vote_counts.columns else 0.0

    if isinstance(yea_frac, pd.Series):
        winner_frac = np.maximum(yea_frac, nay_frac)
    else:
        winner_frac = nay_frac.copy()

    comp = pd.Series(index=vote_counts.index, dtype="object")
    if contested_upper == moderate_upper:
        comp[:] = np.where(winner_frac.values >= moderate_upper, "unanimous", "contested")
    else:
        comp[:] = np.where(
            winner_frac.values >= moderate_upper, "unanimous",
            np.where(winner_frac.values >= contested_upper, "moderate", "contested"),
        )

    votes_df["competitiveness"] = votes_df["vote_id"].map(comp)
    return votes_df


def _stratum_key(item: dict) -> tuple:
    return (
        item["congress_era"],
        item["chamber"],
        item["party_norm"],
        item["true_label"],
        item["competitiveness"],
    )


def _build_stratum_pool(pre_filtered: list) -> dict:
    pool = defaultdict(list)
    for c in pre_filtered:
        pool[_stratum_key(c)].append(c)
    return dict(pool)


def _allocate_quotas(stratum_sizes: dict, n: int, min_per: int) -> dict:
    """
    Proportional allocation with a per-stratum floor, scaled down if
    over-allocated. Mirrors pipeline.py's _build_stratified_sample Phase 1/2,
    generalized to a plain {stratum_key: size} dict.
    """
    total_pool = sum(stratum_sizes.values())
    if total_pool == 0:
        return {}

    allocations = {}
    for key, size in stratum_sizes.items():
        proportional = int(n * size / total_pool)
        alloc = max(proportional, min_per)
        alloc = min(alloc, size)
        allocations[key] = alloc

    total_allocated = sum(allocations.values())

    if total_allocated > n:
        deficit = total_allocated - n
        above_min = {k: v for k, v in allocations.items() if v > min_per}
        total_above_min_excess = sum(v - min_per for v in above_min.values())

        if total_above_min_excess > 0:
            for k in above_min:
                reduction = int(deficit * (above_min[k] - min_per) / total_above_min_excess)
                allocations[k] = max(min_per, allocations[k] - reduction)

        total_allocated = sum(allocations.values())
        while total_allocated > n:
            k_max = max(allocations, key=allocations.get)
            allocations[k_max] -= 1
            total_allocated -= 1

    return allocations


def _select_stratified_batch(pool_by_stratum, seen_keys, need, batch_size, rng):
    """
    Pick up to batch_size candidates across strata, proportional to each
    stratum's remaining need. Strata with heavier attrition automatically
    get a larger share of later batches since their need stays high —
    this is what makes the loop self-correcting without knowing attrition
    rates in advance.
    """
    remaining = {}
    for key in need:
        avail = [c for c in pool_by_stratum.get(key, []) if (c["bioguide"], c["vote_id"]) not in seen_keys]
        if avail:
            remaining[key] = avail
    if not remaining:
        return []

    total_need = sum(need[k] for k in remaining)
    draw_counts = {}
    for key, avail in remaining.items():
        share = batch_size * need[key] / total_need if total_need > 0 else 0
        draw_counts[key] = min(round(share), need[key], len(avail))

    total_capacity = sum(min(need[k], len(remaining[k])) for k in remaining)
    target_total = min(batch_size, total_capacity)
    leftover = target_total - sum(draw_counts.values())
    if leftover > 0:
        for key in sorted(remaining, key=lambda k: need[k], reverse=True):
            capacity = min(need[key], len(remaining[key])) - draw_counts[key]
            if capacity <= 0:
                continue
            add = min(leftover, capacity)
            draw_counts[key] += add
            leftover -= add
            if leftover <= 0:
                break

    batch = []
    for key, count in draw_counts.items():
        if count <= 0:
            continue
        batch.extend(rng.sample(remaining[key], count))
    return batch


def _trim_to_quota(accumulated: list, target_quota: dict, rng) -> list:
    """
    Per-stratum trim (replaces the flat rng.sample trim used in flat mode)
    so the achieved proportions from the adaptive loop are preserved rather
    than re-randomized away at the end.
    """
    by_stratum = defaultdict(list)
    for t in accumulated:
        by_stratum[_stratum_key(t["item"])].append(t)

    trimmed = []
    for key, items in by_stratum.items():
        quota = target_quota.get(key, len(items))
        if len(items) > quota:
            trimmed.extend(rng.sample(items, quota))
        else:
            trimmed.extend(items)
    return trimmed


# ---------------------------------------------------------------------------
# Batch processing — shared by both the flat and stratified loops
# ---------------------------------------------------------------------------

def _process_batch(batch: list, batch_num: int, args, votes_df, tweet_index) -> list:
    """Feature-collect, retrieve, and min-tweets-filter one batch. Returns
    the tasks that passed --min-tweets."""
    tasks = []
    for item in tqdm(batch, desc=f"Batch {batch_num} features", unit="candidate"):
        bioguide  = item["bioguide"]
        vote_date = item["vote_date"]

        profile = fb.get_member_profile(bioguide, votes_df)
        raw_tweets = fb.get_recent_tweets(
            bioguide, vote_date, tweet_index,
            n_days=args.n_days, max_k=args.max_tweets_raw,
        )
        past_votes = fb.get_past_votes(
            bioguide, vote_date, votes_df, n=config.N_PAST_VOTES
        )

        vote_date_str = (
            vote_date.strftime("%Y-%m-%d")
            if hasattr(vote_date, "strftime") else str(vote_date)
        )

        tasks.append({
            "item":          item,
            "profile":       profile,
            "raw_tweets":    raw_tweets,
            "past_votes":    past_votes,
            "vote_date_str": vote_date_str,
        })

    # Filter out candidates with no raw tweets
    tasks = [t for t in tasks if len(t["raw_tweets"]) > 0]
    logger.info("  %d/%d candidates have raw tweets in batch", len(tasks), len(batch))

    if not tasks:
        return []

    # Retrieve relevant tweets
    if args.retrieval_method == "topic":
        # Batched vector scoring — no thread pool, no network.
        from pipeline import topic_model
        from pipeline import topic_retriever

        logger.info("  Topic retrieval: pre-embedding texts for %d candidates …", len(tasks))
        texts = []
        for t in tasks:
            texts.extend(t["raw_tweets"])
            texts.extend(topic_model.build_vote_query_texts(
                t["item"]["bill_title"],
                t["item"].get("bill_summary", ""),
                t["item"]["question"],
            ))
        topic_retriever.warm_texts(texts, show_progress=True)

        for task in tqdm(tasks, desc=f"Batch {batch_num} scoring", unit="candidate"):
            filtered_tweets, _scores = topic_retriever.retrieve_with_scores(
                task["raw_tweets"],
                bill_title=task["item"]["bill_title"],
                question=task["item"]["question"],
                bill_summary=task["item"].get("bill_summary", ""),
                top_k=args.max_tweets_prompt,
            )
            task["filtered_tweets"] = filtered_tweets[:args.max_tweets_prompt]
            task["n_tweets_raw"] = len(task["raw_tweets"])
            task["n_tweets_used"] = len(task["filtered_tweets"])
    else:
        logger.info("  Retrieving tweets for %d candidates (concurrency=%d) …", len(tasks), config.MAX_CONCURRENCY)
        with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENCY) as pool:
            futures = {
                pool.submit(
                    tweet_retriever.retrieve_relevant_tweets,
                    t["raw_tweets"],
                    t["item"]["bill_number"],
                    t["item"]["bill_title"],
                    t["item"]["question"],
                    t["profile"]["member_name"],
                    args.model,
                    t["item"].get("bill_summary", ""),
                ): t
                for t in tasks
            }
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc=f"Batch {batch_num} retrieval", unit="candidate",
            ):
                task = futures[future]
                filtered_tweets, _n = future.result()
                task["filtered_tweets"] = filtered_tweets[:args.max_tweets_prompt]
                task["n_tweets_raw"] = len(task["raw_tweets"])
                task["n_tweets_used"] = len(task["filtered_tweets"])

    # Filter by min_tweets
    passed = [t for t in tasks if t["n_tweets_used"] >= args.min_tweets]
    return passed


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    mode = "stratified" if args.stratify else "flat"

    # ---- Load data --------------------------------------------------------
    logger.info("Loading vote data …")
    votes_df = data_loader.load_votes(args.chamber)
    if config.DROP_NOMINATION_VOTES:
        votes_df = vote_filters.filter_nominations(votes_df)

    logger.info("Computing vote competitiveness …")
    votes_df = _add_competitiveness(votes_df)

    if args.stratify:
        logger.info("Stratify: computing party_norm/congress_era …")
        votes_df["party_norm"] = votes_df["party"].apply(_normalize_party)
        votes_df["congress_era"] = votes_df["congress"].apply(_congress_to_era)

    logger.info("Loading tweets …")
    tweets_df = data_loader.load_tweets()
    tweet_index = data_loader.build_tweet_index(tweets_df)

    # ---- Build candidates -------------------------------------------------
    logger.info("Building candidate pool …")
    candidates = []
    for _, row in votes_df.iterrows():
        if pd.isna(row["bill_title"]) or str(row["bill_title"]).strip() == "":
            continue
        cand = {
            "bioguide":    row["bioguide"],
            "vote_id":     row["vote_id"],
            "vote_date":   row["vote_date"],
            "bill_number": row["bill_number"],
            "bill_title":  row["bill_title"],
            "bill_summary": row.get("bill_summary", ""),
            "question":    row["question"],
            "chamber":     row["chamber"],
            "congress":    row["congress"],
            "session":     row["session"],
            "true_label":  row["vote_cast"],
            "competitiveness": row.get("competitiveness", ""),
        }
        if args.stratify:
            cand["party_norm"] = row["party_norm"]
            cand["congress_era"] = row["congress_era"]
        candidates.append(cand)
    logger.info("Total candidates with bill_title: %d", len(candidates))

    if args.stratify:
        # Drop Independent-party candidates — too few for meaningful strata
        # (mirrors pipeline.py's _build_stratified_sample).
        n_before = len(candidates)
        candidates = [c for c in candidates if c["party_norm"] != "I"]
        n_after = len(candidates)
        if n_before != n_after:
            logger.info(
                "Stratify: dropped %d I-party candidates (%d → %d)",
                n_before - n_after, n_before, n_after,
            )

    # ---- Pre-filter: only candidates with raw tweets in window -------------
    logger.info("Pre-filtering by raw tweets in %d-day window …", args.n_days)
    pre_filtered = []
    for c in candidates:
        if _has_tweets_in_window(c["bioguide"], c["vote_date"], tweet_index, args.n_days):
            pre_filtered.append(c)
    logger.info(
        "Pre-filtered: %d/%d candidates have raw tweets (%.1f%%)",
        len(pre_filtered), len(candidates),
        len(pre_filtered) / len(candidates) * 100 if candidates else 0,
    )

    if not pre_filtered:
        logger.error("No candidates have tweets in the window. Nothing to do.")
        return

    # ---- Stratified quota allocation (stratify mode only) ------------------
    pool_by_stratum = {}
    target_quota = {}
    if args.stratify:
        pool_by_stratum = _build_stratum_pool(pre_filtered)
        stratum_sizes = {k: len(v) for k, v in pool_by_stratum.items()}
        target_quota = _allocate_quotas(stratum_sizes, args.target_sample, args.min_per_stratum)
        logger.info(
            "Stratified allocation: %d strata, %d pool candidates, target=%d, min_per_stratum=%d",
            len(stratum_sizes), len(pre_filtered), args.target_sample, args.min_per_stratum,
        )
        for key, quota in sorted(target_quota.items()):
            logger.info("  stratum %s: pool=%d quota=%d", key, stratum_sizes[key], quota)

    # ---- Iterative retrieval -----------------------------------------------
    checkpoint_path = args.output.parent / (args.output.name + ".checkpoint.pkl")

    if checkpoint_path.exists():
        state = sample_cache.load_checkpoint(checkpoint_path)
        checkpoint_mode = state.get("mode", "flat")
        if checkpoint_mode != mode:
            logger.error(
                "Checkpoint %s was created in '%s' mode, but this run requests "
                "'%s' mode (--stratify=%s). Delete the checkpoint or use a "
                "different --output to start fresh in the new mode.",
                checkpoint_path, checkpoint_mode, mode, args.stratify,
            )
            return
        accumulated = state["accumulated"]
        seen_keys = state["seen_keys"]
        batch_num = state["batch_num"]
        prior_elapsed = state["elapsed_seconds"]
        logger.info(
            "Resuming from checkpoint %s: batch %d, %d/%d accumulated, %.1fh elapsed so far.",
            checkpoint_path, batch_num, len(accumulated), args.target_sample,
            prior_elapsed / 3600,
        )
    else:
        accumulated = []  # candidates that passed min_tweets
        seen_keys = set()
        batch_num = 0
        prior_elapsed = 0.0

    start_time = time.time()

    while True:
        if args.stratify:
            achieved = Counter(_stratum_key(t["item"]) for t in accumulated)
            need = {
                key: target_quota[key] - achieved.get(key, 0)
                for key in target_quota
                if target_quota[key] - achieved.get(key, 0) > 0
            }
            if not need:
                logger.info("All strata satisfied. Stopping with %d accumulated.", len(accumulated))
                break

            batch = _select_stratified_batch(pool_by_stratum, seen_keys, need, args.batch_size, rng)
            if not batch:
                logger.warning(
                    "No more candidates available for any stratum still in need. "
                    "Stopping with %d accumulated (target was %d).",
                    len(accumulated), args.target_sample,
                )
                break

            batch_num += 1
            batch_keys = {(c["bioguide"], c["vote_id"]) for c in batch}
            seen_keys.update(batch_keys)

            logger.info(
                "Batch %d: sampling %d candidates across %d strata still in need "
                "(accumulated: %d/%d, elapsed: %.1fh)",
                batch_num, len(batch), len(need), len(accumulated), args.target_sample,
                prior_elapsed / 3600 + (time.time() - start_time) / 3600,
            )
        else:
            if len(accumulated) >= args.target_sample:
                break

            available = [c for c in pre_filtered if (c["bioguide"], c["vote_id"]) not in seen_keys]
            if not available:
                logger.warning("No more candidates available. Stopping with %d accumulated.", len(accumulated))
                break

            batch_num += 1
            batch_size = min(args.batch_size, len(available))
            batch = rng.sample(available, batch_size)
            batch_keys = {(c["bioguide"], c["vote_id"]) for c in batch}
            seen_keys.update(batch_keys)

            logger.info(
                "Batch %d: sampling %d candidates (accumulated: %d/%d, elapsed: %.1fh)",
                batch_num, batch_size, len(accumulated), args.target_sample,
                prior_elapsed / 3600 + (time.time() - start_time) / 3600,
            )

        passed = _process_batch(batch, batch_num, args, votes_df, tweet_index)
        accumulated.extend(passed)
        logger.info(
            "  Batch %d: %d candidates passed min_tweets=%d (total accumulated: %d)",
            batch_num, len(passed), args.min_tweets, len(accumulated),
        )

        # Checkpoint after every batch — a crash loses at most this batch.
        sample_cache.save_checkpoint(
            {
                "accumulated": accumulated,
                "seen_keys": seen_keys,
                "batch_num": batch_num,
                "elapsed_seconds": prior_elapsed + (time.time() - start_time),
                "mode": mode,
            },
            checkpoint_path,
        )

    # ---- Finalize -----------------------------------------------------------
    elapsed_hours = prior_elapsed / 3600 + (time.time() - start_time) / 3600
    logger.info("Retrieval complete: %d candidates accumulated in %.1f hours.", len(accumulated), elapsed_hours)

    if args.stratify:
        achieved = Counter(_stratum_key(t["item"]) for t in accumulated)
        for key, quota in sorted(target_quota.items()):
            got = achieved.get(key, 0)
            if got < quota:
                logger.warning("Stratum %s: got %d/%d (shortfall %d)", key, got, quota, quota - got)
        if len(accumulated) < args.target_sample:
            logger.warning(
                "Only %d candidates accumulated across all strata (target was %d). Using all.",
                len(accumulated), args.target_sample,
            )
        accumulated = _trim_to_quota(accumulated, target_quota, rng)
    else:
        if len(accumulated) < args.target_sample:
            logger.warning(
                "Only %d candidates passed min_tweets=%d (target was %d). Using all.",
                len(accumulated), args.min_tweets, args.target_sample,
            )
        if len(accumulated) > args.target_sample:
            accumulated = rng.sample(accumulated, args.target_sample)

    # ---- Save cache -------------------------------------------------------
    sample_cache.save(accumulated, args.output)
    checkpoint_path.unlink(missing_ok=True)
    logger.info("Done. Sample cache saved to %s (%d candidates).", args.output, len(accumulated))


if __name__ == "__main__":
    main()
