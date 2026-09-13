#!/usr/bin/env python3
"""
rerank_cache.py — Re-filter an existing sample cache with topic retrieval.

The sample cache stores `raw_tweets` separately from `filtered_tweets` and
never mutates the former, so a different retrieval method can be applied to
the *identical* candidate set with the *identical* raw tweet pool — a
perfectly matched A/B against the cached LLM run, at zero LLM cost.

Two modes:

  --sweep   Score every candidate once, then report the zero-relevant rate
            and tweet-count distribution across a range of thresholds, next
            to the cached LLM baseline. Use this to pick --min-sim
            deliberately rather than accepting the default.

  default   Write a new cache with topic-selected tweets at --min-sim, ready
            to feed straight into run_pipeline.py --sample-cache.

Usage:
    python rerank_cache.py --cache data/sample_cache.pkl --sweep
    python rerank_cache.py --cache data/sample_cache.pkl \
        --output data/sample_cache_topic.pkl --min-sim 0.25
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rerank_cache")

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from pipeline import config           # noqa: E402
from pipeline import sample_cache     # noqa: E402
from pipeline import topic_model      # noqa: E402
from pipeline import topic_retriever  # noqa: E402

DEFAULT_THRESHOLDS = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-filter a sample cache with topic-model retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=config.DATA_DIR / "sample_cache.pkl",
        help="Existing sample cache to re-filter.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the re-filtered cache. Omit with --sweep.",
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=config.TOPIC_MIN_SIM,
        dest="min_sim",
        help="Score floor. 0.0 = pure top-K, never empty.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.TOPIC_TOP_K,
        dest="top_k",
        help="Max tweets kept per candidate.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Report threshold sweep instead of writing a cache.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N cached candidates (for a quick look).",
    )
    return parser.parse_args()


def score_all(tasks: list) -> list:
    """
    Score every tweet of every candidate once. Returns a list of score arrays
    aligned with each task's raw_tweets.
    """
    logger.info("Pre-embedding all tweet and vote texts …")
    texts = []
    for t in tasks:
        texts.extend(t["raw_tweets"])
        texts.extend(topic_model.build_vote_query_texts(
            t["item"]["bill_title"],
            t["item"].get("bill_summary", ""),
            t["item"]["question"],
        ))
    topic_retriever.warm_texts(texts, show_progress=True)

    all_scores = []
    for t in tqdm(tasks, desc="Scoring", unit="pair"):
        all_scores.append(topic_retriever.score_tweets(
            t["raw_tweets"],
            bill_title=t["item"]["bill_title"],
            question=t["item"]["question"],
            bill_summary=t["item"].get("bill_summary", ""),
        ))
    return all_scores


def report_sweep(tasks: list, all_scores: list, top_k: int, thresholds=None) -> None:
    """Print zero-relevant rate and tweet counts per threshold vs the cache."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    n = len(tasks)

    cached_used = np.array([t.get("n_tweets_used", 0) for t in tasks])
    cached_zero = float((cached_used == 0).mean() * 100)

    print("\n" + "=" * 74)
    print(f"Cached retrieval baseline  (n={n})")
    print("=" * 74)
    print(f"  zero-relevant : {cached_zero:5.1f}%")
    print(f"  median used   : {np.median(cached_used):.0f}")
    print(f"  mean used     : {cached_used.mean():.2f}")

    print("\n" + "=" * 74)
    print("Topic retrieval by threshold")
    print("=" * 74)
    print(f"{'min_sim':>8}  {'zero-rel':>9}  {'median':>7}  {'mean':>7}  {'>=1 tweet':>10}")
    print("-" * 74)

    for thr in thresholds:
        counts = np.array([
            int(min(int((s >= thr).sum()), top_k)) for s in all_scores
        ])
        zero = float((counts == 0).mean() * 100)
        print(f"{thr:8.2f}  {zero:8.1f}%  {np.median(counts):7.0f}  "
              f"{counts.mean():7.2f}  {100 - zero:9.1f}%")

    # Score distribution — where the mass actually sits.
    flat = np.concatenate([s for s in all_scores if len(s)]) if all_scores else np.zeros(0)
    if flat.size:
        print("\nScore distribution over all (tweet, vote) pairs:")
        for p in [50, 75, 90, 95, 99, 99.9]:
            print(f"  p{p:<5} = {np.percentile(flat, p):.4f}")
        print(f"  max    = {flat.max():.4f}")
    print()


def main() -> None:
    args = parse_args()

    if not args.sweep and args.output is None:
        raise SystemExit("Provide --output to write a cache, or pass --sweep to report only.")

    if not args.cache.exists():
        raise SystemExit(f"Cache not found: {args.cache}")

    tasks = sample_cache.load(args.cache)
    logger.info("Loaded %d cached candidates from %s", len(tasks), args.cache)
    if args.limit:
        tasks = tasks[: args.limit]
        logger.info("Limited to first %d candidates.", len(tasks))

    all_scores = score_all(tasks)

    if args.sweep:
        report_sweep(tasks, all_scores, args.top_k)
        return

    # Apply the chosen threshold and write a new cache.
    n_empty = 0
    for t, scores in zip(tasks, all_scores):
        order = np.argsort(-scores)
        selected, sel_scores = [], []
        for idx in order[: args.top_k]:
            if scores[idx] < args.min_sim:
                break
            selected.append(t["raw_tweets"][idx])
            sel_scores.append(float(scores[idx]))

        t["filtered_tweets"] = selected
        t["n_tweets_raw"]    = len(t["raw_tweets"])
        t["n_tweets_used"]   = len(selected)
        t["mean_similarity"] = round(float(np.mean(sel_scores)), 4) if sel_scores else ""
        if not selected:
            n_empty += 1

    logger.info(
        "Topic retrieval at min_sim=%.2f, top_k=%d: %d/%d candidates empty (%.1f%%).",
        args.min_sim, args.top_k, n_empty, len(tasks),
        n_empty / len(tasks) * 100 if tasks else 0,
    )

    sample_cache.save(tasks, args.output)
    logger.info("Wrote re-filtered cache to %s", args.output)


if __name__ == "__main__":
    main()
