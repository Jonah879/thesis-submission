#!/usr/bin/env python3
"""
run_pipeline.py — CLI entry point for the Congressional Vote Prediction Pipeline.

Examples
--------
# Predict 500 random (member, vote) pairs from both chambers
python run_pipeline.py --sample 500

# Run on House only, wider tweet window, and evaluate afterwards
python run_pipeline.py --chamber house --n-days 60 --n-past-votes 15 --sample 200 --evaluate

# Use a different OpenRouter model
python run_pipeline.py --model meta-llama/llama-3-70b-instruct --sample 100 --evaluate

# Only run evaluation on an existing results file
python run_pipeline.py --evaluate-only

# Dry-run: print the prompt for the first candidate pair without calling the LLM
python run_pipeline.py --dry-run --sample 1

Environment
-----------
OPENROUTER_API_KEY  must be set before running (except with --dry-run / --evaluate-only).
"""

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup — do this before importing pipeline modules so that
# their module-level loggers pick up the configuration.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")

# ---------------------------------------------------------------------------
# Add src/ to path so pipeline package is importable regardless of CWD
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from pipeline import config                          # noqa: E402
from pipeline.pipeline import VotePredictionPipeline # noqa: E402
from pipeline.evaluator import evaluate, bootstrap_evaluate  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict congressional votes with an LLM via OpenRouter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data scope --------------------------------------------------------
    parser.add_argument(
        "--chamber",
        choices=["house", "senate", "both"],
        default="both",
        help="Which chamber(s) to include.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N (member, vote) pairs. Omit to process all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for --sample.",
    )

    # ---- Feature settings --------------------------------------------------
    parser.add_argument(
        "--n-days",
        type=int,
        default=config.N_DAYS_TWEETS,
        dest="n_days",
        help="Days before vote date to look back for tweets.",
    )
    parser.add_argument(
        "--n-past-votes",
        type=int,
        default=config.N_PAST_VOTES,
        dest="n_past_votes",
        help="Number of past votes to include as context.",
    )
    parser.add_argument(
        "--max-tweets-raw",
        type=int,
        default=config.MAX_TWEETS_RAW,
        dest="max_tweets_raw",
        help="Max tweets collected from window and sent to retrieval LLM.",
    )
    parser.add_argument(
        "--max-tweets-prompt",
        type=int,
        default=config.MAX_TWEETS_PROMPT,
        dest="max_tweets_prompt",
        help="Max tweets in final prediction prompt (after retrieval).",
    )
    parser.add_argument(
        "--min-tweets",
        type=int,
        default=0,
        dest="min_tweets",
        help="Minimum relevant tweets required for prediction. Candidates with fewer are skipped.",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        dest="skip_retrieval",
        help="Skip tweet relevance filtering — use raw tweets directly.",
    )
    parser.add_argument(
        "--retrieval-method",
        choices=["llm", "topic", "none"],
        default=config.RETRIEVAL_METHOD,
        dest="retrieval_method",
        help="Tweet retrieval strategy: LLM relevance judgment, topic-model "
             "gating (needs a fitted topic space — see fit_topic_model.py), "
             "or none.",
    )
    parser.add_argument(
        "--topic-top-k",
        type=int,
        default=config.TOPIC_TOP_K,
        dest="topic_top_k",
        help="Max tweets kept per candidate by topic retrieval.",
    )
    parser.add_argument(
        "--topic-min-sim",
        type=float,
        default=config.TOPIC_MIN_SIM,
        dest="topic_min_sim",
        help="Score floor for topic retrieval. 0.0 = pure top-K, never empty.",
    )
    parser.add_argument(
        "--only-tweets",
        action="store_true",
        dest="only_tweets",
        help="Exclude voting history from prompts — use only tweets as context.",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        dest="no_context",
        help="Hide member profile and bill info from the prompt — LLM only sees tweets and voting history.",
    )

    # ---- LLM settings ------------------------------------------------------
    parser.add_argument(
        "--model",
        default=config.DEFAULT_MODEL,
        help="OpenRouter model identifier.",
    )

    # ---- Output ------------------------------------------------------------
    parser.add_argument(
        "--output",
        type=Path,
        default=config.PREDICTIONS_CSV,
        help="Path to the predictions CSV (supports resume).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing results and start fresh.",
    )
    parser.add_argument(
        "--sample-cache",
        type=Path,
        default=None,
        dest="sample_cache",
        help="Path to save/load sample cache (pickle). Skips feature collection + retrieval on load.",
    )

    # ---- Run modes ---------------------------------------------------------
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after predictions complete.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        dest="evaluate_only",
        help="Skip predictions; only run evaluation on existing results CSV.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Compute bootstrap confidence intervals (B=10000, 95%% CI) after evaluation.",
    )
    parser.add_argument(
        "--restrict-to",
        type=Path,
        default=None,
        dest="restrict_to",
        help="Restrict evaluation to (vote_id, bioguide) pairs with a "
             "non-ERROR prediction in this reference results CSV (e.g. "
             "results/baseline_dw_nominate.csv), for apples-to-apples "
             "comparison across methods.",
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Use proportional stratified sampling across congress, chamber, "
             "party, vote_cast, and vote competitiveness. Requires --sample N.",
    )
    parser.add_argument(
        "--allow-zero-tweets",
        action="store_true",
        dest="allow_zero_tweets",
        help="Disable the requirement that a candidate have >=1 raw tweet in the "
             "n-days pre-vote window. Off by default, every sampling mode (plain "
             "--sample and --stratify alike) drops any candidate with 0 raw tweets "
             "in that window -- which silently excludes any vote before the tweet "
             "corpus starts (2010-11-06), regardless of sampling mode. Pass this "
             "for a true unconditioned random or stratified sample proportional to "
             "the full vote population; combine with --stratify for both at once.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Build prompts and print them without calling the LLM. "
            "Useful for inspecting prompt quality. Combine with --sample N."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dry-run helper
# ---------------------------------------------------------------------------

def dry_run(args: argparse.Namespace) -> None:
    """Load data, build prompts, and print them — no LLM calls."""
    import random
    import pandas as pd
    from pipeline import data_loader
    from pipeline import feature_builder as fb
    from pipeline import prompt_builder
    from pipeline import tweet_retriever

    logger.info("DRY RUN — loading data …")
    votes_df   = data_loader.load_votes(args.chamber)
    tweets_df  = data_loader.load_tweets()
    tweet_idx  = data_loader.build_tweet_index(tweets_df)

    n = args.sample or 3
    random.seed(args.seed)
    rows = votes_df.sample(min(n, len(votes_df)), random_state=args.seed)

    for _, row in rows.iterrows():
        bioguide  = row["bioguide"]
        vote_date = row["vote_date"]

        profile    = fb.get_member_profile(bioguide, votes_df)
        raw_tweets = fb.get_recent_tweets(bioguide, vote_date, tweet_idx,
                                          n_days=args.n_days, max_k=args.max_tweets_raw)
        past_votes = [] if args.only_tweets else fb.get_past_votes(bioguide, vote_date, votes_df, n=args.n_past_votes)

        bill_summary = row.get("bill_summary", "")

        # Tweet retrieval (or skip)
        use_retrieval = raw_tweets and not args.skip_retrieval and args.retrieval_method != "none"

        if use_retrieval and args.retrieval_method == "topic":
            # No prompt to show — topic retrieval scores vectors, it does not
            # call a model.
            from pipeline import topic_retriever

            retrieval_prompt = None
            filtered_tweets, scores = topic_retriever.retrieve_with_scores(
                raw_tweets,
                bill_title=row["bill_title"],
                question=row["question"],
                bill_summary=bill_summary,
                top_k=args.max_tweets_prompt,
                min_sim=args.topic_min_sim,
            )
            tweets = filtered_tweets
            n_used = len(tweets)
            mean_score = sum(scores) / len(scores) if scores else 0.0
            logger.info(
                "Topic retrieval: %d/%d tweets (mean score %.3f) for %s",
                n_used, len(raw_tweets), mean_score, row["vote_id"],
            )
        elif use_retrieval:
            retrieval_prompt = tweet_retriever._RETRIEVAL_USER_TEMPLATE.format(
                bill_number=row["bill_number"],
                bill_title=row["bill_title"],
                bill_summary=bill_summary.strip() if bill_summary else "(no official summary available)",
                question=row["question"],
                member_name=profile["member_name"],
                tweet_block=tweet_retriever._build_tweet_block(raw_tweets),
            )
            filtered_tweets, n_used = tweet_retriever.retrieve_relevant_tweets(
                raw_tweets,
                row["bill_number"], row["bill_title"], row["question"],
                profile["member_name"], args.model,
                bill_summary,
            )
            tweets = filtered_tweets[:args.max_tweets_prompt]
            logger.info("Retrieved %d/%d tweets for %s", n_used, len(raw_tweets), row["vote_id"])
        else:
            retrieval_prompt = None
            tweets = raw_tweets[:args.max_tweets_prompt]
            n_used = len(tweets)

        bill_info = {
            "vote_id":     row["vote_id"],
            "vote_date":   vote_date,
            "bill_number": row["bill_number"],
            "bill_title":  row["bill_title"],
            "bill_summary": bill_summary,
            "question":    row["question"],
            "chamber":     row["chamber"],
            "congress":    row["congress"],
            "session":     row["session"],
        }

        prompt = prompt_builder.build_prompt(profile, bill_info, past_votes, tweets, no_context=args.no_context)

        print("\n" + "=" * 70)
        print(f"TRUE LABEL : {row['vote_cast']}")
        print(f"TWEETS USED: {n_used} (raw={len(raw_tweets)}, prompt={len(tweets)})")
        print("=" * 70)
        if retrieval_prompt is not None:
            print("--- RETRIEVAL PROMPT (system prompt omitted) ---")
            print(retrieval_prompt)
            print("--- PREDICTION PROMPT ---")
        print(prompt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Evaluate-only mode ------------------------------------------------
    if args.evaluate_only:
        evaluate(args.output, restrict_to=args.restrict_to)
        if args.bootstrap:
            bootstrap_evaluate(args.output, restrict_to=args.restrict_to)
        return

    # ---- Dry-run mode ------------------------------------------------------
    if args.dry_run:
        dry_run(args)
        return

    # ---- Normal prediction run ---------------------------------------------
    logger.info(
        "Starting pipeline | chamber=%s | model=%s | sample=%s | "
        "n_days=%d | n_past_votes=%d | raw=%d | prompt=%d | min_tweets=%d | "
        "retrieval=%s | skip_retrieval=%s | stratify=%s",
        args.chamber, args.model, args.sample,
        args.n_days, args.n_past_votes,
        args.max_tweets_raw, args.max_tweets_prompt,
        args.min_tweets, args.retrieval_method, args.skip_retrieval, args.stratify,
    )
    if args.retrieval_method == "topic":
        config.TOPIC_TOP_K = args.topic_top_k
        config.TOPIC_MIN_SIM = args.topic_min_sim
        logger.info(
            "Topic retrieval | top_k=%d | min_sim=%.3f | weights topic=%.2f emb=%.2f",
            config.TOPIC_TOP_K, config.TOPIC_MIN_SIM,
            config.TOPIC_WEIGHT, config.EMB_WEIGHT,
        )

    pipe = VotePredictionPipeline(
        chamber=args.chamber,
        n_days=args.n_days,
        n_past_votes=args.n_past_votes,
        max_tweets_raw=args.max_tweets_raw,
        max_tweets_prompt=args.max_tweets_prompt,
        skip_retrieval=args.skip_retrieval,
        retrieval_method=args.retrieval_method,
        min_tweets=args.min_tweets,
        model=args.model,
        sample=args.sample,
        resume=not args.no_resume,
        output_csv=args.output,
        sample_cache=args.sample_cache,
        random_seed=args.seed,
        only_tweets=args.only_tweets,
        no_context=args.no_context,
        stratify=args.stratify,
        require_tweets=not args.allow_zero_tweets,
    )
    pipe.run()

    # ---- Optional evaluation after predictions -----------------------------
    if args.evaluate:
        evaluate(args.output, restrict_to=args.restrict_to)
    if args.bootstrap:
        bootstrap_evaluate(args.output, restrict_to=args.restrict_to)


if __name__ == "__main__":
    main()
