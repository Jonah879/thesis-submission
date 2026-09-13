"""
pipeline.py — Main orchestration class.

Usage:
    from pipeline import VotePredictionPipeline

    pipe = VotePredictionPipeline(
        chamber="both",
        n_days=30,
        n_past_votes=10,
        model="openai/gpt-4o-mini",
        sample=500,
        resume=True,
    )
    pipe.run()
"""

import csv
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from . import data_loader
from . import feature_builder as fb
from . import llm_client
from . import prompt_builder
from . import sample_cache
from . import tweet_retriever
from . import vote_filters

logger = logging.getLogger(__name__)

# CSV columns written to predictions.csv
#
# NOTE: the header is only written when the output file does not yet exist
# (see _open_writer), so adding a column here breaks appending onto a CSV
# produced by an earlier version. Use a fresh --output path.
RESULT_COLUMNS = [
    "vote_id",
    "bioguide",
    "member_name",
    "party",
    "state",
    "chamber",
    "congress",
    "bill_number",
    "bill_title",
    "vote_date",
    "true_label",
    "predicted_label",
    "reasoning",
    "prompt_tokens",
    "completion_tokens",
    "n_tweets_raw",
    "n_tweets_used",
    "competitiveness",
    "had_bill_summary",
    "retrieval_method",
    "mean_similarity",
    "timestamp",
]


class VotePredictionPipeline:
    def __init__(
        self,
        chamber: Literal["house", "senate", "both"] = "both",
        n_days: int = config.N_DAYS_TWEETS,
        n_past_votes: int = config.N_PAST_VOTES,
        max_tweets_raw: int = config.MAX_TWEETS_RAW,
        max_tweets_prompt: int = config.MAX_TWEETS_PROMPT,
        skip_retrieval: bool = config.SKIP_RETRIEVAL,
        retrieval_method: str = config.RETRIEVAL_METHOD,
        min_tweets: int = 0,
        model: str = config.DEFAULT_MODEL,
        sample: Optional[int] = None,
        resume: bool = True,
        output_csv: Optional[Path] = None,
        sample_cache: Optional[Path] = None,
        random_seed: int = 42,
        only_tweets: bool = False,
        no_context: bool = False,
        stratify: bool = False,
        require_tweets: bool = True,
    ):
        self.chamber           = chamber
        self.n_days            = n_days
        self.n_past_votes      = n_past_votes
        self.max_tweets_raw    = max_tweets_raw
        self.max_tweets_prompt = max_tweets_prompt
        self.skip_retrieval    = skip_retrieval
        self.retrieval_method  = retrieval_method
        self.min_tweets        = min_tweets
        self.model             = model
        self.sample            = sample
        self.resume            = resume
        self.output_csv        = Path(output_csv) if output_csv else config.PREDICTIONS_CSV
        self.sample_cache      = Path(sample_cache) if sample_cache else None
        self.random_seed       = random_seed
        self.only_tweets       = only_tweets
        self.no_context        = no_context
        self.stratify          = stratify
        self.require_tweets    = require_tweets

        # Loaded lazily in run()
        self.votes_df: Optional[pd.DataFrame] = None
        self.tweet_index: Optional[dict]       = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        logger.info("Loading vote data …")
        votes_df = data_loader.load_votes(self.chamber)

        # Nomination/confirmation votes have no bill text to match tweets
        # against. Filtering here covers both candidate builders at once.
        if config.DROP_NOMINATION_VOTES:
            votes_df = vote_filters.filter_nominations(votes_df)

        logger.info("Adding vote competitiveness …")
        votes_df = self._add_competitiveness(votes_df)
        logger.info("Loading tweets …")
        tweets_df = data_loader.load_tweets()

        logger.info("Building tweet index …")
        self.tweet_index = data_loader.build_tweet_index(tweets_df)
        data_loader.log_coverage(votes_df, self.tweet_index)

        data_loader.save_member_splits(votes_df, self.tweet_index)

        self.votes_df = data_loader.filter_to_covered(votes_df, self.tweet_index)
        data_loader.save_loaded_votes(self.votes_df)

    # ------------------------------------------------------------------
    # Candidate pairs
    # ------------------------------------------------------------------

    def _build_candidates(self, already_done: set) -> list[dict]:
        """
        Return a list of dicts, one per (bioguide, vote_id) pair to predict.

        Each dict contains all metadata needed to build the prompt and
        write the result row.
        """
        df = self.votes_df

        candidates = []
        for _, row in df.iterrows():
            key = (row["bioguide"], row["vote_id"])
            if key in already_done:
                continue
            # Skip candidates with missing bill title
            if pd.isna(row["bill_title"]) or str(row["bill_title"]).strip() == "":
                continue
            candidates.append({
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
            })

        if self.sample is not None and self.sample < len(candidates):
            random.seed(self.random_seed)
            candidates = random.sample(candidates, self.sample)
            logger.info("Sampled %d candidate pairs.", len(candidates))
        else:
            logger.info("Processing all %d candidate pairs.", len(candidates))

        return candidates

    # ------------------------------------------------------------------
    # Competitiveness & stratified sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_party(party: str) -> str:
        """Normalize raw party codes to D / R / I."""
        if pd.isna(party):
            return "I"
        p = str(party).strip().upper()
        if p.startswith("D"):
            return "D"
        if p.startswith("R"):
            return "R"
        return "I"

    @staticmethod
    def _congress_to_era(congress: int) -> str:
        """Group congresses into political eras for stratified sampling."""
        if congress <= 111:
            return "1_Bush_Obama_start"    # 110-111: 2007-2011
        elif congress <= 114:
            return "2_Obama_polarization"  # 112-114: 2011-2017
        elif congress <= 116:
            return "3_Trump1"              # 115-116: 2017-2021
        elif congress <= 118:
            return "4_Biden"               # 117-118: 2021-2025
        else:
            return "5_Trump2"              # 119:      2025-2027

    def _add_competitiveness(self, votes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'competitiveness' column to votes_df.

        Computed per vote_id: winner_fraction = max(yea_frac, nay_frac).
        Classification uses config.COMPETITIVENESS_THRESHOLDS:
            (contested_upper, moderate_upper)
            winner < contested_upper  -> "contested"
            contested_upper <= winner < moderate_upper -> "moderate"
            winner >= moderate_upper -> "unanimous"
        """
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

    def _build_stratified_sample(self, n: int, already_done: set) -> list[dict]:
        """
        Proportional stratified sampling across 5 dimensions:
            congress x chamber x party_norm x vote_cast x competitiveness

        Allocation: proportional to stratum size with a minimum floor
        (config.MIN_PER_STRATUM). If total exceeds n, above-floor
        allocations are scaled down.

        Saves the sample to config.STRATIFIED_SAMPLE_CSV for DW-NOMINATE
        analysis.
        """
        df = self.votes_df.copy()

        # Filter out candidates with missing bill title
        n_before = len(df)
        df = df[df["bill_title"].notna() & (df["bill_title"].str.strip() != "")]
        n_after = len(df)
        if n_before != n_after:
            logger.info(
                "Stratified sample: dropped %d rows with missing bill_title (%d → %d)",
                n_before - n_after, n_before, n_after,
            )

        # Filter out already-done pairs
        done_mask = df.apply(
            lambda r: (r["bioguide"], r["vote_id"]) in already_done, axis=1
        )
        df = df[~done_mask].copy()
        total_pool = len(df)

        if total_pool == 0:
            logger.warning("No candidates left after filtering already-done pairs.")
            return []

        # Add party_norm
        df["party_norm"] = df["party"].apply(self._normalize_party)

        # Drop I-party (Independents) — too few members (10) for meaningful stratification
        n_before = len(df)
        df = df[df["party_norm"] != "I"].copy()
        n_after = len(df)
        if n_before != n_after:
            logger.info(
                "Stratified sample: dropped %d I-party rows (%d → %d)",
                n_before - n_after, n_before, n_after,
            )

        # Add congress era for stratification
        df["congress_era"] = df["congress"].apply(self._congress_to_era)

        strata_cols = ["congress_era", "chamber", "party_norm", "vote_cast", "competitiveness"]
        strata = df.groupby(strata_cols, sort=False)

        n_strata = len(strata)
        min_per = config.MIN_PER_STRATUM

        logger.info(
            "Stratified sampling: %d strata, %d pool rows, target n=%d, min_per_stratum=%d",
            n_strata, total_pool, n, min_per,
        )

        # Phase 1: compute allocation per stratum
        allocations = {}
        for name, group in strata:
            proportional = int(n * len(group) / total_pool)
            alloc = max(proportional, min_per)
            alloc = min(alloc, len(group))
            allocations[name] = alloc

        total_allocated = sum(allocations.values())

        # Phase 2: if over-allocated, scale down above-minimum strata
        if total_allocated > n:
            deficit = total_allocated - n
            above_min = {k: v for k, v in allocations.items() if v > min_per}
            total_above_min_excess = sum(v - min_per for v in above_min.values())

            if total_above_min_excess > 0:
                for k in above_min:
                    reduction = int(deficit * (above_min[k] - min_per) / total_above_min_excess)
                    allocations[k] = max(min_per, allocations[k] - reduction)

            total_allocated = sum(allocations.values())
            # If still over, trim largest strata
            while total_allocated > n:
                k_max = max(allocations, key=allocations.get)
                allocations[k_max] -= 1
                total_allocated -= 1

        # Phase 3: sample within each stratum
        sampled_parts = []
        rng = random.Random(self.random_seed)
        for name, group in strata:
            alloc = allocations.get(name, 0)
            if alloc <= 0:
                continue
            alloc = min(alloc, len(group))
            sampled_idx = rng.sample(range(len(group)), alloc)
            sampled_parts.append(group.iloc[sampled_idx])

        sample_df = pd.concat(sampled_parts, ignore_index=True)

        # Shuffle for mixed processing order
        sample_df = sample_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        # Save for DW-NOMINATE analysis
        sample_df.to_csv(config.STRATIFIED_SAMPLE_CSV, index=False)
        logger.info(
            "Stratified sample saved to %s (%d rows, %d strata)",
            config.STRATIFIED_SAMPLE_CSV, len(sample_df), n_strata,
        )

        # Log stratum distribution
        dist = sample_df.groupby(strata_cols).size()
        logger.info("Sample distribution across strata:")
        for name, count in dist.items():
            logger.info("  %s: %d", name, count)

        # Convert to list of dicts (same format as _build_candidates)
        candidates = []
        for _, row in sample_df.iterrows():
            candidates.append({
                "bioguide":       row["bioguide"],
                "vote_id":        row["vote_id"],
                "vote_date":      row["vote_date"],
                "bill_number":    row["bill_number"],
                "bill_title":     row["bill_title"],
                "bill_summary":   row.get("bill_summary", ""),
                "question":       row["question"],
                "chamber":        row["chamber"],
                "congress":       row["congress"],
                "session":        row["session"],
                "true_label":     row["vote_cast"],
                "competitiveness": row.get("competitiveness", ""),
            })

        return candidates

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def _load_already_done(self) -> set:
        """Return set of (bioguide, vote_id) tuples already present in output CSV."""
        if not self.resume or not self.output_csv.exists():
            return set()
        try:
            existing = pd.read_csv(self.output_csv, usecols=["vote_id", "bioguide"])
            done = set(zip(existing["bioguide"], existing["vote_id"]))
            logger.info("Resuming: %d pairs already completed.", len(done))
            return done
        except Exception as exc:
            logger.warning("Could not read existing results (%s). Starting fresh.", exc)
            return set()

    # ------------------------------------------------------------------
    # CSV writer
    # ------------------------------------------------------------------

    def _open_writer(self):
        """Open the output CSV for appending, writing the header if new."""
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.output_csv.exists()
        fh = open(self.output_csv, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        return fh, writer

    # ------------------------------------------------------------------
    # Phase 2b — minimum-tweets filter
    # ------------------------------------------------------------------

    def _apply_min_tweets(self, tasks: list) -> list:
        """Drop candidates with fewer than min_tweets relevant tweets."""
        if self.min_tweets <= 0:
            return tasks
        if self.max_tweets_prompt == 0:
            logger.warning(
                "--min-tweets %d with --max-tweets-prompt 0 will drop EVERY candidate "
                "(no tweets reach the prompt, so n_tweets_used is 0 for all). "
                "Use --min-tweets 0 for a tweets-ablation run.",
                self.min_tweets,
            )
        n_before = len(tasks)
        kept = [t for t in tasks if t["n_tweets_used"] >= self.min_tweets]
        n_dropped = n_before - len(kept)
        if n_dropped > 0:
            logger.info(
                "Phase 2b: dropped %d candidates with < %d relevant tweets (%d → %d)",
                n_dropped, self.min_tweets, n_before, len(kept),
            )
        if not kept:
            logger.info("No candidates with >= %d tweets remaining.", self.min_tweets)
        return kept

    # ------------------------------------------------------------------
    # Phase 2 — retrieval strategies
    # ------------------------------------------------------------------

    def _set_retrieval_result(self, task: dict, filtered: list, scores=None) -> None:
        """Attach retrieval outputs to a task in one place."""
        task["filtered_tweets"] = filtered[: self.max_tweets_prompt]
        task["n_tweets_raw"]    = len(task["raw_tweets"])
        task["n_tweets_used"]   = len(task["filtered_tweets"])
        if scores:
            used = scores[: self.max_tweets_prompt]
            task["mean_similarity"] = round(float(np.mean(used)), 4) if used else ""
        else:
            task["mean_similarity"] = task.get("mean_similarity", "")

    def _retrieve_none(self, tasks: list) -> None:
        """No filtering — pass the raw tweets straight through."""
        logger.info("Retrieval disabled — using raw tweets for %d candidates.", len(tasks))
        for t in tasks:
            self._set_retrieval_result(t, t["raw_tweets"])

    def _retrieve_llm(self, tasks: list) -> None:
        """LLM relevance judgment — one call per candidate, thread-pooled."""
        logger.info(
            "Retrieving relevant tweets via LLM for %d candidates (concurrency=%d) …",
            len(tasks), config.MAX_CONCURRENCY,
        )
        with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENCY) as pool:
            futures = {
                pool.submit(
                    tweet_retriever.retrieve_relevant_tweets,
                    t["raw_tweets"],
                    t["item"]["bill_number"],
                    t["item"]["bill_title"],
                    t["item"]["question"],
                    t["profile"]["member_name"],
                    self.model,
                    t["item"].get("bill_summary", ""),
                ): t
                for t in tasks
            }
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Retrieving tweets", unit="pair",
            ):
                task = futures[future]
                filtered_tweets, _n_retrieved = future.result()
                self._set_retrieval_result(task, filtered_tweets)

    def _retrieve_topic(self, tasks: list) -> None:
        """
        Topic-model gating.

        Everything is embedded in one batched pass first, so the per-candidate
        step is pure vector arithmetic — no thread pool, no network.
        """
        from . import topic_model
        from . import topic_retriever

        logger.info("Topic retrieval: pre-embedding texts for %d candidates …", len(tasks))
        texts = []
        for t in tasks:
            texts.extend(t["raw_tweets"])
            texts.extend(topic_model.build_vote_query_texts(
                t["item"]["bill_title"],
                t["item"].get("bill_summary", ""),
                t["item"]["question"],
            ))
        topic_retriever.warm_texts(texts, show_progress=True)

        n_empty = 0
        for t in tqdm(tasks, desc="Scoring tweets", unit="pair"):
            selected, scores = topic_retriever.retrieve_with_scores(
                t["raw_tweets"],
                bill_title=t["item"]["bill_title"],
                question=t["item"]["question"],
                bill_summary=t["item"].get("bill_summary", ""),
                top_k=self.max_tweets_prompt,
            )
            self._set_retrieval_result(t, selected, scores)
            if not selected:
                n_empty += 1

        logger.info(
            "Topic retrieval: %d/%d candidates got zero relevant tweets (%.1f%%) "
            "at min_sim=%.2f.",
            n_empty, len(tasks), n_empty / len(tasks) * 100 if tasks else 0,
            config.TOPIC_MIN_SIM,
        )

    # ------------------------------------------------------------------
    # Phase 3 — prompt assembly (shared by the cache and fresh paths)
    # ------------------------------------------------------------------

    def _build_prompt_for(self, task: dict) -> str:
        """
        Assemble the prediction prompt for one task.

        Feature ablations are applied HERE, at the point of use, rather than
        during feature collection. That is deliberate: the sample cache stores
        fully-populated features, and only this method is reached by both the
        cached and the fresh path. Suppressing a feature earlier would silently
        do nothing on a cached run — which is exactly how the original
        predictions_no_voting.csv ended up identical to the full run.

        Ablation levers (no dedicated flags needed):
            --only-tweets           -> past_votes = []
            --n-past-votes 0        -> past_votes = []
            --max-tweets-prompt 0   -> recent_tweets = []

        An emptied feature still renders its section header plus a
        "(no … available)" placeholder; see prompt_builder.build_prompt.
        """
        item = task["item"]
        bill_info = {
            "vote_id":     item["vote_id"],
            "vote_date":   item["vote_date"],
            "bill_number": item["bill_number"],
            "bill_title":  item["bill_title"],
            "bill_summary": item.get("bill_summary", ""),
            "question":    item["question"],
            "chamber":     item["chamber"],
            "congress":    item["congress"],
            "session":     item["session"],
        }

        past_votes = [] if self.only_tweets else task["past_votes"][: self.n_past_votes]
        # Normally a no-op (_set_retrieval_result already truncated), but keeps
        # the ablation explicit and stays correct if the cache was built with a
        # larger max_tweets_prompt than this run uses.
        tweets = task["filtered_tweets"][: self.max_tweets_prompt]

        return prompt_builder.build_prompt(
            member_profile=task["profile"],
            bill_info=bill_info,
            past_votes=past_votes,
            recent_tweets=tweets,
            no_context=self.no_context,
        )

    def _log_active_ablation(self) -> None:
        """
        State the active condition up front, so a run's intent is recoverable
        from its log. RESULT_COLUMNS has no field for this, and the absence of
        such a record is what made the earlier failed ablation hard to spot.
        """
        active = []
        if self.only_tweets:
            active.append("--only-tweets (past voting history suppressed)")
        if self.n_past_votes == 0:
            active.append("--n-past-votes 0 (past voting history suppressed)")
        if self.max_tweets_prompt == 0:
            active.append("--max-tweets-prompt 0 (tweets suppressed)")
        if self.no_context:
            active.append("--no-context (member profile + bill info suppressed)")
        if not self.require_tweets:
            active.append("--allow-zero-tweets (candidates with 0 raw tweets are kept, "
                           "not dropped -- this run is NOT tweet-conditioned)")

        logger.info("Condition: %s", " | ".join(active) if active else "FULL (no ablation)")
        logger.info("Output:    %s", self.output_csv)
        if active:
            logger.info(
                "Verify afterwards: mean prompt_tokens must differ from the full run "
                "(no voting history ≈ -40%, no tweets ≈ -20%)."
            )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full pipeline end-to-end."""
        self._log_active_ablation()
        self._load_data()

        already_done = self._load_already_done()

        if self.stratify and self.sample is not None:
            candidates = self._build_stratified_sample(self.sample, already_done)
        else:
            candidates   = self._build_candidates(already_done)

        if not candidates:
            logger.info("Nothing to do — all pairs already predicted.")
            return

        fh, writer = self._open_writer()

        try:
            # ---- Phase 0: Load sample cache (skip Phase 1+2 if cached) ----
            tasks = None
            if self.sample_cache and self.sample_cache.exists():
                logger.info("Loading sample cache from %s …", self.sample_cache)
                tasks = sample_cache.load(self.sample_cache)
                # Filter out already-done pairs
                tasks = [t for t in tasks if (t["item"]["bioguide"], t["item"]["vote_id"]) not in already_done]
                if not tasks:
                    logger.info("All cached pairs already predicted.")
                    return
                logger.info("Loaded %d candidates from cache (skipping feature collection).", len(tasks))

                # The cache keeps raw_tweets separate from filtered_tweets and
                # never mutates them, so a different retrieval method can be
                # applied to the identical candidate set at no LLM cost. This
                # is the matched A/B against the cached run.
                if self.retrieval_method == "topic":
                    logger.info("Re-filtering cached candidates with topic retrieval …")
                    self._retrieve_topic(tasks)
                elif self.skip_retrieval or self.retrieval_method == "none":
                    self._retrieve_none(tasks)
                else:
                    # Reuse the cached LLM selection, but honour the current
                    # max_tweets_prompt rather than the cache-time value.
                    for t in tasks:
                        self._set_retrieval_result(t, t.get("filtered_tweets", []))

                # min_tweets is otherwise a no-op on the cache path.
                tasks = self._apply_min_tweets(tasks)
                if not tasks:
                    return

            # ---- Phase 1: Collect features sequentially -------------------
            if tasks is None:
                logger.info("Collecting features for %d candidates …", len(candidates))
                tasks = []
                for item in tqdm(candidates, desc="Collecting features", unit="pair"):
                    bioguide  = item["bioguide"]
                    vote_date = item["vote_date"]

                    profile = fb.get_member_profile(bioguide, self.votes_df)
                    raw_tweets = fb.get_recent_tweets(
                        bioguide, vote_date, self.tweet_index,
                        n_days=self.n_days, max_k=self.max_tweets_raw,
                    )

                    # Always collect the full history. Ablation happens at
                    # prompt-assembly time (_build_prompt_for) so it applies on
                    # the cached path too — and so a cache saved below never
                    # bakes in a condition-specific feature set.
                    past_votes = fb.get_past_votes(
                        bioguide, vote_date, self.votes_df, n=config.N_PAST_VOTES
                    )

                    vote_date_str = (
                        vote_date.strftime("%Y-%m-%d")
                        if isinstance(vote_date, pd.Timestamp)
                        else str(vote_date)
                    )

                    tasks.append({
                        "item":          item,
                        "profile":       profile,
                        "raw_tweets":    raw_tweets,
                        "past_votes":    past_votes,
                        "vote_date_str": vote_date_str,
                    })

                # ---- Phase 1b: Filter candidates with no tweets --------------
                # Every sampling mode (plain --sample and --stratify alike) reaches
                # this same point, so require_tweets=False is what makes a *true*
                # unconditioned random/stratified sample possible -- without it, no
                # vote before the tweet corpus starts (2010-11-06) can ever survive,
                # in any sampling mode, because its window has nothing to look at.
                if self.require_tweets:
                    n_before = len(tasks)
                    tasks = [t for t in tasks if len(t["raw_tweets"]) > 0]
                    n_dropped = n_before - len(tasks)
                    if n_dropped > 0:
                        logger.info(
                            "Dropped %d candidates with no tweets in %d-day window (%d → %d)",
                            n_dropped, self.n_days, n_before, len(tasks),
                        )
                    if not tasks:
                        logger.info("No candidates with tweets remaining. Nothing to do.")
                        return
                else:
                    n_zero = sum(1 for t in tasks if len(t["raw_tweets"]) == 0)
                    logger.info(
                        "require_tweets=False: keeping all %d candidates, including %d "
                        "with zero raw tweets in the %d-day window.",
                        len(tasks), n_zero, self.n_days,
                    )

                # ---- Phase 2: Batch-retrieve relevant tweets (optional) -------
                if self.skip_retrieval or self.retrieval_method == "none":
                    self._retrieve_none(tasks)
                elif self.retrieval_method == "topic":
                    self._retrieve_topic(tasks)
                else:
                    self._retrieve_llm(tasks)

                # ---- Save sample cache after Phase 2 ---------------------------
                if self.sample_cache:
                    sample_cache.save(tasks, self.sample_cache)

                # ---- Phase 2b: Filter by minimum tweets ----------------------
                tasks = self._apply_min_tweets(tasks)
                if not tasks:
                    return

            # ---- Phase 3: Build prediction prompts -------------------------
            logger.info("Building prompts for %d candidates …", len(tasks))
            for t in tasks:
                t["prompt"] = self._build_prompt_for(t)

            # ---- Phase 4: Concurrent LLM inference -------------------------
            logger.info(
                "Sending %d prompts to LLM (concurrency=%d) …",
                len(tasks), config.MAX_CONCURRENCY,
            )
            with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENCY) as pool:
                futures = {
                    pool.submit(llm_client.predict, t["prompt"], self.model): t
                    for t in tasks
                }
                for future in tqdm(
                    as_completed(futures), total=len(futures),
                    desc="Predicting votes", unit="pair",
                ):
                    task    = futures[future]
                    item    = task["item"]
                    profile = task["profile"]
                    result  = future.result()

                    writer.writerow({
                        "vote_id":           item["vote_id"],
                        "bioguide":          item["bioguide"],
                        "member_name":       profile.get("member_name", ""),
                        "party":             profile.get("party", ""),
                        "state":             profile.get("state", ""),
                        "chamber":           item["chamber"],
                        "congress":          item["congress"],
                        "bill_number":       item["bill_number"],
                        "bill_title":        item["bill_title"],
                        "vote_date":         task["vote_date_str"],
                        "true_label":        item["true_label"],
                        "predicted_label":   result["prediction"],
                        "reasoning":         result["reasoning"],
                        "prompt_tokens":     result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "n_tweets_raw":      task["n_tweets_raw"],
                        "n_tweets_used":     task["n_tweets_used"],
                        "competitiveness":   item.get("competitiveness", ""),
                        "had_bill_summary":  bool(str(item.get("bill_summary", "")).strip()),
                        "retrieval_method":  "none" if self.skip_retrieval else self.retrieval_method,
                        "mean_similarity":   task.get("mean_similarity", ""),
                        "timestamp":         datetime.utcnow().isoformat(),
                    })
                    fh.flush()

        finally:
            fh.close()

        logger.info("Done. Results written to %s", self.output_csv)
