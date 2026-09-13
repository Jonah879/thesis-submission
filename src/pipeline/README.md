# Congressional Vote Prediction Pipeline

LLM-based pipeline that predicts how a member of Congress will vote on a bill, using their recent tweets and past voting history as context.

## Prerequisites

1. **Preprocess vote data** (run once from `src/`):
   ```bash
   python preprocess_votes.py
   ```
   Generates `data/house_votes_members.csv` and `data/senate_votes_members.csv`.

2. **Tweets data**: `data/tweets_full.pkl` must exist with a `bioguide_id` column mapping tweets to members.

3. **API key**: Set the OpenRouter API key (unless using a local model):
   ```bash
   export OPENROUTER_API_KEY="sk-or-..."
   ```

4. **Dependencies**:
   ```bash
   pip install pandas numpy openai python-dotenv scikit-learn tqdm pyyaml

   # additionally, for topic-based retrieval:
   pip install sentence-transformers bertopic umap-learn hdbscan
   ```

5. **Topic space** (only for `--retrieval-method topic`): fit it once, on a
   GPU machine. Reads `config.TWEETS_TOPIC_PKL` — a cleaned corpus with
   `id`, `bioguide_id`, `created_at`, `text`.
   ```bash
   python fit_topic_model.py
   ```

## Quick Start

```bash
cd src

# Predict 500 random (member, vote) pairs from both chambers
python run_pipeline.py --sample 500

# Predict and evaluate
python run_pipeline.py --sample 500 --evaluate

# House only, with a specific model
python run_pipeline.py --chamber house --model meta-llama/llama-3.3-70b-instruct:free --sample 200

# Skip tweet retrieval (use raw tweets directly)
python run_pipeline.py --sample 500 --skip-retrieval

# Dry-run: inspect prompts without calling the LLM
python run_pipeline.py --dry-run --sample 3 --skip-retrieval

# Stratified sample with 15,000 pairs
python run_pipeline.py --stratify --sample 15000 --evaluate

# Save sample cache for ablation studies (run once, reuse for many configs)
python run_pipeline.py --sample 10000 --sample-cache data/sample_cache.pkl

# Topic-model retrieval instead of the LLM retriever (needs fit_topic_model.py first)
python run_pipeline.py --retrieval-method topic --sample 10000 \
    --output results/predictions_topic.csv --evaluate

# Matched A/B: re-filter an existing cache with topic retrieval — no LLM cost,
# identical candidate set and identical raw tweet pool as the cached LLM run
python run_pipeline.py --retrieval-method topic \
    --sample-cache data/sample_cache.pkl \
    --output results/predictions_topic.csv --evaluate --bootstrap

# Reuse cache for different configs (skips feature collection + retrieval)
python run_pipeline.py --sample 10000 --only-tweets --sample-cache data/sample_cache.pkl
python run_pipeline.py --sample 10000 --no-context --sample-cache data/sample_cache.pkl
```

## Pipeline Architecture

The pipeline runs in 4 phases:

1. **Collect features** — For each (member, vote) pair, gather recent tweets, past votes, and member profile
2. **Retrieve tweets** — Use an LLM to filter tweets to only those directly relevant to the bill (skippable with `--skip-retrieval`)
3. **Build prompts** — Construct the full LLM prompt with member profile, bill info, tweets, and voting history
4. **Predict** — Call the LLM in parallel to get Yea/Nay predictions

### Sample Cache

For ablation studies, use `--sample-cache` to save the full sampled dataset (with retrieved tweets) to a pickle file. Future runs with different configurations can load this cache and skip feature collection + retrieval entirely.

**Building a high-quality cache (one-time):**

Use `build_sample_cache.py` to iteratively build a cache with guaranteed minimum tweets per candidate:

```bash
# Build a 10k sample where each candidate has at least 1 relevant tweet
python build_sample_cache.py --target-sample 10000 --min-tweets 1

# Build with stricter filter (5+ relevant tweets)
python build_sample_cache.py --target-sample 10000 --min-tweets 5
```

This script:
1. Pre-filters candidates by raw tweets (cheap, no LLM)
2. Iteratively retrieves tweets until target is reached
3. Saves a complete cache for all future runs

**Using the cache:**
```bash
# Run with cache (fast, skips feature collection + retrieval)
python run_pipeline.py --sample 10000 --sample-cache data/sample_cache.pkl

# Same data, different configs
python run_pipeline.py --sample 10000 --only-tweets --sample-cache data/sample_cache.pkl
python run_pipeline.py --sample 10000 --no-context --sample-cache data/sample_cache.pkl
```

**Behavior:**
- Cache is only saved when Phase 2 (retrieval) actually runs
- Cache is never overwritten once saved
- If cache exists, Phase 1 is skipped entirely
- Candidate matching is by `(bioguide, vote_id)` pairs
- The cache stores `raw_tweets` **separately** from `filtered_tweets` and never
  mutates the former. Loading a cache with `--retrieval-method topic` therefore
  **re-filters from the raw tweets** rather than reusing the cached LLM
  selection — that is the matched A/B. With `--retrieval-method llm` the cached
  selection is reused as before.
- `--min-tweets` is applied on the cache path too (it used to be skipped)

### Completeness Filters

The pipeline automatically filters out candidates with incomplete data:
- **Missing `bill_title`** — procedural votes (ADJOURN, JOURNAL, MOTION) are excluded at candidate selection
- **No tweets in window** — candidates with zero tweets in the 100-day window are dropped after feature collection

## Tweet Retrieval

Retrieval narrows a member's raw tweets to those relevant to the bill being voted
on. Pick the strategy with `--retrieval-method {llm,topic,none}`.

### Why there are two methods

The LLM retriever returns **zero relevant tweets for 92.8% of candidates**
(measured across both 10,664-row runs in `src/old/`: median `n_tweets_used` = 0
against a median of 147 raw tweets available). The tweet-augmented condition was
therefore almost never actually augmented — those runs are effectively
party + voting-history results.

The cause is structural, not a tuning issue: the prompt requires tweets to
*explicitly* mention the bill number/title/policy, rejects broadly-related
topics, and instructs "better to return too few"; a parse failure also yields an
empty list.

### `llm` — LLM relevance judgment (original)

One lightweight LLM call per candidate, thread-pooled at `MAX_CONCURRENCY`.
Deliberately strict: only tweets explicitly naming the bill or its specific
policy are kept; when in doubt, tweets are dropped.

### `topic` — topic-model gating

Scores every tweet against the vote instead of judging it in or out:

```
score = TOPIC_WEIGHT * topic_sim + EMB_WEIGHT * emb_sim
```

- `topic_sim` — cosine between the tweet's and the vote's topic distributions.
  This is the gate: a tweet about immigration matches a bill about immigration
  whether the tweet is for or against it. Stance is the prediction LLM's job.
- `emb_sim` — cosine between sentence embeddings, max-pooled over the vote's
  text chunks, so a long summary does not dilute a strong match on one section.

Selection is top-K above `--topic-min-sim`. Setting it to `0.0` gives pure
top-K, which never returns empty — the empty rate becomes a threshold you choose
rather than an outcome you get.

Requires a fitted topic space (`fit_topic_model.py`). Costs no LLM calls: all
texts are embedded once in a batched pass, after which scoring is vector
arithmetic.

### `none`

Pass raw tweets straight through. Equivalent to `--skip-retrieval`.

### Calibrating the threshold

Do this before running predictions — it is free and needs no LLM:

```bash
python rerank_cache.py --cache data/sample_cache.pkl --sweep
```

Prints the zero-relevant rate, tweet counts, and score distribution across
thresholds, next to the cached LLM baseline.

## Vote Scope: Nominations Excluded

`config.DROP_NOMINATION_VOTES` (default `True`) drops Senate nomination and
confirmation votes ("… Nancy L. Moritz, to be U.S. Circuit Judge"). They have no
bill text for the topic method to match against — 1,182 of the 1,196 Senate votes
lacking a summary are nominations — and they inflate accuracy (91.8% vs 82.6% on
bill votes) on a task the method cannot address. Detection is Senate-only, since
the House does not confirm nominations.

Consequences:
- Senate drops from 2,128 to ~947 votes, i.e. from 46% to ~21% of the vote
  population. Re-check stratum sizes against `MIN_PER_STRATUM = 30`.
- Prior runs included nominations, so their headline numbers are **not**
  directly comparable. Re-run the baselines on the filtered candidate set.
- Treaty votes are deliberately kept — substantive votes that merely lack a
  congress.gov summary.

### Tweet Caps

Two separate caps control tweet flow:
- `--max-tweets-raw` (default: 150) — tweets collected from the time window, sent to the retrieval LLM
- `--max-tweets-prompt` (default: 20) — tweets placed in the final prediction prompt (after retrieval)

These are decoupled because a wider net catches more relevant tweets during retrieval, while the prompt stays focused.

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--chamber` | `both` | `house`, `senate`, or `both` |
| `--sample N` | all | Randomly sample N (member, vote) pairs |
| `--seed` | `42` | Random seed for `--sample` |
| `--model` | `Qwen/Qwen2.5-32B-Instruct-AWQ` | OpenRouter model identifier |
| `--n-days` | `100` | Days before vote to look for tweets |
| `--n-past-votes` | `25` | Number of past votes to include |
| `--max-tweets-raw` | `150` | Max tweets collected from window and sent to retrieval |
| `--max-tweets-prompt` | `20` | Max tweets in final prediction prompt (after retrieval) |
| `--min-tweets` | `0` | Minimum relevant tweets required for prediction. Candidates with fewer are skipped. |
| `--retrieval-method` | `llm` | `llm`, `topic`, or `none`. `topic` needs a fitted topic space. |
| `--topic-top-k` | `25` | Max tweets kept per candidate by topic retrieval |
| `--topic-min-sim` | `0.25` | Score floor for topic retrieval. `0.0` = pure top-K, never empty |
| `--skip-retrieval` | off | Skip tweet relevance filtering — use raw tweets directly |
| `--only-tweets` | off | Exclude voting history from context |
| `--no-context` | off | Hide member profile and bill info |
| `--stratify` | off | Stratified sampling across congress_era/chamber/party/vote/competitiveness (requires `--sample`) |
| `--allow-zero-tweets` | off | Keep candidates with 0 raw tweets in the `--n-days` window instead of dropping them. Needed for a sample that isn't conditioned on tweet availability — see "True random / true stratified sampling" below. |
| `--sample-cache PATH` | None | Save/load sample cache (pickle). Skips feature collection + retrieval on load. |
| `--evaluate` | off | Run evaluation after predictions |
| `--evaluate-only` | off | Skip predictions, evaluate existing CSV |
| `--bootstrap` | off | Compute bootstrap confidence intervals (B=10000, 95% CI) |
| `--dry-run` | off | Print prompts without calling LLM |
| `--output` | `results/predictions.csv` | Output CSV path |
| `--no-resume` | off | Start fresh, ignore existing results |

## Output

Results are written to `src/results/predictions.csv` with columns:

| Column | Description |
|--------|-------------|
| `vote_id` | Unique vote identifier |
| `bioguide` | Member's Bioguide ID |
| `member_name` | Member's name |
| `party` | Party affiliation (D/R) |
| `state` | State |
| `chamber` | House or Senate |
| `congress` | Congress number |
| `bill_number` | Bill identifier |
| `bill_title` | Bill title |
| `vote_date` | Date of the vote |
| `true_label` | Actual vote cast (Yea/Nay) |
| `predicted_label` | LLM prediction (or ERROR for unmatched/invalid) |
| `reasoning` | LLM's explanation |
| `prompt_tokens` | Prompt token usage |
| `completion_tokens` | Completion token usage |
| `n_tweets_raw` | Number of tweets in the 100-day window |
| `n_tweets_used` | Number of tweets in the prediction prompt (after retrieval) |
| `competitiveness` | Vote competitiveness (unanimous/moderate/contested) |
| `had_bill_summary` | Whether a congress.gov summary was available for this vote |
| `retrieval_method` | `llm`, `topic`, or `none` — which retriever produced the tweets |
| `mean_similarity` | Mean topic-retrieval score of the selected tweets (blank for `llm`/`none`) |
| `timestamp` | Prediction timestamp |

> **Note:** the CSV header is only written when the output file does not yet
> exist, so `retrieval_method` and `mean_similarity` break appending onto a
> `predictions.csv` produced before they were added. Use a fresh `--output`.

## Baselines

All baselines mirror the `(bioguide, vote_id)` candidate set of an existing
predictions CSV, so row counts line up with the LLM run.

### Causal baselines — comparable to the LLM

These use only information available strictly before the vote's date, the same
constraint the pipeline itself runs under. Report these against the LLM.

```bash
# writes all four, then evaluates each
python run_baseline.py --variant causal --mirror results/predictions_full.csv --evaluate
```

| `--variant` | Prediction rule |
|---|---|
| `always_yea` | The more frequent class among all votes cast before this date. Resolves to Yea for effectively every row — the conventional majority-class floor. |
| `member_prior` | Yea if the member voted Yea in ≥50% of their own prior roll calls. |
| `party_prior` | Yea if the member's party voted Yea in ≥50% of prior roll calls in the same chamber and congress. |
| `chamber_majority` | Yea if the member's party holds the most seats in that chamber/congress, else Nay. |

Each is built by aggregating the corpus to `(key…, vote_date)` totals, taking a
cumulative sum within the key group, then **shifting by one step** — so the
totals attached to a `vote_date` cover only votes decided before it, and no
candidate can see its own vote.

`vote_date` is a full timestamp for Senate rows but date-only for House rows, so
resolution differs by chamber: a Senate candidate sees earlier votes from the
same day, a House candidate does not. This matches `feature_builder.py:58`
(`votes_df["vote_date"] < vote_date`), which is how the LLM's past-votes block
is assembled — so both sides of the comparison draw on the same information.

They key off the *vote row* rather than the mirror CSV: the mirror's `party`
column comes from `feature_builder.get_member_profile`, which takes the member's
most recent record corpus-wide and is therefore wrong for party switchers.

**One baseline set per candidate sample.** Baselines mirror the `(bioguide,
vote_id)` pairs of whatever predictions CSV you point `--mirror` at, so a new
sample (stratified, random, …) needs its own baselines or there is nothing to
compare it against. Name the outputs with the sample as a suffix and
`results/model_comparison.ipynb` will discover them automatically:

```bash
python run_baseline.py --variant causal --mirror results/predictions_full_random.csv \
    --output-always-yea       results/baseline_always_yea_random.csv \
    --output-member-prior     results/baseline_member_prior_random.csv \
    --output-party-prior      results/baseline_party_prior_random.csv \
    --output-chamber-majority results/baseline_chamber_majority_random.csv
python run_dw_nominate.py --mirror results/predictions_full_random.csv \
    --output results/baseline_dw_nominate_random.csv
```

### Legacy variants — NOT comparable to the LLM

`--variant majority` and `--variant party` predict from the realized tally of
the *same* roll call, and count the member's own vote in that tally. That
information does not exist at prediction time. `party` in particular measures
party cohesion, not predictive skill — a single-member party would score 100% by
construction. Retained for reference; do not table them against the LLM.

```bash
python run_baseline.py --variant both     # the two legacy variants
python run_baseline.py --variant all      # everything
```

### DW-NOMINATE

**Build the roll-call crosswalk first** — this is a one-time step and the
baseline will refuse to run without it:

```bash
python build_rollcall_crosswalk.py
python run_dw_nominate.py --mirror results/predictions_full.csv --evaluate
```

Our `vote_id` carries the Clerk's roll-call number, which restarts at 1 every
session; Voteview's `rollnumber` runs continuously across a Congress. Joining
one onto the other directly matches most rows to a *different bill*. The
crosswalk resolves this by joining on `(congress, chamber, session,
clerk_rollnumber)`, which Voteview's `HSall_rollcalls.csv` carries next to its
own `rollnumber`, and validates the result against the recorded votes
(agreement 0.999996 over 1.14M member-votes).

Two prediction rules, selected with `--rule`:

**`geometric` (default) — the reportable baseline.** Classifies each member
against the roll call's cutting line: predict the outcome point on the member's
side of it, i.e. the sign of `(ideal_point − midpoint) · spread`, taking ideal
points from `HSall_members.csv` and the line from `HSall_rollcalls.csv`.
Voteview's spread runs Yea→Nay, so a negative projection predicts Yea. Roll
calls where NOMINATE estimated no cutting line at all (both spread components
exactly 0 — overwhelmingly near-unanimous votes such as post-office namings and
suspension-calendar bills) have no geometry to classify against and are marked
`ERROR` rather than collapsed onto a default label. That is ~12% of rows,
spread across hundreds of bioguides; it is a property of the source data, not a
join defect.

**`prob` — diagnostics only, not a predictor.** Voteview's `prob` is the fitted
probability of *the choice the member actually made*, so thresholding it
reduces to "predict the observed vote when NOMINATE fits it, the opposite when
it doesn't". Accuracy under this rule is identically 1.000 wherever
`prob >= 50` and 0.000 elsewhere — it is NOMINATE's in-sample hit rate, not
predictive accuracy. The script warns when you select it. Never table it
against the LLM.

Both rules share a caveat worth repeating in the write-up: NOMINATE's ideal
points and cutting lines are estimated from these very roll calls, so even
`geometric` is an in-sample fit rather than a forecast the way the LLM pipeline
is.

Diagnostic/tuning flags:

| Flag | Effect |
|---|---|
| `--roc-analysis` | Compute ROC AUC of the reconstructed `p_yea_hat` score against `true_label` (overall + by chamber) and report the Youden's-J-optimal threshold. Reuses the existing output CSV if it already has a `p_yea_hat` column. |
| `--roc-plot PATH` | Save a ROC-curve PNG (requires `matplotlib`). |
| `--tau FLOAT` | Predict `"Yea"` iff `p_yea_hat >= tau`, replacing the default prob/threshold branch-and-flip rule. |

## Evaluation

```bash
# Evaluate existing predictions
python run_pipeline.py --evaluate-only

# With bootstrap confidence intervals
python run_pipeline.py --evaluate-only --bootstrap

# Or use the evaluator directly
python -c "from pipeline.evaluator import evaluate; evaluate()"
```

Reports accuracy, macro/weighted F1, per-class F1, confusion matrix, and breakdowns by party/chamber. Rows with `predicted_label == "ERROR"` are excluded from evaluation.

### Per-Stratum Evaluation

Use `groupby_accuracy()` to evaluate accuracy for specific strata:

```python
from pipeline.evaluator import groupby_accuracy

# Per-era accuracy
groupby_accuracy("results/predictions.csv", ["congress_era"])

# Per-era × chamber
groupby_accuracy("results/predictions.csv", ["congress_era", "chamber"])

# Full stratum breakdown
groupby_accuracy("results/predictions.csv", ["congress_era", "chamber", "party"])
```

The function automatically derives `congress_era` from the `congress` column using the era mapping.

## Stratified Sampling

For balanced evaluation, use `--stratify` to sample proportionally across:
- **Congress era** (5 eras: Bush-Obama start, Obama polarization, Trump1, Biden, Trump2)
- Chamber (House/Senate)
- Party (Democrat/Republican)
- Vote direction (Yea/Nay)
- Competitiveness (unanimous/moderate/contested)

### Congress Eras

Congresses are grouped into political eras to reduce strata and ensure meaningful sample sizes:

| Era | Congresses | Period |
|-----|------------|--------|
| `1_Bush_Obama_start` | 110-111 | 2007-2011 |
| `2_Obama_polarization` | 112-114 | 2011-2017 |
| `3_Trump1` | 115-116 | 2017-2021 |
| `4_Biden` | 117-118 | 2021-2025 |
| `5_Trump2` | 119 | 2025-2027 |

This reduces strata from ~648 (individual congresses) to ~60-80 (eras), ensuring each stratum has meaningful sample sizes.

### Party Filtering

Party "I" (Independents) is excluded from stratified sampling due to insufficient members (~10 unique members, ~0.24% of votes). I-party members are still evaluated in non-stratified runs.

Requires `--sample N` and saves the sample to `data/stratified_sample.csv` for DW-NOMINATE analysis.

### True random / true stratified sampling

By default — with or without `--stratify` — every candidate is required to have at least one raw
tweet in the `--n-days` (100-day) window before its vote (`pipeline.py`, Phase 1b). Since the tweet
corpus starts 2010-11-06, that filter silently excludes any vote before roughly mid-February 2011
**regardless of sampling mode**: a plain `--sample` random draw and a `--stratify` draw both end up
with almost nothing from congresses 110-111, because the filter runs after candidates are already
built. This is *not* the same thing as `build_sample_cache.py`'s `--min-tweets`, which is a separate,
stricter filter on *relevant* (post-retrieval) tweets, only used to build the tweet-conditioned main
sample.

Pass `--allow-zero-tweets` to disable that filter and draw a sample proportional to the full
2007-2026 vote population instead of only its post-2010 tweet-covered slice:

```bash
# True random sample, unconditioned on tweet availability
python run_pipeline.py --sample 15000 --allow-zero-tweets --evaluate

# True stratified sample, unconditioned on tweet availability
python run_pipeline.py --stratify --sample 15000 --allow-zero-tweets --evaluate
```

Candidates with zero raw tweets simply get an empty tweet section in the prompt (`n_tweets_raw=0`,
`n_tweets_used=0`) rather than being dropped — retrieval already handles an empty tweet list
correctly (no LLM call, no error). The active run logs `--allow-zero-tweets` alongside the other
ablation flags so this is recoverable from the log rather than needing to be inferred later.

## File Overview

| File | Purpose |
|------|---------|
| `pipeline.py` | Main orchestration class (4-phase run loop) |
| `config.py` | Paths, LLM settings, defaults |
| `data_loader.py` | Loads votes and tweets |
| `feature_builder.py` | Assembles context features |
| `prompt_builder.py` | Constructs LLM prompts |
| `tweet_retriever.py` | LLM-based tweet relevance filtering |
| `topic_retriever.py` | Topic-model tweet relevance scoring (drop-in for `tweet_retriever`) |
| `topic_model.py` | Shared topic space: embedding, BERTopic fit, centroids, vote query text |
| `vote_filters.py` | Vote-scope filters (nomination/confirmation detection) |
| `sample_cache.py` | Save/load sample cache (pickle) for ablation studies |
| `llm_client.py` | OpenRouter API wrapper (`predict()` + `retrieve_tweets()`) |
| `evaluator.py` | Computes evaluation metrics (`evaluate()`, `groupby_accuracy()`) |
| `run_pipeline.py` (in `src/`) | CLI entry point |
| `build_sample_cache.py` (in `src/`) | One-time script to build high-quality sample cache with iterative retrieval |
| `run_baseline.py` (in `src/`) | Causal baselines (+ legacy same-roll-call variants) |
| `run_dw_nominate.py` (in `src/`) | DW-NOMINATE baseline (spatial rule; needs the crosswalk) |
| `build_rollcall_crosswalk.py` (in `src/`) | One-time build of `data/rollcall_crosswalk.csv`, mapping `vote_id` → Voteview `rollnumber` |
| `thesis_analysis.py` (in `src/`) | Shared analysis helpers for the notebooks: run/sample discovery, metrics, bootstrap, paired tests |
| `fit_topic_model.py` (in `src/`) | One-time BERTopic fit over the tweet corpus (GPU) |
| `rerank_cache.py` (in `src/`) | Re-filter a sample cache with topic retrieval; threshold sweep |
| `topic_analysis.ipynb` (in `src/`) | Topic overview, topics over time / by party, tweet-vs-bill coverage |
