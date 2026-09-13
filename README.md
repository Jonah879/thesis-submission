# More Than Tweets — Code & Data

Reproducibility package for the thesis *"More Than Tweets: Predicting Roll Call Votes
with a Large Language Model"* (Jonah Hartmann, University of Konstanz, 2026): whether an
LLM (Qwen2.5-32B-Instruct-AWQ), prompted zero-shot with a member of Congress's tweets and
voting history, can predict their Yea/Nay vote — compared against DW-NOMINATE and
simpler causal baselines.

## Quick start

```bash
pip install -r requirements.txt
```

`data/` and `results/` live **inside `src/`** (`thesis_analysis.py` resolves both as
fixed children of its own location). Run scripts from inside `src/`
(`cd src && python run_pipeline.py ...`); notebooks in `src/notebooks/` find
`thesis_analysis.py` automatically by walking up from wherever they're opened.

## What's here

- **`src/notebooks/table_figures.ipynb`** — the deliverable. Every table and figure in
  the thesis, each computed live as a DataFrame in one cell and formatted to
  copy-pasteable LaTeX in the next. Search it for a `\label{}` (e.g. `tab:full_vs_dw`) to
  find exactly where a number in the thesis came from.
- **`src/pipeline/`** — the 4-phase prediction pipeline (see its own `README.md` for CLI
  usage). `run_pipeline.py` predicts, `run_baseline.py`/`run_dw_nominate.py` build the
  baselines, `build_sample_cache.py` builds a candidate sample.
- **`src/thesis_analysis.py`** — shared metrics/bootstrap/plotting helpers behind
  `table_figures.ipynb`. Read its module docstring first.
- **`src/methods_section_outline.md`** — the methodology reference, more detailed than
  the thesis's own Methods chapter.
- **`src/notebooks/data_cleaning.ipynb`** — the one-off cleaning step (question-type
  whitelist, text normalisation) already applied to `data/*_votes_members.csv`; kept as a
  historical record, not meant to be re-run.
- **`src/notebooks/topic_analysis.ipynb`** — the topic-model analysis (Appendix). Needs
  `sentence-transformers`/`torch` (not in `requirements.txt`) to re-run, but its last
  server run's output is saved inside the notebook, so it's readable as-is.
- **`src/results/`** — every prediction/baseline CSV (`predictions_*.csv`,
  `baseline_*.csv`) and figure PNG. This is what every table/figure is computed from —
  no LLM calls needed to reproduce anything.
- **`src/data/`** — processed vote/tweet/bill data plus small cached aggregates in
  `data/derived/`.

## What's excluded

Raw multi-GB source data (the tweet corpus, GovTrack XML, Voteview's DW-NOMINATE files)
and intermediate sample caches aren't included — none of it is needed to reproduce a
table or figure, only to rebuild the pipeline's inputs from scratch. `data/house_votes_members.csv`
is additionally `.gitignore`d (224 MB, over GitHub's 100 MB limit) but present on disk if
you have this folder directly rather than a GitHub clone.

## Never add these

`pipeline/.env` (`OPENROUTER_API_KEY`), `data/api_key.txt` (congress.gov key), or any
scraper credentials — none are included, and none should ever be committed.
