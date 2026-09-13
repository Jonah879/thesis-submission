"""One-off backfill: add an `n_past_votes` column to every result CSV in results/.

Reconstructs, without re-running the LLM pipeline, exactly what
`pipeline/feature_builder.get_past_votes` computed at prediction time: the number of
this member's Yea/Nay votes strictly before the candidate's own `vote_date`, capped at
`config.N_PAST_VOTES` (25) since the prompt only ever showed the most recent 25.

    member_votes = votes_df[(votes_df["bioguide"] == bioguide) &
                             (votes_df["vote_date"] <  vote_date)]
    recent = member_votes.sort_values("vote_date", ascending=False).head(n)

`votes_df` is `data/loaded_votes.csv` -- the same Yea/Nay-only, nomination-filtered,
chamber/Congress-unrestricted snapshot the pipeline itself reads from. `n_past_votes == 0`
is the true cold-start indicator `freshman` (a first-term flag, not a zero-history flag)
cannot provide -- see §8b of model_comparison.ipynb.
"""
import glob

import numpy as np
import pandas as pd

N_PAST_VOTES = 25

votes = pd.read_csv("data/loaded_votes.csv", usecols=["bioguide", "vote_date"])
votes["vote_date"] = pd.to_datetime(votes["vote_date"], format="mixed")
sorted_dates = {
    bg: np.sort(g["vote_date"].to_numpy())
    for bg, g in votes.groupby("bioguide")
}

for path in sorted(glob.glob("results/*.csv")):
    df = pd.read_csv(path)
    if df.empty or "bioguide" not in df.columns or "vote_date" not in df.columns:
        print(f"{path}: skipped (empty or missing bioguide/vote_date)")
        continue

    missing = sorted(set(df["bioguide"].unique()) - set(sorted_dates))
    assert not missing, f"{path}: {len(missing)} bioguides not found in loaded_votes.csv: {missing[:5]}"

    vote_date = pd.to_datetime(df["vote_date"], format="mixed")
    n_past = np.empty(len(df), dtype=np.int64)
    for bg, idx in df.groupby("bioguide").groups.items():
        pos = df.index.get_indexer(idx)
        raw_count = np.searchsorted(sorted_dates[bg], vote_date.to_numpy()[pos], side="left")
        n_past[pos] = raw_count

    df["n_past_votes"] = np.clip(n_past, 0, N_PAST_VOTES)
    zero_n = int((df["n_past_votes"] == 0).sum())
    df.to_csv(path, index=False)
    print(f"{path}: {len(df)} rows | n_past_votes==0: {zero_n} "
          f"({100 * zero_n / len(df):.1f}%) | mean={df['n_past_votes'].mean():.2f}")
