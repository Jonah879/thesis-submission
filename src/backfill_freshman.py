"""One-off backfill: add a `freshman` (cold-start indicator) column to every
result CSV in results/. A candidate row is 'freshman' when its `congress`
equals the member's first Congress ever served, per data/hsall_members.csv
(Voteview, congress 1-119 / 1789-2026 -- not windowed to our vote data's
2007+ coverage, so it isn't confounded by our dataset's own start date).
Uses the same member file the DW-NOMINATE baseline is built from.
"""
import glob

import numpy as np
import pandas as pd

hsall = pd.read_csv("data/hsall_members.csv", usecols=["congress", "bioguide_id"])
hsall = hsall.dropna(subset=["bioguide_id"])
first_congress = hsall.groupby("bioguide_id")["congress"].min()

for path in sorted(glob.glob("results/*.csv")):
    df = pd.read_csv(path)
    if df.empty or "bioguide" not in df.columns or "congress" not in df.columns:
        print(f"{path}: skipped (empty or missing bioguide/congress)")
        continue

    fc = df["bioguide"].map(first_congress)
    n_missing = fc.isna().sum()
    assert n_missing == 0, f"{path}: {n_missing} rows have no first_congress match"

    df["freshman"] = np.where(df["congress"] == fc, "freshman", "veteran")
    counts = df["freshman"].value_counts().to_dict()
    df.to_csv(path, index=False)
    print(f"{path}: {len(df)} rows | {counts}")
