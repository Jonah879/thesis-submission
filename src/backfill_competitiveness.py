"""One-off backfill: populate the `competitiveness` column in main-sample result
CSVs that predate the build_sample_cache.py fix (commit f240574) for the flat/
non-stratified sampling path. Values are joined in from data/loaded_votes.csv,
which computes competitiveness identically to the fixed code path.
"""
import pandas as pd

TARGET_FILES = [
    "results/predictions_full.csv",
    "results/predictions_no_context.csv",
    "results/predictions_no_tweets.csv",
    "results/predictions_no_voting.csv",
    "results/predictions_raw_tweets.csv",
    "results/predictions_topic.csv",
    "results/baseline_always_yea.csv",
    "results/baseline_chamber_majority.csv",
    "results/baseline_member_prior.csv",
    "results/baseline_party_prior.csv",
    "results/baseline_dw_nominate.csv",
]

comp_map = (
    pd.read_csv("data/loaded_votes.csv", usecols=["vote_id", "competitiveness"])
    .drop_duplicates()
    .set_index("vote_id")["competitiveness"]
)
assert comp_map.index.is_unique, "loaded_votes.csv has conflicting competitiveness per vote_id"

for path in TARGET_FILES:
    df = pd.read_csv(path)
    before = df["competitiveness"].value_counts(dropna=False).to_dict()

    df["competitiveness"] = df["vote_id"].map(comp_map)
    n_missing = df["competitiveness"].isna().sum()
    assert n_missing == 0, f"{path}: {n_missing} rows have no competitiveness match"

    after = df["competitiveness"].value_counts(dropna=False).to_dict()
    df.to_csv(path, index=False)
    print(f"{path}: {len(df)} rows | before={before} | after={after}")
