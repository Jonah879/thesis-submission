"""
feature_builder.py — Assembles per-(member, vote) context features.

Public API:
    get_recent_tweets(bioguide, vote_date, tweet_index, n_days, max_k) -> list[str]
    get_past_votes(bioguide, vote_date, votes_df, n)                   -> list[dict]
    get_member_profile(bioguide, votes_df)                             -> dict
"""

from datetime import timedelta
from typing import Dict, List

import pandas as pd

from . import config


def get_recent_tweets(
    bioguide: str,
    vote_date: pd.Timestamp,
    tweet_index: Dict[str, pd.DataFrame],
    n_days: int = config.N_DAYS_TWEETS,
    max_k: int = config.MAX_TWEETS_RAW,
) -> List[str]:
    """
    Return up to max_k tweet texts posted by the member in the n_days
    immediately before vote_date, ordered most-recent first.
    """
    member_tweets = tweet_index.get(bioguide)
    if member_tweets is None or member_tweets.empty:
        return []

    cutoff_start = vote_date - timedelta(days=n_days)
    mask = (
        (member_tweets["created_at"] >= cutoff_start) &
        (member_tweets["created_at"] <  vote_date)
    )
    window = member_tweets.loc[mask].head(max_k)

    return window["text"].dropna().tolist()


def get_past_votes(
    bioguide: str,
    vote_date: pd.Timestamp,
    votes_df: pd.DataFrame,
    n: int = config.N_PAST_VOTES,
) -> List[dict]:
    """
    Return the last n votes cast by this member strictly before vote_date,
    ordered most-recent first.

    Each entry is a dict with keys:
        vote_date, bill_number, bill_title, question, vote_cast, chamber
    """
    member_votes = votes_df[
        (votes_df["bioguide"]   == bioguide) &
        (votes_df["vote_date"]  <  vote_date)
    ]
    recent = member_votes.sort_values("vote_date", ascending=False).head(n)

    return recent[[
        "vote_date", "bill_number", "bill_title", "question", "vote_cast", "chamber"
    ]].to_dict(orient="records")


def get_member_profile(
    bioguide: str,
    votes_df: pd.DataFrame,
) -> dict:
    """
    Return a stable profile for the member derived from the votes DataFrame.
    Uses the most recent row for that bioguide.

    Keys: bioguide, member_name, party, state, chamber
    """
    rows = votes_df[votes_df["bioguide"] == bioguide]
    if rows.empty:
        return {
            "bioguide":    bioguide,
            "member_name": "Unknown",
            "party":       "Unknown",
            "state":       "Unknown",
            "chamber":     "Unknown",
        }

    # Use the most recent record for stable party/state info
    latest = rows.sort_values("vote_date", ascending=False).iloc[0]
    return {
        "bioguide":    bioguide,
        "member_name": latest["member_name"],
        "party":       latest["party"],
        "state":       latest["state"],
        "chamber":     latest["chamber"],
    }
