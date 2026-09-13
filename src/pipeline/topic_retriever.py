"""
topic_retriever.py — Topic-model-based tweet relevance filtering.

A drop-in alternative to tweet_retriever.retrieve_relevant_tweets, replacing
the LLM's binary in/out judgment with a continuous relevance score.

Why: the LLM retriever returns zero relevant tweets for 92.8% of candidates
(measured across both 10,664-row runs in src/old/), so the tweet-augmented
condition was, in practice, almost never augmented. A score-based retriever
makes the empty rate a threshold you choose rather than an outcome you get.

Scoring, per (tweet, vote):

    score = TOPIC_WEIGHT * topic_sim + EMB_WEIGHT * emb_sim

    topic_sim — cosine between the tweet's and the vote's topic
                distributions. This is the topical gate: a tweet about
                immigration matches a bill about immigration whether the
                tweet is for or against it. Stance is the prediction LLM's
                job, not retrieval's.
    emb_sim   — cosine between raw sentence embeddings, max-pooled over the
                vote's text chunks. Ranks within a topic, so a bill on one
                specific immigration provision prefers tweets about that
                provision over generic immigration talk.

Selection is top-K above TOPIC_MIN_SIM. Setting TOPIC_MIN_SIM to 0 gives
pure top-K, which never returns empty.

Public API:
    retrieve_relevant_tweets(...)  -> (list[str], int)     # drop-in seam
    retrieve_with_scores(...)      -> (list[str], list[float])
    warm_texts(texts)              -> None                 # batch pre-embed
    reset_cache()                  -> None
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from . import topic_model

logger = logging.getLogger(__name__)

# text -> embedding (L2-normalised). Populated in batch by warm_texts() so
# per-candidate retrieval is pure vector arithmetic.
_EMB_CACHE: Dict[str, np.ndarray] = {}
_DISTR_CACHE: Dict[str, np.ndarray] = {}


def reset_cache() -> None:
    """Drop the embedding/distribution caches (they grow with corpus size)."""
    _EMB_CACHE.clear()
    _DISTR_CACHE.clear()


# ---------------------------------------------------------------------------
# Batch embedding
# ---------------------------------------------------------------------------

def warm_texts(texts: Sequence[str], show_progress: bool = False) -> None:
    """
    Embed any texts not already cached, in one batched call.

    Call this once over every tweet and vote text in a run before retrieving.
    One vectorised encode over all candidates is dramatically faster than the
    per-candidate calls the LLM retriever needed a 40-thread pool to hide.
    """
    unseen = []
    seen = set()
    for t in texts:
        key = str(t)
        if key and key not in _EMB_CACHE and key not in seen:
            unseen.append(key)
            seen.add(key)

    if not unseen:
        return

    logger.info("Embedding %d new texts …", len(unseen))
    emb = topic_model.encode(unseen, show_progress=show_progress)

    space = topic_model.load_topic_space()
    distr = space.topic_distribution(emb)

    for i, key in enumerate(unseen):
        _EMB_CACHE[key] = emb[i]
        _DISTR_CACHE[key] = distr[i]


def _vectors(texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (embeddings, topic_distributions) for texts, embedding on miss."""
    missing = [t for t in texts if str(t) not in _EMB_CACHE]
    if missing:
        warm_texts(missing)

    if not texts:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    emb = np.vstack([_EMB_CACHE[str(t)] for t in texts])
    distr = np.vstack([_DISTR_CACHE[str(t)] for t in texts])
    return emb, distr


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_tweets(
    tweets: Sequence[str],
    bill_title: str,
    question: str = "",
    bill_summary: str = "",
    topic_weight: Optional[float] = None,
    emb_weight: Optional[float] = None,
) -> np.ndarray:
    """
    Return a relevance score per tweet against this vote. Shape (len(tweets),).
    """
    if not tweets:
        return np.zeros(0, dtype=np.float32)

    topic_weight = config.TOPIC_WEIGHT if topic_weight is None else topic_weight
    emb_weight = config.EMB_WEIGHT if emb_weight is None else emb_weight

    vote_texts = topic_model.build_vote_query_texts(bill_title, bill_summary, question)
    vote_emb, vote_distr = _vectors(vote_texts)
    tweet_emb, tweet_distr = _vectors(list(tweets))

    if vote_emb.size == 0 or tweet_emb.size == 0:
        return np.zeros(len(tweets), dtype=np.float32)

    # Embedding similarity: best-matching chunk of the bill wins, so a long
    # summary does not dilute a strong match on one section.
    emb_sim = (tweet_emb @ vote_emb.T).max(axis=1)

    # Topic similarity against the max-pooled, renormalised vote distribution.
    pooled = vote_distr.max(axis=0)
    total = pooled.sum()
    if total > 0:
        pooled = pooled / total
    denom = np.linalg.norm(tweet_distr, axis=1) * np.linalg.norm(pooled)
    topic_sim = np.where(denom > 0, (tweet_distr @ pooled) / np.clip(denom, 1e-12, None), 0.0)

    return (topic_weight * topic_sim + emb_weight * emb_sim).astype(np.float32)


def retrieve_with_scores(
    tweets: Sequence[str],
    bill_number: str = "",
    bill_title: str = "",
    question: str = "",
    member_name: str = "",
    model: str = "",
    bill_summary: str = "",
    top_k: Optional[int] = None,
    min_sim: Optional[float] = None,
) -> Tuple[List[str], List[float]]:
    """
    Select the most topically relevant tweets for this vote.

    `bill_number`, `member_name` and `model` are accepted for signature
    compatibility with the LLM retriever and are deliberately unused — the
    topic method needs no model and no member identity.

    Returns (selected_tweets, their_scores), highest score first.
    """
    if not tweets:
        return [], []

    top_k = config.TOPIC_TOP_K if top_k is None else top_k
    min_sim = config.TOPIC_MIN_SIM if min_sim is None else min_sim

    scores = score_tweets(tweets, bill_title, question, bill_summary)
    order = np.argsort(-scores)

    selected: List[str] = []
    selected_scores: List[float] = []
    for idx in order[:top_k]:
        if scores[idx] < min_sim:
            break
        selected.append(tweets[idx])
        selected_scores.append(float(scores[idx]))

    return selected, selected_scores


def retrieve_relevant_tweets(
    tweets: List[str],
    bill_number: str,
    bill_title: str,
    question: str,
    member_name: str,
    model: str,
    bill_summary: str = "",
) -> Tuple[List[str], int]:
    """
    Drop-in replacement for tweet_retriever.retrieve_relevant_tweets.

    Identical 7-positional-argument signature and (tweets, count) return, so
    it can be swapped in wherever the LLM retriever is called.
    """
    selected, _scores = retrieve_with_scores(
        tweets,
        bill_number=bill_number,
        bill_title=bill_title,
        question=question,
        member_name=member_name,
        model=model,
        bill_summary=bill_summary,
    )
    return selected, len(selected)
