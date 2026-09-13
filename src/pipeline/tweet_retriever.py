"""
tweet_retriever.py — LLM-based tweet relevance filtering.

Public API:
    retrieve_relevant_tweets(tweets, bill_number, bill_title, question,
                             member_name, model) -> tuple[list[str], int]
        Returns (filtered_tweets, n_used).
"""

import json
import logging
from typing import List, Tuple

from . import llm_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retrieval prompt
# ---------------------------------------------------------------------------

_RETRIEVAL_SYSTEM = (
    "You are a political analyst. You will be given a list of tweets from a "
    "member of Congress and a bill they are about to vote on. Your task is to "
    "identify tweets that are substantively about the bill's specific topic.\n\n"
    "A tweet is relevant if it:\n"
    "- Discusses the specific policy area or issue this bill addresses — not "
    "just a passing or tangential mention\n"
    "- Explicitly mentions the bill number, bill title, or specific policy in "
    "the bill (this is sufficient but not required for relevance)\n\n"
    "A tweet is NOT relevant if it:\n"
    "- Only tangentially touches the topic without substantive discussion of it\n"
    "- Is about the member's other policy positions, district activities, "
    "personal life, endorsements, scheduling, or any other unrelated topic\n\n"
    "If a bill summary is provided, use it to understand what the bill actually "
    "does — the bill title alone is often too short to judge relevance accurately."
)

_RETRIEVAL_USER_TEMPLATE = """Bill: {bill_number} — {bill_title}
Bill summary: {bill_summary}
Vote question: {question}
Member: {member_name}

Tweets:
{tweet_block}

Think step by step: for each tweet, decide if it is substantively about the \
specific topic of this bill. Return ONLY a JSON array of indices of \
relevant tweets, e.g. [1, 3, 7].

If fewer than 5 tweets are relevant, return only those. \
If no tweets are relevant, return []."""


def _build_tweet_block(tweets: List[str]) -> str:
    """Format tweets as a numbered list for the retrieval prompt."""
    lines = []
    for i, tweet in enumerate(tweets, 1):
        truncated = tweet[:280] + ("\u2026" if len(tweet) > 280 else "")
        lines.append(f"[{i}] {truncated}")
    return "\n".join(lines)


def _parse_indices(content: str) -> List[int]:
    """Extract a list of integer indices from the LLM response."""
    text = content.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [l for l in lines[1:] if not l.strip().startswith("```")]
        text = "\n".join(inner).strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [int(x) for x in data]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: find the first [ ... ] block
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list):
                return [int(x) for x in data]
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning("Could not parse tweet indices from response: %s", text[:200])
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    Filter tweets to only those relevant to the bill being voted on.

    Args:
        tweets: Raw tweet texts (up to max_tweets_raw).
        bill_number: e.g. "H.R.7521"
        bill_title: e.g. "Protecting Americans from..."
        question: e.g. "On the Motion to Recommit"
        member_name: e.g. "Nancy Pelosi"
        model: LLM model to use for retrieval.
        bill_summary: official congress.gov summary of the bill, if available.

    Returns:
        (filtered_tweets, n_used) — the relevant subset of tweets and the count.
        Falls back to returning all tweets on failure.
    """
    if not tweets:
        return [], 0

    tweet_block = _build_tweet_block(tweets)
    user_prompt = _RETRIEVAL_USER_TEMPLATE.format(
        bill_number=bill_number,
        bill_title=bill_title,
        bill_summary=bill_summary.strip() if bill_summary else "(no official summary available)",
        question=question,
        member_name=member_name,
        tweet_block=tweet_block,
    )

    result = llm_client.retrieve_tweets(
        user_prompt=user_prompt,
        model=model,
        system_prompt=_RETRIEVAL_SYSTEM,
    )

    indices = _parse_indices(result["raw_response"])

    # Validate and filter — indices are 1-based, only keep valid ones
    valid_indices = {i for i in indices if 1 <= i <= len(tweets)}
    filtered = [tweets[i - 1] for i in sorted(valid_indices)]

    if not filtered and tweets:
        logger.debug(
            "Retrieval: 0/%d tweets relevant for %s", len(tweets), member_name,
        )

    return filtered, len(filtered)
