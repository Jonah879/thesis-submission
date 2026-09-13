"""
prompt_builder.py — Constructs the LLM prompt for a single (member, vote) pair.

Public API:
    build_prompt(member_profile, bill_info, past_votes, recent_tweets) -> str
    SYSTEM_PROMPT -> str  (constant)
"""

from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# System prompt — sets the role and output contract for the LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert political analyst specialising in the United States Congress.
Your task is to predict how a specific member of Congress will vote on a given bill.

You will be provided with:
- The member's name, party affiliation, state, and chamber
- The bill number, title, and the specific vote question
- The bill's official summary (if available)
- The member's recent tweets (if available) from the weeks before the vote
- The member's recent voting history (if available)

Respond ONLY with a valid JSON object in the following format — no markdown, no extra text:
{"prediction": "<Yea|Nay>", "reasoning": "<concise explanation, 2-4 sentences>"}

Rules:
- "prediction" must be exactly one of: Yea, Nay
- "reasoning" must reference specific evidence from the tweets or voting history where available
- Do not hedge — always commit to one of the two labels
"""

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_PARTY_MAP = {"D": "Democrat", "R": "Republican", "I": "Independent"}


def _fmt_party(party_code: str) -> str:
    return _PARTY_MAP.get(party_code.upper(), party_code)


def _fmt_past_vote(record: dict) -> str:
    date = record["vote_date"]
    if isinstance(date, pd.Timestamp):
        date = date.strftime("%Y-%m-%d")
    bill   = record.get("bill_number", "") or "N/A"
    title  = record.get("bill_title",  "") or ""
    # Suppress NaN / empty titles cleanly
    if not title or str(title).lower() == "nan":
        title = ""
    q      = record.get("question",    "") or "N/A"
    cast   = record.get("vote_cast",   "N/A")
    chamber = record.get("chamber", "")
    chamber_str = f" [{chamber.upper()}]" if chamber else ""
    title_str   = f" — {title}" if title else ""
    return f"  • {date}{chamber_str}  {bill}{title_str} | Q: {q} | Vote: {cast}"


def build_prompt(
    member_profile: dict,
    bill_info: dict,
    past_votes: List[dict],
    recent_tweets: List[str],
    no_context: bool = False,
) -> str:
    """
    Assemble the user-turn prompt.

    Args:
        member_profile: output of feature_builder.get_member_profile()
        bill_info: dict with keys vote_id, vote_date, bill_number, bill_title, question,
                   chamber, congress, session
        past_votes: output of feature_builder.get_past_votes()
        recent_tweets: output of feature_builder.get_recent_tweets()
        no_context: if True, hide member profile and bill info from prompt

    Returns:
        A formatted string to be sent as the user message to the LLM.
    """
    name    = member_profile.get("member_name", "Unknown")
    party   = _fmt_party(member_profile.get("party", ""))
    state   = member_profile.get("state", "Unknown")
    chamber = member_profile.get("chamber", "Unknown").upper()

    vote_date = bill_info.get("vote_date", "")
    if isinstance(vote_date, pd.Timestamp):
        vote_date = vote_date.strftime("%Y-%m-%d")

    bill_number = bill_info.get("bill_number", "N/A")
    bill_title  = bill_info.get("bill_title",  "N/A")
    bill_summary = bill_info.get("bill_summary", "") or ""
    question    = bill_info.get("question",    "N/A")
    congress    = bill_info.get("congress",    "N/A")
    session     = bill_info.get("session",     "N/A")

    lines = []

    # -----------------------------------------------------------------------
    # Section: Member profile + Bill info (skipped when no_context=True)
    # -----------------------------------------------------------------------
    if not no_context:
        lines += [
            "=== MEMBER PROFILE ===",
            f"Name:    {name}",
            f"Party:   {party}",
            f"State:   {state}",
            f"Chamber: {chamber}",
            "",
            "=== BILL / VOTE INFORMATION ===",
            f"Date:     {vote_date}",
            f"Congress: {congress}, Session: {session}",
            f"Bill:     {bill_number}",
            f"Title:    {bill_title}",
            f"Summary:  {bill_summary.strip() if bill_summary.strip() else '(no official summary available)'}",
            f"Question: {question}",
            "",
        ]

    # -----------------------------------------------------------------------
    # Section: Recent tweets
    # -----------------------------------------------------------------------
    lines.append("=== RECENT TWEETS ===")
    if recent_tweets:
        for i, tweet in enumerate(recent_tweets, 1):
            # Truncate very long tweets to keep prompt compact
            truncated = tweet[:280] + ("…" if len(tweet) > 280 else "")
            lines.append(f"  [{i}] {truncated}")
    else:
        lines.append("  (no tweets available for this member in this time window)")
    lines.append("")

    # -----------------------------------------------------------------------
    # Section: Past voting history
    # -----------------------------------------------------------------------
    lines.append("=== PAST VOTING HISTORY (most recent first) ===")
    if past_votes:
        for record in past_votes:
            lines.append(_fmt_past_vote(record))
    else:
        lines.append("  (no past voting history available)")
    lines.append("")

    # -----------------------------------------------------------------------
    # Task instruction
    # -----------------------------------------------------------------------
    lines.append("=== TASK ===")
    if no_context:
        lines += [
            "Based on the above, predict how this member of Congress will vote on this bill.",
            "",
            'Respond with a JSON object: {"prediction": "<Yea|Nay>", "reasoning": "<explanation>"}',
        ]
    else:
        lines += [
            f"Based on the above, predict how {name} ({party}, {state}) will vote on:",
            f'"{bill_title}" ({question})',
            "",
            'Respond with a JSON object: {"prediction": "<Yea|Nay>", "reasoning": "<explanation>"}',
        ]

    return "\n".join(lines)
