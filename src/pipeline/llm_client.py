"""
llm_client.py — OpenRouter API wrapper for vote predictions.

Public API:
    predict(prompt, model, system_prompt) -> dict
        Returns: {
            "prediction":        "Yea" | "Nay" | "ERROR",
            "reasoning":         str,
            "prompt_tokens":     int,
            "completion_tokens": int,
            "raw_response":      str,
        }
    retrieve_tweets(user_prompt, model, system_prompt) -> dict
        Returns: {
            "prompt_tokens":     int,
            "completion_tokens": int,
            "raw_response":      str,
        }
"""

import json
import logging
import time
from typing import Optional

from openai import OpenAI, RateLimitError, APIError

from . import config
from .prompt_builder import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

VALID_PREDICTIONS = {"Yea", "Nay"}

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Return a process-wide singleton OpenAI client (thread-safe)."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.get_api_key(),
            base_url=config.get_base_url(),
        )
    return _client


def _parse_response(content: str) -> dict:
    """
    Extract {"prediction": ..., "reasoning": ...} from the model's reply.
    Handles cases where the model wraps JSON in markdown code fences.
    """
    # Strip markdown code fences if present
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop opening fence (```json or ```) and closing fence
        inner = [l for l in lines[1:] if not l.strip().startswith("```")]
        text = "\n".join(inner).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fall back: try to find the first { ... } block
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                logger.warning("Could not parse JSON from response: %s", text[:200])
                return {"prediction": "ERROR", "reasoning": text[:500]}
        else:
            return {"prediction": "ERROR", "reasoning": text[:500]}

    prediction = str(data.get("prediction", "ERROR")).strip()
    # Normalise minor variations
    _norm = {"yea": "Yea", "aye": "Yea", "nay": "Nay", "no": "Nay"}
    prediction = _norm.get(prediction.lower(), prediction)

    if prediction not in VALID_PREDICTIONS:
        logger.warning("Unexpected prediction value '%s', marking as ERROR.", prediction)
        prediction = "ERROR"

    reasoning = str(data.get("reasoning", "")).strip()
    return {"prediction": prediction, "reasoning": reasoning}


def predict(
    prompt: str,
    model: str = config.DEFAULT_MODEL,
    system_prompt: str = SYSTEM_PROMPT,
    max_retries: int = config.MAX_RETRIES,
) -> dict:
    """
    Call the OpenRouter API and return a structured prediction dict.

    Returns:
        {
            "prediction":        str,   # Yea | Nay | ERROR
            "reasoning":         str,
            "prompt_tokens":     int,
            "completion_tokens": int,
            "raw_response":      str,
        }
    """
    client = _get_client()

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )

            content = response.choices[0].message.content or ""
            usage   = response.usage

            result = _parse_response(content)
            result["prompt_tokens"]     = usage.prompt_tokens     if usage else 0
            result["completion_tokens"] = usage.completion_tokens if usage else 0
            result["raw_response"]      = content
            return result

        except RateLimitError as exc:
            wait = 2 ** attempt  # 2, 4, 8 seconds
            logger.warning(
                "Rate limit hit (attempt %d/%d). Retrying in %ds …",
                attempt, max_retries, wait,
            )
            last_error = exc
            time.sleep(wait)

        except APIError as exc:
            logger.warning(
                "API error on attempt %d/%d: %s", attempt, max_retries, exc
            )
            last_error = exc
            time.sleep(2 ** attempt)

    # All retries exhausted
    logger.error("All %d retries failed. Last error: %s", max_retries, last_error)
    return {
        "prediction":        "ERROR",
        "reasoning":         f"API call failed after {max_retries} retries: {last_error}",
        "prompt_tokens":     0,
        "completion_tokens": 0,
        "raw_response":      "",
    }


def retrieve_tweets(
    user_prompt: str,
    model: str = config.DEFAULT_MODEL,
    system_prompt: str = "",
    max_retries: int = config.MAX_RETRIES,
) -> dict:
    """
    Call the LLM for tweet relevance filtering (lightweight, small output).

    Returns:
        {
            "prompt_tokens":     int,
            "completion_tokens": int,
            "raw_response":      str,   # raw text containing the index list
        }
    """
    client = _get_client()

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=config.TEMPERATURE,
            )

            content = response.choices[0].message.content or ""
            usage   = response.usage

            return {
                "prompt_tokens":     usage.prompt_tokens     if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "raw_response":      content,
            }

        except RateLimitError as exc:
            wait = 2 ** attempt
            logger.warning(
                "Rate limit hit on retrieval (attempt %d/%d). Retrying in %ds …",
                attempt, max_retries, wait,
            )
            last_error = exc
            time.sleep(wait)

        except APIError as exc:
            logger.warning(
                "API error on retrieval attempt %d/%d: %s", attempt, max_retries, exc,
            )
            last_error = exc
            time.sleep(2 ** attempt)

    logger.error("Retrieval: all %d retries failed. Last error: %s", max_retries, last_error)
    return {
        "prompt_tokens":     0,
        "completion_tokens": 0,
        "raw_response":      "",
    }
