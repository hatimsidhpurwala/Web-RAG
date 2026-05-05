"""
Resilient Groq client with automatic model fallback.

Usage (replaces direct `Groq()` calls in agents):
    from src.core.groq_client import groq_chat

    response = groq_chat(messages=[...], temperature=0.2, max_tokens=500)
    text = response.choices[0].message.content

The function automatically retries with the next model in MODEL_FALLBACK_CHAIN
when it encounters a rate-limit (429) or decommissioned-model (400) error.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from groq import Groq, RateLimitError

from config.settings import GROQ_API_KEY, LLM_MAX_TOKENS, LLM_TEMPERATURE, MODEL_FALLBACK_CHAIN

logger = logging.getLogger(__name__)

_client = Groq(api_key=GROQ_API_KEY)


def groq_chat(
    messages: List[dict],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[dict] = None,
    model_chain: Optional[List[str]] = None,
):
    """Call Groq chat completions with automatic model fallback.

    Parameters
    ----------
    messages : list[dict]
        Standard OpenAI-format messages.
    temperature : float, optional
        Sampling temperature (defaults to LLM_TEMPERATURE).
    max_tokens : int, optional
        Max tokens for the response (defaults to LLM_MAX_TOKENS).
    response_format : dict, optional
        E.g. ``{"type": "json_object"}`` for structured JSON output.
    model_chain : list[str], optional
        Override the default MODEL_FALLBACK_CHAIN.

    Returns
    -------
    groq ChatCompletion object
        The first successful completion response.

    Raises
    ------
    Exception
        Re-raises the last error if ALL models in the chain fail.
    """
    chain = model_chain or MODEL_FALLBACK_CHAIN
    temperature = temperature if temperature is not None else LLM_TEMPERATURE
    max_tokens = max_tokens or LLM_MAX_TOKENS

    last_exc: Exception | None = None

    for model in chain:
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format:
            kwargs["response_format"] = response_format

        try:
            logger.debug("groq_chat: trying model '%s'", model)
            response = _client.chat.completions.create(**kwargs)
            if model != chain[0]:
                logger.info("groq_chat: succeeded with fallback model '%s'", model)
            return response

        except Exception as exc:
            err_msg = str(exc)
            is_rate_limit   = "429" in err_msg or "rate_limit" in err_msg.lower() or isinstance(exc, RateLimitError)
            is_decommission = "decommissioned" in err_msg.lower()
            is_overloaded   = "overloaded" in err_msg.lower() or "503" in err_msg

            if is_rate_limit or is_decommission or is_overloaded:
                logger.warning(
                    "groq_chat: model '%s' unavailable (%s), trying next fallback...",
                    model, "rate_limit" if is_rate_limit else ("decommissioned" if is_decommission else "overloaded"),
                )
                last_exc = exc
                time.sleep(0.5)   # small pause before retry
                continue
            else:
                # Unknown error — don't swallow it
                raise

    raise last_exc or RuntimeError("All models in fallback chain failed")
