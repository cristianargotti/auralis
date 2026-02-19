"""AURALIS Gemini Client — Multi-model AI with automatic fallback.

Uses Gemini 3 Pro (most intelligent) → Gemini 3 Flash (fast) → Gemini 2.5 Pro (stable).
Handles quota limits (429) by automatically falling back to the next model.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any

import structlog

logger = structlog.get_logger()

# Model priority: most intelligent first, fastest last
MODEL_PRIORITY = [
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
]


def _get_api_key() -> str:
    """Get the Gemini API key from environment."""
    key = os.getenv("AURALIS_GEMINI_API_KEY", "")
    if not key:
        key = os.getenv("GEMINI_API_KEY", "")
    return key


def generate(
    prompt: str,
    system_prompt: str = "",
    json_mode: bool = True,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    preferred_model: str | None = None,
) -> dict[str, Any] | str:
    """Generate content using Gemini with automatic model fallback.

    Args:
        prompt: User prompt
        system_prompt: System instruction
        json_mode: If True, parse response as JSON
        max_tokens: Max output tokens
        temperature: Creativity level (0.0 - 1.0)
        preferred_model: Override model selection

    Returns:
        Parsed JSON dict if json_mode, otherwise raw string
    """
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("No Gemini API key found. Set AURALIS_GEMINI_API_KEY in .env")

    # Always try all models — preferred first, then fallback chain
    if preferred_model and preferred_model in MODEL_PRIORITY:
        # Put preferred first, then the rest in priority order
        models_to_try = [preferred_model] + [m for m in MODEL_PRIORITY if m != preferred_model]
    elif preferred_model:
        # Unknown model — try it first, then fallback chain
        models_to_try = [preferred_model] + MODEL_PRIORITY
    else:
        models_to_try = MODEL_PRIORITY

    contents = []
    if system_prompt:
        contents.append({
            "role": "user",
            "parts": [{"text": f"[SYSTEM INSTRUCTION]\n{system_prompt}\n[END SYSTEM INSTRUCTION]"}],
        })
        contents.append({
            "role": "model",
            "parts": [{"text": "Understood. I will follow these instructions precisely."}],
        })

    contents.append({
        "role": "user",
        "parts": [{"text": prompt}],
    })

    body = json.dumps({
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            **({"responseMimeType": "application/json"} if json_mode else {}),
        },
    }).encode()

    last_error = None

    for model in models_to_try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent?key={api_key}"
        )

        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
        )

        try:
            resp = urllib.request.urlopen(req, timeout=60)
            data = json.loads(resp.read())

            # Extract text from response
            candidates = data.get("candidates", [])
            if not candidates:
                logger.warning("gemini_empty_response", model=model)
                continue

            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")

            if not text:
                logger.warning("gemini_empty_text", model=model)
                continue

            logger.info(
                "gemini_success",
                model=model,
                response_length=len(text),
            )

            if json_mode:
                # Clean markdown code fences if present
                cleaned = text.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.split("\n")
                    cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    logger.warning(
                        "gemini_json_parse_fail",
                        model=model,
                        text_preview=text[:200],
                    )
                    # Return raw text wrapped in dict
                    return {"raw_response": text, "parse_error": True}

            return text

        except urllib.error.HTTPError as e:
            error_body = e.read().decode()[:300]
            if e.code == 429:
                logger.info("gemini_quota_exceeded", model=model)
                last_error = f"Quota exceeded for {model}"
                continue  # Try next model
            else:
                logger.error(
                    "gemini_http_error",
                    model=model,
                    status=e.code,
                    body=error_body,
                )
                last_error = f"HTTP {e.code} for {model}: {error_body}"
                continue
        except Exception as e:
            logger.error("gemini_error", model=model, error=str(e))
            last_error = str(e)
            continue

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


def get_available_model() -> str | None:
    """Check which model is currently available (not quota-limited)."""
    api_key = _get_api_key()
    if not api_key:
        return None

    body = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
        "generationConfig": {"maxOutputTokens": 5},
    }).encode()

    for model in MODEL_PRIORITY:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent?key={api_key}"
        )
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req, timeout=10)
            return model
        except urllib.error.HTTPError as e:
            if e.code == 429:
                continue
            return None
        except Exception:
            continue

    return None
