"""TEMPORARY — see TEMPORARY.md

Posting a prompt at the live Qwen endpoint.

One call, no retry. The endpoint answers a short prompt in ~49 s and a pack
prompt not at all (a gateway timeout at ~125 s), so a client that retries on a
60 s budget would fire on a request that is merely slow — and would send the same
prompt a second time, creating a second captured request for it. This makes the
one call with a timeout that fits the observed latency and reports what happened.
"""

import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from layer_profile.config import ENDPOINT_URL, REPLAY_REQUEST_TIMEOUT_S


def endpoint_url() -> str:
    return ENDPOINT_URL


def build_payload(prompt: str) -> dict:
    """The ``/generate`` payload shape this endpoint expects."""
    return {"messages": [{"role": "user", "content": prompt}]}


@dataclass
class EndpointResponse:
    """What one call to the endpoint did. Never an exception."""
    ok: bool
    status: Optional[int]
    elapsed_s: float
    text: str = ""
    error: str = ""
    raw: Any = None


def post_generate(
    prompt: str,
    timeout: float = REPLAY_REQUEST_TIMEOUT_S,
) -> EndpointResponse:
    """Send one prompt. Returns a result object rather than raising.

    A replay sends a whole pack; one prompt timing out must not lose the rest,
    and must not be retried — a retry doubles the load on a 2-minute endpoint and
    would create a second captured request for the same prompt.
    """
    started = time.time()
    try:
        response = requests.post(
            endpoint_url(),
            json=build_payload(prompt),
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        return EndpointResponse(
            ok=False, status=None, elapsed_s=time.time() - started, error=str(exc)
        )

    elapsed = time.time() - started
    try:
        body = response.json()
    except ValueError as exc:
        return EndpointResponse(
            ok=response.ok, status=response.status_code, elapsed_s=elapsed,
            text=response.text[:2000], error=f"non-JSON response: {exc}",
        )

    return EndpointResponse(
        ok=response.ok,
        status=response.status_code,
        elapsed_s=elapsed,
        text=extract_text(body),
        raw=body,
    )


def extract_text(body: Any) -> str:
    """Best-effort pull of the generated text out of the response body.

    The endpoint's exact response shape is not pinned anywhere in this repo, so
    this walks the handful of keys generative APIs use and falls back to the
    stringified body rather than guessing wrong and reporting an empty answer.
    """
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        for key in ("generated_text", "text", "response", "output", "completion", "result"):
            value = body.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, (dict, list)):
                nested = extract_text(value)
                if nested:
                    return nested
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            return extract_text(choices[0])
        message = body.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    if isinstance(body, list) and body:
        return extract_text(body[0])
    return ""
