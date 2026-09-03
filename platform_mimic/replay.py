"""TEMPORARY — see TEMPORARY.md

Send the pack at the live endpoint and compare what the platform recorded.

This is the step that would make the whole workaround unnecessary: if the
platform captures the replayed prompts, the pack can be demonstrated in the
product itself and the local capture is only a cross-check. As of the last run
the endpoint answers without recording, so :func:`run_replay` is built to report
"captured 0/N" and **exit 0** — a platform that records nothing is the finding,
not a crash.
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

import pandas as pd

from layer_profile.config import (
    HF_MODEL,
    MAX_NEW_TOKENS,
    PARITY_PASSES,
    PARITY_STD_RTOL,
    PARITY_STD_TOL,
    QWEN_STG_MODEL_ID,
    REPLAY_POLL_INTERVAL_S,
    REPLAY_POLL_TIMEOUT_S,
    REPLAY_REQUEST_TIMEOUT_S,
)
from layer_profile.pack import load_pack

from .diff import diff_rows, summarise, write_parity_report
from .endpoint import post_generate

SEND_LOG_COLUMNS = [
    "prompt_id", "sent_utc", "ok", "status", "elapsed_s", "n_chars", "error", "text",
]


@dataclass
class SendOutcome:
    """Result of sending a whole pack."""
    log: pd.DataFrame
    window_start: str
    window_end: str
    n_sent: int = 0
    n_ok: int = 0
    errors: List[str] = field(default_factory=list)


def _db_timestamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------

def send_pack(
    prompts: Dict[str, str],
    out_dir: str,
    *,
    timeout: float = REPLAY_REQUEST_TIMEOUT_S,
    progress: bool = True,
    pad_seconds: int = 120,
) -> SendOutcome:
    """Send every prompt once, logging each outcome. Never raises for one prompt.

    The returned window is padded either side of the send so a poll for the
    platform's rows cannot miss a request whose timestamp is written slightly
    before or after the call returns.
    """
    os.makedirs(out_dir, exist_ok=True)
    started = datetime.now(timezone.utc)
    rows, errors, n_ok = [], [], 0

    for i, (prompt_id, prompt) in enumerate(prompts.items(), 1):
        response = post_generate(prompt, timeout=timeout)
        n_ok += int(response.ok)
        if not response.ok:
            errors.append(f"{prompt_id}: {response.error or response.status}")
        rows.append({
            "prompt_id": prompt_id,
            "sent_utc": datetime.now(timezone.utc).isoformat(),
            "ok": response.ok,
            "status": response.status,
            "elapsed_s": round(response.elapsed_s, 1),
            "n_chars": len(prompt),
            "error": response.error,
            "text": response.text,
        })
        if progress:
            print(f"  [{i}/{len(prompts)}] {prompt_id:<34} "
                  f"{'ok' if response.ok else 'FAILED'} "
                  f"{response.elapsed_s:.0f}s {response.error}")

    finished = datetime.now(timezone.utc)
    log = pd.DataFrame(rows, columns=SEND_LOG_COLUMNS)
    log.to_csv(os.path.join(out_dir, "send_log.csv"), index=False)

    return SendOutcome(
        log=log,
        window_start=_db_timestamp(started - timedelta(seconds=pad_seconds)),
        window_end=_db_timestamp(finished + timedelta(seconds=pad_seconds)),
        n_sent=len(prompts),
        n_ok=n_ok,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Poll
# ---------------------------------------------------------------------------

def fetch_platform_rows(
    start_time: str,
    end_time: str,
    model_id: Optional[str] = QWEN_STG_MODEL_ID,
) -> pd.DataFrame:
    """Scalar rows the platform recorded in the replay window."""
    from .stg import fetch_scalar_rows
    return fetch_scalar_rows(start_time, end_time, model_id=model_id)


def fetch_platform_prompts(
    start_time: str,
    end_time: str,
    model_id: Optional[str] = QWEN_STG_MODEL_ID,
) -> pd.DataFrame:
    from .stg import fetch_prompts
    return fetch_prompts(start_time, end_time, model_id)


def poll_captured(
    start_time: str,
    end_time: str,
    expected: int,
    *,
    model_id: Optional[str] = QWEN_STG_MODEL_ID,
    timeout_s: float = REPLAY_POLL_TIMEOUT_S,
    interval_s: float = REPLAY_POLL_INTERVAL_S,
    progress: bool = True,
) -> pd.DataFrame:
    """Poll until the platform has rows for every sent prompt, or time out.

    Polls on the *condition* (rows present) rather than sleeping a fixed
    duration, and returns whatever exists when the budget runs out — a partial
    capture is a result worth reporting, not a failure to wait longer for.
    """
    deadline = time.time() + timeout_s
    rows = pd.DataFrame()
    while True:
        rows = fetch_platform_rows(start_time, end_time, model_id)
        seen = rows["request_id"].nunique() if not rows.empty else 0
        if progress:
            print(f"  polled: {seen}/{expected} requests captured")
        if seen >= expected or time.time() >= deadline:
            return rows
        time.sleep(min(interval_s, max(deadline - time.time(), 0)))


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def _restamp_by_prompt(
    platform_rows: pd.DataFrame,
    platform_prompts: pd.DataFrame,
    prompts: Dict[str, str],
) -> pd.DataFrame:
    """Relabel platform rows with our prompt ids, matched on the prompt text.

    The platform assigns its own request ids, so the two sides have no key in
    common until the text is used to join them. Whitespace is normalised because
    the pack ships one prompt per line.
    """
    if platform_rows.empty or platform_prompts.empty:
        return platform_rows

    def norm(text: str) -> str:
        return " ".join(str(text).split())

    by_text = {norm(text): pid for pid, text in prompts.items()}
    request_to_prompt = {
        row["request_id"]: by_text[norm(row["prompt"])]
        for _, row in platform_prompts.iterrows()
        if norm(row["prompt"]) in by_text
    }
    rows = platform_rows[platform_rows["request_id"].isin(request_to_prompt)].copy()
    rows["request_id"] = rows["request_id"].map(request_to_prompt)
    return rows


def capture_pack_locally(
    prompts: Dict[str, str],
    *,
    model: str = HF_MODEL,
    device: Optional[str] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    progress: bool = True,
) -> pd.DataFrame:
    """Local rows for the pack — the side of the replay diff we control."""
    from .capture import LocalCapture

    frames = []
    with LocalCapture(model, device=device, max_new_tokens=max_new_tokens) as capture:
        for i, (prompt_id, prompt) in enumerate(prompts.items(), 1):
            result = capture.run(prompt, request_id=prompt_id)
            frames.append(result.rows)
            if progress:
                print(f"  [{i}/{len(prompts)}] captured {prompt_id}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _zero_capture_explanation(send: SendOutcome, dry_run: bool) -> str:
    """Why nothing was captured — the three cases are genuinely different.

    Conflating "the endpoint never answered" with "the endpoint answered but did
    not record" would misreport the state of the platform, which is the one thing
    this run exists to establish.
    """
    if dry_run:
        return ("Dry run: nothing was sent, so the platform side is empty by "
                "construction. This exercises the report, not the platform.")
    if send.n_ok == 0 and send.n_sent:
        return (f"The endpoint did not answer: {send.n_sent}/{send.n_sent} requests "
                "failed. This says nothing about whether it captures — it was not "
                "reached. Check the status codes in send_log.csv; a 524 at ~125 s "
                "is a gateway timeout in front of the model, not a rejection of "
                "the prompt.")
    if send.n_ok < send.n_sent:
        return (f"Only {send.n_ok}/{send.n_sent} requests were answered, and none "
                "of those were recorded. Both problems are live at once.")
    return ("The endpoint answered every request and the platform recorded none "
            "of them. That is the condition this folder works around; it is a "
            "finding, not a failure.")


def run_replay(
    pack_dir: str,
    out_dir: str,
    *,
    dry_run: bool = False,
    model: str = HF_MODEL,
    device: Optional[str] = None,
    model_id: Optional[str] = QWEN_STG_MODEL_ID,
    tol: float = PARITY_STD_TOL,
    rtol: float = PARITY_STD_RTOL,
    progress: bool = True,
) -> Dict:
    """Send the pack, poll for what the platform recorded, and diff it.

    Returns a summary dict. **Exit code 0 regardless of how many rows the
    platform captured**: zero captured means the endpoint still does not record,
    which is the very condition this folder exists to work around, and reporting
    it is the point.
    """
    os.makedirs(out_dir, exist_ok=True)
    prompts = load_pack(pack_dir)
    if progress:
        print(f"{len(prompts)} prompts from {pack_dir}")

    if dry_run:
        if progress:
            print("--dry-run: not sending; platform side will be empty")
        send = SendOutcome(log=pd.DataFrame(columns=SEND_LOG_COLUMNS),
                           window_start="", window_end="")
        platform_rows = pd.DataFrame()
    else:
        if progress:
            print("\nsending:")
        send = send_pack(prompts, out_dir, progress=progress)
        if progress:
            print(f"\npolling {send.window_start} .. {send.window_end}")
        platform_rows = poll_captured(
            send.window_start, send.window_end, len(prompts),
            model_id=model_id, progress=progress,
        )
        platform_prompts = fetch_platform_prompts(
            send.window_start, send.window_end, model_id
        )
        platform_rows = _restamp_by_prompt(platform_rows, platform_prompts, prompts)

    if progress:
        print("\ncapturing locally:")
    local_rows = capture_pack_locally(
        prompts, model=model, device=device, progress=progress
    )

    diff = diff_rows(local_rows, platform_rows, passes=PARITY_PASSES)
    summary = summarise(diff, tol=tol, rtol=rtol)
    paths = write_parity_report(
        os.path.join(out_dir, "parity_replay"), diff, summary,
        title="Replay: local capture vs platform capture",
        notes=[
            f"pack: {pack_dir}",
            f"prompts sent: {send.n_sent} ({send.n_ok} ok)",
            f"window: {send.window_start} .. {send.window_end}" if send.window_start
            else "dry run: nothing sent",
            f"model: {model}",
            _zero_capture_explanation(send, dry_run) if summary.n_requests_captured == 0
            else f"{summary.n_requests_captured} requests were captured by the platform",
        ] + [f"send error — {e}" for e in send.errors[:10]],
    )

    captured = summary.n_requests_captured
    if progress:
        print(f"\ncaptured {captured}/{len(prompts)}")
        if captured == 0:
            print(_zero_capture_explanation(send, dry_run))
        print(f"report -> {paths['md']}")

    return {
        "n_prompts": len(prompts),
        "n_sent": send.n_sent,
        "n_ok": send.n_ok,
        "n_captured": captured,
        "dry_run": dry_run,
        "report": paths["md"],
        "rows": paths["csv"],
        "summary": summary,
    }
