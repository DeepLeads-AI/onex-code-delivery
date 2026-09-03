"""TEMPORARY — see TEMPORARY.md

Reads the platform's own capture of the 2026-09-03 Qwen batch and freezes it on
disk, so everything downstream (the healthy band, the manifold, the parity check)
runs against a fixed snapshot rather than a live query.

Every function here opens a database connection; importing the module does not.
"""

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor

from layer_profile.config import (
    QWEN_STG_MODEL_ID,
    REFERENCE_SIGNAL,
    STG_WINDOW_END,
    STG_WINDOW_START,
)
from layer_profile.db import dsn
from layer_profile.db_scalars import get_signal_scalars_df
from layer_profile.reference import complete_request_ids


def stg_window() -> Tuple[str, str]:
    """The frozen capture window. A constant, not ``now`` — the batch is history."""
    return STG_WINDOW_START, STG_WINDOW_END


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_prompts(
    start_time: str = STG_WINDOW_START,
    end_time: str = STG_WINDOW_END,
    model_id: Optional[str] = QWEN_STG_MODEL_ID,
) -> pd.DataFrame:
    """Request ids, timestamps and the user turn of each /generate payload.

    Qwen requests store the prompt at ``payload->raw->messages->-1->>content``;
    the ``->>'text'`` shape belongs to the /predict family and is checked first
    so the same query works if this is ever pointed at a BERT model.
    """
    sql = """
        SELECT request_id, timestamp,
               COALESCE(
                 payload->'raw'->>'text',
                 payload->'raw'->'messages'->-1->>'content'
               ) AS prompt
        FROM request_payloads
        WHERE timestamp >= %s AND timestamp < %s
          AND deleted_at IS NULL
    """
    params: list = [start_time, end_time]
    if model_id is not None:
        sql += " AND model_id = %s"
        params.append(model_id)
    sql += " ORDER BY timestamp"

    conn = psycopg2.connect(dsn())
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame(columns=["request_id", "timestamp", "prompt"])
    return pd.DataFrame(rows)[["request_id", "timestamp", "prompt"]]


def fetch_scalar_rows(
    start_time: str = STG_WINDOW_START,
    end_time: str = STG_WINDOW_END,
    model_id: Optional[str] = QWEN_STG_MODEL_ID,
    signal_type_filter: Optional[List[str]] = None,
    passes: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """All per-layer scalar rows in the window (every forward pass)."""
    return get_signal_scalars_df(
        start_time, end_time,
        signal_type_filter=signal_type_filter,
        model_id=model_id,
        passes=passes,
    )


def fetch_lastpass_vectors(
    start_time: str = STG_WINDOW_START,
    end_time: str = STG_WINDOW_END,
    model_id: Optional[str] = QWEN_STG_MODEL_ID,
    signal_type: str = REFERENCE_SIGNAL,
) -> Tuple[List[str], np.ndarray]:
    """The stored 768-dim embedding of ``signal_type`` at each request's last pass.

    The pass is the last **generated** token, selected here rather than by the
    platform's own signal reader — that takes each request's
    ``MAX(forward_pass_index)``, which for a request that only logged the prefill
    is the pass-1 vector. See ``layer_profile.reference.fit_manifold`` for why
    that one vector is destructive.
    """
    return fetch_vectors_at_last_generation_pass(
        start_time, end_time, model_id, signal_type
    )


def fetch_vectors_at_last_generation_pass(
    start_time: str = STG_WINDOW_START,
    end_time: str = STG_WINDOW_END,
    model_id: Optional[str] = QWEN_STG_MODEL_ID,
    signal_type: str = REFERENCE_SIGNAL,
    min_pass: int = 2,
) -> Tuple[List[str], np.ndarray]:
    """One vector per request: ``signal_type`` at its highest generated-token pass.

    ``DISTINCT ON`` with a numeric cast in the ORDER BY, because
    ``forward_pass_index`` is a varchar where ``'10' < '8'``.
    """
    sql = """
        SELECT DISTINCT ON (ns.request_id)
               ns.request_id, ns.forward_pass_index, ns.embedding_vector
        FROM neural_signals ns
        JOIN request_payloads rp ON rp.request_id = ns.request_id
        WHERE rp.timestamp >= %s AND rp.timestamp < %s
          AND ns.deleted_at IS NULL
          AND rp.deleted_at IS NULL
          AND ns.signal_type = %s
          AND ns.embedding_vector IS NOT NULL
          AND array_length(ns.embedding_vector, 1) > 0
          AND ns.forward_pass_index IS NOT NULL
          AND ns.forward_pass_index ~ '^[0-9]+$'
          AND ns.forward_pass_index::int >= %s
    """
    params: list = [start_time, end_time, signal_type, min_pass]
    if model_id is not None:
        sql += " AND rp.model_id = %s"
        params.append(model_id)
    sql += " ORDER BY ns.request_id, ns.forward_pass_index::int DESC, ns.id"

    conn = psycopg2.connect(dsn())
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return [], np.empty((0, 0))
    ids = [r["request_id"] for r in rows]
    vectors = np.array([list(r["embedding_vector"]) for r in rows], dtype=np.float64)
    return ids, vectors


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------

def pull_and_freeze(
    out_dir: str,
    version: str,
    start_time: str = STG_WINDOW_START,
    end_time: str = STG_WINDOW_END,
    model_id: Optional[str] = QWEN_STG_MODEL_ID,
) -> dict:
    """Write the frozen STG snapshot and return a summary of what landed.

    Two files: the prompts (gitignored — they are client request payloads, only
    their sha256 is committed) and the scalar rows. The ``layer_14`` embeddings
    are not frozen: ``fit-manifold`` fetches them at fit time, so there is no
    second copy of them to drift out of step with the manifold itself.
    """
    from layer_profile.reference import reference_paths

    os.makedirs(out_dir, exist_ok=True)
    paths = reference_paths(out_dir, version)

    prompts = fetch_prompts(start_time, end_time, model_id)
    scalars = fetch_scalar_rows(start_time, end_time, model_id)

    prompts.to_csv(paths["prompts"], index=False)
    scalars.to_csv(paths["scalars"], index=False)

    complete = complete_request_ids(scalars)
    return {
        "version": version,
        "window": [start_time, end_time],
        "model_id": model_id,
        "n_requests": int(len(prompts)),
        "n_requests_with_signals": int(scalars["request_id"].nunique()) if not scalars.empty else 0,
        "n_requests_complete": len(complete),
        "n_signal_types": int(scalars["signal_type"].nunique()) if not scalars.empty else 0,
        "n_scalar_rows": int(len(scalars)),
        "complete_request_ids": complete,
        "paths": paths,
    }
