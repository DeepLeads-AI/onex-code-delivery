"""Scalar-column reader for ``neural_signals``.

The platform's existing signal reader returns ``embedding_vector``, and only for
each request's *last* forward pass — neither of which is what a per-layer profile
needs. This is the missing reader: one JOIN, every pass, the five scalar columns
the platform already stores. It adds no table, no column and no write.
"""

from typing import List, Optional, Sequence

import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor

from .config import SCALAR_COLUMNS
from .db import dsn

#: Columns returned, in order.
SCALAR_ROW_COLUMNS = [
    "request_id", "signal_type", "forward_pass_index",
] + SCALAR_COLUMNS + ["time"]


def get_signal_scalars_df(
    start_time: str,
    end_time: str,
    signal_type_filter: Optional[List[str]] = None,
    model_id: Optional[str] = None,
    passes: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Per-(request, signal_type, forward_pass_index) scalars in a time window.

    Unlike the platform's own reader this keeps **all** forward passes — the
    layer profile is an average over generated-token passes, so collapsing
    to the last pass would throw the metric away.

    ``DISTINCT ON`` deduplicates: the platform writes
    ``embeddings.model.embed_tokens`` twice per pass with byte-identical values
    (two hooks on the tied embedding module), and a duplicated row would double
    that layer's weight in any aggregate.

    Returns an empty frame with the right columns when nothing matches, matching
    the convention of the dashboard readers beside it.
    """
    sql = """
        SELECT DISTINCT ON (ns.request_id, ns.signal_type, ns.forward_pass_index)
               ns.request_id, ns.signal_type, ns.forward_pass_index,
               ns.std, ns.mean, ns.norm, ns.sparsity, ns.saturation, ns.time
        FROM neural_signals ns
        JOIN request_payloads rp ON rp.request_id = ns.request_id
        WHERE rp.timestamp >= %s AND rp.timestamp < %s
          AND ns.deleted_at IS NULL
          AND rp.deleted_at IS NULL
          AND ns.forward_pass_index IS NOT NULL
    """
    params: list = [start_time, end_time]
    if model_id is not None:
        sql += " AND rp.model_id = %s"
        params.append(model_id)
    if signal_type_filter:
        sql += " AND ns.signal_type = ANY(%s)"
        params.append(list(signal_type_filter))
    if passes:
        sql += " AND ns.forward_pass_index = ANY(%s)"
        params.append([str(p) for p in passes])
    sql += " ORDER BY ns.request_id, ns.signal_type, ns.forward_pass_index, ns.id"

    conn = psycopg2.connect(dsn())
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=SCALAR_ROW_COLUMNS)
    return pd.DataFrame([dict(r) for r in rows])[SCALAR_ROW_COLUMNS]
