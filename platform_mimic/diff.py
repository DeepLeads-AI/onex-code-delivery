"""TEMPORARY — see TEMPORARY.md

Row-for-row comparison of a local capture against platform-captured rows.

Two callers, one code path:

* **parity** (`parity_check_live.py`) — do the local hooks reproduce the SDK's
  numbers on prompts the platform already captured? If they do not, every score
  this folder produces is measuring the local model rather than the platform's.
* **replay** (:mod:`replay`) — after sending the pack at the live endpoint, do the
  rows the platform recorded match what we captured locally?

Both are "same key, compare five scalars", so both use :func:`diff_rows`.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from layer_profile.config import (
    EMBED_SIGNAL,
    PARITY_PASSES,
    PARITY_STD_RTOL,
    PARITY_STD_TOL,
    SCALAR_COLUMNS,
    TOKEN_MATCH_TOL,
)

#: What makes a row comparable across the two sides.
KEY_COLUMNS = ["request_id", "signal_type", "forward_pass_index"]

NORMALISED_COLUMNS = KEY_COLUMNS + SCALAR_COLUMNS


def normalise_rows(df: pd.DataFrame, request_id: Optional[str] = None) -> pd.DataFrame:
    """Reduce either side to the comparable columns, with matching dtypes.

    ``forward_pass_index`` is a varchar on the platform and an int in most local
    code paths; both become strings here so the merge does not silently miss.
    Local-only columns (``n_positions``, ``dim``, ``time``) are dropped — they
    have no platform counterpart and would just be noise in the report.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=NORMALISED_COLUMNS)

    out = df.copy()
    if request_id is not None:
        out["request_id"] = request_id
    for col in NORMALISED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[NORMALISED_COLUMNS]
    out["request_id"] = out["request_id"].astype(str)
    out["signal_type"] = out["signal_type"].astype(str)
    out["forward_pass_index"] = out["forward_pass_index"].astype(str)
    for col in SCALAR_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.drop_duplicates(subset=KEY_COLUMNS, keep="first").reset_index(drop=True)


def comparability(
    merged: pd.DataFrame,
    *,
    token_match_tol: float = TOKEN_MATCH_TOL,
) -> pd.DataFrame:
    """Per (request, pass): was the same token fed, and was it fed all along?

    The platform sampled its generated tokens, so local greedy decoding follows
    it only until the first sampled token differs. The embedding signal is the
    fingerprint that detects this: ``embeddings.model.embed_tokens`` is a table
    lookup with no accumulation, so the same token gives the same std to
    floating-point noise (~1e-8 measured) while a different token is off by
    ~1e-3.

    Comparability is **cumulative**. A pass whose token happens to re-match after
    an earlier divergence is still not comparable — the KV cache holds different
    tokens, so its hidden states cannot be expected to agree. Judging such a row
    as a parity failure would blame the capture for the platform's sampling.

    Requests with no embedding row at all have no fingerprint; they are treated
    as comparable and counted in ``without_fingerprint`` so the assumption is
    always visible in the report rather than silent.

    Returns a frame indexed by (request_id, pass) with ``token_match``,
    ``comparable`` and ``has_fingerprint``.
    """
    index = merged[["request_id", "pass"]].drop_duplicates()
    fingerprint = merged[merged["signal_type"] == EMBED_SIGNAL]

    records = []
    for rid, group in index.groupby("request_id"):
        fp = fingerprint[fingerprint["request_id"] == rid].sort_values("pass")
        has_fp = not fp.empty
        matched_so_far = True
        matches = {}
        if has_fp:
            # Not itertuples(): "pass" is a Python keyword and gets renamed.
            for pass_index, d_std in zip(fp["pass"], fp["d_std"]):
                delta = abs(d_std) if pd.notna(d_std) else float("inf")
                matches[pass_index] = delta < token_match_tol
        for pass_index in sorted(group["pass"].dropna().unique()):
            token_match = matches.get(pass_index, True) if has_fp else True
            matched_so_far = matched_so_far and bool(token_match)
            records.append({
                "request_id": rid,
                "pass": pass_index,
                "token_match": bool(token_match),
                "comparable": bool(matched_so_far),
                "has_fingerprint": has_fp,
            })
    if not records:
        return pd.DataFrame(columns=["request_id", "pass", "token_match", "comparable", "has_fingerprint"])
    return pd.DataFrame(records)


def diff_rows(
    local: pd.DataFrame,
    platform: pd.DataFrame,
    *,
    passes: Optional[Sequence[int]] = PARITY_PASSES,
    columns: Sequence[str] = SCALAR_COLUMNS,
) -> pd.DataFrame:
    """Outer-join the two sides and difference each scalar.

    Outer rather than left: a row the platform recorded and we did not is just as
    much a parity failure as the reverse, and a left join would hide it. Rows
    present on only one side get NaN deltas and ``captured=False``, so the report
    is still written — an empty platform side must produce a report saying so,
    not an exception.
    """
    left = normalise_rows(local)
    right = normalise_rows(platform)
    if passes is not None:
        wanted = {str(p) for p in passes}
        left = left[left["forward_pass_index"].isin(wanted)]
        right = right[right["forward_pass_index"].isin(wanted)]

    merged = left.merge(
        right, on=KEY_COLUMNS, how="outer", suffixes=("_local", "_platform"), indicator=True
    )
    for col in columns:
        merged[f"d_{col}"] = merged[f"{col}_local"] - merged[f"{col}_platform"]
    merged["captured"] = merged["_merge"] == "both"
    merged["side"] = merged["_merge"].map(
        {"both": "both", "left_only": "local_only", "right_only": "platform_only"}
    )
    merged = merged.drop(columns=["_merge"])
    merged["pass"] = pd.to_numeric(merged["forward_pass_index"], errors="coerce")

    flags = comparability(merged)
    if flags.empty:
        merged["token_match"] = True
        merged["comparable"] = True
        merged["has_fingerprint"] = False
    else:
        merged = merged.merge(flags, on=["request_id", "pass"], how="left")
        merged["token_match"] = merged["token_match"].fillna(True).astype(bool)
        merged["comparable"] = merged["comparable"].fillna(True).astype(bool)
        merged["has_fingerprint"] = merged["has_fingerprint"].fillna(False).astype(bool)
    # A row can only be judged if both sides have it AND the decode still agreed.
    merged["comparable"] = merged["comparable"] & merged["captured"]
    return merged.sort_values(KEY_COLUMNS).reset_index(drop=True)


@dataclass
class ParitySummary:
    """Verdict over a diff frame."""
    n_local_rows: int
    n_platform_rows: int
    n_matched: int
    n_local_only: int
    n_platform_only: int
    n_requests_captured: int
    n_requests_local: int
    max_abs_d_std: float
    mean_abs_d_std: float
    n_over_tol: int
    within_tol: bool
    first_divergent_pass: Optional[int]
    tol: float
    rtol: float = PARITY_STD_RTOL
    max_rel_d_std: float = float("nan")
    n_comparable: int = 0
    n_decode_diverged: int = 0
    n_requests_decode_diverged: int = 0
    n_requests_without_fingerprint: int = 0
    passes: List[int] = field(default_factory=list)
    max_abs_d_std_by_pass: Dict[int, float] = field(default_factory=dict)
    decode_divergence_by_pass: Dict[int, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Needs comparable rows, all inside tolerance, and nothing platform-only.

        Judged on *comparable* rows only — rows past a sampled-token divergence
        describe a different generation, not a capture error.
        """
        return self.within_tol and self.n_comparable > 0 and self.n_platform_only == 0


def summarise(
    diff: pd.DataFrame,
    *,
    tol: float = PARITY_STD_TOL,
    rtol: float = PARITY_STD_RTOL,
    column: str = "std",
) -> ParitySummary:
    """Collapse a diff frame to a verdict.

    ``first_divergent_pass`` is the diagnostic that matters when this fails:
    divergence at pass 1 only means the chat template differs (the prefill sees a
    different token sequence); divergence across every pass means the numerics
    differ — dtype, device or attention kernel.
    """
    if diff.empty:
        return ParitySummary(
            n_local_rows=0, n_platform_rows=0, n_matched=0, n_local_only=0,
            n_platform_only=0, n_requests_captured=0, n_requests_local=0,
            max_abs_d_std=float("nan"), mean_abs_d_std=float("nan"), n_over_tol=0,
            within_tol=False, first_divergent_pass=None, tol=tol, rtol=rtol, passes=[],
        )

    matched = diff[diff["captured"]]
    comparable = diff[diff["comparable"]]
    delta = comparable[f"d_{column}"].abs()
    scale = comparable[f"{column}_platform"].abs()
    # numpy.isclose semantics: an absolute floor plus a relative term, because
    # the stored std spans four orders of magnitude in bfloat16.
    allowed = tol + rtol * scale
    over = comparable[delta > allowed]

    by_pass: Dict[int, float] = {}
    for pass_index, group in comparable.groupby("pass"):
        by_pass[int(pass_index)] = float(group[f"d_{column}"].abs().max())

    diverged = matched[~matched["comparable"]]
    divergence_by_pass = {
        int(p): int(g["request_id"].nunique()) for p, g in diverged.groupby("pass")
    }
    no_fingerprint = diff.loc[~diff["has_fingerprint"], "request_id"].nunique()

    relative = (delta / scale.clip(lower=1e-12)) if len(delta) else pd.Series(dtype=float)

    return ParitySummary(
        n_local_rows=int((diff["side"] != "platform_only").sum()),
        n_platform_rows=int((diff["side"] != "local_only").sum()),
        n_matched=int(len(matched)),
        n_local_only=int((diff["side"] == "local_only").sum()),
        n_platform_only=int((diff["side"] == "platform_only").sum()),
        n_requests_captured=int(diff.loc[diff["captured"], "request_id"].nunique()),
        n_requests_local=int(diff.loc[diff["side"] != "platform_only", "request_id"].nunique()),
        max_abs_d_std=float(delta.max()) if len(delta) else float("nan"),
        mean_abs_d_std=float(delta.mean()) if len(delta) else float("nan"),
        n_over_tol=int(len(over)),
        within_tol=bool(len(delta) > 0 and (delta <= allowed).all()),
        first_divergent_pass=int(over["pass"].min()) if len(over) else None,
        tol=tol,
        rtol=rtol,
        max_rel_d_std=float(relative.max()) if len(relative) else float("nan"),
        n_comparable=int(len(comparable)),
        n_decode_diverged=int(len(diverged)),
        n_requests_decode_diverged=int(diverged["request_id"].nunique()),
        n_requests_without_fingerprint=int(no_fingerprint),
        passes=sorted(int(p) for p in comparable["pass"].dropna().unique()),
        max_abs_d_std_by_pass=by_pass,
        decode_divergence_by_pass=divergence_by_pass,
    )


def write_parity_report(
    out_prefix: str,
    diff: pd.DataFrame,
    summary: ParitySummary,
    *,
    title: str = "Parity",
    notes: Sequence[str] = (),
) -> Dict[str, str]:
    """Write ``<prefix>.csv`` and ``<prefix>.md``. Always writes, even on failure."""
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)) or ".", exist_ok=True)
    csv_path = f"{out_prefix}.csv"
    md_path = f"{out_prefix}.md"

    diff.to_csv(csv_path, index=False)

    verdict = "PASS" if summary.passed else "FAIL"
    lines = [
        f"# {title}: {verdict}",
        "",
        f"- tolerance: `|delta| <= {summary.tol} + {summary.rtol} * |platform|`",
        f"- passes compared: `{summary.passes or 'none'}`",
        f"- requests local / captured by platform: {summary.n_requests_local} / "
        f"{summary.n_requests_captured}",
        f"- rows local / platform / matched: {summary.n_local_rows} / "
        f"{summary.n_platform_rows} / {summary.n_matched}",
        f"- rows only on one side: {summary.n_local_only} local, "
        f"{summary.n_platform_only} platform",
        f"- **rows comparable (same tokens fed): {summary.n_comparable}**",
        f"- rows excluded as decode divergence: {summary.n_decode_diverged} "
        f"(in {summary.n_requests_decode_diverged} requests)",
        f"- max |delta std|: {summary.max_abs_d_std:.3e}",
        f"- max relative delta: {summary.max_rel_d_std:.3e}",
        f"- mean |delta std|: {summary.mean_abs_d_std:.3e}",
        f"- rows over tolerance: {summary.n_over_tol}",
    ]

    if summary.n_decode_diverged:
        lines += [
            "",
            "> The platform **sampled** its generated tokens, so local greedy decoding "
            "follows it only until the first token differs. Passes at or after that "
            "point are **not comparable** — the KV cache holds different tokens — and "
            "are excluded from the verdict rather than counted as capture errors. "
            "Divergence is detected from the `embeddings.model.embed_tokens` "
            "fingerprint, and is cumulative: a token that re-matches later does not "
            "restore comparability.",
        ]
        if summary.decode_divergence_by_pass:
            lines += [
                "",
                "| first incomparable pass | requests |",
                "| --- | --- |",
            ] + [
                f"| {p} | {n} |"
                for p, n in sorted(summary.decode_divergence_by_pass.items())
            ]

    if summary.n_requests_without_fingerprint:
        lines += [
            "",
            f"> {summary.n_requests_without_fingerprint} request(s) carried no "
            "`embeddings.model.embed_tokens` row, so decode divergence could not be "
            "detected for them; their passes were assumed comparable.",
        ]
    if summary.first_divergent_pass is not None:
        lines += [
            f"- first divergent pass: **{summary.first_divergent_pass}**",
            "",
            "> Divergence at pass 1 only points at the chat template (the prefill sees a "
            "different token sequence). Divergence across every pass points at numerics — "
            "dtype, device or attention kernel.",
        ]
    if summary.n_matched == 0:
        lines += ["", "> No rows matched. The platform captured nothing for these requests."]
    elif summary.n_comparable == 0:
        lines += ["", "> No comparable rows: every pass diverged in its decoded tokens."]

    if summary.max_abs_d_std_by_pass:
        lines += ["", "## max |delta std| by forward pass", "", "| pass | max abs delta |", "| --- | --- |"]
        lines += [
            f"| {p} | {v:.3e} |" for p, v in sorted(summary.max_abs_d_std_by_pass.items())
        ]

    if notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]

    lines += ["", f"Rows: `{os.path.basename(csv_path)}`", ""]
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines))
    return {"csv": csv_path, "md": md_path}
