"""Score one captured request: the layer-profile metric plus the output-side gate.

The capture is supplied by the caller, so this module has no torch dependency and
no opinion about where the request ran. It takes any result object exposing
``prompt``, ``text``, ``rows``, ``vectors``, ``n_prompt_tokens`` and
``n_generated_tokens``.
"""

from typing import Dict, Optional, Sequence

import numpy as np

from .config import (
    FLAT_Z_CV_MAX,
    METRIC_LAYERS,
    METRIC_PASSES,
    OOD_PROB_MAX,
    REFERENCE_SIGNAL,
)
from .gate import run_gate
from .layer_profile import (
    LayerProfileReference,
    compute_stats,
    is_flattened,
    std_matrix,
)

#: Columns of a score row, in order, so a CSV of several scores is stable.
SCORE_COLUMNS = [
    "prompt_id",
    "n_chars", "n_prompt_tokens", "n_generated_tokens",
    "cv", "slope", "corr", "z_cv", "z_slope", "max_abs_z_layer",
    "cv_pass_slope", "n_passes_used",
    "ood_dist", "ood_prob", "is_ood",
    "text_ok", "text_reasons", "flag", "error",
    "prompt", "generated_text",
]


def score_capture(
    prompt_id: str,
    result,
    ref: LayerProfileReference,
    manifold=None,
    *,
    z_cv_max: float = FLAT_Z_CV_MAX,
    ood_prob_max: float = OOD_PROB_MAX,
    layers: Sequence[int] = METRIC_LAYERS,
    passes: Sequence[int] = METRIC_PASSES,
):
    """Score one captured request. Returns ``(row, std_matrix)``.

    Never raises for a single prompt: a prompt that hits EOS before enough
    passes has no profile, and that is a fact about the prompt, recorded in
    ``error``, not a reason to lose the rest of a batch.
    """
    row: Dict[str, object] = {
        "prompt_id": prompt_id,
        "n_chars": len(result.prompt),
        "prompt": result.prompt,
        "error": "",
    }
    row["n_prompt_tokens"] = result.n_prompt_tokens
    row["n_generated_tokens"] = result.n_generated_tokens
    row["generated_text"] = result.text

    matrix = std_matrix(result.rows, layers=layers, passes=passes)
    try:
        stats = compute_stats(matrix, ref)
    except ValueError as exc:
        # A prompt that stops generating early is excluded, with a reason.
        row["error"] = f"no profile: {exc}"
        return row, matrix

    row.update({
        "cv": stats.cv,
        "slope": stats.slope,
        "corr": stats.corr,
        "z_cv": stats.z_cv,
        "z_slope": stats.z_slope,
        "max_abs_z_layer": stats.max_abs_z_layer,
        "cv_pass_slope": stats.cv_pass_slope,
        "n_passes_used": stats.n_passes_used,
        "flag": is_flattened(stats, z_cv_max=z_cv_max),
    })

    vector = last_reference_vector(result.vectors)
    gate = run_gate(result.text, vector, manifold, ood_prob_max=ood_prob_max)
    row["text_ok"] = gate.text.ok
    row["text_reasons"] = "; ".join(gate.text.reasons)
    if gate.ood is not None:
        row.update({
            "ood_dist": gate.ood.dist,
            "ood_prob": gate.ood.ood_prob,
            "is_ood": gate.ood.is_ood,
        })
    return row, matrix


def last_reference_vector(vectors: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """The reference signal's vector at the highest pass — what the OOD panel scores."""
    keys = [k for k in vectors if k.startswith(f"{REFERENCE_SIGNAL}|")]
    if not keys:
        return None
    return vectors[max(keys, key=lambda k: int(k.rsplit("|", 1)[1]))]
