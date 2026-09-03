"""Turns the frozen platform snapshot into the two things everything downstream
scores against: the **healthy layer-profile band** and the **OOD manifold**.

Also owns provenance. A reference ``.npz`` with no record of the window, the
model or the code that produced it is unauditable a month later, so every
reference file is hashed into a sidecar — including the prompts CSV, which is
never committed and whose sha256 is therefore the only record of it.
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_DB_NAME,
    MANIFOLD_PCA_K,
    MANIFOLD_SHRINKAGE,
    MANIFOLD_THRESHOLD_QUANTILE,
    METRIC_LAYERS,
    METRIC_PASSES,
    REFERENCE_SIGNAL,
)
from .layer_profile import (
    LayerProfileReference,
    fit_reference,
    layer_index,
    std_matrix,
)

#: Files that make up a versioned reference. The prompts CSV is gitignored;
#: everything else is committed.
FILE_TEMPLATES = {
    "prompts": "qwen_stg_prompts_{v}.csv",
    "scalars": "qwen_stg_layer_scalars_{v}.csv",
    "profile_ref": "qwen_stg_layer_profile_ref_{v}.npz",
    "manifold": "manifold_qwen_stg_layer14_{v}.npz",
    "provenance": "qwen_stg_reference_{v}.provenance.json",
}

TRACKED_FILES = ["scalars", "profile_ref", "manifold"]
UNTRACKED_FILES = ["prompts"]


def reference_paths(ref_dir: str, version: str) -> Dict[str, str]:
    return {
        key: os.path.join(ref_dir, template.format(v=version))
        for key, template in FILE_TEMPLATES.items()
    }


# ---------------------------------------------------------------------------
# The healthy band
# ---------------------------------------------------------------------------

def build_reference(
    scalars: pd.DataFrame,
    request_ids: Optional[Sequence[str]] = None,
    *,
    layers: Sequence[int] = METRIC_LAYERS,
    passes: Sequence[int] = METRIC_PASSES,
    version: str = "v1",
) -> LayerProfileReference:
    """Fit the healthy layer-profile band from platform-captured scalar rows.

    ``request_ids`` defaults to the requests with a complete (layer, pass) grid —
    partial requests would bias the band towards early-generation values.
    """
    ids = list(request_ids) if request_ids is not None else complete_request_ids(
        scalars, layers, passes
    )
    if not ids:
        raise ValueError("No complete requests to fit a reference band from")

    matrices = [
        std_matrix(scalars[scalars["request_id"] == rid], layers=layers, passes=passes)
        for rid in ids
    ]
    return fit_reference(matrices, layers=layers, passes=passes, version=version)


# ---------------------------------------------------------------------------
# The OOD manifold
# ---------------------------------------------------------------------------

def fit_manifold(
    embeddings: np.ndarray,
    output_path: str,
    *,
    pca_k: int = MANIFOLD_PCA_K,
    shrinkage: float = MANIFOLD_SHRINKAGE,
    quantile: float = MANIFOLD_THRESHOLD_QUANTILE,
) -> dict:
    """Fit and save the output-side OOD manifold from already-fetched vectors.

    Vectors come in as an argument rather than being queried here, so this stays
    pure: the caller decides which pass each vector is taken from, and that
    decision matters. Feed it each request's last **generated**-token vector.
    Taking the platform's own ``MAX(forward_pass_index)`` instead includes, for a
    request that only ever logged its prefill, the pass-1 vector — at position 0
    the constant ``<|im_start|>`` token, ``layer_14`` std ~61.9 against ~0.66 for
    a generated token. Exactly one such vector among 49 inflated the fitted
    covariance so far that every live request scored ``ood_prob`` ~0: a gate that
    cannot fail.

    The dimension and shrinkage policy matches the platform's own seeder, so a
    manifold fitted here scores the way the delivered OOD panel does.
    """
    from .manifold import DistanceToManifoldEmbeddings

    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 generated-token vectors, got {embeddings.shape}"
        )

    n_samples, n_features = embeddings.shape
    safe_k = min(pca_k, n_features, max(n_samples - 2, 2))
    if n_samples < safe_k + 2:
        raise ValueError(
            f"Need at least {safe_k + 2} samples to fit a {safe_k}-dim manifold, "
            f"got {n_samples}."
        )
    safe_shrinkage = shrinkage if n_samples > 2 * safe_k else max(shrinkage, 0.3)

    detector = DistanceToManifoldEmbeddings(
        shrinkage=safe_shrinkage, threshold_quantile=quantile,
        use_pca=True, pca_k=safe_k,
    )
    stats = detector.fit(embeddings)
    detector.save(output_path)
    return stats


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(repo_dir: Optional[str] = None) -> str:
    """HEAD of the working tree, or ``"unknown"`` outside a repo.

    Never raises: a missing git sha must not be able to fail a reference build.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir or os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def write_provenance(
    ref_dir: str,
    version: str,
    summary: dict,
    extra: Optional[dict] = None,
) -> str:
    """Record what produced this reference, and the sha256 of every file in it."""
    paths = reference_paths(ref_dir, version)
    files = {}
    for key, path in paths.items():
        if key == "provenance" or not os.path.exists(path):
            continue
        files[key] = {
            "file": os.path.basename(path),
            "sha256": sha256_file(path),
            "bytes": os.path.getsize(path),
            "tracked": key in TRACKED_FILES,
        }

    record = {
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "source": DEFAULT_DB_NAME,
        "window": summary.get("window"),
        "model_id": summary.get("model_id"),
        "reference_signal": REFERENCE_SIGNAL,
        "metric_layers": [int(k) for k in METRIC_LAYERS],
        "metric_passes": [int(p) for p in METRIC_PASSES],
        "counts": {k: v for k, v in summary.items() if k.startswith("n_")},
        "complete_request_ids": summary.get("complete_request_ids", []),
        "files": files,
    }
    record.update(extra or {})

    out = paths["provenance"]
    with open(out, "w") as fh:
        json.dump(record, fh, indent=2)
    return out


def verify_provenance(ref_dir: str, version: str) -> Dict[str, List[str]]:
    """Re-hash every file the sidecar names.

    Returns ``{"ok": [...], "changed": [...], "missing": [...]}``. Reports rather
    than raises, so a single tampered file still tells you about the others.
    """
    paths = reference_paths(ref_dir, version)
    with open(paths["provenance"]) as fh:
        record = json.load(fh)

    result: Dict[str, List[str]] = {"ok": [], "changed": [], "missing": []}
    for key, entry in record.get("files", {}).items():
        path = os.path.join(ref_dir, entry["file"])
        if not os.path.exists(path):
            result["missing"].append(entry["file"])
        elif sha256_file(path) == entry["sha256"]:
            result["ok"].append(entry["file"])
        else:
            result["changed"].append(entry["file"])
    return result


def load_manifold(path: str):
    """Load a saved manifold. Scoring needs no database connection."""
    from .manifold import DistanceToManifoldEmbeddings
    return DistanceToManifoldEmbeddings.load(path)


def load_reference_band(ref_dir: str, version: str) -> LayerProfileReference:
    return LayerProfileReference.load(reference_paths(ref_dir, version)["profile_ref"])


# ---------------------------------------------------------------------------
# Request selection
# ---------------------------------------------------------------------------

def complete_request_ids(
    scalars: pd.DataFrame,
    layers: Sequence[int] = METRIC_LAYERS,
    passes: Sequence[int] = METRIC_PASSES,
) -> List[str]:
    """Requests that have every (metric layer, metric pass) cell populated.

    A prompt that hit EOS after three tokens has a genuinely shorter profile;
    including it would pull the healthy band towards early-generation values.
    """
    if scalars.empty:
        return []
    wanted = {f"{k}|{p}" for k in layers for p in passes}
    df = scalars.copy()
    df["_k"] = [_safe_layer_index(s) for s in df["signal_type"]]
    df = df[df["_k"].isin(set(layers))]
    df["_cell"] = df["_k"].astype(str) + "|" + df["forward_pass_index"].astype(str)
    have = df.groupby("request_id")["_cell"].apply(lambda s: wanted <= set(s))
    return sorted(have[have].index.tolist())


def _safe_layer_index(signal_type: str) -> int:
    try:
        return layer_index(str(signal_type))
    except ValueError:
        return -99
