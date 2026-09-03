"""The layer-profile metric family — the panel this delivery adds.

Given a time window, how many requests show the cross-layer activation std
collapsing toward constant, and how far below the healthy band they sit.

Written in the shape of the platform's existing dashboard metric functions —
window strings in, dataclasses out, an optional ``model_id`` and
``signal_type_filter``, dynamic equal-width bands for the distribution — so it
sits beside the delivered OOD and drift panels rather than needing a rewrite to
join them.

Scalar rows are read through :func:`_read_scalars`, a one-line indirection whose
only purpose is that importing this module must not open a database connection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    DISTRIBUTION_NUM_BANDS,
    FLAT_Z_CV_MAX,
    LAYER_SIGNAL_PREFIX,
    METRIC_LAYERS,
    METRIC_PASSES,
    N_LAYERS,
)
from .layer_profile import (
    LayerProfileReference,
    compute_stats,
    is_flattened,
    std_matrix,
)


@dataclass
class ScoreBand:
    """One band in a score distribution, shaped like the dashboard's own."""
    lower: float
    upper: float
    count: int
    fraction: float


@dataclass
class LayerProfileSummary:
    """Summary layer-profile health for a time window."""
    flattened_rate: float        # fraction whose per-layer std collapsed (0-1)
    healthy_rate: float          # 1 - flattened_rate
    sample_count: int            # requests with a scoreable profile
    mean_cv: float               # mean cross-layer coefficient of variation
    mean_z_cv: float             # mean z of that CV against the healthy band
    mean_corr: float             # mean corr of the profile against the healthy shape
    reference_cv_mean: float     # the band itself, so the number is interpretable
    reference_cv_sd: float
    reference_n_requests: int
    n_incomplete: int = 0        # requests dropped for too few usable passes


@dataclass
class LayerProfileDistribution:
    """Distribution of per-request z_cv across dynamic bands."""
    bands: List[ScoreBand] = field(default_factory=list)
    sample_count: int = 0


@dataclass
class RequestLayerProfile:
    """Per-request layer-profile score."""
    request_id: str
    cv: float
    z_cv: float
    corr: float
    slope: float
    max_abs_z_layer: float
    cv_pass_slope: float
    n_passes_used: int
    is_flattened: bool


def default_signal_types(n_layers: int = N_LAYERS) -> List[str]:
    """The decoder-layer signal types the profile is defined over."""
    return [f"{LAYER_SIGNAL_PREFIX}{k}" for k in range(n_layers)]


def _read_scalars(
    start_time: str,
    end_time: str,
    signal_type_filter: Optional[List[str]] = None,
    model_id: Optional[str] = None,
    passes: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Indirection over :mod:`db_scalars` so importing this module binds nothing.

    Dropped into the platform, this becomes a direct import of whichever reader
    supplies the scalar columns.
    """
    from .db_scalars import get_signal_scalars_df
    return get_signal_scalars_df(
        start_time, end_time,
        signal_type_filter=signal_type_filter, model_id=model_id, passes=passes,
    )


def _score_requests(
    rows: pd.DataFrame,
    ref: LayerProfileReference,
    *,
    layers: Sequence[int],
    passes: Sequence[int],
    z_cv_max: float,
):
    """Score every request in a scalar frame. Returns (scores, n_incomplete)."""
    scores: List[RequestLayerProfile] = []
    incomplete = 0
    if rows.empty:
        return scores, incomplete

    for request_id, group in rows.groupby("request_id"):
        matrix = std_matrix(group, layers=layers, passes=passes)
        try:
            stats = compute_stats(matrix, ref)
        except ValueError:
            # Too few usable passes — a real property of short generations, and
            # counted rather than dropped silently.
            incomplete += 1
            continue
        scores.append(RequestLayerProfile(
            request_id=str(request_id),
            cv=stats.cv,
            z_cv=stats.z_cv,
            corr=stats.corr,
            slope=stats.slope,
            max_abs_z_layer=stats.max_abs_z_layer,
            cv_pass_slope=stats.cv_pass_slope,
            n_passes_used=stats.n_passes_used,
            is_flattened=is_flattened(stats, z_cv_max=z_cv_max),
        ))
    return scores, incomplete


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_layer_profile_scores(
    start_time: str,
    end_time: str,
    reference_path: str,
    model_id: Optional[str] = None,
    signal_type_filter: Optional[List[str]] = None,
    passes: Sequence[int] = METRIC_PASSES,
    layers: Sequence[int] = METRIC_LAYERS,
    z_cv_max: float = FLAT_Z_CV_MAX,
) -> List[RequestLayerProfile]:
    """Per-request layer-profile scores for a window."""
    ref = LayerProfileReference.load(reference_path)
    rows = _read_scalars(
        start_time, end_time,
        signal_type_filter=signal_type_filter or default_signal_types(),
        model_id=model_id, passes=passes,
    )
    scores, _ = _score_requests(
        rows, ref, layers=layers, passes=passes, z_cv_max=z_cv_max,
    )
    return scores


def get_layer_profile_summary(
    start_time: str,
    end_time: str,
    reference_path: str,
    model_id: Optional[str] = None,
    signal_type_filter: Optional[List[str]] = None,
    passes: Sequence[int] = METRIC_PASSES,
    layers: Sequence[int] = METRIC_LAYERS,
    z_cv_max: float = FLAT_Z_CV_MAX,
) -> LayerProfileSummary:
    """Summary layer-profile health for a time window.

    Args:
        start_time: Start of the monitoring window (DB timestamp format).
        end_time: End of the monitoring window.
        reference_path: Path to a saved healthy-band ``.npz``
            (``LayerProfileReference``), fitted with ``fit-reference``.
        model_id: Optional model_id to filter to a specific deployed model.
        signal_type_filter: Optional signal types; defaults to every decoder layer.
        passes: Forward passes to average over. Pass 1 is the chat-template
            prefill and carries no prompt-dependent signal, so it is excluded.
        layers: Decoder layers in the profile. Layer 23 is excluded — it feeds
            the final norm and dips on every request, healthy or not.
        z_cv_max: Flattening threshold on the z of the cross-layer CV.

    Returns:
        LayerProfileSummary. An empty window returns a zero-count summary
        carrying the reference band, not an exception — the panel still has
        something to render.
    """
    ref = LayerProfileReference.load(reference_path)
    rows = _read_scalars(
        start_time, end_time,
        signal_type_filter=signal_type_filter or default_signal_types(),
        model_id=model_id, passes=passes,
    )
    scores, incomplete = _score_requests(
        rows, ref, layers=layers, passes=passes, z_cv_max=z_cv_max,
    )

    if not scores:
        return LayerProfileSummary(
            flattened_rate=0.0, healthy_rate=1.0, sample_count=0,
            mean_cv=0.0, mean_z_cv=0.0, mean_corr=0.0,
            reference_cv_mean=ref.cv_mean, reference_cv_sd=ref.cv_sd,
            reference_n_requests=ref.n_requests, n_incomplete=incomplete,
        )

    flattened = float(np.mean([s.is_flattened for s in scores]))
    return LayerProfileSummary(
        flattened_rate=flattened,
        healthy_rate=1.0 - flattened,
        sample_count=len(scores),
        mean_cv=float(np.mean([s.cv for s in scores])),
        mean_z_cv=float(np.mean([s.z_cv for s in scores])),
        mean_corr=float(np.mean([s.corr for s in scores])),
        reference_cv_mean=ref.cv_mean,
        reference_cv_sd=ref.cv_sd,
        reference_n_requests=ref.n_requests,
        n_incomplete=incomplete,
    )


def get_layer_profile_distribution(
    start_time: str,
    end_time: str,
    reference_path: str,
    num_bands: int = DISTRIBUTION_NUM_BANDS,
    model_id: Optional[str] = None,
    signal_type_filter: Optional[List[str]] = None,
    passes: Sequence[int] = METRIC_PASSES,
    layers: Sequence[int] = METRIC_LAYERS,
    z_cv_max: float = FLAT_Z_CV_MAX,
) -> LayerProfileDistribution:
    """Distribution of per-request ``z_cv`` across ``num_bands`` equal-width bands.

    Bands span the observed [min, max], as the other dashboard distributions do.
    The final band is closed at the top so the maximum is counted.
    """
    scores = get_layer_profile_scores(
        start_time, end_time, reference_path,
        model_id=model_id, signal_type_filter=signal_type_filter,
        passes=passes, layers=layers, z_cv_max=z_cv_max,
    )
    if not scores:
        return LayerProfileDistribution(bands=[], sample_count=0)

    values = np.array([s.z_cv for s in scores], dtype=np.float64)
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        return LayerProfileDistribution(
            bands=[ScoreBand(lower=lo, upper=hi, count=len(values), fraction=1.0)],
            sample_count=len(values),
        )

    edges = np.linspace(lo, hi, num_bands + 1)
    bands: List[ScoreBand] = []
    for i in range(num_bands):
        lower, upper = float(edges[i]), float(edges[i + 1])
        if i < num_bands - 1:
            count = int(np.sum((values >= lower) & (values < upper)))
        else:
            count = int(np.sum((values >= lower) & (values <= upper)))
        bands.append(ScoreBand(
            lower=lower, upper=upper, count=count, fraction=count / len(values),
        ))
    return LayerProfileDistribution(bands=bands, sample_count=len(values))
