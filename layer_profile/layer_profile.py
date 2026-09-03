"""The layer-profile metric: pure numpy, no database, no torch.

A *layer profile* is the per-decoder-layer activation std of one request,
averaged over its generated-token forward passes. A healthy Qwen2.5-0.5B request
ramps from ~0.18 at layer 0 to ~2.4 at layer 22; the deterioration the client
asks us to demonstrate is that ramp collapsing towards a constant — the
cross-layer coefficient of variation falling out of the healthy band while the
model's *output* still reads normally.

Nothing here depends on anything else in this package except :mod:`config`
defaults, so it drops into a platform codebase as a single file.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    EMBED_SIGNAL,
    FLAT_Z_CV_MAX,
    LAYER_SIGNAL_PREFIX,
    METRIC_LAYERS,
    METRIC_PASSES,
    MIN_PASSES,
)

#: Sentinel layer index for the token-embedding signal, which sits "before"
#: layer 0 and is excluded from the decoder-layer profile.
EMBED_LAYER_INDEX = -1


# ---------------------------------------------------------------------------
# Signal-type parsing
# ---------------------------------------------------------------------------

def layer_index(signal_type: str) -> int:
    """Map a platform ``signal_type`` to a decoder-layer index.

    ``embeddings.model.embed_tokens`` -> ``-1``;
    ``hidden_states.causal_decoder.layer_K`` -> ``K``.
    Anything else is a signal type this metric has no meaning for, so it raises
    rather than being silently dropped into some default bucket.
    """
    if signal_type == EMBED_SIGNAL:
        return EMBED_LAYER_INDEX
    if signal_type.startswith(LAYER_SIGNAL_PREFIX):
        suffix = signal_type[len(LAYER_SIGNAL_PREFIX):]
        if suffix.isdigit():
            return int(suffix)
    raise ValueError(f"Not a layer-profile signal type: {signal_type!r}")


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------

def std_matrix(
    rows: pd.DataFrame,
    *,
    layers: Sequence[int] = METRIC_LAYERS,
    passes: Sequence[int] = METRIC_PASSES,
    value_column: str = "std",
) -> np.ndarray:
    """Pivot scalar rows for ONE request into a ``(n_passes, n_layers)`` matrix.

    ``rows`` needs the columns ``signal_type``, ``forward_pass_index`` and
    ``value_column``. ``forward_pass_index`` is a varchar in the platform schema
    ('1'..'8'), so it is coerced to int here. Missing (pass, layer) cells are
    NaN — every downstream statistic is NaN-aware, because a prompt that hits EOS
    early genuinely has fewer passes.
    """
    matrix = np.full((len(passes), len(layers)), np.nan, dtype=np.float64)
    if rows is None or len(rows) == 0:
        return matrix

    pass_pos = {p: i for i, p in enumerate(passes)}
    layer_pos = {k: j for j, k in enumerate(layers)}

    for signal_type, fpi, value in zip(
        rows["signal_type"], rows["forward_pass_index"], rows[value_column]
    ):
        try:
            k = layer_index(str(signal_type))
        except ValueError:
            continue          # output_distribution &c: not part of this metric
        j = layer_pos.get(k)
        if j is None:
            continue
        if fpi is None or (isinstance(fpi, float) and np.isnan(fpi)):
            continue
        i = pass_pos.get(int(fpi))
        if i is None:
            continue
        matrix[i, j] = float(value)
    return matrix


def last_generation_pass(
    rows: pd.DataFrame,
    *,
    min_pass: int = 2,
) -> dict:
    """Per request, the highest forward pass that is a *generated* token.

    The platform's own signal reader selects each request's
    ``MAX(forward_pass_index)``. For a request that only ever logged the prefill that is the pass-1 vector,
    which is a categorically different object: at position 0 the prefill is the
    constant ``<|im_start|>`` token, and its layer_14 activation std is ~61.9
    against ~0.66 for a generated token. One of those in a manifold's training
    set inflates the covariance enough that nothing scores as out-of-domain.

    Requests with no generation pass at all are absent from the result rather
    than falling back to the prefill.

    ``forward_pass_index`` is a varchar, so the comparison is numeric on purpose:
    lexicographically ``'10' < '8'``.
    """
    if rows is None or len(rows) == 0:
        return {}
    passes = pd.to_numeric(rows["forward_pass_index"], errors="coerce")
    usable = rows.assign(_pass=passes)
    usable = usable[usable["_pass"] >= min_pass]
    if usable.empty:
        return {}
    return {
        str(rid): int(value)
        for rid, value in usable.groupby("request_id")["_pass"].max().items()
    }


def profile_from_matrix(matrix: np.ndarray, *, min_passes: int = MIN_PASSES) -> np.ndarray:
    """Collapse a ``(P, L)`` std matrix to the ``(L,)`` layer profile.

    Averages over passes, ignoring NaN. Raises when fewer than ``min_passes``
    passes carry any data at all — a one-pass "profile" is a single token's
    hidden state, not a stable signature, and averaging it would quietly produce
    a number that looks like the others.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    usable = int(np.sum(~np.all(np.isnan(matrix), axis=1)))
    if usable < min_passes:
        raise ValueError(
            f"Need at least {min_passes} passes with data, got {usable}"
        )
    with np.errstate(invalid="ignore"):
        return np.nanmean(matrix, axis=0)


def n_usable_passes(matrix: np.ndarray) -> int:
    """How many passes of a ``(P, L)`` matrix carry at least one value."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return int(np.sum(~np.all(np.isnan(matrix), axis=1)))


# ---------------------------------------------------------------------------
# Scalar statistics of a profile
# ---------------------------------------------------------------------------

def cross_layer_cv(profile: np.ndarray) -> float:
    """Coefficient of variation of a layer profile: ``std / mean``.

    Scale-invariant on purpose — a request whose activations are uniformly
    smaller is not flattened, one whose layers all agree is. Population std
    (ddof=0) so the value does not depend on how many layers are in scope.
    """
    profile = np.asarray(profile, dtype=np.float64)
    mean = float(np.nanmean(profile))
    if not np.isfinite(mean) or mean == 0.0:
        return 0.0
    return float(np.nanstd(profile) / abs(mean))


def _ls_slope(y: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    """Least-squares slope of ``y`` against ``x`` (default 0, 1, 2, ...).

    NaN-tolerant; returns 0.0 when fewer than two finite points remain or x is
    constant, which is the honest answer for "no trend measurable".
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.arange(y.size, dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(x)
    if int(ok.sum()) < 2:
        return 0.0
    xs, ys = x[ok], y[ok]
    var = float(np.sum((xs - xs.mean()) ** 2))
    if var == 0.0:
        return 0.0
    return float(np.sum((xs - xs.mean()) * (ys - ys.mean())) / var)


def profile_slope(profile: np.ndarray, layers: Sequence[int] = METRIC_LAYERS) -> float:
    """Slope of the profile against layer index, in std units per layer."""
    return _ls_slope(profile, np.asarray(layers, dtype=np.float64))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, guarded.

    Returns 0.0 when either side has no variance — which is exactly the
    flattened case, where the correlation is genuinely undefined rather than
    perfect. Returning 0.0 keeps ``corr < corr_min`` true for flat profiles.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 2:
        return 0.0
    a, b = a[ok], b[ok]
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def cv_by_pass(matrix: np.ndarray) -> np.ndarray:
    """Per-pass cross-layer CV — a ``(P,)`` vector, NaN where a pass has no data.

    The per-pass series is what shows a profile flattening *during* generation
    rather than being flat from the first token.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    out = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    for i in range(matrix.shape[0]):
        row = matrix[i]
        if np.all(np.isnan(row)):
            continue
        out[i] = cross_layer_cv(row)
    return out


# ---------------------------------------------------------------------------
# Reference band
# ---------------------------------------------------------------------------

@dataclass
class LayerProfileReference:
    """The healthy band, fitted on platform-captured requests.

    ``profile_mean`` / ``profile_sd`` are per-layer; ``cv_*`` and ``slope_*`` are
    the across-request mean and sample sd (ddof=1) of the two scalars a live
    request is z-scored against.
    """
    layers: np.ndarray
    passes: np.ndarray
    profile_mean: np.ndarray
    profile_sd: np.ndarray
    cv_mean: float
    cv_sd: float
    slope_mean: float
    slope_sd: float
    n_requests: int
    version: str = "v1"

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            layers=np.asarray(self.layers, dtype=np.int32),
            passes=np.asarray(self.passes, dtype=np.int32),
            profile_mean=np.asarray(self.profile_mean, dtype=np.float64),
            profile_sd=np.asarray(self.profile_sd, dtype=np.float64),
            cv_mean=np.array([self.cv_mean], dtype=np.float64),
            cv_sd=np.array([self.cv_sd], dtype=np.float64),
            slope_mean=np.array([self.slope_mean], dtype=np.float64),
            slope_sd=np.array([self.slope_sd], dtype=np.float64),
            n_requests=np.array([self.n_requests], dtype=np.int32),
            version=np.array([self.version]),
        )

    @classmethod
    def load(cls, path: str) -> "LayerProfileReference":
        data = np.load(path, allow_pickle=False)
        return cls(
            layers=data["layers"].astype(int),
            passes=data["passes"].astype(int),
            profile_mean=data["profile_mean"].astype(np.float64),
            profile_sd=data["profile_sd"].astype(np.float64),
            cv_mean=float(data["cv_mean"][0]),
            cv_sd=float(data["cv_sd"][0]),
            slope_mean=float(data["slope_mean"][0]),
            slope_sd=float(data["slope_sd"][0]),
            n_requests=int(data["n_requests"][0]),
            version=str(data["version"][0]),
        )


def fit_reference(
    matrices: Sequence[np.ndarray],
    *,
    layers: Sequence[int] = METRIC_LAYERS,
    passes: Sequence[int] = METRIC_PASSES,
    version: str = "v1",
    min_passes: int = MIN_PASSES,
) -> LayerProfileReference:
    """Fit the healthy band from per-request ``(P, L)`` std matrices.

    Sample sd (ddof=1) throughout: the band is an estimate from a finite set of
    captured requests, and z-scores computed against a population sd would read
    slightly too extreme.
    """
    profiles, cvs, slopes = [], [], []
    for m in matrices:
        profile = profile_from_matrix(m, min_passes=min_passes)
        profiles.append(profile)
        cvs.append(cross_layer_cv(profile))
        slopes.append(profile_slope(profile, layers))

    if len(profiles) < 2:
        raise ValueError(f"Need at least 2 requests to fit a band, got {len(profiles)}")

    stack = np.vstack(profiles)
    return LayerProfileReference(
        layers=np.asarray(layers, dtype=int),
        passes=np.asarray(passes, dtype=int),
        profile_mean=np.nanmean(stack, axis=0),
        profile_sd=np.nanstd(stack, axis=0, ddof=1),
        cv_mean=float(np.mean(cvs)),
        cv_sd=float(np.std(cvs, ddof=1)),
        slope_mean=float(np.mean(slopes)),
        slope_sd=float(np.std(slopes, ddof=1)),
        n_requests=len(profiles),
        version=version,
    )


# ---------------------------------------------------------------------------
# Per-request statistics
# ---------------------------------------------------------------------------

@dataclass
class LayerProfileStats:
    """Everything the search and the charts need about one request's profile."""
    cv: float                    # cross-layer coefficient of variation
    slope: float                 # std units per layer
    corr: float                  # Pearson corr of the profile vs the healthy mean
    z_cv: float                  # (cv - ref.cv_mean) / ref.cv_sd  — the headline number
    z_slope: float
    z_by_layer: np.ndarray       # (L,) per-layer z against the band
    max_abs_z_layer: float
    cv_by_pass: np.ndarray       # (P,)
    cv_pass_slope: float         # trend of the per-pass CV during generation
    n_passes_used: int
    profile: np.ndarray          # (L,) the profile itself, for charting


def _safe_z(value: float, mean: float, sd: float) -> float:
    """z-score that degrades to 0.0 rather than inf when the band has zero width."""
    if not np.isfinite(sd) or sd == 0.0:
        return 0.0
    return float((value - mean) / sd)


def compute_stats(
    matrix: np.ndarray,
    ref: LayerProfileReference,
    *,
    min_passes: int = MIN_PASSES,
) -> LayerProfileStats:
    """Score one request's ``(P, L)`` std matrix against the healthy band."""
    profile = profile_from_matrix(matrix, min_passes=min_passes)
    cv = cross_layer_cv(profile)
    slope = profile_slope(profile, ref.layers)
    per_pass = cv_by_pass(matrix)

    sd = np.asarray(ref.profile_sd, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_by_layer = np.where(sd > 0, (profile - ref.profile_mean) / sd, 0.0)
    finite_z = z_by_layer[np.isfinite(z_by_layer)]

    return LayerProfileStats(
        cv=cv,
        slope=slope,
        corr=_corr(profile, ref.profile_mean),
        z_cv=_safe_z(cv, ref.cv_mean, ref.cv_sd),
        z_slope=_safe_z(slope, ref.slope_mean, ref.slope_sd),
        z_by_layer=z_by_layer,
        max_abs_z_layer=float(np.max(np.abs(finite_z))) if finite_z.size else 0.0,
        cv_by_pass=per_pass,
        cv_pass_slope=_ls_slope(per_pass),
        n_passes_used=n_usable_passes(matrix),
        profile=profile,
    )


def is_flattened(stats: LayerProfileStats, *, z_cv_max: float = FLAT_Z_CV_MAX) -> bool:
    """The signature asked for: the cross-layer std has collapsed toward constant.

    One condition, on the collapse itself. The metric was designed with a second
    one — ``corr < 0.99``, meant to confirm the profile's *shape* had changed
    rather than the whole curve merely being scaled — but measured over 270
    scored requests the correlation never fell below 0.993, so that half can
    never fire on this model. ``corr`` stays in :class:`LayerProfileStats` as a
    diagnostic; it is no longer part of the verdict. See ``config.FLAT_Z_CV_MAX``.
    """
    return bool(stats.z_cv <= z_cv_max)
