"""Planted scalar frames in, literal summary values out. ``_read_scalars`` is
monkeypatched at the module boundary, so no database is touched and no
connection module is imported.
"""

import math

import numpy as np
import pandas as pd
import pytest

from layer_profile import dashboard_metrics as dm
from layer_profile import layer_profile as lp

LAYERS = (0, 1, 2)
PASSES = (2, 3)

# Same planted band as test_layer_profile: profiles [1,2,3] and [3,4,5].
CV1 = math.sqrt(2.0 / 3.0) / 2.0
CV2 = math.sqrt(2.0 / 3.0) / 4.0
CV_MEAN = (CV1 + CV2) / 2.0
CV_SD = abs(CV1 - CV2) / math.sqrt(2.0)

Z_HEALTHY = 1.0 / math.sqrt(2.0)        # profile [1,2,3]
Z_FLAT = -3.0 / math.sqrt(2.0)          # profile [2,2,2]


@pytest.fixture
def reference_path(tmp_path):
    band = lp.fit_reference(
        [np.array([[1.0, 2.0, 3.0]] * 2), np.array([[3.0, 4.0, 5.0]] * 2)],
        layers=LAYERS, passes=PASSES, version="test",
    )
    path = str(tmp_path / "band.npz")
    band.save(path)
    return path


def scalar_rows(*profiles_by_request):
    """Build a platform-shaped scalar frame from {request_id: {pass: [stds]}}."""
    rows = []
    for request_id, per_pass in profiles_by_request:
        for pass_index, values in per_pass.items():
            for layer, value in enumerate(values):
                rows.append({
                    "request_id": request_id,
                    "signal_type": f"hidden_states.causal_decoder.layer_{layer}",
                    "forward_pass_index": str(pass_index),
                    "std": value,
                    "mean": 0.0, "norm": 1.0, "sparsity": 0.0, "saturation": 0.0,
                    "time": "2026-09-03 14:40:00+00",
                })
    return pd.DataFrame(rows)


HEALTHY = ("r_healthy", {2: [1.0, 2.0, 3.0], 3: [1.0, 2.0, 3.0]})
FLAT = ("r_flat", {2: [2.0, 2.0, 2.0], 3: [2.0, 2.0, 2.0]})
SHORT = ("r_short", {2: [1.0, 2.0, 3.0]})          # one usable pass only


@pytest.fixture
def plant(monkeypatch):
    def _plant(frame):
        captured = {}

        def fake(start_time, end_time, signal_type_filter=None, model_id=None, passes=None):
            captured.update({
                "start_time": start_time, "end_time": end_time,
                "signal_type_filter": signal_type_filter,
                "model_id": model_id, "passes": passes,
            })
            return frame
        monkeypatch.setattr(dm, "_read_scalars", fake)
        return captured
    return _plant


WINDOW = ("2026-09-03 14:30:00+00", "2026-09-03 15:00:00+00")


def summary(reference_path, **kw):
    return dm.get_layer_profile_summary(
        *WINDOW, reference_path, passes=PASSES, layers=LAYERS,
        z_cv_max=-2.0, **kw
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def test_summary_literals(reference_path, plant):
    plant(scalar_rows(HEALTHY, FLAT))
    s = summary(reference_path)

    assert s.sample_count == 2
    assert s.flattened_rate == 0.5
    assert s.healthy_rate == 0.5
    assert s.mean_cv == pytest.approx((CV1 + 0.0) / 2)
    assert s.mean_z_cv == pytest.approx((Z_HEALTHY + Z_FLAT) / 2)
    assert s.mean_corr == pytest.approx(0.5)        # 1.0 healthy, 0.0 flat (guarded)
    assert s.reference_cv_mean == pytest.approx(CV_MEAN)
    assert s.reference_cv_sd == pytest.approx(CV_SD)
    assert s.reference_n_requests == 2
    assert s.n_incomplete == 0


def test_requests_with_too_few_passes_are_counted_not_dropped_silently(reference_path, plant):
    plant(scalar_rows(HEALTHY, FLAT, SHORT))
    s = summary(reference_path)
    assert s.sample_count == 2
    assert s.n_incomplete == 1


def test_an_all_healthy_window_reports_a_zero_rate(reference_path, plant):
    plant(scalar_rows(HEALTHY))
    s = summary(reference_path)
    assert s.flattened_rate == 0.0 and s.healthy_rate == 1.0 and s.sample_count == 1


def test_an_empty_window_still_carries_the_band(reference_path, plant):
    plant(pd.DataFrame())
    s = summary(reference_path)
    assert s.sample_count == 0
    assert s.flattened_rate == 0.0 and s.healthy_rate == 1.0
    assert s.reference_cv_mean == pytest.approx(CV_MEAN)


def test_the_reader_is_given_the_window_model_and_default_signal_types(reference_path, plant):
    captured = plant(scalar_rows(HEALTHY))
    summary(reference_path, model_id="model_K1WST_TPJHBETCDoYlhDkg")
    assert captured["start_time"] == WINDOW[0]
    assert captured["end_time"] == WINDOW[1]
    assert captured["model_id"] == "model_K1WST_TPJHBETCDoYlhDkg"
    assert captured["passes"] == PASSES
    assert captured["signal_type_filter"] == dm.default_signal_types()


def test_default_signal_types_are_every_decoder_layer():
    types = dm.default_signal_types()
    assert len(types) == 24
    assert types[0] == "hidden_states.causal_decoder.layer_0"
    assert types[-1] == "hidden_states.causal_decoder.layer_23"
    assert "embeddings.model.embed_tokens" not in types


def test_a_caller_can_override_the_signal_types(reference_path, plant):
    captured = plant(scalar_rows(HEALTHY))
    summary(reference_path, signal_type_filter=["hidden_states.causal_decoder.layer_0"])
    assert captured["signal_type_filter"] == ["hidden_states.causal_decoder.layer_0"]


# ---------------------------------------------------------------------------
# Per-request scores
# ---------------------------------------------------------------------------

def test_per_request_scores(reference_path, plant):
    plant(scalar_rows(HEALTHY, FLAT))
    scores = {s.request_id: s for s in dm.get_layer_profile_scores(
        *WINDOW, reference_path, passes=PASSES, layers=LAYERS,
        z_cv_max=-2.0,
    )}
    assert scores["r_healthy"].is_flattened is False
    assert scores["r_healthy"].z_cv == pytest.approx(Z_HEALTHY)
    assert scores["r_flat"].is_flattened is True
    assert scores["r_flat"].z_cv == pytest.approx(Z_FLAT)
    assert scores["r_flat"].n_passes_used == 2


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

def test_distribution_bands(reference_path, plant):
    plant(scalar_rows(HEALTHY, FLAT))
    dist = dm.get_layer_profile_distribution(
        *WINDOW, reference_path, num_bands=2, passes=PASSES, layers=LAYERS,
        z_cv_max=-2.0,
    )
    assert dist.sample_count == 2
    assert len(dist.bands) == 2
    assert dist.bands[0].lower == pytest.approx(Z_FLAT)
    assert dist.bands[-1].upper == pytest.approx(Z_HEALTHY)
    assert [b.count for b in dist.bands] == [1, 1]
    assert [b.fraction for b in dist.bands] == [0.5, 0.5]


def test_distribution_last_band_includes_the_maximum(reference_path, plant):
    plant(scalar_rows(HEALTHY, FLAT))
    dist = dm.get_layer_profile_distribution(
        *WINDOW, reference_path, num_bands=10, passes=PASSES, layers=LAYERS,
    )
    assert sum(b.count for b in dist.bands) == dist.sample_count == 2


def test_identical_scores_collapse_to_one_band(reference_path, plant):
    plant(scalar_rows(HEALTHY, ("r_healthy2", {2: [1.0, 2.0, 3.0], 3: [1.0, 2.0, 3.0]})))
    dist = dm.get_layer_profile_distribution(
        *WINDOW, reference_path, num_bands=10, passes=PASSES, layers=LAYERS,
    )
    assert len(dist.bands) == 1
    assert dist.bands[0].count == 2 and dist.bands[0].fraction == 1.0


def test_an_empty_window_has_no_bands(reference_path, plant):
    plant(pd.DataFrame())
    dist = dm.get_layer_profile_distribution(*WINDOW, reference_path)
    assert dist.bands == [] and dist.sample_count == 0
