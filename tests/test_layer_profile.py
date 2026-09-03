"""Literals here are hand-computed, not captured from a run: see the comment above
each one for the arithmetic.
"""

import math

import numpy as np
import pandas as pd
import pytest

from layer_profile import layer_profile as lp


# ---------------------------------------------------------------------------
# signal_type -> layer index
# ---------------------------------------------------------------------------

def test_layer_index_parses_decoder_layers():
    assert lp.layer_index("hidden_states.causal_decoder.layer_0") == 0
    assert lp.layer_index("hidden_states.causal_decoder.layer_23") == 23


def test_layer_index_maps_embeddings_below_layer_zero():
    assert lp.layer_index("embeddings.model.embed_tokens") == lp.EMBED_LAYER_INDEX == -1


@pytest.mark.parametrize("bad", [
    "output_distribution",
    "hidden_states.causal_decoder.layer_",
    "hidden_states.causal_decoder.layer_x",
    "",
])
def test_layer_index_rejects_everything_else(bad):
    with pytest.raises(ValueError):
        lp.layer_index(bad)


# ---------------------------------------------------------------------------
# std_matrix
# ---------------------------------------------------------------------------

def test_std_matrix_pivots_and_drops_out_of_scope_rows():
    rows = pd.DataFrame([
        {"signal_type": "hidden_states.causal_decoder.layer_0", "forward_pass_index": "2", "std": 0.5},
        {"signal_type": "hidden_states.causal_decoder.layer_1", "forward_pass_index": "2", "std": 1.5},
        {"signal_type": "hidden_states.causal_decoder.layer_0", "forward_pass_index": "3", "std": 0.7},
        # dropped: embedding signal is layer -1, outside the requested layers
        {"signal_type": "embeddings.model.embed_tokens", "forward_pass_index": "2", "std": 9.9},
        # dropped: layer outside the requested range
        {"signal_type": "hidden_states.causal_decoder.layer_5", "forward_pass_index": "2", "std": 3.3},
        # dropped: not a layer-profile signal type at all
        {"signal_type": "output_distribution", "forward_pass_index": "2", "std": 1.0},
        # dropped: pass outside the requested range
        {"signal_type": "hidden_states.causal_decoder.layer_1", "forward_pass_index": "9", "std": 2.0},
    ])
    m = lp.std_matrix(rows, layers=(0, 1), passes=(2, 3))
    assert m.shape == (2, 2)
    np.testing.assert_allclose(m[0], [0.5, 1.5])
    assert m[1, 0] == 0.7
    assert math.isnan(m[1, 1])


def test_std_matrix_of_nothing_is_all_nan():
    m = lp.std_matrix(pd.DataFrame(), layers=(0, 1), passes=(2, 3))
    assert m.shape == (2, 2) and np.all(np.isnan(m))


# ---------------------------------------------------------------------------
# profile statistics — linear ramp 0.1 * (k + 1), k = 0..4
# ---------------------------------------------------------------------------

RAMP = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
# mean 0.3; population var = (0.04+0.01+0+0.01+0.04)/5 = 0.02; sd = sqrt(0.02)
RAMP_CV = math.sqrt(0.02) / 0.3          # 0.4714045207910317


def test_cross_layer_cv_of_a_linear_ramp():
    assert lp.cross_layer_cv(RAMP) == pytest.approx(RAMP_CV)
    assert lp.cross_layer_cv(RAMP) == pytest.approx(0.4714045207910317)


def test_cv_is_scale_invariant():
    """A uniformly smaller request is not a flattened one."""
    assert lp.cross_layer_cv(RAMP * 17.0) == pytest.approx(RAMP_CV)


def test_profile_slope_of_a_linear_ramp_is_the_step():
    assert lp.profile_slope(RAMP, layers=(0, 1, 2, 3, 4)) == pytest.approx(0.1)


def test_flat_profile_has_zero_cv_and_zero_slope():
    flat = np.full(5, 0.3)
    assert lp.cross_layer_cv(flat) == 0.0
    assert lp.profile_slope(flat, layers=(0, 1, 2, 3, 4)) == 0.0


def test_corr_is_guarded_to_zero_when_a_side_is_flat():
    assert lp._corr(np.full(5, 0.3), RAMP) == 0.0
    assert lp._corr(RAMP, RAMP) == pytest.approx(1.0)


def test_cv_of_an_all_zero_profile_does_not_divide_by_zero():
    assert lp.cross_layer_cv(np.zeros(5)) == 0.0


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_profile_averages_over_passes_ignoring_nan():
    m = np.array([[1.0, 2.0, 3.0], [3.0, np.nan, 5.0]])
    np.testing.assert_allclose(lp.profile_from_matrix(m, min_passes=2), [2.0, 2.0, 4.0])


def test_profile_raises_when_too_few_passes_carry_data():
    m = np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]])
    assert lp.n_usable_passes(m) == 1
    with pytest.raises(ValueError, match="at least 2 passes"):
        lp.profile_from_matrix(m, min_passes=2)
    np.testing.assert_allclose(lp.profile_from_matrix(m, min_passes=1), [1.0, 2.0, 3.0])


def test_cv_by_pass_is_nan_for_empty_passes():
    m = np.array([[0.1, 0.2, 0.3], [np.nan, np.nan, np.nan]])
    per_pass = lp.cv_by_pass(m)
    assert per_pass[0] == pytest.approx(lp.cross_layer_cv(np.array([0.1, 0.2, 0.3])))
    assert math.isnan(per_pass[1])


# ---------------------------------------------------------------------------
# Reference band — two requests with profiles [1,2,3] and [3,4,5]
# ---------------------------------------------------------------------------

M1 = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
M2 = np.array([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]])
LAYERS = (0, 1, 2)
PASSES = (2, 3)

CV1 = math.sqrt(2.0 / 3.0) / 2.0          # 0.408248290463863
CV2 = math.sqrt(2.0 / 3.0) / 4.0          # 0.2041241452319315
CV_MEAN = (CV1 + CV2) / 2.0               # 0.30618621784789726
CV_SD = abs(CV1 - CV2) / math.sqrt(2.0)   # ddof=1 over two points


@pytest.fixture
def ref():
    return lp.fit_reference([M1, M2], layers=LAYERS, passes=PASSES, version="test")


def test_fit_reference_uses_sample_sd(ref):
    np.testing.assert_allclose(ref.profile_mean, [2.0, 3.0, 4.0])
    # ddof=1 over {1,3}: sqrt(((1-2)^2 + (3-2)^2) / 1) = sqrt(2)
    np.testing.assert_allclose(ref.profile_sd, [math.sqrt(2.0)] * 3)
    assert ref.cv_mean == pytest.approx(CV_MEAN)
    assert ref.cv_sd == pytest.approx(CV_SD)
    assert ref.slope_mean == pytest.approx(1.0)
    assert ref.slope_sd == pytest.approx(0.0)
    assert ref.n_requests == 2


def test_fit_reference_needs_more_than_one_request():
    with pytest.raises(ValueError, match="at least 2 requests"):
        lp.fit_reference([M1], layers=LAYERS, passes=PASSES)


def test_reference_round_trips_through_npz(tmp_path, ref):
    path = str(tmp_path / "ref.npz")
    ref.save(path)
    back = lp.LayerProfileReference.load(path)
    np.testing.assert_allclose(back.profile_mean, ref.profile_mean)
    np.testing.assert_allclose(back.profile_sd, ref.profile_sd)
    np.testing.assert_array_equal(back.layers, np.array(LAYERS))
    np.testing.assert_array_equal(back.passes, np.array(PASSES))
    assert (back.cv_mean, back.cv_sd) == (ref.cv_mean, ref.cv_sd)
    assert (back.slope_mean, back.slope_sd) == (ref.slope_mean, ref.slope_sd)
    assert back.n_requests == 2
    assert back.version == "test"


# ---------------------------------------------------------------------------
# compute_stats / is_flattened
# ---------------------------------------------------------------------------

def test_stats_of_a_flat_request_against_the_band(ref):
    flat = np.full((2, 3), 2.0)
    stats = lp.compute_stats(flat, ref)

    assert stats.cv == 0.0
    assert stats.slope == 0.0
    assert stats.corr == 0.0                        # guarded: flat side has no variance
    # z_cv = (0 - CV_MEAN) / CV_SD = -3/sqrt(2) since CV1 = 2 * CV2
    assert stats.z_cv == pytest.approx(-3.0 / math.sqrt(2.0))
    assert stats.z_cv == pytest.approx(-2.1213203435596424)
    assert stats.z_slope == 0.0                     # band has zero width -> guarded, not inf
    # profile [2,2,2] vs mean [2,3,4] over sd sqrt(2)
    np.testing.assert_allclose(
        stats.z_by_layer, [0.0, -1 / math.sqrt(2.0), -2 / math.sqrt(2.0)]
    )
    assert stats.max_abs_z_layer == pytest.approx(math.sqrt(2.0))
    np.testing.assert_allclose(stats.cv_by_pass, [0.0, 0.0])
    assert stats.cv_pass_slope == 0.0
    assert stats.n_passes_used == 2


def test_stats_of_a_healthy_request_sit_inside_the_band(ref):
    stats = lp.compute_stats(M1, ref)
    assert stats.corr == pytest.approx(1.0)
    # cv1 sits half the CV1-CV2 gap above the mean, and the ddof=1 sd over two
    # points is that gap / sqrt(2), so z = sqrt(2)/2.
    assert stats.z_cv == pytest.approx(1.0 / math.sqrt(2.0))
    assert not lp.is_flattened(stats, z_cv_max=-2.0)


def test_is_flattened_is_z_cv_only(ref):
    """The corr half of the designed rule is gone; the verdict is the collapse."""
    flat = lp.compute_stats(np.full((2, 3), 2.0), ref)
    assert flat.z_cv == pytest.approx(-2.1213203435596424)
    assert lp.is_flattened(flat, z_cv_max=-2.0)
    # same request, stricter collapse threshold -> not flagged
    assert not lp.is_flattened(flat, z_cv_max=-3.0)

    # A request that keeps the healthy shape (corr 1.0) IS flagged once the
    # threshold is loose enough to admit its z_cv. Under the old two-condition
    # rule the corr test vetoed exactly this case. Nothing here is a regression:
    # on real data corr never fell below 0.993, so the veto could never fire,
    # and keeping it would have meant a metric with a permanently dead half.
    healthy = lp.compute_stats(M1, ref)
    assert healthy.corr == pytest.approx(1.0)
    assert healthy.z_cv == pytest.approx(1.0 / math.sqrt(2.0))
    assert lp.is_flattened(healthy, z_cv_max=2.0)
    assert not lp.is_flattened(healthy, z_cv_max=-2.0)


def test_is_flattened_defaults_to_the_shipped_threshold(ref):
    """The default is -2.0, chosen because 0 of 45 healthy controls reach it."""
    from layer_profile.config import FLAT_Z_CV_MAX

    assert FLAT_Z_CV_MAX == -2.0
    flat = lp.compute_stats(np.full((2, 3), 2.0), ref)
    assert lp.is_flattened(flat)
