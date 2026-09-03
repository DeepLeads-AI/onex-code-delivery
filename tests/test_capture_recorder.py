"""TEMPORARY — see ../TEMPORARY.md

Recorder tests only: no torch, no model, no network. Literals are hand-computed
over the planted 4-element vector [0.0, 0.5, 1.0, -2.0].
"""

import math

import numpy as np
import pytest

from platform_mimic import capture
from layer_profile.config import EMBED_SIGNAL


# ---------------------------------------------------------------------------
# stats_768
# ---------------------------------------------------------------------------

def test_stats_use_only_the_first_keep_dims():
    # 9.9 sits past keep_dims=4 and must not move any statistic
    stats = capture.stats_768([0.0, 0.5, 1.0, -2.0, 9.9], keep_dims=4)
    assert stats["mean"] == pytest.approx(-0.125)              # (0 + .5 + 1 - 2) / 4
    assert stats["norm"] == pytest.approx(math.sqrt(5.25))     # sqrt(0 + .25 + 1 + 4)
    # population var = (0.015625 + 0.390625 + 1.265625 + 3.515625) / 4 = 1.296875
    assert stats["std"] == pytest.approx(math.sqrt(1.296875))
    assert stats["sparsity"] == 0.25                           # one exact zero of four
    assert stats["saturation"] == 0.5                          # |1.0| and |-2.0| exceed 0.95


def test_saturation_boundary_is_strictly_greater_than_threshold():
    assert capture.stats_768([0.95, 0.95], keep_dims=2)["saturation"] == 0.0
    assert capture.stats_768([0.951, 0.95], keep_dims=2)["saturation"] == 0.5


def test_sparsity_counts_exact_zeros_only():
    assert capture.stats_768([0.0, 1e-12, 1.0, 1.0], keep_dims=4)["sparsity"] == 0.25


def test_stats_of_an_empty_vector_raise():
    with pytest.raises(ValueError):
        capture.stats_768([], keep_dims=4)


# ---------------------------------------------------------------------------
# SignalRecorder
# ---------------------------------------------------------------------------

def planted_hidden(t: int = 3, d: int = 10) -> np.ndarray:
    """(1, T, D) where position 0 is the vector the literals above are built on."""
    hidden = np.arange(t * d, dtype=np.float64).reshape(1, t, d) * -1.0   # decoys
    hidden[0, 0, :4] = [0.0, 0.5, 1.0, -2.0]
    return hidden


# The platform reduces sequence positions differently per signal family — solved
# against STG prefill rows, where the embedding std matches the mean over positions
# to 1e-8 while every decoder layer matches position 0 exactly.

def test_decoder_layers_reduce_to_position_zero():
    assert capture.reduction_for("hidden_states.causal_decoder.layer_0") == "first"
    assert capture.reduction_for("hidden_states.causal_decoder.layer_23") == "first"


def test_the_embedding_signal_reduces_to_the_mean_over_positions():
    assert capture.reduction_for(EMBED_SIGNAL) == "mean"


def test_embedding_rows_average_the_sequence():
    """Prefill: platform embed std tracks mean-over-positions, not position 0."""
    rec = capture.SignalRecorder(keep_dims=4, max_passes=8)
    rec.begin_pass()
    hidden = np.zeros((1, 2, 4))
    hidden[0, 0, :] = [0.0, 0.0, 0.0, 0.0]
    hidden[0, 1, :] = [0.0, 1.0, 2.0, -3.0]
    row = rec.record(EMBED_SIGNAL, hidden)
    # mean over the two positions = [0, 0.5, 1.0, -1.5]; mean of that = 0.0
    assert row["mean"] == pytest.approx(0.0)
    assert row["norm"] == pytest.approx(math.sqrt(0.25 + 1.0 + 2.25))


def test_single_position_passes_are_identical_under_both_reductions():
    """Generation passes have T=1, so the two rules agree — which is why only
    prefill rows exposed the difference."""
    rec = capture.SignalRecorder(keep_dims=4, max_passes=8)
    rec.begin_pass()
    hidden = np.array([[[0.0, 0.5, 1.0, -2.0]]])
    embed = rec.record(EMBED_SIGNAL, hidden)
    layer = rec.record("hidden_states.causal_decoder.layer_0", hidden)
    assert embed["std"] == pytest.approx(layer["std"])


def test_reduction_can_be_overridden_per_call():
    rec = capture.SignalRecorder(keep_dims=4, max_passes=8)
    rec.begin_pass()
    row = rec.record(EMBED_SIGNAL, planted_hidden(), reduction="first")
    assert row["mean"] == pytest.approx(-0.125)


def test_recorder_reads_batch_zero_position_zero():
    rec = capture.SignalRecorder(keep_dims=4, max_passes=8)
    rec.begin_pass()
    row = rec.record("hidden_states.causal_decoder.layer_0", planted_hidden())

    assert row["forward_pass_index"] == "1"
    assert row["std"] == pytest.approx(math.sqrt(1.296875))
    assert row["mean"] == pytest.approx(-0.125)
    assert row["norm"] == pytest.approx(math.sqrt(5.25))
    assert row["sparsity"] == 0.25
    assert row["saturation"] == 0.5
    assert row["n_positions"] == 3      # T, recorded but never compared to platform
    assert row["dim"] == 10             # full width before the 768-dim slice


def test_recorder_numbers_passes_from_one():
    rec = capture.SignalRecorder(keep_dims=4, max_passes=8)
    for _ in range(3):
        rec.begin_pass()
        rec.record(EMBED_SIGNAL, planted_hidden())
    assert [r["forward_pass_index"] for r in rec.rows] == ["1", "2", "3"]


def test_recorder_drops_passes_past_the_budget():
    rec = capture.SignalRecorder(keep_dims=4, max_passes=2)
    for _ in range(5):
        rec.begin_pass()
        rec.record(EMBED_SIGNAL, planted_hidden())
    assert [r["forward_pass_index"] for r in rec.rows] == ["1", "2"]
    assert rec.active is False


def test_recorder_is_inactive_before_the_first_pass():
    rec = capture.SignalRecorder(keep_dims=4, max_passes=8)
    assert rec.active is False
    assert rec.record(EMBED_SIGNAL, planted_hidden()) is None
    assert rec.rows == []


def test_recorder_keeps_vectors_only_for_requested_signal_types():
    rec = capture.SignalRecorder(
        keep_dims=4, max_passes=8, keep_vectors_for=("hidden_states.causal_decoder.layer_1",)
    )
    rec.begin_pass()
    rec.record("hidden_states.causal_decoder.layer_0", planted_hidden())
    rec.record("hidden_states.causal_decoder.layer_1", planted_hidden())
    assert list(rec.vectors) == ["hidden_states.causal_decoder.layer_1|1"]
    np.testing.assert_allclose(
        rec.vectors["hidden_states.causal_decoder.layer_1|1"], [0.0, 0.5, 1.0, -2.0]
    )


def test_recorder_accepts_two_and_one_dimensional_hidden_states():
    rec = capture.SignalRecorder(keep_dims=4, max_passes=8)
    rec.begin_pass()
    two_d = rec.record("hidden_states.causal_decoder.layer_0", planted_hidden()[0])
    one_d = rec.record("hidden_states.causal_decoder.layer_1", np.array([0.0, 0.5, 1.0, -2.0]))
    assert two_d["std"] == pytest.approx(one_d["std"])
    assert one_d["n_positions"] == 1


def test_reset_clears_passes_rows_and_vectors():
    rec = capture.SignalRecorder(keep_dims=4, max_passes=8, keep_vectors_for=(EMBED_SIGNAL,))
    rec.begin_pass()
    rec.record(EMBED_SIGNAL, planted_hidden())
    rec.reset()
    assert (rec.pass_index, rec.rows, rec.vectors) == (0, [], {})


# ---------------------------------------------------------------------------
# rows_to_df
# ---------------------------------------------------------------------------

def test_rows_to_df_has_the_platform_columns_plus_local_extras():
    rec = capture.SignalRecorder(keep_dims=4, max_passes=8)
    rec.begin_pass()
    rec.record(EMBED_SIGNAL, planted_hidden())
    df = capture.rows_to_df(rec.rows, request_id="req-1")

    assert list(df.columns) == capture.ROW_COLUMNS
    assert capture.STG_ROW_COLUMNS == [
        "request_id", "signal_type", "forward_pass_index",
        "std", "mean", "norm", "sparsity", "saturation",
    ]
    assert df["request_id"].tolist() == ["req-1"]


def test_rows_to_df_of_nothing_still_has_the_columns():
    df = capture.rows_to_df([])
    assert df.empty and list(df.columns) == capture.ROW_COLUMNS


def test_rows_round_trip_through_parquet(tmp_path):
    rec = capture.SignalRecorder(keep_dims=4, max_passes=8)
    rec.begin_pass()
    rec.record(EMBED_SIGNAL, planted_hidden())
    df = capture.rows_to_df(rec.rows, request_id="req-1")
    path = capture.save_rows(df, str(tmp_path / "rows.parquet"))
    back = capture.load_rows(path)
    assert back["std"].tolist() == df["std"].tolist()
    assert list(back.columns) == capture.ROW_COLUMNS


def test_layer_signal_type_matches_the_platform_naming():
    assert capture.layer_signal_type(0) == "hidden_states.causal_decoder.layer_0"
    assert capture.layer_signal_type(23) == "hidden_states.causal_decoder.layer_23"
