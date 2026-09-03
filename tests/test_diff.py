"""TEMPORARY — see ../TEMPORARY.md

Planted frames only — deltas here are exact by construction.
"""

import math
import os

import numpy as np
import pandas as pd
import pytest

from platform_mimic import diff as D


def row(rid, layer, fpi, std, **over):
    base = {
        "request_id": rid,
        "signal_type": f"hidden_states.causal_decoder.layer_{layer}",
        "forward_pass_index": fpi,
        "std": std, "mean": 0.0, "norm": 1.0, "sparsity": 0.0, "saturation": 0.0,
    }
    base.update(over)
    return base


LOCAL = pd.DataFrame([
    row("r1", 0, "1", 1.000), row("r1", 1, "1", 2.000),
    row("r1", 0, "2", 1.001), row("r1", 1, "2", 2.010),
])
PLATFORM = pd.DataFrame([
    row("r1", 0, "1", 1.000), row("r1", 1, "1", 2.000),
    row("r1", 0, "2", 1.000), row("r1", 1, "2", 2.000),
])


# ---------------------------------------------------------------------------
# normalise_rows
# ---------------------------------------------------------------------------

def test_normalise_keeps_only_comparable_columns_and_strings_the_pass():
    local = LOCAL.copy()
    local["n_positions"] = 7
    local["dim"] = 896
    local["forward_pass_index"] = [1, 1, 2, 2]        # int on the local side
    out = D.normalise_rows(local)
    assert list(out.columns) == D.NORMALISED_COLUMNS
    assert out["forward_pass_index"].tolist() == ["1", "1", "2", "2"]


def test_normalise_can_restamp_the_request_id():
    out = D.normalise_rows(LOCAL, request_id="pack-01")
    assert out["request_id"].unique().tolist() == ["pack-01"]


def test_normalise_drops_duplicate_keys():
    doubled = pd.concat([LOCAL, LOCAL], ignore_index=True)
    assert len(D.normalise_rows(doubled)) == len(LOCAL)


def test_normalise_of_nothing_has_the_columns():
    out = D.normalise_rows(pd.DataFrame())
    assert out.empty and list(out.columns) == D.NORMALISED_COLUMNS


# ---------------------------------------------------------------------------
# diff_rows
# ---------------------------------------------------------------------------

def test_deltas_are_exact():
    d = D.diff_rows(LOCAL, PLATFORM, passes=(1, 2))
    by_key = {(r.signal_type[-1], r.forward_pass_index): r for r in d.itertuples()}
    assert by_key[("0", "1")].d_std == pytest.approx(0.0)
    assert by_key[("0", "2")].d_std == pytest.approx(0.001)
    assert by_key[("1", "2")].d_std == pytest.approx(0.010)
    assert d["captured"].all()


def test_passes_outside_the_gate_are_excluded():
    d = D.diff_rows(LOCAL, PLATFORM, passes=(1,))
    assert d["forward_pass_index"].unique().tolist() == ["1"]
    assert len(d) == 2


def test_passes_none_compares_everything():
    assert len(D.diff_rows(LOCAL, PLATFORM, passes=None)) == 4


def test_a_row_only_the_local_side_has_is_not_captured():
    platform = PLATFORM.iloc[:3]
    d = D.diff_rows(LOCAL, platform, passes=(1, 2))
    missing = d[~d["captured"]]
    assert len(missing) == 1
    assert missing.iloc[0]["side"] == "local_only"
    assert math.isnan(missing.iloc[0]["d_std"])


def test_a_row_only_the_platform_side_has_is_reported_not_hidden():
    """A left join would swallow this; parity must notice it."""
    extra = pd.concat([PLATFORM, pd.DataFrame([row("r1", 2, "2", 3.0)])], ignore_index=True)
    d = D.diff_rows(LOCAL, extra, passes=(1, 2))
    assert (d["side"] == "platform_only").sum() == 1


def test_an_empty_platform_side_yields_nan_deltas_not_an_exception():
    d = D.diff_rows(LOCAL, pd.DataFrame(), passes=(1, 2))
    assert len(d) == 4
    assert not d["captured"].any()
    assert d["d_std"].isna().all()


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------

def test_summary_of_an_exact_match_passes():
    s = D.summarise(D.diff_rows(PLATFORM, PLATFORM, passes=(1, 2)), tol=5e-3)
    assert s.passed is True
    assert s.max_abs_d_std == 0.0
    assert s.n_matched == 4
    assert s.first_divergent_pass is None
    assert s.passes == [1, 2]


def test_summary_reports_the_first_divergent_pass():
    # rtol=0 isolates the absolute bound: the 0.010 delta sits on a signal of
    # 2.0, so the default relative term would admit it.
    s = D.summarise(D.diff_rows(LOCAL, PLATFORM, passes=(1, 2)), tol=5e-3, rtol=0.0)
    assert s.passed is False
    assert s.n_over_tol == 1                     # only the 0.010 delta exceeds 5e-3
    assert s.first_divergent_pass == 2
    assert s.max_abs_d_std == pytest.approx(0.010)
    assert s.max_abs_d_std_by_pass == {1: 0.0, 2: pytest.approx(0.010)}


def test_a_loose_tolerance_admits_the_same_diff():
    s = D.summarise(D.diff_rows(LOCAL, PLATFORM, passes=(1, 2)), tol=5e-2, rtol=0.0)
    assert s.passed is True and s.n_over_tol == 0


def test_the_default_relative_term_admits_that_same_diff():
    s = D.summarise(D.diff_rows(LOCAL, PLATFORM, passes=(1, 2)))
    assert s.passed is True          # 0.010 <= 0.005 + 0.01 * 2.0


def test_summary_with_nothing_captured_does_not_pass():
    s = D.summarise(D.diff_rows(LOCAL, pd.DataFrame(), passes=(1, 2)), tol=5e-3)
    assert s.passed is False
    assert s.n_matched == 0 and s.n_requests_captured == 0
    assert s.n_local_only == 4


def test_summary_of_an_empty_diff_does_not_pass():
    s = D.summarise(pd.DataFrame(), tol=5e-3)
    assert s.passed is False and s.n_matched == 0


def test_platform_only_rows_fail_even_when_matched_rows_are_exact():
    extra = pd.concat([PLATFORM, pd.DataFrame([row("r1", 2, "2", 3.0)])], ignore_index=True)
    s = D.summarise(D.diff_rows(PLATFORM, extra, passes=(1, 2)), tol=5e-3)
    assert s.n_comparable > 0
    assert s.within_tol is True and s.passed is False
    assert s.n_platform_only == 1


# ---------------------------------------------------------------------------
# write_parity_report
# ---------------------------------------------------------------------------

def test_report_is_written_on_a_pass(tmp_path):
    d = D.diff_rows(PLATFORM, PLATFORM, passes=(1, 2))
    paths = D.write_parity_report(str(tmp_path / "parity"), d, D.summarise(d))
    body = open(paths["md"]).read()
    assert body.startswith("# Parity: PASS")
    assert "max |delta std| by forward pass" in body
    assert os.path.exists(paths["csv"])
    assert len(pd.read_csv(paths["csv"])) == 4


def test_report_is_still_written_when_the_platform_captured_nothing(tmp_path):
    d = D.diff_rows(LOCAL, pd.DataFrame(), passes=(1, 2))
    paths = D.write_parity_report(str(tmp_path / "parity"), d, D.summarise(d))
    body = open(paths["md"]).read()
    assert body.startswith("# Parity: FAIL")
    assert "The platform captured nothing" in body


def test_report_explains_a_pass_one_only_divergence(tmp_path):
    local = pd.DataFrame([row("r1", 0, "1", 9.0), row("r1", 0, "2", 1.0)])
    platform = pd.DataFrame([row("r1", 0, "1", 1.0), row("r1", 0, "2", 1.0)])
    d = D.diff_rows(local, platform, passes=(1, 2))
    paths = D.write_parity_report(str(tmp_path / "parity"), d, D.summarise(d))
    body = open(paths["md"]).read()
    assert "first divergent pass: **1**" in body
    assert "chat template" in body


def test_report_carries_notes(tmp_path):
    d = D.diff_rows(PLATFORM, PLATFORM, passes=(1, 2))
    paths = D.write_parity_report(
        str(tmp_path / "parity"), d, D.summarise(d), notes=["device=cpu"]
    )
    assert "- device=cpu" in open(paths["md"]).read()


# ---------------------------------------------------------------------------
# Decode comparability
#
# The platform sampled its tokens, so local greedy decoding diverges from it at
# some pass. Rows after that point are not a parity failure — they are a
# different generation. The embedding signal is the fingerprint: it is a table
# lookup, so the same token gives the same std to floating-point noise.
# ---------------------------------------------------------------------------

EMBED = "embeddings.model.embed_tokens"


def embed_row(rid, fpi, std):
    return row(rid, 0, fpi, std, signal_type=EMBED)


def frames_with_fingerprint(local_embed_stds, platform_embed_stds, layer_std=1.0):
    """Build both sides with a controllable per-pass embedding fingerprint."""
    local, platform = [], []
    for i, (le, pe) in enumerate(zip(local_embed_stds, platform_embed_stds), start=1):
        local += [embed_row("r1", str(i), le), row("r1", 5, str(i), layer_std)]
        platform += [embed_row("r1", str(i), pe), row("r1", 5, str(i), layer_std)]
    return pd.DataFrame(local), pd.DataFrame(platform)


def test_matching_tokens_make_a_pass_comparable():
    local, platform = frames_with_fingerprint([0.01, 0.02, 0.03], [0.01, 0.02, 0.03])
    d = D.diff_rows(local, platform, passes=(1, 2, 3))
    assert d["comparable"].all()


def test_a_diverged_token_makes_that_pass_incomparable():
    local, platform = frames_with_fingerprint([0.01, 0.02], [0.01, 0.09])
    d = D.diff_rows(local, platform, passes=(1, 2))
    by_pass = d.groupby("pass")["comparable"].first()
    assert by_pass[1] is np.True_ or by_pass[1]
    assert not by_pass[2]


def test_comparability_is_cumulative_not_per_pass():
    """A token that re-matches after a divergence is still incomparable: the KV
    cache holds different tokens, so its hidden states cannot agree."""
    local, platform = frames_with_fingerprint([0.01, 0.02, 0.03], [0.01, 0.09, 0.03])
    d = D.diff_rows(local, platform, passes=(1, 2, 3))
    assert d.groupby("pass")["comparable"].first().tolist() == [True, False, False]


def test_incomparable_rows_are_excluded_from_the_verdict():
    local, platform = frames_with_fingerprint([0.01, 0.02], [0.01, 0.09], layer_std=1.0)
    # make the pass-2 layer row wildly wrong; it must not fail parity
    local.loc[(local["forward_pass_index"] == "2") & (local["signal_type"] != EMBED), "std"] = 99.0
    s = D.summarise(D.diff_rows(local, platform, passes=(1, 2)))
    assert s.passed is True
    assert s.n_comparable == 2                 # pass 1 only: embedding + layer
    assert s.n_decode_diverged == 2


def test_summary_counts_requests_that_diverged():
    local, platform = frames_with_fingerprint([0.01, 0.02], [0.01, 0.09])
    s = D.summarise(D.diff_rows(local, platform, passes=(1, 2)))
    assert s.n_requests_decode_diverged == 1
    assert s.decode_divergence_by_pass == {2: 1}


def test_requests_without_a_fingerprint_are_treated_as_comparable_and_counted():
    """The planted frames elsewhere in this file carry no embedding row. That is
    reported, never silently assumed away."""
    s = D.summarise(D.diff_rows(LOCAL, PLATFORM, passes=(1, 2)))
    assert s.n_requests_without_fingerprint == 1
    assert s.n_comparable == 4


# ---------------------------------------------------------------------------
# Two-sided tolerance
# ---------------------------------------------------------------------------

def test_relative_tolerance_admits_bfloat16_noise_on_a_large_signal():
    local = pd.DataFrame([row("r1", 23, "2", 3.4138)])
    platform = pd.DataFrame([row("r1", 23, "2", 3.4000)])
    d = D.diff_rows(local, platform, passes=(2,))
    assert D.summarise(d, tol=5e-3, rtol=1e-2).passed is True      # 0.0138 <= 0.005 + 0.034
    assert D.summarise(d, tol=5e-3, rtol=0.0).passed is False


def test_absolute_floor_still_governs_near_zero_signals():
    local = pd.DataFrame([row("r1", 0, "2", 0.0100)])
    platform = pd.DataFrame([row("r1", 0, "2", 0.0040)])
    d = D.diff_rows(local, platform, passes=(2,))
    # 0.006 > 0.005 + 0.01 * 0.004 -> the relative term cannot rescue it
    assert D.summarise(d, tol=5e-3, rtol=1e-2).passed is False


def test_summary_reports_the_worst_relative_error():
    local = pd.DataFrame([row("r1", 23, "2", 3.4138)])
    platform = pd.DataFrame([row("r1", 23, "2", 3.4000)])
    s = D.summarise(D.diff_rows(local, platform, passes=(2,)), tol=5e-3, rtol=1e-2)
    assert s.max_rel_d_std == pytest.approx(0.0138 / 3.4, rel=1e-3)


def test_report_explains_decode_divergence(tmp_path):
    local, platform = frames_with_fingerprint([0.01, 0.02], [0.01, 0.09])
    d = D.diff_rows(local, platform, passes=(1, 2))
    paths = D.write_parity_report(str(tmp_path / "parity"), d, D.summarise(d))
    body = open(paths["md"]).read()
    assert "sampled" in body and "not comparable" in body
