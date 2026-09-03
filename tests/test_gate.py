"""The OOD half is exercised against a real DistanceToManifoldEmbeddings fitted on
a planted 2-D Gaussian (``use_pca=False`` so the fit is exactly the cloud) — the
vendored detector itself, not a stand-in, so the gate is tested against what
actually scores in production.
"""

import numpy as np
import pytest

from layer_profile.manifold import DistanceToManifoldEmbeddings

from layer_profile import gate


@pytest.fixture
def manifold():
    rng = np.random.default_rng(0)
    cloud = rng.normal(loc=[0.0, 0.0], scale=[1.0, 1.0], size=(400, 2))
    det = DistanceToManifoldEmbeddings(shrinkage=1e-2, threshold_quantile=0.99, use_pca=False)
    det.fit(cloud)
    return det


# ---------------------------------------------------------------------------
# OOD side
# ---------------------------------------------------------------------------

def test_a_point_at_the_centre_is_in_domain(manifold):
    verdict = gate.ood_gate(np.array([0.0, 0.0]), manifold)
    assert verdict.is_ood is False
    assert verdict.dist == pytest.approx(0.0, abs=0.1)
    assert verdict.ood_prob < 0.05


def test_a_far_point_is_flagged(manifold):
    verdict = gate.ood_gate(np.array([12.0, -12.0]), manifold)
    assert verdict.is_ood is True
    assert verdict.ood_prob == pytest.approx(1.0)


def test_ood_prob_max_can_flag_a_point_the_manifold_threshold_admits(manifold):
    """A borderline point still moves the dashboard, so the gate is stricter."""
    borderline = np.array([2.0, 2.0])
    assert gate.ood_gate(borderline, manifold).is_ood is False
    strict = gate.ood_gate(borderline, manifold, ood_prob_max=0.5)
    assert strict.is_ood is True
    assert strict.ood_prob > 0.5


def test_ood_gate_accepts_a_row_vector(manifold):
    assert gate.ood_gate(np.array([[0.0, 0.0]]), manifold).dist == pytest.approx(
        gate.ood_gate(np.array([0.0, 0.0]), manifold).dist
    )


# ---------------------------------------------------------------------------
# Text side
# ---------------------------------------------------------------------------

HEALTHY = (
    "You can bake sourdough without a Dutch oven by preheating a baking stone and "
    "creating steam with a tray of boiling water on the lower rack. Score the loaf "
    "before it goes in, and bake at 230C for about 35 minutes until the crust is deep brown."
)


def test_a_normal_answer_passes():
    verdict = gate.text_checks(HEALTHY)
    assert verdict.ok is True and verdict.reasons == []


def test_truncated_output_is_rejected():
    verdict = gate.text_checks("Sure.")
    assert verdict.ok is False
    assert any("too short" in r for r in verdict.reasons)


def test_overlong_output_is_rejected():
    verdict = gate.text_checks("word " * 2000, max_chars=100)
    assert any("too long" in r for r in verdict.reasons)


def test_a_stuck_character_is_rejected():
    verdict = gate.text_checks("The answer is aaaaaaaaaaaaaaaaaaaa and that is all there is to say.")
    assert any("repeated" in r for r in verdict.reasons)


def test_a_collapsed_vocabulary_is_rejected():
    verdict = gate.text_checks("yes no " * 40)
    assert verdict.ok is False
    assert any("distinct-word ratio" in r for r in verdict.reasons)


def test_a_stuck_word_is_rejected():
    text = "the " * 30 + "lighthouse stood alone against the storm grey horizon blinking once"
    verdict = gate.text_checks(text)
    assert any("top word" in r for r in verdict.reasons)


def test_reasons_accumulate():
    verdict = gate.text_checks("aaaaaaaaaaaaaaaaaaaaaaaa")
    assert len(verdict.reasons) >= 2      # too short AND a stuck character


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------

def test_run_gate_passes_when_both_sides_pass(manifold):
    result = gate.run_gate(HEALTHY, np.array([0.0, 0.0]), manifold)
    assert result.ok is True and result.ood.is_ood is False


def test_run_gate_fails_on_the_ood_side_alone(manifold):
    result = gate.run_gate(HEALTHY, np.array([12.0, -12.0]), manifold)
    assert result.ok is False and result.text.ok is True


def test_run_gate_fails_on_the_text_side_alone(manifold):
    result = gate.run_gate("yes no " * 40, np.array([0.0, 0.0]), manifold)
    assert result.ok is False and result.ood.is_ood is False


def test_run_gate_without_a_manifold_skips_rather_than_passes_the_ood_half():
    result = gate.run_gate(HEALTHY)
    assert result.ood is None and result.ok is True
