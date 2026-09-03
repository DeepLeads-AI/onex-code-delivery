"""Scoring one captured request. The capture is planted, so every number below
is arithmetic over the same [1,2,3] / [3,4,5] band as ``test_layer_profile``.
"""

import math

import numpy as np
import pandas as pd
import pytest

from layer_profile import layer_profile as lp
from layer_profile.score import SCORE_COLUMNS, last_reference_vector, score_capture

LAYERS = (0, 1, 2)
PASSES = (2, 3)

SIGNAL = "hidden_states.causal_decoder.layer_"
REF_SIGNAL = "hidden_states.causal_decoder.layer_14"

HEALTHY_TEXT = (
    "You can bake sourdough without a Dutch oven by preheating a baking stone and "
    "creating steam with a tray of boiling water on the lower rack. Score the loaf "
    "before it goes in, and bake at 230C for about 35 minutes until the crust is deep brown."
)


class FakeResult:
    """The duck type ``score_capture`` documents, and nothing more."""

    def __init__(self, profile_by_pass, text=HEALTHY_TEXT, vectors=None, prompt="a prompt"):
        self.prompt = prompt
        self.text = text
        self.vectors = {} if vectors is None else vectors
        self.n_prompt_tokens = 11
        self.n_generated_tokens = 64
        self.rows = pd.DataFrame([
            {"signal_type": f"{SIGNAL}{k}", "forward_pass_index": str(p), "std": value}
            for p, values in profile_by_pass.items()
            for k, value in zip(LAYERS, values)
        ])


@pytest.fixture
def ref():
    m1 = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    m2 = np.array([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]])
    return lp.fit_reference([m1, m2], layers=LAYERS, passes=PASSES, version="test")


def score(result, ref, manifold=None, **kw):
    return score_capture(
        "p-00", result, ref, manifold, layers=LAYERS, passes=PASSES, **kw
    )


# ---------------------------------------------------------------------------
# The metric half
# ---------------------------------------------------------------------------

def test_a_flat_request_is_flagged_with_the_literal_scores(ref):
    """Profile [2,2,2]: cv 0, and z_cv = (0 - CV_MEAN) / CV_SD = -3/sqrt(2)."""
    row, matrix = score(FakeResult({2: [2.0, 2.0, 2.0], 3: [2.0, 2.0, 2.0]}), ref)

    assert row["cv"] == 0.0
    assert row["slope"] == 0.0
    assert row["corr"] == 0.0                      # guarded: a flat side has no variance
    assert row["z_cv"] == pytest.approx(-3.0 / math.sqrt(2.0))
    assert row["z_cv"] == pytest.approx(-2.1213203435596424)
    assert row["n_passes_used"] == 2
    assert row["flag"] is True                     # -2.12 <= the shipped -2.0
    assert row["error"] == ""
    assert matrix.shape == (2, 3)


def test_a_healthy_request_is_not_flagged(ref):
    row, _ = score(FakeResult({2: [1.0, 2.0, 3.0], 3: [1.0, 2.0, 3.0]}), ref)

    assert row["corr"] == pytest.approx(1.0)
    assert row["z_cv"] == pytest.approx(1.0 / math.sqrt(2.0))
    assert row["flag"] is False


def test_the_threshold_is_overridable(ref):
    result = FakeResult({2: [2.0, 2.0, 2.0], 3: [2.0, 2.0, 2.0]})
    assert score(result, ref, z_cv_max=-3.0)[0]["flag"] is False


def test_prompt_length_comes_from_the_prompt_not_the_answer(ref):
    result = FakeResult({2: [1.0, 2.0, 3.0], 3: [1.0, 2.0, 3.0]}, prompt="12345")
    row, _ = score(result, ref)
    assert row["n_chars"] == 5
    assert row["prompt"] == "12345"


# ---------------------------------------------------------------------------
# Too few passes — recorded, not raised
# ---------------------------------------------------------------------------

def test_one_usable_pass_records_an_error_instead_of_raising(ref):
    """A prompt that hits EOS early has no profile. That is a fact about the
    prompt, and losing the rest of a batch over it would be worse."""
    row, matrix = score(FakeResult({2: [1.0, 2.0, 3.0]}), ref)

    assert "no profile" in row["error"]
    assert "at least 2 passes" in row["error"]
    assert "z_cv" not in row and "flag" not in row
    assert matrix is not None      # the matrix is still returned, for charting


# ---------------------------------------------------------------------------
# The gate half
# ---------------------------------------------------------------------------

class PlantedManifold:
    """Scores whatever it is given, so the gate columns are predictable."""

    def __init__(self, prob):
        self.prob = prob

    def predict(self, Z):
        n = np.asarray(Z).shape[0]
        return {
            "dist": np.full(n, 4.2),
            "ood_prob": np.full(n, self.prob),
            "is_ood": np.full(n, self.prob > 0.99),
        }


def test_without_a_manifold_the_ood_columns_are_absent(ref):
    row, _ = score(FakeResult({2: [1.0, 2.0, 3.0], 3: [1.0, 2.0, 3.0]}), ref)

    assert "ood_prob" not in row and "ood_dist" not in row and "is_ood" not in row
    assert row["text_ok"] is True      # the text half still ran


def test_with_a_manifold_the_ood_columns_are_filled(ref):
    result = FakeResult(
        {2: [1.0, 2.0, 3.0], 3: [1.0, 2.0, 3.0]},
        vectors={f"{REF_SIGNAL}|2": np.zeros(4), f"{REF_SIGNAL}|3": np.ones(4)},
    )
    row, _ = score(result, ref, PlantedManifold(0.5))

    assert row["ood_dist"] == pytest.approx(4.2)
    assert row["ood_prob"] == pytest.approx(0.5)
    assert row["is_ood"] is False


def test_degenerate_text_fails_the_text_half(ref):
    result = FakeResult({2: [1.0, 2.0, 3.0], 3: [1.0, 2.0, 3.0]}, text="yes no " * 40)
    row, _ = score(result, ref)

    assert row["text_ok"] is False
    assert "distinct-word ratio" in row["text_reasons"]


# ---------------------------------------------------------------------------
# Reference-vector selection
# ---------------------------------------------------------------------------

def test_the_reference_vector_is_the_highest_pass():
    """Numeric, not lexicographic: '10' must beat '8'."""
    vectors = {
        f"{REF_SIGNAL}|8": np.array([8.0]),
        f"{REF_SIGNAL}|10": np.array([10.0]),
        "hidden_states.causal_decoder.layer_3|12": np.array([99.0]),
    }
    assert last_reference_vector(vectors) == np.array([10.0])


def test_no_reference_vector_is_none():
    assert last_reference_vector({}) is None
    assert last_reference_vector({"embeddings.model.embed_tokens|2": np.zeros(3)}) is None


# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------

def test_every_key_a_score_row_produces_is_a_declared_column(ref):
    result = FakeResult(
        {2: [1.0, 2.0, 3.0], 3: [1.0, 2.0, 3.0]},
        vectors={f"{REF_SIGNAL}|2": np.zeros(4)},
    )
    row, _ = score(result, ref, PlantedManifold(0.5))
    assert set(row) <= set(SCORE_COLUMNS)
