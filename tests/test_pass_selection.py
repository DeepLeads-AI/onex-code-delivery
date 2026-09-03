"""Which forward pass represents a request.

The platform's own signal reader picks each request's MAX forward pass. For a
request that only ever logged the prefill that is the pass-1 vector — a
categorically different object (layer_14 std ~61.9 against ~0.66 for a generated
token), and one of them in a manifold's training set inflates the covariance
until nothing scores as out-of-domain. That is not hypothetical: it happened to
this reference's v1 manifold.
"""

import pandas as pd

from layer_profile.layer_profile import last_generation_pass


def rows(*pairs):
    return pd.DataFrame(
        [{"request_id": rid, "forward_pass_index": fpi} for rid, fpi in pairs]
    )


def test_the_prefill_is_never_selected():
    assert last_generation_pass(
        rows(("r1", "1"), ("r1", "8"), ("r2", "1"), ("r2", "3"))
    ) == {"r1": 8, "r2": 3}


def test_a_prefill_only_request_is_dropped_not_fallen_back_on():
    assert last_generation_pass(rows(("r_prefill_only", "1"))) == {}


def test_selection_is_numeric_not_lexicographic():
    """forward_pass_index is a varchar, where '10' sorts before '8'."""
    assert last_generation_pass(rows(("r1", "8"), ("r1", "10"))) == {"r1": 10}


def test_the_minimum_pass_is_configurable():
    assert last_generation_pass(rows(("r1", "1"), ("r1", "2")), min_pass=3) == {}
    assert last_generation_pass(rows(("r1", "1"), ("r1", "2")), min_pass=2) == {"r1": 2}


def test_nothing_in_nothing_out():
    assert last_generation_pass(pd.DataFrame()) == {}
    assert last_generation_pass(None) == {}


def test_unparseable_pass_indices_are_ignored():
    assert last_generation_pass(rows(("r1", "x"), ("r1", "4"))) == {"r1": 4}
