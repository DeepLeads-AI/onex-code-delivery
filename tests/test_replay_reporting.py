"""TEMPORARY — see ../TEMPORARY.md

Zero captured rows has three genuinely different causes, and conflating them
would misreport the state of the platform — which is the one thing a replay run
exists to establish. No network, no model: the send outcome is planted.
"""

import pandas as pd
import pytest

from platform_mimic.replay import SendOutcome, _zero_capture_explanation


def outcome(n_sent=5, n_ok=5):
    return SendOutcome(
        log=pd.DataFrame(), window_start="a", window_end="b",
        n_sent=n_sent, n_ok=n_ok,
    )


def test_a_dry_run_says_nothing_was_sent():
    message = _zero_capture_explanation(outcome(0, 0), dry_run=True)
    assert "Dry run" in message
    assert "not the platform" in message


def test_an_unreachable_endpoint_is_not_reported_as_a_capture_failure():
    """524 at the gateway says nothing about whether the platform records."""
    message = _zero_capture_explanation(outcome(n_sent=5, n_ok=0), dry_run=False)
    assert "did not answer" in message
    assert "says nothing about whether it captures" in message


def test_a_partial_answer_reports_both_problems():
    message = _zero_capture_explanation(outcome(n_sent=5, n_ok=3), dry_run=False)
    assert "3/5" in message and "Both problems" in message


def test_answered_but_unrecorded_is_the_condition_this_folder_works_around():
    message = _zero_capture_explanation(outcome(n_sent=5, n_ok=5), dry_run=False)
    assert "answered every request" in message
    assert "recorded none" in message


@pytest.mark.parametrize("n_sent,n_ok,dry", [(0, 0, True), (5, 0, False), (5, 3, False), (5, 5, False)])
def test_every_case_produces_a_non_empty_explanation(n_sent, n_ok, dry):
    assert _zero_capture_explanation(outcome(n_sent, n_ok), dry_run=dry).strip()
