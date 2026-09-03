"""Provenance round trip and tamper detection. No DB: the reference directory is
planted on disk, so this exercises the hashing and the sidecar, not the pull.
"""

import json
import os

import numpy as np
import pytest

from layer_profile import reference as ref

# `printf 'layer_profile reference test\n' | shasum -a 256`
KNOWN_CONTENT = "layer_profile reference test\n"
KNOWN_SHA256 = "f28a8e27b94a2e8786a7f23ee035446f52e12ce72ca6168c9bdb258f58c56295"


@pytest.fixture
def ref_dir(tmp_path):
    """A reference directory with every file the sidecar expects to hash."""
    directory = str(tmp_path / "reference")
    os.makedirs(directory)
    paths = ref.reference_paths(directory, "v1")
    for key in ("prompts", "scalars"):
        with open(paths[key], "w") as fh:
            fh.write(KNOWN_CONTENT)
    np.savez_compressed(paths["profile_ref"], profile_mean=np.zeros(3))
    np.savez_compressed(paths["manifold"], mu=np.zeros(3))
    return directory


SUMMARY = {
    "window": ["2026-09-03 14:30:00+00", "2026-09-03 15:00:00+00"],
    "model_id": "model_K1WST_TPJHBETCDoYlhDkg",
    "n_requests": 100,
    "n_requests_with_signals": 49,
    "n_requests_complete": 44,
    "complete_request_ids": ["a", "b"],
}


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def test_sha256_file_matches_the_known_digest(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text(KNOWN_CONTENT)
    assert ref.sha256_file(str(path)) == KNOWN_SHA256


def test_git_sha_never_raises(tmp_path):
    """A missing git sha must not be able to fail a reference build."""
    assert isinstance(ref.git_sha(str(tmp_path)), str)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def test_reference_paths_are_versioned():
    paths = ref.reference_paths("/ref", "v2")
    assert paths["scalars"].endswith("qwen_stg_layer_scalars_v2.csv")
    assert paths["prompts"].endswith("qwen_stg_prompts_v2.csv")
    assert paths["manifold"].endswith("manifold_qwen_stg_layer14_v2.npz")


def test_the_prompts_file_is_the_only_untracked_one():
    assert ref.UNTRACKED_FILES == ["prompts"]
    assert set(ref.TRACKED_FILES) | set(ref.UNTRACKED_FILES) | {"provenance"} == set(
        ref.FILE_TEMPLATES
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_provenance_records_a_digest_for_every_file(ref_dir):
    ref.write_provenance(ref_dir, "v1", SUMMARY)
    with open(ref.reference_paths(ref_dir, "v1")["provenance"]) as fh:
        record = json.load(fh)

    assert set(record["files"]) == {"prompts", "scalars", "profile_ref", "manifold"}
    assert record["files"]["scalars"]["sha256"] == KNOWN_SHA256
    assert record["counts"]["n_requests_complete"] == 44
    assert record["model_id"] == SUMMARY["model_id"]
    assert record["complete_request_ids"] == ["a", "b"]


def test_provenance_keeps_the_sha_of_the_gitignored_prompts(ref_dir):
    """The prompts CSV is never committed, so its digest is the only record."""
    ref.write_provenance(ref_dir, "v1", SUMMARY)
    with open(ref.reference_paths(ref_dir, "v1")["provenance"]) as fh:
        record = json.load(fh)
    assert record["files"]["prompts"]["tracked"] is False
    assert record["files"]["prompts"]["sha256"] == KNOWN_SHA256
    assert record["files"]["scalars"]["tracked"] is True


def test_verify_passes_on_an_untouched_reference(ref_dir):
    ref.write_provenance(ref_dir, "v1", SUMMARY)
    result = ref.verify_provenance(ref_dir, "v1")
    assert len(result["ok"]) == 4
    assert result["changed"] == [] and result["missing"] == []


def test_verify_detects_a_tampered_file(ref_dir):
    ref.write_provenance(ref_dir, "v1", SUMMARY)
    paths = ref.reference_paths(ref_dir, "v1")
    with open(paths["scalars"], "a") as fh:
        fh.write("one extra row\n")

    result = ref.verify_provenance(ref_dir, "v1")
    assert result["changed"] == ["qwen_stg_layer_scalars_v1.csv"]
    assert len(result["ok"]) == 3          # the others still report, not just the first failure


def test_verify_detects_a_missing_file(ref_dir):
    ref.write_provenance(ref_dir, "v1", SUMMARY)
    os.remove(ref.reference_paths(ref_dir, "v1")["manifold"])

    result = ref.verify_provenance(ref_dir, "v1")
    assert result["missing"] == ["manifold_qwen_stg_layer14_v1.npz"]
    assert result["changed"] == []


def test_provenance_skips_files_that_were_not_produced(tmp_path):
    directory = str(tmp_path / "partial")
    os.makedirs(directory)
    with open(ref.reference_paths(directory, "v1")["scalars"], "w") as fh:
        fh.write(KNOWN_CONTENT)

    ref.write_provenance(directory, "v1", SUMMARY)
    result = ref.verify_provenance(directory, "v1")
    assert result["ok"] == ["qwen_stg_layer_scalars_v1.csv"]
