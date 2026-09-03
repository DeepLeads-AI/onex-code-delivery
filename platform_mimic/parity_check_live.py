"""TEMPORARY — see TEMPORARY.md

Does the local capture reproduce the platform's numbers?

Runs the frozen STG prompts through :class:`capture.LocalCapture` and diffs the
resulting scalar rows against what the platform recorded for the same requests.
This is the load-bearing check for the whole folder: if the local hooks do not
reproduce the SDK's statistic, every score produced here describes a local model
rather than the platform's, and the prompt pack would not transfer.

Ad-hoc script, not pytest — it needs the model and either the frozen reference or
a live database, which the unit suite must never touch.

    python -m layer_profile parity                 # frozen CSV
    python -m layer_profile parity --live-db       # query the platform database
    python -m layer_profile parity --limit 3       # quick look
"""

import argparse
import os
import sys
import time
from typing import List, Optional

import pandas as pd

# Run as a bare script (`python platform_mimic/parity_check_live.py`) as well as
# through `python -m layer_profile parity`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __package__ in (None, ""):
    sys.path.insert(0, REPO_ROOT)
    __package__ = "platform_mimic"

from layer_profile.config import (                                   # noqa: E402
    HF_MODEL, MAX_NEW_TOKENS, PARITY_PASSES, PARITY_STD_RTOL, PARITY_STD_TOL,
)
from layer_profile.reference import reference_paths                  # noqa: E402

from .capture import LocalCapture                                    # noqa: E402
from .diff import diff_rows, summarise, write_parity_report          # noqa: E402

DEFAULT_REFERENCE_DIR = os.path.join(REPO_ROOT, "reference")
DEFAULT_OUT = os.path.join(REPO_ROOT, "runs", "parity")


def load_frozen(reference_dir: str, version: str):
    paths = reference_paths(reference_dir, version)
    if not os.path.exists(paths["prompts"]):
        raise SystemExit(
            f"Missing {paths['prompts']}.\n"
            "It is gitignored (client request payloads); re-create it with\n"
            f"  python -m layer_profile pull-reference --version {version}\n"
            "then check its sha256 against the provenance sidecar."
        )
    prompts = pd.read_csv(paths["prompts"])
    scalars = pd.read_csv(paths["scalars"], dtype={"forward_pass_index": str})
    return prompts, scalars


def load_live():
    from .stg import fetch_prompts, fetch_scalar_rows
    return fetch_prompts(), fetch_scalar_rows()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--version", default="v1")
    parser.add_argument("--reference-dir", default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--live-db", action="store_true",
                        help="query STG instead of reading the frozen CSVs")
    parser.add_argument("--limit", type=int, default=None,
                        help="only check the first N requests (smoke)")
    parser.add_argument("--model", default=HF_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--tol", type=float, default=PARITY_STD_TOL,
                        help="absolute floor of the two-sided tolerance")
    parser.add_argument("--rtol", type=float, default=PARITY_STD_RTOL,
                        help="relative term; bfloat16 carries ~4e-3 relative precision")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from layer_profile.reference import complete_request_ids

    prompts, platform_rows = load_live() if args.live_db else load_frozen(
        args.reference_dir, args.version
    )

    ids = complete_request_ids(platform_rows)
    if args.limit:
        ids = ids[: args.limit]
    prompt_by_id = dict(zip(prompts["request_id"], prompts["prompt"]))
    ids = [rid for rid in ids if isinstance(prompt_by_id.get(rid), str)]
    if not ids:
        raise SystemExit("No complete requests with a recoverable prompt.")

    if args.device != "cpu":
        print(f"WARNING: parity is only gated on cpu; device={args.device} may "
              "shift the low-order digits through a different attention kernel.")

    print(f"Checking {len(ids)} requests, passes {list(PARITY_PASSES)}, "
          f"tol |d| <= {args.tol} + {args.rtol} * |platform|")
    captured = []
    started = time.time()
    with LocalCapture(args.model, device=args.device,
                      max_new_tokens=args.max_new_tokens) as cap:
        for i, rid in enumerate(ids, 1):
            result = cap.run(prompt_by_id[rid], request_id=rid)
            captured.append(result.rows)
            print(f"  [{i}/{len(ids)}] {rid}  {result.n_prompt_tokens} prompt tokens, "
                  f"{result.n_generated_tokens} generated  ({time.time() - started:.0f}s)")

    local_rows = pd.concat(captured, ignore_index=True)
    platform_subset = platform_rows[platform_rows["request_id"].isin(ids)]

    diff = diff_rows(local_rows, platform_subset, passes=PARITY_PASSES)
    summary = summarise(diff, tol=args.tol, rtol=args.rtol)
    paths = write_parity_report(
        os.path.join(args.out, "parity"), diff, summary,
        title="Local capture vs platform capture",
        notes=[
            f"model: {args.model}",
            f"device: {args.device}",
            f"source: {'live STG' if args.live_db else f'frozen reference {args.version}'}",
            f"requests: {len(ids)}",
            "The platform sampled its generated tokens, so passes at or after the "
            "first divergence are excluded from the verdict; see the note above.",
        ],
    )

    print()
    print(f"requests             : {summary.n_requests_captured}/{len(ids)}")
    print(f"rows matched         : {summary.n_matched}")
    print(f"rows comparable      : {summary.n_comparable}")
    print(f"rows excluded (decode divergence): {summary.n_decode_diverged} "
          f"in {summary.n_requests_decode_diverged} requests")
    print(f"max |delta std|      : {summary.max_abs_d_std:.3e}")
    print(f"max relative delta   : {summary.max_rel_d_std:.3e}")
    print(f"rows over tolerance  : {summary.n_over_tol}")
    if summary.n_requests_without_fingerprint:
        print(f"requests without a decode fingerprint: "
              f"{summary.n_requests_without_fingerprint}")
    if summary.first_divergent_pass is not None:
        print(f"first divergent pass: {summary.first_divergent_pass}")
    print(f"report -> {paths['md']}")
    print()
    print("PASS" if summary.passed else "FAIL")
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
