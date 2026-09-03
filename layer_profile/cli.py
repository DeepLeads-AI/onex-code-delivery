"""Command line for the layer-profile metric.

Every handler imports its heavy dependencies lazily, so ``--help`` never loads
torch and the pure-numpy commands never open a database connection.

Five commands need the temporary local capture in ``platform_mimic/`` and are
marked ``# TEMPORARY`` where they import it. Nothing else in this package refers
to that directory, so deleting it costs those five handlers and nothing more —
see TEMPORARY.md.
"""

import argparse
import os
from typing import List, Optional

from .config import (
    FLAT_Z_CV_MAX,
    HF_MODEL,
    MAX_NEW_TOKENS,
    PARITY_STD_RTOL,
    PARITY_STD_TOL,
    QWEN_STG_MODEL_ID,
    REF_VERSION,
    STG_WINDOW_END,
    STG_WINDOW_START,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_REFERENCE_DIR = os.path.join(REPO_ROOT, "reference")
DEFAULT_RUNS_DIR = os.path.join(REPO_ROOT, "runs")
DEFAULT_PACK_DIR = os.path.join(REPO_ROOT, "prompt_packs", "layer_flat_v1")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

#: Printed for each scored prompt, in this order.
TABLE_COLUMNS = [
    "prompt_id", "cv", "slope", "corr", "z_cv", "z_slope",
    "ood_prob", "is_ood", "text_ok", "n_generated_tokens", "flag",
]


def cmd_score(args: argparse.Namespace) -> int:
    """Run one prompt, or a whole pack, through the metric and the gate."""
    import pandas as pd

    from .config import REFERENCE_SIGNAL
    from .pack import load_pack
    from .reference import load_manifold, load_reference_band, reference_paths
    from .score import SCORE_COLUMNS, score_capture

    # TEMPORARY: the only place this package reaches into platform_mimic/. When
    # the platform's own endpoint records again, capture comes from there and
    # this import goes away with the directory.
    from platform_mimic.capture import LocalCapture

    if args.pack:
        prompts = load_pack(args.pack)
    elif args.prompt:
        prompts = {"prompt": args.prompt}
    else:
        raise SystemExit("Give either --prompt or --pack.")

    band = load_reference_band(args.reference_dir, args.version)
    manifold = None if args.no_gate else load_manifold(
        reference_paths(args.reference_dir, args.version)["manifold"]
    )

    rows = []
    with LocalCapture(
        args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        keep_vectors_for=(REFERENCE_SIGNAL,),
    ) as capture:
        for i, (prompt_id, prompt) in enumerate(prompts.items(), 1):
            print(f"[{i}/{len(prompts)}] {prompt_id}")
            result = capture.run(prompt, request_id=prompt_id)
            row, _ = score_capture(
                prompt_id, result, band, manifold, z_cv_max=args.z_cv_max,
            )
            rows.append(row)

    frame = pd.DataFrame(rows).reindex(columns=SCORE_COLUMNS)
    print()
    print(frame[[c for c in TABLE_COLUMNS if c in frame.columns]].to_string(index=False))

    out = args.out or os.path.join(DEFAULT_RUNS_DIR, "scores.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"\nscores -> {out}")
    return 0


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def cmd_metrics(args: argparse.Namespace) -> int:
    """The dashboard panel: layer-profile health over a window of the client DB."""
    from .dashboard_metrics import (
        get_layer_profile_distribution,
        get_layer_profile_summary,
    )
    from .reference import reference_paths

    reference_path = reference_paths(args.reference_dir, args.version)["profile_ref"]
    summary = get_layer_profile_summary(
        args.start, args.end, reference_path,
        model_id=args.model_id, z_cv_max=args.z_cv_max,
    )

    print(f"window                    : {args.start} .. {args.end}")
    print(f"model_id                  : {args.model_id}")
    print(f"sample_count              : {summary.sample_count}")
    print(f"n_incomplete              : {summary.n_incomplete}")
    print(f"flattened_rate            : {summary.flattened_rate:.4f}")
    print(f"healthy_rate              : {summary.healthy_rate:.4f}")
    print(f"mean_cv                   : {summary.mean_cv:.4f}")
    print(f"mean_z_cv                 : {summary.mean_z_cv:.4f}")
    print(f"mean_corr                 : {summary.mean_corr:.4f}")
    print(f"reference band            : {summary.reference_cv_mean:.4f} "
          f"+/- {summary.reference_cv_sd:.4f} (n={summary.reference_n_requests})")

    distribution = get_layer_profile_distribution(
        args.start, args.end, reference_path,
        model_id=args.model_id, z_cv_max=args.z_cv_max,
    )
    if distribution.bands:
        print("\nz_cv distribution")
        for band in distribution.bands:
            bar = "#" * band.count
            print(f"  [{band.lower:+.2f}, {band.upper:+.2f})  "
                  f"{band.count:>4}  {band.fraction:.3f}  {bar}")
    return 0


# ---------------------------------------------------------------------------
# reference lifecycle
# ---------------------------------------------------------------------------

def cmd_fit_reference(args: argparse.Namespace) -> int:
    """Fit the healthy layer-profile band from the frozen scalar rows."""
    import pandas as pd

    from .reference import (
        build_reference, complete_request_ids, reference_paths,
    )

    paths = reference_paths(args.reference_dir, args.version)
    scalars = pd.read_csv(paths["scalars"], dtype={"forward_pass_index": str})
    ids = complete_request_ids(scalars)
    band = build_reference(scalars, ids, version=args.version)
    band.save(paths["profile_ref"])

    print(f"requests (complete)       : {band.n_requests}")
    print(f"layers                    : {band.layers[0]}..{band.layers[-1]}")
    print(f"passes                    : {band.passes[0]}..{band.passes[-1]}")
    print(f"cross-layer CV            : {band.cv_mean:.4f} +/- {band.cv_sd:.4f}")
    print(f"CV band (mean +/- 3 sd)   : "
          f"[{band.cv_mean - 3 * band.cv_sd:.4f}, {band.cv_mean + 3 * band.cv_sd:.4f}]")
    print(f"slope                     : {band.slope_mean:.5f} +/- {band.slope_sd:.5f}")
    print(f"profile L0 -> L{band.layers[-1]}       : "
          f"{band.profile_mean[0]:.4f} -> {band.profile_mean[-1]:.4f}")
    print(f"band -> {paths['profile_ref']}")

    refresh_provenance(args.reference_dir, args.version)
    return 0


def cmd_fit_manifold(args: argparse.Namespace) -> int:
    """Fit the output-side OOD manifold on the last generated-token vectors."""
    from .reference import fit_manifold, reference_paths

    # TEMPORARY: the vectors are fetched from the platform database through
    # platform_mimic/. The fit itself is durable and takes the vectors as data.
    from platform_mimic.stg import fetch_vectors_at_last_generation_pass

    paths = reference_paths(args.reference_dir, args.version)
    ids, embeddings = fetch_vectors_at_last_generation_pass(
        args.start, args.end, args.model_id,
    )
    print(f"vectors (last generated pass): {len(ids)}")
    stats = fit_manifold(embeddings, paths["manifold"])
    for key in ("n_train", "orig_dim", "fit_dim", "shrinkage", "threshold"):
        print(f"{key:<26}: {stats[key]}")
    print(f"manifold -> {paths['manifold']}")

    refresh_provenance(args.reference_dir, args.version)
    return 0


def cmd_verify_reference(args: argparse.Namespace) -> int:
    """Re-hash every file the provenance sidecar names."""
    from .reference import verify_provenance

    result = verify_provenance(args.reference_dir, args.version)
    for name in result["ok"]:
        print(f"OK       {name}")
    for name in result["changed"]:
        print(f"CHANGED  {name}")
    for name in result["missing"]:
        print(f"MISSING  {name}")
    if result["changed"] or result["missing"]:
        print("\nFAIL: reference does not match its provenance sidecar")
        return 1
    print(f"\nPASS: {len(result['ok'])} files match")
    return 0


def cmd_refresh_provenance(args: argparse.Namespace) -> int:
    """Re-hash the reference files into the sidecar, keeping its other fields."""
    from .reference import reference_paths

    extra = {}
    if args.source_created_utc:
        extra["source_created_utc"] = args.source_created_utc
    path = refresh_provenance(args.reference_dir, args.version, extra=extra or None)
    if path is None:
        print(f"No sidecar at {reference_paths(args.reference_dir, args.version)['provenance']}")
        return 1
    print(f"provenance -> {path}")
    return 0


def refresh_provenance(
    reference_dir: str,
    version: str,
    extra: Optional[dict] = None,
) -> Optional[str]:
    """Re-hash after a step adds a file, keeping the sidecar the whole manifest.

    Returns the sidecar path, or ``None`` when there is no sidecar to refresh —
    a reference built from scratch has nothing to carry forward yet.
    """
    import json

    from .reference import reference_paths, write_provenance

    paths = reference_paths(reference_dir, version)
    if not os.path.exists(paths["provenance"]):
        return None
    with open(paths["provenance"]) as fh:
        previous = json.load(fh)
    summary = {
        "window": previous.get("window"),
        "model_id": previous.get("model_id"),
        "complete_request_ids": previous.get("complete_request_ids", []),
        **previous.get("counts", {}),
    }
    carried = {k: previous[k] for k in ("source_created_utc",) if k in previous}
    carried.update(extra or {})
    return write_provenance(reference_dir, version, summary, extra=carried or None)


# ---------------------------------------------------------------------------
# TEMPORARY: the local capture mimic — see TEMPORARY.md
# ---------------------------------------------------------------------------

def cmd_pull_reference(args: argparse.Namespace) -> int:
    """TEMPORARY. Freeze the platform's own capture of the Qwen batch."""
    import json

    from .reference import write_provenance
    from platform_mimic.stg import pull_and_freeze

    summary = pull_and_freeze(args.reference_dir, args.version)
    for key in (
        "n_requests", "n_requests_with_signals", "n_requests_complete",
        "n_signal_types", "n_scalar_rows",
    ):
        print(f"{key:<26}: {summary[key]}")
    provenance = write_provenance(args.reference_dir, args.version, summary)
    print(f"provenance -> {provenance}")
    if args.verbose:
        print(json.dumps(summary["paths"], indent=2))
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    """TEMPORARY. Run one prompt locally and write platform-shaped rows."""
    from .config import REFERENCE_SIGNAL
    from platform_mimic.capture import LocalCapture, save_rows, write_capture_meta

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    with LocalCapture(
        args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        keep_vectors_for=(REFERENCE_SIGNAL,),
    ) as cap:
        result = cap.run(args.prompt, request_id=args.request_id)

    rows_path = save_rows(result.rows, os.path.join(out_dir, "rows.parquet"))
    write_capture_meta(os.path.join(out_dir, "capture_meta.json"), result)

    print(f"prompt tokens      : {result.n_prompt_tokens}")
    print(f"generated tokens   : {result.n_generated_tokens}")
    print(f"signal types       : {len(result.signal_types())}")
    print(f"rows               : {len(result.rows)}")
    print(f"passes             : {sorted(set(result.rows['forward_pass_index']), key=int)}")
    print(f"rows -> {rows_path}")
    print("\n--- generated text ---")
    print(result.text)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """TEMPORARY. Send the pack at the live endpoint and diff what was recorded.

    Always returns 0: "the platform captured nothing" is the finding this whole
    arrangement exists because of, not an error condition.
    """
    from datetime import datetime, timezone

    from platform_mimic.replay import run_replay

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or os.path.join(DEFAULT_RUNS_DIR, f"{stamp}-replay")
    result = run_replay(
        args.pack, out_dir,
        dry_run=args.dry_run, model=args.model, device=args.device,
    )
    print(f"\ncaptured {result['n_captured']}/{result['n_prompts']}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    """TEMPORARY. Diff a local rows file against a platform scalar file."""
    import pandas as pd

    from .config import PARITY_PASSES
    from platform_mimic.capture import load_rows
    from platform_mimic.diff import diff_rows, summarise, write_parity_report

    local = (load_rows(args.local) if args.local.endswith(".parquet")
             else pd.read_csv(args.local, dtype={"forward_pass_index": str}))
    platform = (load_rows(args.platform) if args.platform.endswith(".parquet")
                else pd.read_csv(args.platform, dtype={"forward_pass_index": str}))

    diff = diff_rows(local, platform, passes=PARITY_PASSES)
    summary = summarise(diff, tol=args.tol, rtol=args.rtol)
    paths = write_parity_report(args.out, diff, summary, title="Diff")
    print(f"rows comparable    : {summary.n_comparable}")
    print(f"max |delta std|    : {summary.max_abs_d_std:.3e}")
    print(f"max relative delta : {summary.max_rel_d_std:.3e}")
    print(f"report -> {paths['md']}")
    print("PASS" if summary.passed else "FAIL")
    return 0 if summary.passed else 1


def cmd_parity(args: argparse.Namespace) -> int:
    """TEMPORARY. Local capture of the frozen prompts vs what the platform recorded."""
    from platform_mimic.parity_check_live import main as parity_main

    argv = ["--version", args.version, "--reference-dir", args.reference_dir]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.live_db:
        argv += ["--live-db"]
    if args.device:
        argv += ["--device", args.device]
    return parity_main(argv)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m layer_profile",
        description="The layer-profile metric: score prompts against the healthy "
                    "band, and report window health over the platform database.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("score", help="score a prompt or a whole pack")
    _add_reference_args(p)
    p.add_argument("--prompt", default=None, help="a single prompt to score")
    p.add_argument("--pack", nargs="?", const=DEFAULT_PACK_DIR, default=None,
                   help=f"a pack directory (default {DEFAULT_PACK_DIR})")
    p.add_argument("--out", default=None, help="scores CSV (default runs/scores.csv)")
    p.add_argument("--z-cv-max", type=float, default=FLAT_Z_CV_MAX)
    p.add_argument("--no-gate", action="store_true",
                   help="skip the OOD half of the gate (no manifold)")
    p.add_argument("--model", default=HF_MODEL)
    p.add_argument("--device", default=None,
                   help="default cpu; parity is only gated on cpu")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("metrics", help="layer-profile health over a database window")
    _add_reference_args(p)
    p.add_argument("--start", required=True, help="window start, e.g. '2026-09-03 14:30:00+00'")
    p.add_argument("--end", required=True, help="window end")
    p.add_argument("--model-id", default=None, help="filter to one deployed model")
    p.add_argument("--z-cv-max", type=float, default=FLAT_Z_CV_MAX)
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("fit-reference", help="fit the healthy layer-profile band")
    _add_reference_args(p)
    p.set_defaults(func=cmd_fit_reference)

    p = sub.add_parser("fit-manifold", help="fit the output-side OOD manifold")
    _add_reference_args(p)
    p.add_argument("--start", default=STG_WINDOW_START)
    p.add_argument("--end", default=STG_WINDOW_END)
    p.add_argument("--model-id", default=QWEN_STG_MODEL_ID)
    p.set_defaults(func=cmd_fit_manifold)

    p = sub.add_parser("verify-reference", help="re-hash the frozen reference files")
    _add_reference_args(p)
    p.set_defaults(func=cmd_verify_reference)

    p = sub.add_parser("refresh-provenance", help="re-hash the reference into its sidecar")
    _add_reference_args(p)
    p.add_argument("--source-created-utc", default=None,
                   help="when the underlying snapshot was taken")
    p.set_defaults(func=cmd_refresh_provenance)

    # --- TEMPORARY: these five need platform_mimic/ — see TEMPORARY.md ---

    p = sub.add_parser("pull-reference",
                       help="TEMPORARY: freeze the platform snapshot to reference/")
    _add_reference_args(p)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_pull_reference)

    p = sub.add_parser("capture", help="TEMPORARY: capture one prompt locally")
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", default=os.path.join(DEFAULT_RUNS_DIR, "capture"))
    p.add_argument("--model", default=HF_MODEL)
    p.add_argument("--device", default=None,
                   help="default cpu; parity is only gated on cpu")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--request-id", default="local")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("replay", help="TEMPORARY: send the pack at the live endpoint")
    p.add_argument("--pack", default=DEFAULT_PACK_DIR)
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="skip the endpoint; produces the empty-platform report")
    p.add_argument("--model", default=HF_MODEL)
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("diff", help="TEMPORARY: diff a local rows file against platform rows")
    p.add_argument("--local", required=True, help=".parquet or .csv")
    p.add_argument("--platform", required=True, help=".parquet or .csv")
    p.add_argument("--out", default=os.path.join(DEFAULT_RUNS_DIR, "diff", "diff"))
    p.add_argument("--tol", type=float, default=PARITY_STD_TOL)
    p.add_argument("--rtol", type=float, default=PARITY_STD_RTOL)
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("parity",
                       help="TEMPORARY: local capture vs platform capture on the same prompts")
    _add_reference_args(p)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--live-db", action="store_true")
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_parity)

    return parser


def _add_reference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--version", default=REF_VERSION)
    parser.add_argument("--reference-dir", default=DEFAULT_REFERENCE_DIR)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
