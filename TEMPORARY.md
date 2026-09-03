# TEMPORARY: `platform_mimic/`

**Everything in `platform_mimic/` is duplicate infrastructure.** It replicates,
locally, the neural-signal capture your platform already performs — because the
public Qwen endpoint does not record the requests it answers. Delete it once it
does.

Nothing else in this repo depends on it. `layer_profile/` never imports it except
inside five CLI handlers, each marked `# TEMPORARY` at the import line.

## What it mimics, and why

| Here | The real thing | Why the duplicate exists |
| --- | --- | --- |
| `capture.py` — forward hooks on `model.model.layers[k]`, five scalars per pass | your SDK's neural-signal capture | The endpoint answers, but writes no `neural_signals` rows for what it answers. Without a local capture there is no way to score a new prompt at all. |
| `stg.py` | your own signal readers | Pulls the platform's capture of the 2026-09-03 Qwen batch and freezes it, so the healthy band is fitted against a fixed snapshot rather than a live query. |
| `endpoint.py`, `replay.py` | — | Sends the pack at the live endpoint and reports what the platform recorded for it. This is the step that makes the rest of this directory unnecessary. |
| `diff.py`, `parity_check_live.py` | — | The proof that the local capture reproduces your numbers. Without it, every score here would describe a local model rather than yours. |

## Does the local capture actually match the platform?

**Yes — PASS, 44 of 44 requests**, 4,550 comparable rows, 0 over tolerance, max
relative delta 1.08e-2. `python -m layer_profile parity` re-runs it.

Two things that check needs:

- **The tolerance is two-sided** (`|delta| <= 5e-3 + 1e-2 * |platform|`). An
  absolute bound cannot judge bfloat16 data whose stored std spans 0.004 to 61.9.
- **The platform sampled its generated tokens**, so local greedy decoding follows
  it only until the first differing token — pass 2 for 12 of the 44 requests,
  never within passes 1..6 for 22. Those passes are excluded via the embedding
  fingerprint, cumulatively: a token that re-matches later does not restore
  comparability, because the KV cache still holds different tokens.

## Deletion checklist

When `https://qwen.getonex.ai/generate` records the requests it answers:

1. Delete `platform_mimic/`.
2. Delete the five CLI handlers that import it — `pull-reference`, `capture`,
   `replay`, `diff`, `parity` — and their subparsers. They are grouped together
   in `layer_profile/cli.py` under a `TEMPORARY` heading.
3. Re-point the two `# TEMPORARY` imports inside the durable handlers:
   `cmd_score`'s `LocalCapture` (score requests from your own captured rows
   instead) and `cmd_fit_manifold`'s `fetch_vectors_at_last_generation_pass`.
4. Delete `tests/test_capture_recorder.py`, `tests/test_diff.py` and
   `tests/test_replay_reporting.py` — the three test modules whose docstrings
   start `TEMPORARY`.
5. Drop the `capture` extra from `pyproject.toml` and the `[capture]` block from
   `requirements.txt`. That removes torch and transformers: the metric itself
   needs neither.

What survives all of that: `layer_profile/`, `reference/`,
`prompt_packs/layer_flat_v1/` and the other seven test modules. The prompt pack
in particular states its own provenance in full and stays valid once this
directory is gone.

## Known platform quirks this directory works around

These are worth reading whether or not you keep the code — each one is a real
property of the captured data that anyone reimplementing this metric will hit.

- **Sequence positions are not reduced the same way for every signal type.**
  Decoder layers store position 0; `embeddings.model.embed_tokens` stores the
  **mean over positions**. Generation passes have T=1, where the two rules
  coincide, so the difference is only visible on the prefill pass — where
  position 0 is the constant `<|im_start|>` embedding and cannot vary by prompt,
  yet the stored value does. Solved against real prefill rows to ~1e-8; see
  `capture.POSITION_REDUCTIONS`.
- **Pass 1 carries no prompt-dependent signal.** It is the chat-template start
  token, and its decoder states sit on a constant std ~61.9 plateau across layers
  3..20 on every request. The metric therefore uses passes 2..8. A metric that
  averaged pass 1 in would be measuring the template.
- **Seeding an OOD manifold through `MAX(forward_pass_index)` poisons it.** For a
  request that only ever logged its prefill, that maximum *is* the pass-1 vector
  — `layer_14` std **61.9** and max |x| **1696**, against ~0.66 and ~11 for a
  generated token. Exactly one of the 49 Qwen requests is like that, and that
  single vector inflated the fitted covariance so far that **every** live request
  scored `ood_prob` ~0: a gate that cannot fail. `fit-manifold` here selects each
  request's last **generated**-token vector instead. Worth checking any manifold
  you have already seeded the other way.
- **Even corrected, a 48-vector `layer_14` manifold does not discriminate.**
  `PCA_K = 32` fitted on 48 samples spans nearly the whole training subspace, so
  new points project close to the centre: measured, planted gibberish scores like
  an in-domain coding prompt. "Not OOD" is a necessary condition here, not
  evidence, and the pack's README says so rather than presenting it as proof.
- **`embeddings.model.embed_tokens` is written twice per forward pass**, with
  byte-identical scalars (two hooks on the tied embedding module). Complete
  requests therefore have 208 rows, not 25 x 8 = 200. `db_scalars` deduplicates
  with `DISTINCT ON (request_id, signal_type, forward_pass_index)`; an aggregate
  that does not would double that layer's weight.
- **`forward_pass_index` is a varchar.** Comparisons and ordering must cast to
  int, because lexicographically `'10' < '8'`.
- **No `output_distribution` rows exist for this Qwen batch** — the platform
  captured the 25 hidden-state signal types only. That is why the output-side
  gate is the `layer_14` manifold plus text heuristics rather than logit-based.
- **The gateway times out on long prompts.** All five pack prompts returned HTTP
  524 (Cloudflare gateway timeout) at ~125 s on 2026-09-03, while a 10-character
  prompt returned 200 in 49 s. So the pack cannot currently be replayed from
  outside your network; `replay` still exits 0 and distinguishes "the endpoint
  did not answer" from "answered but recorded nothing", because those are
  different findings.
