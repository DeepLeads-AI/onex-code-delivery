# Layer-flattening prompt pack (v1)

5 prompts for **Qwen/Qwen2.5-0.5B-Instruct**, each paired with a matched healthy control, selected from 225 candidates across 5 families and levels [1, 2, 3, 4, 5].

## What it demonstrates

Each prompt's **per-decoder-layer activation std** — the layer profile — flattens relative to the healthy band, while the answer still reads normally and the platform's OOD metric stays quiet. The headline number is `z_cv`: the z-score of a request's cross-layer coefficient of variation against the band fitted from the platform's own capture (CV 0.7572 +/- 0.0130, n=44).

A **lower `z_cv` means a flatter profile**: the layers' activation stds are becoming more alike, which is the signature in question.

## How to read the evidence

- `pack.txt` — the prompts, one per line.
- `controls.txt` — the matched controls, same order. Each control is its candidate's own question with the padding removed, so a difference between the two is attributable to the padding and nothing else.
- `pack.csv` — per-prompt metrics, and the verbatim prompt text (`pack.txt` is the same text with newlines escaped so each prompt fits on one line). `z_cv` is the absolute score, `control_z_cv` its control's, `dz_cv` the difference — the controlled effect; `dz_slope` is the same comparison on the profile's slope. `ood_prob` and `text_ok` are the output-side gate; see the caveat on `ood_prob` below.
- `charts/<prompt_id>.png` — **left**: the layer profile against the healthy band, with the control overlaid; both sit inside the band, because the band is wide relative to the effect. **Middle**: the candidate minus its own control, per layer — this is where the effect is visible, and its shape is the finding: the padding raises early-layer std and lowers late-layer std, compressing the ramp. **Right**: cross-layer CV per generated token.
- `charts/pack_overview.png` — every pack prompt and control ranked by `z_cv`.

> **`ood_prob` is a weak signal here.** The manifold it comes from is fitted on 48 seed vectors reduced to 32 PCA dimensions, and measured against planted gibberish it does not discriminate. "Not OOD" is a necessary condition, not evidence; the matched control and the text checks carry the output-side argument. See [`TEMPORARY.md`](../../TEMPORARY.md).

## Which rule chose these prompts

**`paired`**. The **fallback** rule, used because the absolute signature was not prompt-inducible in this grid. Prompts are ranked by `dz_cv` — their `z_cv` minus their **own matched control's** — which isolates the padding's effect from where that particular question already sat in the band. This is a weaker claim than the absolute rule and should be presented as one. Threshold: `dz_cv <= -1.0`.

## What this run actually found — read this before presenting it

- The **strongest absolute score** reached was `z_cv = -2.37` against a selection threshold of `-3.0`.
- The **strongest controlled effect** (candidate minus its own control) was `-3.87` z.
- The profile's **correlation with the healthy shape never fell below 0.993**. The cross-layer ramp is largely architectural: padding compresses it, but does not reorder or destroy it. A literal "activation std identical across all decoder layers" was not produced by any prompt in this grid.

- Across all 270 scored requests the correlation **spanned only 0.0067** (0.9932-0.9999), so the `corr < 0.99` half of the absolute rule can never fire for this model: the ramp is compressed, never reshaped.

### What drives the effect: the kind of padding, not the amount

Mean controlled effect per family (candidate minus its **own matched control**, one-sample t-test against zero over every pair in the grid). More negative is more flattening.

| family | n | mean dz_cv | t | p | mean dz_slope | p |
| --- | --- | --- | --- | --- | --- | --- |
| `uniform_list` | 45 | -0.614 | -5.58 | 0.0000 | -1.005 | 0.0000 |
| `boilerplate` | 45 | -0.464 | -3.29 | 0.0020 | -0.829 | 0.0008 |
| `repeated_phrase` | 45 | -0.422 | -2.23 | 0.0310 | -0.512 | 0.0310 |
| `degenerate_code_loop` | 45 | -0.024 | -0.10 | 0.9177 | +0.343 | 0.2147 |
| `paraphrase_chain` | 45 | +1.293 | +6.53 | 0.0000 | -0.155 | 0.4287 |

Two of these deserve saying out loud:

- **`paraphrase_chain` goes the other way.** Restating the same request in different words makes the profile *sharper*, significantly so. Repeating the **meaning** is not what flattens the layer profile; repeating low-entropy **surface form** is.
- **`degenerate_code_loop` does nothing** — despite being the closest family to the coding batch the healthy band was fitted on.

### Padding volume does not matter

Comparing level 5 against level 1 **within each column** — same question, same filler, 33x the padding — over 45 paired columns: mean difference **-0.060** z, t -0.55, **p = 0.588**.

So there is no dose-response. A prompt is not made flatter by adding more of the same padding; what matters is whether the context is low-entropy surface repetition at all. Level is retained in the prompt ids as provenance, not as a dial.

### Is the pack cherry-picked?

It is the strongest handful out of 225 candidates, so yes — these sit in the tail, not at the typical effect size. Three things stop that from making them meaningless, and the census below shows what each removed:

1. **Only families with a significant mean effect are eligible.** The most negative prompt of a family centred on zero is noise, however exactly it reproduces. This is what keeps `paraphrase_chain` out.
2. **The prompt must look flat on its own chart** (`z_cv <= -1.0`), not merely flatter than its control. Without this the ranking was won by pairs whose *control* was anomalously sharp — true as arithmetic, misleading as a demo.
3. **One prompt per question.** Without this, 9 of the top 10 were the same two questions at different padding levels.

The effect is also **deterministic**: decoding is greedy with a fixed seed, so each of these prompts reproduces its numbers exactly. Quote a pack prompt to show the effect; quote the per-family table to show it is not a lucky draw.

### Selection census

| condition | candidates remaining |
| --- | --- |
| candidates | 225 |
| scored | 225 |
| flattened (absolute rule) | 0 |
| z_cv <= -3.0 | 0 |
| dz_cv <= -1.0 | 44 |
| text_ok | 225 |
| not_ood | 225 |
| control_not_flagged | 225 |
| selected_absolute | 0 |
| selected_paired | 5 |
| selected_paired_without_family_guard | 5 |
| selected_paired_without_any_guard | 44 |

## Provenance

- model: `Qwen/Qwen2.5-0.5B-Instruct`, device `cpu`, greedy decoding
- seed: `20260903`
- reference: version `v1`, healthy band fitted on 44 platform-captured requests (layers 0..22, passes 2..8)
- thresholds: selection-time thresholds `z_cv <= -3.0`, `corr < 0.99`; the shipped metric flags at `z_cv <= -2.0` (see the top-level README)
- run: run artefacts not shipped
- created: 2026-09-03T18:32:22.352836+00:00

Produced with `platform_mimic/` — a temporary local replication of the platform's capture, used because the public Qwen endpoint does not record the requests it answers. The repo's [`TEMPORARY.md`](../../TEMPORARY.md) explains the arrangement; this pack does not depend on it and stays valid once it is deleted.
