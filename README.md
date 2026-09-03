# OneX code delivery — layer-profile metric and prompt pack

Delivered to you for your use. This is not an open-source release, and there is
no licence file: the code is yours to run, change and fold into your platform.

## 1. What is in this repo

Three things, kept apart on purpose.

| | What it is | Keep it? |
| --- | --- | --- |
| **`prompt_packs/layer_flat_v1/`** | The evidence. Five fixed prompts, each with a matched control, and the measured numbers for all ten. No code. | Yes — it is the deliverable. |
| **`layer_profile/`** | The metric. A new monitoring panel: for a window of requests, how many show the per-layer activation std collapsing, and by how much. Reads scalar columns your platform already stores. | Yes — this is what goes on the platform. |
| **`platform_mimic/`** | A local stand-in for your capture, because your public endpoint answers requests without recording them. | **No — delete it** once the endpoint records. See [TEMPORARY.md](TEMPORARY.md). |

## 2. The 24 layers, in plain words

Qwen2.5-0.5B is **24 decoder layers in a row**. Each layer takes a vector of 896
numbers, does its work, and hands a new vector of 896 numbers to the next one.

**Activation std** is one number describing each of those vectors: how spread out
its 896 numbers are. A large std means the layer is producing a wide range of
values; a small std means they are bunched together.

On a healthy request the std **ramps up** as you go deeper — about **0.18** at
layer 0, about **2.43** at layer 22. That 23-number curve is what we call the
request's **layer profile**, and its rising shape is the normal state of this
model. Layer 23 is left out of the profile: it feeds the model's final
normalisation step and dips sharply on *every* request, healthy or not, so
including it would add noise and no signal.

The deterioration you asked about is that ramp **flattening** — the layers
starting to agree with each other — while the answer the model gives still reads
perfectly normally.

## 3. Where the platform captures

Your SDK already records, per request:

- **8 forward passes.** Pass 1 is the *prefill* (the whole prompt read at once);
  passes 2–8 are one generated token each.
- **25 signal types per pass** — the token embeddings plus each of the 24 decoder
  layers.
- For each of those, **the first 768 of the 896 dimensions at sequence
  position 0**, and five scalars of that vector: `std`, `mean`, `norm`,
  `sparsity`, `saturation`.

The metric here uses the `std` column, over **layers 0–22** and **passes 2–8**.

Pass 1 is excluded deliberately. At position 0 the prefill is always the same
chat-template start token, so its hidden states are the same on every request —
a flat std ~61.9 plateau across layers 3–20 that says nothing about the prompt. A
metric that averaged pass 1 in would be measuring the template.

**No new capture, no schema change, no extra storage is needed.** Everything the
metric reads is already in your `neural_signals` table.

## 4. Why a new metric was needed

The drift and out-of-distribution panels already delivered compare **one layer
across many requests**. Your question is the other axis: **all layers within one
request**. A request can sit comfortably inside every existing panel and still
have a collapsed layer profile — and the pack prompts are exactly that case.

Measured on this pack, against the delivered panels:

- pack versus controls, MMD **0.0000**;
- **no** pack request flagged by the existing panels;
- pack distances **below the healthy median**.

The existing panels are quiet on the pack. That is not a fault in them; they are
answering a different question. Hence a new one.

## 5. The metric

For one request:

1. Build the **layer profile**: the `std` for layers 0–22, averaged over passes
   2–8.
2. **CV** = `std(profile) / mean(profile)` — the cross-layer coefficient of
   variation. It measures how *unequal* the layers are. A perfect ramp has a high
   CV; a flattened profile has a low one. Using CV rather than raw spread makes it
   scale-invariant: a request whose activations are uniformly smaller is not
   flattened, one whose layers all agree is.
3. **z_cv** = how many standard deviations that CV sits below the healthy band.
   The band is **CV 0.7572 ± 0.0130**, fitted on **44** of your own captured
   requests.
4. **Flag when `z_cv <= -2.0`.**

### Where -2.0 comes from — read this before quoting the threshold

The metric was designed as `z_cv <= -3 AND corr < 0.99`. Both halves were guesses
made before any prompt had been scored, and measurement moved both:

- **The `corr` half was dropped.** Across all 270 scored requests the profile's
  correlation with the healthy shape spanned **0.9932–0.9999** — it never came
  close to 0.99, so that condition can never fire on this model. The ramp gets
  *compressed* by a flattening prompt, never reshaped. `corr` is still reported
  as a diagnostic; it is no longer part of the verdict.
- **-3 was never reached.** The best absolute score in the whole grid is
  **-2.37**.

**-2.0** is where **0 of 45** healthy controls flag and **2 of the 5** pack
prompts do (`repeated_phrase-L5-04` at -2.30, `boilerplate-L5-07` at -2.37).
Across the full 225-candidate grid, 7 reach it. The five pack prompts score
-1.90, -1.55, -2.30, -2.37 and -1.30.

So: at the shipped threshold the metric flags **two of five** pack prompts and
**none** of the 45 matched controls. All five show the controlled effect — each
is flatter than its own matched control — but only two clear the absolute bar.
Stating it the other way round would overclaim.

### What the threshold does on your own traffic

The 45 controls above are *synthetic* — the pack's questions with the padding
removed. Run the panel over your own captured batch and the base rate is not
zero. Over the reference window (2026-09-03 14:30–15:00 UTC, model_id
`model_K1WST_TPJHBETCDoYlhDkg`):

```
sample_count     48
n_incomplete      0
flattened_rate    0.0417     # 2 of 48
mean_z_cv         0.0501
```

**Two of your 48 real requests flag**, at `z_cv` **-2.37** and **-2.04** — about
a **4% false-positive rate** if you treat every flag as a problem. That is the
expected shape of a 2-sigma threshold on a 48-request sample and it is not a
defect, but it does mean the metric is a **rate to watch, not a per-request
alarm**. Use `flattened_rate` moving against its own baseline; a single flagged
request means very little on its own.

Note also that `sample_count` is **48**, not the 44 the band was fitted on. Those
are different sets on purpose: the band is fitted only on requests with a
*complete* layer x pass grid, so partial generations cannot drag it toward
early-generation values, while the panel scores every request with at least 2
usable passes — a short answer is still a request you want counted. Requests with
fewer than 2 usable passes are reported separately as `n_incomplete`.

## 6. Your asks, and what is delivered against each

| You asked for | What is delivered | Honest status |
| --- | --- | --- |
| "activation std identical across all decoder layers" | A measurable **compression** of the ramp — best `z_cv` **-2.37** — but not identity. `corr` with the healthy shape never fell below **0.993**. | **Not achieved as stated.** The rising ramp is architectural: prompts compress it, nothing in this grid destroys it. |
| "prompts that trigger it while the output looks normal" | **5 prompts with matched controls.** Text checks pass, the OOD gate is quiet, and the delivered drift/OOD panels do not flag any of them. Paired effect significant for `uniform_list` (-0.61, p<1e-4), `boilerplate` (-0.46, p=0.002) and `repeated_phrase` (-0.42, p=0.03). | **Delivered**, with two findings you should know: `paraphrase_chain` goes the *other* way (+1.29 — restating a request in different words makes the profile *sharper*), and `degenerate_code_loop` does nothing (p=0.92). Padding **volume** is irrelevant: level 5 vs level 1 is p=0.59. What matters is low-entropy surface repetition, not how much of it. |
| "a metric on the platform" | `layer_profile/dashboard_metrics.py` — window in, summary and distribution out, shaped like your existing metric functions. Reads the `std` scalars you already store. | **Delivered. No platform change needed** — no new capture, no schema change. |
| "show it live" | `python -m layer_profile replay` sends the pack at the endpoint and diffs what was recorded. | **Blocked at your gateway.** All five pack prompts get Cloudflare **524** at ~125 s; a 10-character prompt answers **200** in 49 s. The prompts are long, the gateway is not waiting. Run the replay from inside your network. |

## 7. The temporary capture infrastructure

`platform_mimic/` exists only because the public endpoint answers requests
without recording them, so there is no other way to score a *new* prompt. It
reproduces your capture faithfully — parity **PASS, 44/44 requests**, 4,550
comparable rows, 0 over tolerance — and it is meant to be deleted. The deletion
checklist, and the platform quirks it works around (several worth reading
regardless), are in [TEMPORARY.md](TEMPORARY.md).

## 8. How to run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[capture,db,replay,test]"

# the unit suite: no database, no model, no network, a couple of seconds
pytest -q

# the frozen reference still matches its recorded hashes
python -m layer_profile verify-reference --version v1

# score one prompt, or the whole pack
python -m layer_profile score --prompt "Explain what a hash map is."
python -m layer_profile score --pack prompt_packs/layer_flat_v1 --device cpu

# the panel, over a window of your database (needs .env — see .env.example)
python -m layer_profile metrics \
    --start "2026-09-03 14:30:00+00" --end "2026-09-03 15:00:00+00" \
    --model-id model_K1WST_TPJHBETCDoYlhDkg

# send the pack at the live endpoint and diff what was recorded
python -m layer_profile replay --pack prompt_packs/layer_flat_v1 --dry-run
python -m layer_profile replay --pack prompt_packs/layer_flat_v1
```

The first `score` downloads the model (~1 GB from Hugging Face). Scoring runs on
CPU with greedy decoding, so the numbers reproduce exactly.

Installing without the extras (`pip install -e .`) gives you the metric alone —
no torch, no database driver. `python -m layer_profile --help` and
`verify-reference` work either way; `pytest` needs `[replay]` as well, because
one test module exercises the endpoint reporting.

`score` prints `RuntimeWarning: ... encountered in matmul` from the manifold's
PCA step on macOS/Apple Accelerate. It is benign and was checked rather than
assumed against the shipped manifold: `mu`, `inv_cov` and the PCA components
contain no NaN, the five pack requests score finite probabilities (4e-8 to
1.3e-4), and planted noise scores 1.0 and is correctly flagged. It is not
silenced, because the detector is vendored from your platform byte for byte and
editing it would let the two copies drift.

## 9. Provenance

- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`, CPU, greedy decoding, seed `20260903`.
- **Reference `v1`**: the healthy band fitted on **44** platform-captured
  requests, window **2026-09-03 14:30–15:00 UTC**, model_id
  `model_K1WST_TPJHBETCDoYlhDkg`, database
  `stg_onex_observability_core_backend`. Layers 0–22, passes 2–8.
- **The check is the sha256s**, not a git commit. `reference/*.provenance.json`
  records a digest for every reference file, and `verify-reference` re-hashes
  them. It covers `qwen_stg_prompts_v1.csv` too, which is **not committed** — it
  holds request payloads — so its digest there is the only record of it.

  **On a fresh clone, `verify-reference` therefore prints `MISSING
  qwen_stg_prompts_v1.csv`, says `FAIL`, and exits 1.** That is expected, not a
  broken delivery: the sidecar names a file the repo deliberately does not carry.
  What matters is that the other three read `OK`. If you restore the prompts CSV
  from your own database (`pull-reference`), check it against the digest in the
  sidecar and the command will pass with all four.
