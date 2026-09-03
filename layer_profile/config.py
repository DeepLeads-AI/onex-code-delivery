"""Constants for the layer-profile metric, the reference freeze and the replay.

Pure data — this module imports nothing, so importing it can never bind a
database or load a model.
"""

# ---------------------------------------------------------------------------
# Platform coordinates
# ---------------------------------------------------------------------------

# Qwen is registered on the STG observability database, not on the PROD-named
# one. Override per-run with $LAYER_PROFILE_DB_NAME, or per-deployment with
# DB_NAME in .env — see db.dsn().
DEFAULT_DB_NAME = "stg_onex_observability_core_backend"
QWEN_STG_MODEL_ID = "model_K1WST_TPJHBETCDoYlhDkg"

# The 100-prompt coding batch captured 2026-09-03. Padded by a minute either side.
STG_WINDOW_START = "2026-09-03 14:30:00+00"
STG_WINDOW_END = "2026-09-03 15:00:00+00"

# The public generation endpoint the replay posts at.
ENDPOINT_URL = "https://qwen.getonex.ai/generate"

# ---------------------------------------------------------------------------
# Model + capture
# ---------------------------------------------------------------------------

HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_LAYERS = 24                     # Qwen2.5-0.5B decoder layers
HIDDEN_DIM = 896                  # model hidden size
KEEP_DIMS = 768                   # the platform stores the first 768 dims only
MAX_PASSES = 8                    # 1 prefill + 7 generated tokens, as the platform records
MAX_NEW_TOKENS = 64               # generation length for the readable output text

EMBED_SIGNAL = "embeddings.model.embed_tokens"
LAYER_SIGNAL_PREFIX = "hidden_states.causal_decoder.layer_"

# Scalar columns the platform persists per (request, signal_type, forward_pass_index).
SCALAR_COLUMNS = ["std", "mean", "norm", "sparsity", "saturation"]

# ---------------------------------------------------------------------------
# Layer-profile metric
# ---------------------------------------------------------------------------

# Pass 1 is the chat-template start token: its hidden states sit on a constant
# std ~61.9 plateau across layers 3..20, which is an artefact of the template,
# not of the prompt. Only generated-token passes carry prompt-dependent signal.
METRIC_PASSES = tuple(range(2, MAX_PASSES + 1))     # 2..8

# Layer 23 dips sharply on every request (it feeds the final norm), so it is
# excluded from the cross-layer profile but still drawn on the charts.
METRIC_LAYERS = tuple(range(0, N_LAYERS - 1))       # 0..22

MIN_PASSES = 2                    # fewer usable passes than this -> no metric

# The flattening threshold, on the z of the cross-layer CV against the healthy
# band. The metric was designed as `z_cv <= -3 AND corr < 0.99`; both halves were
# guesses made before any prompt had been scored, and measurement moved them:
#
#   * `corr` never fell below 0.993 across all 270 scored requests (range
#     0.9932-0.9999), so `corr < 0.99` can never fire on this model — the layer
#     ramp is compressed by a flattening prompt, never reshaped. That half is
#     dropped; `corr` remains in LayerProfileStats as a diagnostic.
#   * No prompt reached -3 (best -2.37). -2.0 is where 0 of 45 healthy controls
#     flag and 2 of the 5 pack prompts do (7 of 225 grid candidates), i.e. the
#     tightest threshold that still separates the pack from the controls.
FLAT_Z_CV_MAX = -2.0

# ---------------------------------------------------------------------------
# Output-side gate (the outputs must still look normal)
# ---------------------------------------------------------------------------

REFERENCE_SIGNAL = "hidden_states.causal_decoder.layer_14"   # best-AUC layer
OOD_PROB_MAX = 0.99               # above this the request would light up the OOD panel

TEXT_MIN_CHARS = 40
TEXT_MAX_CHARS = 4000
TEXT_MAX_CHAR_REPEAT = 12         # longest run of one character
TEXT_MIN_DISTINCT_WORD_RATIO = 0.25   # distinct words / total words
TEXT_MAX_TOP_WORD_SHARE = 0.35    # share of the single most frequent word

# ---------------------------------------------------------------------------
# OOD manifold fit
# ---------------------------------------------------------------------------

# The platform's own OOD detector settings, so a manifold fitted here scores the
# way the delivered OOD panel does.
MANIFOLD_PCA_K = 32
MANIFOLD_SHRINKAGE = 1e-1
MANIFOLD_THRESHOLD_QUANTILE = 0.99

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

# Decoding is greedy, so this only fixes torch/numpy state; it is recorded
# because the pack's numbers are reproducible only under a fixed seed.
RANDOM_SEED = 20260903

# ---------------------------------------------------------------------------
# Parity + replay
# ---------------------------------------------------------------------------

# The platform SAMPLED its generated tokens, so local greedy decoding follows it
# only until the first sampled token differs — measured over the 44 complete
# requests, that is pass 2 for 12 of them and never (within 1..6) for 22. Passes
# after a divergence are not comparable at all: the KV cache holds different
# tokens. `diff.diff_rows` detects this per request from the embedding
# fingerprint, and parity is judged on the cumulatively-matched cells only.
PARITY_PASSES = tuple(range(1, 7))

# Two-sided tolerance, numpy.isclose semantics: |delta| <= atol + rtol * |platform|.
# An absolute-only bound cannot work here — the platform runs bfloat16 on CPU and
# the stored std spans 0.004 (embeddings) to 61.9 (the prefill plateau), so 5e-3
# is 125% of one signal and 0.008% of another. bfloat16 carries 8 mantissa bits
# (2^-8 = 3.9e-3 relative), and the measured relative error over 4,550
# comparable rows is q99 5.9e-3 / max 1.08e-2, growing with layer depth exactly
# as accumulated rounding should. rtol 1e-2 clears all 4,550 with headroom;
# atol 5e-3 remains the floor for near-zero signals.
PARITY_STD_TOL = 5e-3
PARITY_STD_RTOL = 1e-2

# Two embedding rows agreeing this closely means the same token was fed: the
# embedding is a table lookup with no accumulation, so a match is exact to
# floating-point noise (~1e-8 observed) and a different token is off by ~1e-3.
TOKEN_MATCH_TOL = 1e-6

REPLAY_REQUEST_TIMEOUT_S = 180
REPLAY_POLL_TIMEOUT_S = 300
REPLAY_POLL_INTERVAL_S = 15

# ---------------------------------------------------------------------------
# Frozen reference
# ---------------------------------------------------------------------------

REF_VERSION = "v1"

# ---------------------------------------------------------------------------
# Dashboard metric family
# ---------------------------------------------------------------------------

# Matches the dashboard's DISTRIBUTION_NUM_BANDS, so this metric bands the same
# way as the OOD and drift distributions beside it.
DISTRIBUTION_NUM_BANDS = 10
