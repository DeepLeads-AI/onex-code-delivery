"""TEMPORARY — see TEMPORARY.md

Local replication of the platform's per-layer neural-signal capture.

The platform SDK stores, for every forward pass and every instrumented module,
the **first 768 of the 896 hidden dims at sequence position 0** of that module's
output, plus five scalars derived from that vector. This module reproduces that
exactly (see :func:`stats_768` — the scalar definitions were solved against real
STG rows, not guessed) so that a locally captured request and a platform-captured
one are directly comparable row for row.

:class:`SignalRecorder` and :func:`stats_768` are torch-free and are what the
unit tests exercise; :class:`LocalCapture` imports torch/transformers lazily so
that importing this module stays cheap.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from layer_profile.config import (
    EMBED_SIGNAL,
    HF_MODEL,
    KEEP_DIMS,
    LAYER_SIGNAL_PREFIX,
    MAX_NEW_TOKENS,
    MAX_PASSES,
    N_LAYERS,
    RANDOM_SEED,
    SCALAR_COLUMNS,
)

#: Columns the platform persists per scalar row, in platform order.
STG_ROW_COLUMNS = ["request_id", "signal_type", "forward_pass_index"] + SCALAR_COLUMNS

#: Extra columns only the local capture can supply. They are never compared
#: against the platform — :mod:`diff` drops them before diffing.
LOCAL_EXTRA_COLUMNS = ["n_positions", "dim"]

ROW_COLUMNS = STG_ROW_COLUMNS + LOCAL_EXTRA_COLUMNS

#: |x| above this counts as saturated. Solved exactly against 12 STG rows
#: (zero error); 0.9, 0.92, 0.98, 0.99 and 1.0 all mismatch.
SATURATION_THRESHOLD = 0.95

#: How the platform collapses a module's sequence positions into the one vector
#: it stores. **These are not the same for every signal type.** Solved against
#: STG prefill rows (T = 44..50), where per request:
#:
#:   * every decoder layer matches position 0 exactly (layer_10 to 0.0e+00), and
#:   * ``embeddings.model.embed_tokens`` matches the mean over positions to
#:     ~1e-8 while position 0 is off by 6e-3 — position 0 is the constant
#:     ``<|im_start|>`` embedding and cannot vary by prompt, but the stored value
#:     does.
#:
#: Generation passes have T = 1, where the two rules coincide, which is why this
#: asymmetry only shows up on the prefill pass.
POSITION_REDUCTIONS = {EMBED_SIGNAL: "mean"}
DEFAULT_REDUCTION = "first"


def reduction_for(signal_type: str) -> str:
    """Which sequence-position reduction the platform applies to this signal."""
    return POSITION_REDUCTIONS.get(signal_type, DEFAULT_REDUCTION)


def layer_signal_type(index: int) -> str:
    """``0 -> 'hidden_states.causal_decoder.layer_0'``."""
    return f"{LAYER_SIGNAL_PREFIX}{index}"


# ---------------------------------------------------------------------------
# The five scalars
# ---------------------------------------------------------------------------

def stats_768(vector: np.ndarray, keep_dims: int = KEEP_DIMS) -> Dict[str, float]:
    """The platform's five scalars over the first ``keep_dims`` dims of a vector.

    Definitions verified against stored STG rows whose ``embedding_vector`` was
    still present, so these are the platform's, not a reimplementation guess:

    ==========  ==========================================
    std         population std (ddof=0)
    mean        arithmetic mean
    norm        L2 norm
    sparsity    fraction of entries exactly equal to zero
    saturation  fraction of entries with ``|x| > 0.95``
    ==========  ==========================================
    """
    v = np.asarray(vector, dtype=np.float64).ravel()[:keep_dims]
    if v.size == 0:
        raise ValueError("Cannot compute signal statistics of an empty vector")
    return {
        "std": float(np.std(v)),
        "mean": float(np.mean(v)),
        "norm": float(np.linalg.norm(v)),
        "sparsity": float(np.mean(v == 0.0)),
        "saturation": float(np.mean(np.abs(v) > SATURATION_THRESHOLD)),
    }


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

@dataclass
class SignalRecorder:
    """Accumulates platform-shaped scalar rows across forward passes.

    Torch-free by design: hooks hand it plain arrays, so the whole row-building
    path is unit-testable without loading a model. Passes are numbered from 1 to
    match the platform's ``forward_pass_index`` varchar, and anything beyond
    ``max_passes`` is dropped so a 64-token generation still yields exactly the
    8 passes the platform records.
    """

    keep_dims: int = KEEP_DIMS
    max_passes: int = MAX_PASSES
    keep_vectors_for: Sequence[str] = ()

    pass_index: int = 0
    rows: List[Dict[str, Any]] = field(default_factory=list)
    vectors: Dict[str, np.ndarray] = field(default_factory=dict)

    def begin_pass(self) -> int:
        """Called once per model forward; returns the new 1-based pass index."""
        self.pass_index += 1
        return self.pass_index

    @property
    def active(self) -> bool:
        """False once the pass budget is spent, so hooks can return immediately."""
        return 1 <= self.pass_index <= self.max_passes

    def record(
        self,
        signal_type: str,
        hidden: np.ndarray,
        reduction: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Record one module output. ``hidden`` is (B, T, D), (T, D) or (D,).

        Batch 0 always; the sequence positions are collapsed by the rule in
        :data:`POSITION_REDUCTIONS` — position 0 for decoder layers, the mean
        over positions for the embedding signal. Using the wrong one silently
        produces numbers that cannot be compared with the platform's.
        """
        if not self.active:
            return None
        arr = np.asarray(hidden, dtype=np.float64)
        while arr.ndim > 2:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr[None, :]
        n_positions = int(arr.shape[0])

        how = reduction or reduction_for(signal_type)
        if how == "mean":
            reduced = arr.mean(axis=0)
        elif how == "first":
            reduced = arr[0]
        else:
            raise ValueError(f"Unknown position reduction: {how!r}")
        vector = reduced[: self.keep_dims]

        row = {
            "request_id": None,
            "signal_type": signal_type,
            # varchar in the platform schema; kept as a string so the frames join
            "forward_pass_index": str(self.pass_index),
            **stats_768(vector, keep_dims=self.keep_dims),
            "n_positions": n_positions,
            "dim": int(reduced.size),
        }
        self.rows.append(row)
        if signal_type in self.keep_vectors_for:
            self.vectors[f"{signal_type}|{self.pass_index}"] = vector.astype(np.float64)
        return row

    def reset(self) -> None:
        self.pass_index = 0
        self.rows = []
        self.vectors = {}


def rows_to_df(rows: Iterable[Dict[str, Any]], request_id: Optional[str] = None) -> pd.DataFrame:
    """Materialise recorder rows as a DataFrame with the fixed column set."""
    rows = list(rows)
    if not rows:
        return pd.DataFrame(columns=ROW_COLUMNS)
    df = pd.DataFrame(rows)
    if request_id is not None:
        df["request_id"] = request_id
    for col in ROW_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[ROW_COLUMNS]


def save_rows(df: pd.DataFrame, path: str) -> str:
    """Persist a capture frame. Parquet keeps the dtypes; the caller picks the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_rows(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Local model capture
# ---------------------------------------------------------------------------

@dataclass
class CaptureResult:
    """One locally captured request."""
    prompt: str
    text: str
    rows: pd.DataFrame
    vectors: Dict[str, np.ndarray]
    n_prompt_tokens: int
    n_generated_tokens: int
    device: str

    def signal_types(self) -> List[str]:
        return sorted(self.rows["signal_type"].unique())


class LocalCapture:
    """Runs Qwen locally with the platform's instrumentation replicated by hooks.

    Deliberately *not* ``output_hidden_states=True``: that tuple's last entry is
    the post-final-norm hidden state, which is not what the platform stores for
    ``layer_23``. Hooks on ``model.model.layers[k]`` read the pre-norm output the
    platform actually records.
    """

    def __init__(
        self,
        model_name: str = HF_MODEL,
        *,
        device: Optional[str] = None,
        max_passes: int = MAX_PASSES,
        max_new_tokens: int = MAX_NEW_TOKENS,
        keep_dims: int = KEEP_DIMS,
        keep_vectors_for: Sequence[str] = (),
        seed: int = RANDOM_SEED,
    ):
        self.model_name = model_name
        self.device = device
        self.max_passes = max_passes
        self.max_new_tokens = max_new_tokens
        self.keep_dims = keep_dims
        self.keep_vectors_for = tuple(keep_vectors_for)
        self.seed = seed
        self.model = None
        self.tokenizer = None
        self._handles: List[Any] = []
        self._recorder = SignalRecorder(
            keep_dims=keep_dims, max_passes=max_passes, keep_vectors_for=self.keep_vectors_for
        )

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> "LocalCapture":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # CPU by default: parity against the platform was established on CPU, and
        # MPS changes the attention kernel enough to move the low-order digits.
        self.device = self.device or "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()
        self._register_hooks()
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.model = None
        self.tokenizer = None

    def __enter__(self) -> "LocalCapture":
        return self.load()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- hooks -------------------------------------------------------------

    def _register_hooks(self) -> None:
        import torch

        def to_numpy(tensor) -> np.ndarray:
            # bf16 has no numpy dtype; float32 is exact for a bf16 value, and the
            # platform stores float32 too, so this cast introduces no difference.
            return tensor.detach().to(torch.float32).cpu().numpy()

        def module_hook(signal_type: str):
            def hook(_module, _inputs, output):
                if not self._recorder.active:
                    return
                # Decoder layers return (hidden_states, ...) in some transformers
                # versions and a bare tensor in others.
                hidden = output[0] if isinstance(output, tuple) else output
                self._recorder.record(signal_type, to_numpy(hidden))
            return hook

        def pass_hook(_module, _args, _kwargs):
            self._recorder.begin_pass()

        base = self.model.model
        self._handles.append(
            self.model.register_forward_pre_hook(pass_hook, with_kwargs=True)
        )
        self._handles.append(
            base.embed_tokens.register_forward_hook(module_hook(EMBED_SIGNAL))
        )
        for k, layer in enumerate(base.layers):
            self._handles.append(layer.register_forward_hook(module_hook(layer_signal_type(k))))

    # -- run ---------------------------------------------------------------

    def run(self, prompt: str, request_id: Optional[str] = None) -> CaptureResult:
        """Generate greedily for one prompt and return its platform-shaped rows."""
        import torch

        if self.model is None:
            raise RuntimeError("Call load() first")

        # Same chat template the platform's /generate path applies, so pass 1
        # sees the same start token it does.
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        n_prompt_tokens = int(inputs["input_ids"].shape[-1])

        self._recorder.reset()
        torch.manual_seed(self.seed)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = out[0][n_prompt_tokens:]
        completion = self.tokenizer.decode(generated, skip_special_tokens=True)
        rows = rows_to_df(self._recorder.rows, request_id=request_id or "local")
        return CaptureResult(
            prompt=prompt,
            text=completion,
            rows=rows,
            vectors=dict(self._recorder.vectors),
            n_prompt_tokens=n_prompt_tokens,
            n_generated_tokens=int(generated.shape[-1]),
            device=str(self.device),
        )


def write_capture_meta(path: str, result: CaptureResult, extra: Optional[dict] = None) -> str:
    """Sidecar describing a capture, so a rows file is never orphaned."""
    meta = {
        "model": HF_MODEL,
        "device": result.device,
        "prompt": result.prompt,
        "text": result.text,
        "n_prompt_tokens": result.n_prompt_tokens,
        "n_generated_tokens": result.n_generated_tokens,
        "n_rows": int(len(result.rows)),
        "signal_types": result.signal_types(),
        "keep_dims": KEEP_DIMS,
        "max_passes": MAX_PASSES,
        "n_layers": N_LAYERS,
    }
    meta.update(extra or {})
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return path
