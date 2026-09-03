"""The output-side gate: "does this request still look fine from the outside?"

The whole point of the prompt pack is a request whose *internal* per-layer
signature has deteriorated while every outward-facing number stays in range. So
a candidate is only usable if it passes this gate — which is deliberately the
platform's own OOD metric (a manifold fitted on ``layer_14`` embeddings, the same
detector the delivered OOD panel scores against) plus cheap text sanity checks.

``ood_gate`` takes any object exposing ``predict(Z) -> {"dist", "is_ood",
"ood_prob"}``, which is :class:`.manifold.DistanceToManifoldEmbeddings`'s
interface. Duck-typing it means the tests can plant a manifold of their own.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from .config import (
    OOD_PROB_MAX,
    TEXT_MAX_CHAR_REPEAT,
    TEXT_MAX_CHARS,
    TEXT_MAX_TOP_WORD_SHARE,
    TEXT_MIN_CHARS,
    TEXT_MIN_DISTINCT_WORD_RATIO,
)

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class OODVerdict:
    """What the platform's OOD panel would say about this request."""
    dist: float
    ood_prob: float
    is_ood: bool


@dataclass
class TextVerdict:
    """Whether the generated text reads like a normal answer."""
    ok: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class GateResult:
    """Combined output-side verdict."""
    ood: Optional[OODVerdict]
    text: TextVerdict
    ok: bool


# ---------------------------------------------------------------------------
# OOD side
# ---------------------------------------------------------------------------

def ood_gate(
    vector: np.ndarray,
    manifold: Any,
    *,
    ood_prob_max: float = OOD_PROB_MAX,
) -> OODVerdict:
    """Score one embedding against a fitted manifold.

    ``is_ood`` combines the manifold's own threshold with our stricter
    ``ood_prob_max``: a candidate that is merely *close* to the threshold would
    still make the dashboard's OOD rate twitch, which defeats the demo.
    """
    z = np.asarray(vector, dtype=np.float64)
    if z.ndim == 1:
        z = z[None, :]
    pred = manifold.predict(z)
    dist = float(np.asarray(pred["dist"]).ravel()[0])
    prob = float(np.asarray(pred["ood_prob"]).ravel()[0])
    flagged = bool(np.asarray(pred["is_ood"]).ravel()[0])
    return OODVerdict(dist=dist, ood_prob=prob, is_ood=flagged or prob > ood_prob_max)


# ---------------------------------------------------------------------------
# Text side
# ---------------------------------------------------------------------------

def _longest_char_run(text: str) -> int:
    longest = run = 0
    previous = None
    for ch in text:
        run = run + 1 if ch == previous else 1
        previous = ch
        longest = max(longest, run)
    return longest


def text_checks(
    text: str,
    *,
    min_chars: int = TEXT_MIN_CHARS,
    max_chars: int = TEXT_MAX_CHARS,
    max_char_repeat: int = TEXT_MAX_CHAR_REPEAT,
    min_distinct_word_ratio: float = TEXT_MIN_DISTINCT_WORD_RATIO,
    max_top_word_share: float = TEXT_MAX_TOP_WORD_SHARE,
) -> TextVerdict:
    """Heuristics for "a human skimming this would not notice anything wrong".

    Catches the obvious tells of a degenerate generation — truncation, a stuck
    character, a stuck word, a vocabulary that has collapsed. It does not judge
    answer *quality*: that needs a human or an LLM judge, and "the text checks
    passed" is never a claim that the answer was good.
    """
    reasons: List[str] = []
    stripped = text.strip()

    if len(stripped) < min_chars:
        reasons.append(f"too short: {len(stripped)} < {min_chars} chars")
    if len(stripped) > max_chars:
        reasons.append(f"too long: {len(stripped)} > {max_chars} chars")

    run = _longest_char_run(stripped)
    if run > max_char_repeat:
        reasons.append(f"character repeated {run} times in a row")

    words = [w.lower() for w in _WORD_RE.findall(stripped)]
    if words:
        distinct_ratio = len(set(words)) / len(words)
        if distinct_ratio < min_distinct_word_ratio:
            reasons.append(f"only {distinct_ratio:.2f} distinct-word ratio")
        top_share = Counter(words).most_common(1)[0][1] / len(words)
        if top_share > max_top_word_share:
            reasons.append(f"top word is {top_share:.2f} of all words")
    elif stripped:
        reasons.append("no words found")

    return TextVerdict(ok=not reasons, reasons=reasons)


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def run_gate(
    text: str,
    vector: Optional[np.ndarray] = None,
    manifold: Any = None,
    *,
    ood_prob_max: float = OOD_PROB_MAX,
) -> GateResult:
    """Both sides at once. With no manifold, the OOD half is skipped, not passed."""
    text_verdict = text_checks(text)
    ood_verdict = (
        ood_gate(vector, manifold, ood_prob_max=ood_prob_max)
        if manifold is not None and vector is not None
        else None
    )
    ok = text_verdict.ok and (ood_verdict is None or not ood_verdict.is_ood)
    return GateResult(ood=ood_verdict, text=text_verdict, ok=ok)
