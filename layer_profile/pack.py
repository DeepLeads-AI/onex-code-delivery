"""Reading a prompt pack off disk, and the escaping its text file uses."""

import os
from typing import Dict

import pandas as pd


def escape_prompt(text: str) -> str:
    r"""Put a multi-line prompt on one line, reversibly.

    The corpus convention is one prompt per line, but these prompts contain
    newlines and those newlines are part of what is being tested — collapsing
    them to spaces would ship a prompt that is not the prompt that was scored.
    So newlines become a literal ``\n`` and backslashes are escaped, and
    :func:`unescape_prompt` inverts it exactly.
    """
    return str(text).replace("\\", "\\\\").replace("\r\n", "\n").replace("\n", "\\n")


def unescape_prompt(line: str) -> str:
    r"""Inverse of :func:`escape_prompt`: ``\n`` back to a newline."""
    out, i = [], 0
    text = str(line)
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == "n":
                out.append("\n")
                i += 2
                continue
            if nxt == "\\":
                out.append("\\")
                i += 2
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def load_pack(pack_dir: str) -> Dict[str, str]:
    """Prompt ids and verbatim texts from a pack directory.

    ``pack.csv``'s ``prompt`` column is the authority — it holds the text exactly
    as it was scored. ``pack.txt`` is the one-prompt-per-line view of the same
    thing with newlines escaped, and is only used when the CSV is absent. Reading
    a whitespace-collapsed prompt would score something other than what was
    measured.
    """
    pack_csv = os.path.join(pack_dir, "pack.csv")
    pack_txt = os.path.join(pack_dir, "pack.txt")

    if os.path.exists(pack_csv):
        frame = pd.read_csv(pack_csv)
        if "prompt" in frame.columns:
            return dict(zip(frame["prompt_id"].astype(str), frame["prompt"].astype(str)))

    if not os.path.exists(pack_txt):
        raise FileNotFoundError(f"No pack.csv or pack.txt in {pack_dir}")
    with open(pack_txt) as fh:
        texts = [unescape_prompt(line.rstrip("\n")) for line in fh if line.strip()]
    ids = (pd.read_csv(pack_csv)["prompt_id"].astype(str).tolist()
           if os.path.exists(pack_csv) else [f"pack-{i:02d}" for i in range(len(texts))])
    return dict(zip(ids, texts))
