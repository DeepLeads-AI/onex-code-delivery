"""Reading a pack off disk, and the escaping its .txt file uses."""

import pandas as pd
import pytest

from layer_profile.pack import escape_prompt, load_pack, unescape_prompt

MULTILINE = "Summarise this:\nitem one\nitem two\n\nIn one sentence."


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------

def test_escaping_puts_a_multiline_prompt_on_one_line():
    escaped = escape_prompt(MULTILINE)
    assert "\n" not in escaped
    assert escaped.count("\\n") == 4


def test_the_round_trip_is_exact():
    assert unescape_prompt(escape_prompt(MULTILINE)) == MULTILINE


def test_a_literal_backslash_n_survives_the_round_trip():
    r"""The reason backslashes are escaped: `\n` typed by a user is not a newline."""
    text = "a windows path C:\\new and a real\nnewline"
    assert unescape_prompt(escape_prompt(text)) == text


def test_carriage_returns_are_normalised_before_escaping():
    assert escape_prompt("a\r\nb") == "a\\nb"


def test_a_trailing_backslash_is_not_swallowed():
    assert unescape_prompt(escape_prompt("ends with a backslash \\")) == (
        "ends with a backslash \\"
    )


# ---------------------------------------------------------------------------
# load_pack
# ---------------------------------------------------------------------------

def write_pack(directory, *, csv=True, txt=True, prompt_column=True):
    directory.mkdir(exist_ok=True)
    if csv:
        columns = {"prompt_id": ["a-00", "b-01"]}
        if prompt_column:
            columns["prompt"] = [MULTILINE, "plain prompt"]
        pd.DataFrame(columns).to_csv(directory / "pack.csv", index=False)
    if txt:
        (directory / "pack.txt").write_text(
            escape_prompt(MULTILINE) + "\n" + escape_prompt("plain prompt") + "\n"
        )
    return directory


def test_the_csv_is_preferred_and_returns_the_verbatim_text(tmp_path):
    """The CSV holds the text exactly as scored; the .txt is the escaped view."""
    pack = write_pack(tmp_path / "pack")
    assert load_pack(str(pack)) == {"a-00": MULTILINE, "b-01": "plain prompt"}


def test_the_txt_is_used_when_there_is_no_csv(tmp_path):
    pack = write_pack(tmp_path / "pack", csv=False)
    loaded = load_pack(str(pack))
    assert list(loaded.values()) == [MULTILINE, "plain prompt"]
    assert list(loaded) == ["pack-00", "pack-01"]


def test_a_csv_without_a_prompt_column_falls_through_to_the_txt(tmp_path):
    """Ids still come from the CSV — only the text comes from the .txt."""
    pack = write_pack(tmp_path / "pack", prompt_column=False)
    assert load_pack(str(pack)) == {"a-00": MULTILINE, "b-01": "plain prompt"}


def test_an_empty_directory_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No pack.csv or pack.txt"):
        load_pack(str(empty))
