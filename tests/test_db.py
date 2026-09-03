"""Connection-string assembly: precedence, the missing-variable error, redaction.

No connection is opened — ``dsn()`` only builds a string, which is exactly why
it is safe to unit-test.
"""

import pytest

from layer_profile import db
from layer_profile.config import DEFAULT_DB_NAME

ALL_VARS = {
    "DB_USER": "u",
    "DB_PASSWORD": "secret",
    "DB_HOST": "h",
    "DB_PORT": "5433",
}


@pytest.fixture
def env(monkeypatch):
    """A clean environment with the four connection variables set."""
    for name, value in ALL_VARS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("DB_NAME", raising=False)
    monkeypatch.delenv(db.DB_NAME_ENV_VAR, raising=False)
    # .env must not reintroduce anything the test just cleared.
    monkeypatch.setattr(db, "_load_dotenv", lambda: None)
    return monkeypatch


# ---------------------------------------------------------------------------
# Database-name precedence
# ---------------------------------------------------------------------------

def test_falls_back_to_the_configured_default(env):
    assert db.dsn() == f"postgresql://u:secret@h:5433/{DEFAULT_DB_NAME}"


def test_db_name_from_the_environment_beats_the_default(env):
    env.setenv("DB_NAME", "some_restore")
    assert db.dsn().endswith("/some_restore")


def test_the_override_variable_beats_db_name(env):
    """The point of the override: .env names one database, a run needs another."""
    env.setenv("DB_NAME", "onex_observability_core_backend")
    env.setenv(db.DB_NAME_ENV_VAR, "from_override")
    assert db.dsn().endswith("/from_override")


def test_the_argument_beats_everything(env):
    env.setenv("DB_NAME", "from_env")
    env.setenv(db.DB_NAME_ENV_VAR, "from_override")
    assert db.dsn("from_arg").endswith("/from_arg")


# ---------------------------------------------------------------------------
# Missing variables
# ---------------------------------------------------------------------------

def test_missing_variables_are_all_named(env):
    """Naming every missing variable beats failing once per fix-and-retry."""
    env.delenv("DB_HOST")
    env.delenv("DB_PORT")
    with pytest.raises(RuntimeError) as exc:
        db.dsn()
    message = str(exc.value)
    assert "DB_HOST" in message and "DB_PORT" in message
    assert "DB_USER" not in message


def test_an_empty_variable_counts_as_missing(env):
    env.setenv("DB_PASSWORD", "")
    with pytest.raises(RuntimeError, match="DB_PASSWORD"):
        db.dsn()


def test_the_error_does_not_leak_the_password(env):
    env.delenv("DB_HOST")
    with pytest.raises(RuntimeError) as exc:
        db.dsn()
    assert "secret" not in str(exc.value)


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

def test_redact_removes_the_password_and_keeps_the_host():
    redacted = db._redact("postgresql://u:secret@h:5433/somedb")
    assert "secret" not in redacted
    assert redacted == "postgresql://***@h:5433/somedb"


def test_redact_passes_through_anything_that_is_not_a_dsn():
    assert db._redact("not a dsn") == "not a dsn"
