"""The one place a database connection string is built.

Reads the five connection variables from the environment, falling back to a
``.env`` file at the repo root. Nothing is built at import time: importing this
module must never open a connection or fail because a variable is unset, so the
pure-numpy parts of the package stay importable with no database at all.
"""

import os
from pathlib import Path
from typing import Optional

from .config import DEFAULT_DB_NAME

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Environment variable that redirects this package at another database (a local
#: restore, say) without editing ``.env`` or passing an argument.
DB_NAME_ENV_VAR = "LAYER_PROFILE_DB_NAME"

#: Connection variables, in DSN order. All five are required.
REQUIRED_VARS = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT"]


def dsn(db_name: Optional[str] = None) -> str:
    """Build a PostgreSQL DSN for the observability database.

    Database-name precedence: the ``db_name`` argument, then
    ``$LAYER_PROFILE_DB_NAME``, then ``$DB_NAME``, then
    :data:`config.DEFAULT_DB_NAME`. The argument wins so a caller that knows
    which database it needs cannot be redirected by the environment.

    Raises ``RuntimeError`` naming every missing variable, rather than building a
    DSN with the string ``"None"`` in it and failing at connect time.
    """
    _load_dotenv()

    resolved_db = (
        db_name
        or os.environ.get(DB_NAME_ENV_VAR)
        or os.environ.get("DB_NAME")
        or DEFAULT_DB_NAME
    )

    missing = [name for name in REQUIRED_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing database settings: " + ", ".join(missing) + ". "
            f"Set them in the environment or in {REPO_ROOT / '.env'} "
            "(see .env.example)."
        )

    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    host = os.environ["DB_HOST"]
    port = os.environ["DB_PORT"]
    return f"postgresql://{user}:{password}@{host}:{port}/{resolved_db}"


def _load_dotenv() -> None:
    """Load ``.env`` without overriding anything already in the environment.

    ``override=False`` on purpose: a variable exported in the shell is a
    deliberate act and must beat a file. Missing python-dotenv is not an error —
    the variables may well come from the environment alone — so the import is
    local and its failure is silent.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


def _redact(text: str) -> str:
    """Strip the password out of a DSN so it is safe to put in a message."""
    if "://" not in text or "@" not in text:
        return text
    scheme, rest = text.split("://", 1)
    _, host_part = rest.rsplit("@", 1)
    return f"{scheme}://***@{host_part}"
