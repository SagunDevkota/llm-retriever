"""Centralised configuration and secret loading.

Secrets (API keys, DSNs) are read from the environment so they don't live in
source. A ``.env`` file is loaded automatically if ``python-dotenv`` is present.
Defaults mirror the values already used in the notebook / db modules — nothing
new is introduced here.
"""

import os

from core.errors import ConfigError

try:  # optional dependency — fine if it's not installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def get_env(name: str, default=None, *, required: bool = False):
    """Return an environment variable, raising :class:`ConfigError` if a
    ``required`` variable is unset/empty."""
    value = os.environ.get(name, default)
    if required and not value:
        raise ConfigError(
            f"Missing required environment variable: {name}. "
            f"Set it (e.g. `export {name}=...`) or add it to a .env file."
        )
    return value


# --- Values that already exist in the codebase, now env-overridable --------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")  # was hardcoded in the notebook
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
SUMMARIZER_MODEL = os.environ.get("SUMMARIZER_MODEL", "google/gemma-4-31b-it:free")
# Base DSN (host/user/port + a fallback database name). Entrypoints that read or
# write data override the database name from a required CLI arg via
# ``dsn_with_db``; see save_checkpoint.py / run_retrieve.py.
POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "postgresql://rag:rag@localhost:5432/ragdb")


def dsn_with_db(base_dsn: str, db_name: str) -> str:
    """Return ``base_dsn`` with its database name (last path segment) replaced."""
    base, sep, _ = base_dsn.rpartition("/")
    if not sep:
        raise ConfigError(f"Malformed DSN, cannot locate database name: {base_dsn!r}")
    return f"{base}/{db_name}"# "postgresql://rag:rag@localhost:5432/ragdb"
CHECKPOINT_FILE = os.environ.get("CHECKPOINT_FILE", "hierarchical_checkpoint.json")
