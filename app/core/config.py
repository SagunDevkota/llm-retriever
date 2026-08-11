"""Centralised configuration and secret loading.

Secrets (API keys, DSNs) are read from the environment so they don't live in
source. A ``.env`` file is loaded automatically if ``python-dotenv`` is present.
Defaults mirror the values already used in the notebook / db modules — nothing
new is introduced here.
"""

import os
from pathlib import Path

from core.errors import ConfigError

try:  # optional dependency — fine if it's not installed
    from dotenv import load_dotenv

    # cwd first (an entrypoint's own .env wins), then the two fixed locations —
    # app/.env holds the keys and callers outside app/ (llm_evaluation/) need it
    # too, so it is located relative to this file rather than the cwd.
    _APP_DIR = Path(__file__).resolve().parents[1]
    load_dotenv()
    load_dotenv(_APP_DIR / ".env")
    load_dotenv(_APP_DIR.parent / ".env")
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


# --- Providers -------------------------------------------------------------
# Both are OpenAI-compatible, so a single ``openai.OpenAI`` client talks to
# either one — only the key and base URL differ.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")  # was hardcoded in the notebook
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
HETZNER_API_KEY = os.environ.get("HETZNER_API_KEY")
HETZNER_BASE_URL = os.environ.get(
    "HETZNER_BASE_URL", "https://inference.hetzner.com/api/v1"
)

# --- The four models ------------------------------------------------------
# One model per job, so each can be swapped independently:
#   summarizer  — phase 3, writes evidence_/routing_ summaries per node
#   routing     — phase 5, the per-level tree-navigation decisions
#   answer      — phase 5, the single final generation call
#   evaluation  — llm_evaluation/, the RAGAS judge
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"

SUMMARIZER_MODEL = os.environ.get("SUMMARIZER_MODEL", DEFAULT_MODEL)
ROUTING_MODEL = os.environ.get("ROUTING_MODEL", DEFAULT_MODEL)
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", DEFAULT_MODEL)
EVALUATION_MODEL = os.environ.get("EVALUATION_MODEL", DEFAULT_MODEL)

# role -> (provider, default model). Summarizer/answer run on OpenRouter,
# routing/evaluation on Hetzner.
MODEL_ROLES = {
    "summarizer": ("openrouter", SUMMARIZER_MODEL),
    "answer": ("openrouter", ANSWER_MODEL),
    "routing": ("hetzner", ROUTING_MODEL),
    "evaluation": ("openrouter", EVALUATION_MODEL),
}

_PROVIDERS = {
    "openrouter": ("OPENROUTER_API_KEY", OPENROUTER_API_KEY, OPENROUTER_BASE_URL),
    "hetzner": ("HETZNER_API_KEY", HETZNER_API_KEY, HETZNER_BASE_URL),
}


def init_model(
    role: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    async_client: bool = False,
):
    """Build the OpenAI-compatible client for a model *role* and announce it.

    Returns ``(client, model)``. Explicit ``model`` / ``api_key`` / ``base_url``
    arguments win over the role's defaults; everything else comes from the
    environment. Every initialisation prints one line naming the role, the model
    id and the provider it is talking to, so a run's model wiring is visible in
    the log without reading the config.
    """
    from openai import AsyncOpenAI, OpenAI  # local: keeps core.config import-light

    try:
        provider, default_model = MODEL_ROLES[role]
    except KeyError:
        raise ConfigError(
            f"Unknown model role {role!r}. Known roles: {sorted(MODEL_ROLES)}."
        ) from None

    env_name, provider_key, provider_base_url = _PROVIDERS[provider]
    model = model or default_model
    api_key = api_key or provider_key
    base_url = base_url or provider_base_url

    if not api_key:
        raise ConfigError(
            f"No API key for the {role} model. Set {env_name} (env/.env) "
            f"or pass api_key=... explicitly."
        )

    client_cls = AsyncOpenAI if async_client else OpenAI
    client = client_cls(api_key=api_key, base_url=base_url)
    print(f"[model] {role:<10} -> {model}  (provider={provider}, base_url={base_url})")
    return client, model
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
