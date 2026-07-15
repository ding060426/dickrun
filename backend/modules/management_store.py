"""Select the persistence adapter used by the meeting management module."""

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from modules import meeting_db


BACKEND_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _load_backend_env(environ: MutableMapping[str, str]) -> None:
    if not BACKEND_ENV_PATH.is_file():
        return
    for raw_line in BACKEND_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        environ.setdefault(key.strip(), value.strip())


def select_management_store(environ: Mapping[str, str] | None = None):
    """Use Supabase only when both credentials are configured.

    The local SQLite adapter keeps the integrated application runnable without
    network access or an optional Supabase installation.
    """

    if environ is None:
        _load_backend_env(os.environ)
        environ = os.environ
    if not (environ.get("SUPABASE_URL") and environ.get("SUPABASE_KEY")):
        return meeting_db

    from modules import supabase_db

    return supabase_db
