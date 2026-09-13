"""Load secrets from a local .env into the process environment.

Secrets are read from environment variables only. This loader is the one place
that puts a developer's local .env into that environment; it never logs, echoes,
or persists a value.

A variable that is already set to a non-empty value wins over the file, so a
real environment variable still overrides .env. A variable that is present but
blank — which is what `export OPENAI_API_KEY=` leaves behind — is treated as
unset, because otherwise the file is skipped and the key looks missing.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def parse_env(text: str) -> dict[str, str]:
    """Parse .env text into name/value pairs, tolerating common hand-edits."""
    values: dict[str, str] = {}
    for raw_line in text.lstrip("﻿").splitlines():
        line = raw_line.strip().lstrip("﻿")
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name.lower().startswith("export "):
            name = name[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        value = value.strip()
        if name and value:
            values[name] = value
    return values


def load_env(path: str | Path = DEFAULT_ENV_PATH) -> list[str]:
    """Set any unset-or-blank variables defined in ``path``. Returns names loaded."""
    env_path = Path(path)
    if not env_path.is_file():
        return []

    loaded: list[str] = []
    for name, value in parse_env(env_path.read_text(encoding="utf-8")).items():
        if os.environ.get(name, "").strip():
            continue
        os.environ[name] = value
        loaded.append(name)
    return loaded


def describe(path: str | Path = DEFAULT_ENV_PATH) -> str:
    """Explain what the loader can see, without revealing any value."""
    env_path = Path(path)
    lines = [f"  .env path: {env_path}", f"  .env exists: {env_path.is_file()}"]
    if env_path.is_file():
        names = parse_env(env_path.read_text(encoding="utf-8"))
        lines.append(f"  names parsed from .env: {sorted(names) or '(none)'}")
        for name, value in sorted(names.items()):
            lines.append(f"    {name}: {len(value)} chars")
    shell = os.environ.get("OPENAI_API_KEY")
    if shell is None:
        lines.append("  OPENAI_API_KEY in shell before load: not set")
    elif not shell.strip():
        lines.append("  OPENAI_API_KEY in shell before load: set but BLANK "
                     "(run `unset OPENAI_API_KEY`, or fix the export in your shell profile)")
    else:
        lines.append(f"  OPENAI_API_KEY in shell before load: set, {len(shell)} chars")
    lines.append(f"  working directory: {Path.cwd()}")
    return "\n".join(lines)
