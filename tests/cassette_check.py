"""Check recorded cassettes for secrets and size before they are committed.

Run directly (also used by the pre-commit hook and ``tests/test_cassettes.py``)::

    python tests/cassette_check.py                 # all cassettes under tests/cassettes
    python tests/cassette_check.py path/to/a.yaml  # specific files

Exits non-zero and lists every problem found.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import yaml

CASSETTE_ROOT = Path(__file__).parent / "cassettes"

#: Largest allowed cassette, matching the check-added-large-files pre-commit limit.
MAX_BYTES = 500_000

#: Request headers removed when recording (see ``vcr_config`` in ``tests/conftest.py``).
FILTER_HEADERS = ("authorization", "x-api-key", "api-key", "cookie", "proxy-authorization")

#: Query parameters removed when recording.
FILTER_QUERY_PARAMETERS = ("api_key", "apikey", "token", "key", "access_token")

#: Well-known credential formats that must never appear anywhere in a cassette.
TOKEN_PATTERNS = {
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "Bearer token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    "Private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}

#: Environment variables whose values are secrets (plus any ending in these suffixes).
SECRET_ENV_VARS = ("API_USGS_PAT",)
SECRET_ENV_SUFFIXES = ("_PAT", "_TOKEN", "_API_KEY", "_SECRET", "_PASSWORD")


def _secret_env_values() -> dict[str, str]:
    values = {}
    for name, value in os.environ.items():
        is_secret = name in SECRET_ENV_VARS or name.endswith(SECRET_ENV_SUFFIXES)
        if is_secret and len(value) >= 8:
            values[name] = value
    return values


def check_file(path: Path) -> list[str]:
    """Return a list of problems found in one cassette file (empty if clean)."""
    problems: list[str] = []
    size = path.stat().st_size
    if size > MAX_BYTES:
        problems.append(f"{path}: {size} bytes exceeds the {MAX_BYTES} byte limit")

    text = path.read_text(encoding="utf-8", errors="replace")
    for label, pattern in TOKEN_PATTERNS.items():
        if pattern.search(text):
            problems.append(f"{path}: contains something that looks like a {label}")
    for name, value in _secret_env_values().items():
        if value in text:
            problems.append(f"{path}: contains the value of environment variable {name}")

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        problems.append(f"{path}: not valid YAML ({exc.__class__.__name__})")
        return problems

    for i, interaction in enumerate(data.get("interactions") or []):
        request = interaction.get("request") or {}
        headers = {str(k).lower() for k in (request.get("headers") or {})}
        for header in sorted(headers & set(FILTER_HEADERS)):
            problems.append(f"{path}: interaction {i} request header {header!r} not filtered")
        query = parse_qsl(urlsplit(str(request.get("uri", ""))).query, keep_blank_values=True)
        for key, _ in query:
            if key.lower() in FILTER_QUERY_PARAMETERS:
                problems.append(f"{path}: interaction {i} query parameter {key!r} not filtered")
    return problems


def check_paths(paths: Iterable[Path]) -> list[str]:
    """Check files, or every ``*.yaml``/``*.yml`` file under directories."""
    problems: list[str] = []
    for path in paths:
        files = sorted([*path.rglob("*.yaml"), *path.rglob("*.yml")]) if path.is_dir() else [path]
        for file in files:
            problems.extend(check_file(file))
    return problems


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or [CASSETTE_ROOT]
    problems = check_paths(paths)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
