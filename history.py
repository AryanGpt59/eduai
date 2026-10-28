"""Optional app-local storage for aggregate session summaries only."""

from __future__ import annotations

import json
import os
from contextlib import closing
from pathlib import Path
import sqlite3
from typing import Any


def database_path() -> Path:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    elif os.environ.get("XDG_DATA_HOME"):
        base = Path(os.environ["XDG_DATA_HOME"])
    else:
        base = Path.home() / ".local" / "share"
    return base / "EduAI" / "session_summaries.sqlite3"


def _connect(path: Path | None = None) -> sqlite3.Connection:
    destination = path or database_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(destination)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS summaries ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
        "summary_json TEXT NOT NULL)"
    )
    return connection


def save_summary(summary: dict[str, Any], path: Path | None = None) -> None:
    with closing(_connect(path)) as connection:
        with connection:
            connection.execute("INSERT INTO summaries (summary_json) VALUES (?)", (json.dumps(summary),))


def list_summaries(path: Path | None = None, limit: int = 50) -> list[dict[str, Any]]:
    if limit < 1:
        return []
    with closing(_connect(path)) as connection:
        rows = connection.execute(
            "SELECT saved_at, summary_json FROM summaries ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{"saved_at": saved_at, **json.loads(payload)} for saved_at, payload in rows]


def clear_summaries(path: Path | None = None) -> None:
    with closing(_connect(path)) as connection:
        with connection:
            connection.execute("DELETE FROM summaries")
