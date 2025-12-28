from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .settings import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    base_model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    negative_prompt TEXT NOT NULL,
    params_json TEXT NOT NULL,
    image_files_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_generations_created_at ON generations(created_at DESC);
"""


def _connect() -> sqlite3.Connection:
    db_path = settings.history_db
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


@dataclass(frozen=True)
class HistoryItem:
    id: int
    created_at: int
    seed: int
    base_model: str
    prompt: str
    negative_prompt: str
    params: dict[str, Any]
    image_files: list[str]


def add_generation(
    *,
    seed: int,
    base_model: str,
    prompt: str,
    negative_prompt: str,
    params: dict[str, Any],
    image_files: list[str],
) -> int:
    init_db()
    created_at = int(time.time())
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO generations (created_at, seed, base_model, prompt, negative_prompt, params_json, image_files_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                int(seed),
                str(base_model),
                str(prompt),
                str(negative_prompt or ""),
                json.dumps(params, ensure_ascii=False),
                json.dumps(list(image_files), ensure_ascii=False),
            ),
        )
        return int(cur.lastrowid)


def get_generation(gen_id: int) -> HistoryItem | None:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, created_at, seed, base_model, prompt, negative_prompt, params_json, image_files_json FROM generations WHERE id = ?",
            (int(gen_id),),
        ).fetchone()

    if row is None:
        return None

    return HistoryItem(
        id=int(row["id"]),
        created_at=int(row["created_at"]),
        seed=int(row["seed"]),
        base_model=str(row["base_model"]),
        prompt=str(row["prompt"]),
        negative_prompt=str(row["negative_prompt"]),
        params=json.loads(row["params_json"]),
        image_files=list(json.loads(row["image_files_json"])),
    )


def list_generations(limit: int = 50) -> list[HistoryItem]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, seed, base_model, prompt, negative_prompt, params_json, image_files_json
            FROM generations
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

    items: list[HistoryItem] = []
    for row in rows:
        items.append(
            HistoryItem(
                id=int(row["id"]),
                created_at=int(row["created_at"]),
                seed=int(row["seed"]),
                base_model=str(row["base_model"]),
                prompt=str(row["prompt"]),
                negative_prompt=str(row["negative_prompt"]),
                params=json.loads(row["params_json"]),
                image_files=list(json.loads(row["image_files_json"])),
            )
        )
    return items


def format_choice(item: HistoryItem) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item.created_at))
    prompt = " ".join(item.prompt.strip().split())
    if len(prompt) > 80:
        prompt = prompt[:77] + "..."
    base = Path(item.base_model).name
    return f"{item.id} | {ts} | seed {item.seed} | {base} | {prompt}"
