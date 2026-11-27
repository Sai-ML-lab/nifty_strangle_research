from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")


class LevelStore:
    """Local SQLite persistence for user-marked levels in the Phase 1 MVP."""

    def __init__(self, database_path: str | Path = "data/level_trading/levels.sqlite3"):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS manual_levels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date TEXT NOT NULL,
                name TEXT NOT NULL,
                price REAL NOT NULL CHECK(price > 0),
                side TEXT NOT NULL CHECK(side IN ('SUPPORT','RESISTANCE','REFERENCE')),
                notes TEXT NOT NULL DEFAULT '',
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )''')
            conn.execute("CREATE INDEX IF NOT EXISTS idx_manual_levels_date_active ON manual_levels(session_date, active)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_manual_level(self, *, session_date: date | str, name: str,
                         price: float, side: str, notes: str = "") -> int:
        side = side.upper()
        if side not in {"SUPPORT", "RESISTANCE", "REFERENCE"}:
            raise ValueError("Invalid level side")
        if float(price) <= 0:
            raise ValueError("price must be positive")
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO manual_levels
                (session_date,name,price,side,notes,active,created_at)
                VALUES (?,?,?,?,?,1,?)""",
                (str(pd.Timestamp(session_date).date()), name.strip() or "MANUAL",
                 float(price), side, notes.strip(), datetime.now(IST).isoformat()),
            )
            return int(cur.lastrowid)

    def list_manual_levels(self, session_date: date | str,
                           *, active_only: bool = True) -> list[dict[str, Any]]:
        query = "SELECT * FROM manual_levels WHERE session_date = ?"
        params: list[Any] = [str(pd.Timestamp(session_date).date())]
        if active_only:
            query += " AND active = 1"
        query += " ORDER BY price"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def deactivate(self, level_id: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE manual_levels SET active = 0 WHERE id = ?", (int(level_id),))
