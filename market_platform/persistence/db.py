"""SQLite access for the platform: WAL, migrations, read-only readers.

Two databases with different write profiles (docs/PLATFORM_PLAN.md §3.2):
`app.db` for state, `market.db` for bars and quotes. Each has exactly one
writer (persistence/writer.py); everything else opens read-only
connections, which WAL lets run concurrently with the writer.

Migrations are numbered SQL files under migrations/<db>/. Applied versions
and their checksums are recorded in `schema_migrations`; editing an applied
migration is an error, not a silent drift.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

MIGRATIONS = Path(__file__).resolve().parent / "migrations"


class MigrationError(RuntimeError):
    pass


def connect(path: str | Path, *, readonly: bool = False, timeout: float = 5.0) -> sqlite3.Connection:
    p = Path(path)
    if readonly:
        if not p.exists():
            raise FileNotFoundError(f"{p} does not exist")
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=timeout,
                               check_same_thread=False)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(p, timeout=timeout, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
    conn.row_factory = sqlite3.Row
    return conn


def _migrations(db: str) -> list[tuple[int, str, str]]:
    out = []
    for f in sorted((MIGRATIONS / db).glob("*.sql")):
        version = int(f.name.split("_", 1)[0])
        out.append((version, f.name, f.read_text()))
    return out


def migrate(conn: sqlite3.Connection, db: str) -> list[str]:
    """Apply pending migrations for `db` ('app' | 'market'). Returns names applied."""
    conn.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL,
        applied_at TEXT NOT NULL)""")
    applied = {r["version"]: r for r in conn.execute("SELECT * FROM schema_migrations")}
    done = []
    for version, name, sql in _migrations(db):
        checksum = hashlib.sha256(sql.encode()).hexdigest()
        if version in applied:
            if applied[version]["checksum"] != checksum:
                raise MigrationError(f"{db} migration {name} changed after it was applied "
                                     "— add a new migration instead of editing an old one")
            continue
        try:
            conn.executescript("BEGIN;\n" + sql + "\nCOMMIT;")
        except sqlite3.Error as exc:
            conn.execute("ROLLBACK") if conn.in_transaction else None
            raise MigrationError(f"{db} migration {name} failed: {exc}") from exc
        conn.execute("INSERT INTO schema_migrations VALUES (?,?,?,?)",
                     (version, name, checksum, datetime.now().isoformat(timespec="seconds")))
        conn.commit()
        done.append(name)
    return done


def open_db(path: str | Path, db: str) -> sqlite3.Connection:
    """Writer connection with migrations applied."""
    conn = connect(path)
    migrate(conn, db)
    return conn


def integrity(conn: sqlite3.Connection) -> str:
    return conn.execute("PRAGMA quick_check").fetchone()[0]


class Databases:
    """The pair of platform databases resolved from the config paths."""

    def __init__(self, app_path: str | Path, market_path: str | Path) -> None:
        self.app_path = Path(app_path)
        self.market_path = Path(market_path)
        self.app = open_db(self.app_path, "app")
        self.market = open_db(self.market_path, "market")

    @classmethod
    def from_config(cls, cfg, root: str | Path | None = None) -> Databases:
        base = Path(root) if root else Path.cwd()
        return cls(base / cfg.paths.app_db, base / cfg.paths.market_db)

    def reader(self, which: str) -> sqlite3.Connection:
        return connect(self.app_path if which == "app" else self.market_path, readonly=True)

    def close(self) -> None:
        self.app.close()
        self.market.close()
