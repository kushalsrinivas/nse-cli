"""Single-writer task per database, with batched commits and backpressure.

Producers (candle service, signal engines, risk, execution) never touch
SQLite directly in the event loop. They `submit()` write intents to a
bounded queue; one writer drains it, executes in a worker thread and
commits a batch every `flush_ms` or every `max_batch` statements. This
removes the audit's A7/A8 problems: no commit blocks tick processing and
there is only one writer per file.

When the queue is full `submit()` waits (backpressure), and the wait is
counted, so the health monitor can show that persistence is the
bottleneck. `SyncWriter` has the same interface for scripts, tests and
backtests that run without an event loop.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Write:
    sql: str
    params: tuple | list | None = None
    many: bool = False          # executemany over `params` (a list of tuples)


@dataclass
class WriterStats:
    submitted: int = 0
    committed: int = 0
    batches: int = 0
    errors: int = 0
    dropped: int = 0
    backpressure_waits: int = 0
    last_commit_ms: float = 0.0
    max_commit_ms: float = 0.0
    last_error: str = ""
    recent_commit_ms: list[float] = field(default_factory=list)

    def note_commit(self, ms: float) -> None:
        self.last_commit_ms = ms
        self.max_commit_ms = max(self.max_commit_ms, ms)
        self.recent_commit_ms.append(ms)
        if len(self.recent_commit_ms) > 200:
            del self.recent_commit_ms[:100]


def _apply(conn: sqlite3.Connection, batch: list[Write]) -> int:
    n = 0
    for w in batch:
        if w.many:
            conn.executemany(w.sql, w.params or [])
            n += len(w.params or [])
        else:
            conn.execute(w.sql, w.params or ())
            n += 1
    conn.commit()
    return n


class SyncWriter:
    """Immediate writes; same interface as AsyncWriter.submit (minus await)."""

    def __init__(self, conn: sqlite3.Connection, name: str = "sync") -> None:
        self.conn = conn
        self.name = name
        self.stats = WriterStats()

    def write(self, *items: Write) -> int:
        t = time.perf_counter()
        n = _apply(self.conn, list(items))
        self.stats.submitted += len(items)
        self.stats.committed += n
        self.stats.batches += 1
        self.stats.note_commit((time.perf_counter() - t) * 1000)
        return n


class AsyncWriter:
    def __init__(self, conn: sqlite3.Connection, *, name: str, maxsize: int = 100_000,
                 flush_ms: int = 250, max_batch: int = 5_000, retries: int = 5) -> None:
        self.conn = conn
        self.name = name
        self.queue: asyncio.Queue[Write] = asyncio.Queue(maxsize=maxsize)
        self.flush_s = flush_ms / 1000.0
        self.max_batch = max_batch
        self.retries = retries
        self.stats = WriterStats()
        self._task: asyncio.Task | None = None
        self._stopping = False

    @property
    def depth(self) -> int:
        return self.queue.qsize()

    async def submit(self, item: Write) -> None:
        if self.queue.full():
            self.stats.backpressure_waits += 1
        await self.queue.put(item)
        self.stats.submitted += 1

    def submit_nowait(self, item: Write) -> bool:
        try:
            self.queue.put_nowait(item)
            self.stats.submitted += 1
            return True
        except asyncio.QueueFull:
            self.stats.backpressure_waits += 1
            return False

    def start(self) -> asyncio.Task:
        self._task = asyncio.create_task(self._run(), name=f"writer:{self.name}")
        return self._task

    async def stop(self) -> None:
        self._stopping = True
        if self._task:
            await self._task

    async def flush(self) -> None:
        """Wait until everything submitted so far is committed."""
        while self.queue.qsize() or self._in_flight:
            await asyncio.sleep(0.01)

    _in_flight = False

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch: list[Write] = []
            try:
                first = await asyncio.wait_for(self.queue.get(), timeout=self.flush_s)
                batch.append(first)
            except asyncio.TimeoutError:
                if self._stopping and self.queue.empty():
                    return
                continue
            deadline = loop.time() + self.flush_s
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    if loop.time() >= deadline:
                        break
                    await asyncio.sleep(min(0.01, self.flush_s))
            self._in_flight = True
            try:
                await self._commit(batch)
            finally:
                self._in_flight = False
            if self._stopping and self.queue.empty():
                return

    async def _commit(self, batch: list[Write]) -> None:
        delay = 0.05
        for attempt in range(self.retries + 1):
            t = time.perf_counter()
            try:
                n = await asyncio.to_thread(_apply, self.conn, batch)
                self.stats.committed += n
                self.stats.batches += 1
                self.stats.note_commit((time.perf_counter() - t) * 1000)
                return
            except sqlite3.Error as exc:
                self.conn.rollback()
                self.stats.last_error = str(exc)
                if not _transient(exc):
                    break                                 # a bad statement: isolate it below
                if attempt == self.retries:
                    # Never kill the writer task: drop the batch, count it, and let
                    # the health monitor surface `dropped` / `last_error`.
                    self.stats.errors += 1
                    self.stats.dropped += len(batch)
                    log.error("writer %s dropped a batch of %d after %d retries: %s",
                              self.name, len(batch), self.retries, exc)
                    return
                await asyncio.sleep(delay)
                delay = min(delay * 2, 2.0)
        good = 0
        for w in batch:
            try:
                good += await asyncio.to_thread(_apply, self.conn, [w])
            except sqlite3.Error as inner:
                self.conn.rollback()
                self.stats.errors += 1
                self.stats.dropped += 1
                self.stats.last_error = str(inner)
                log.error("writer %s rejected statement: %s (%s)", self.name, w.sql[:80], inner)
        self.stats.committed += good
        self.stats.batches += 1


_TRANSIENT = ("locked", "busy", "disk i/o")


def _transient(exc: sqlite3.Error) -> bool:
    """Locked/busy/I-O errors are retried; anything else (bad SQL, constraint,
    missing table — SQLite reports some of these as OperationalError too) is a
    statement problem that retrying cannot fix."""
    return isinstance(exc, sqlite3.OperationalError) and any(
        t in str(exc).lower() for t in _TRANSIENT)
