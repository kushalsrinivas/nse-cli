"""In-process pub/sub with bounded queues and an explicit overload policy.

Each subscriber owns one bounded `asyncio.Queue`. A topic has many
subscribers; publishing puts the event on every subscriber's queue. What
happens when a queue is full is decided per subscriber, never implicitly:

* `block`       — the publisher awaits (backpressure). For consumers that
                  must see every event: candle → structure, signal → risk.
* `drop_oldest` — the oldest queued event is discarded and counted. For
                  consumers that only need the latest state (UI, health).
* `drop_new`    — the new event is discarded and counted.

Kafka/Redis are not used: everything runs in one process, events are
re-derivable from market.db, and the audit trail is the `events` table
(plan §3.2). Every drop is counted and exposed to the health monitor.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

POLICIES = ("block", "drop_oldest", "drop_new")


@dataclass
class SubStats:
    delivered: int = 0
    dropped: int = 0
    blocked: int = 0
    high_water: int = 0


@dataclass
class Subscription:
    name: str
    topic: str
    queue: asyncio.Queue
    policy: str
    stats: SubStats = field(default_factory=SubStats)

    @property
    def depth(self) -> int:
        return self.queue.qsize()

    async def get(self):
        return await self.queue.get()

    def get_nowait(self):
        return self.queue.get_nowait()

    def drain(self, limit: int = 10_000) -> list:
        out = []
        while len(out) < limit:
            try:
                out.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return out


class Bus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Subscription]] = {}
        self.published: dict[str, int] = {}

    def subscribe(self, topic: str, name: str, *, maxsize: int = 10_000,
                  policy: str = "block") -> Subscription:
        if policy not in POLICIES:
            raise ValueError(f"policy {policy!r} not in {POLICIES}")
        if maxsize <= 0:
            raise ValueError("queues are bounded: maxsize must be > 0")
        sub = Subscription(name, topic, asyncio.Queue(maxsize=maxsize), policy)
        self._subs.setdefault(topic, []).append(sub)
        return sub

    def subscribe_many(self, topics: tuple[str, ...], name: str, *, maxsize: int = 10_000,
                       policy: str = "block") -> Subscription:
        """One queue fed by several topics, in publish order (e.g. one
        approver for both signal pipelines)."""
        sub = self.subscribe(topics[0], name, maxsize=maxsize, policy=policy)
        for t in topics[1:]:
            self._subs.setdefault(t, []).append(sub)
        sub.topic = "+".join(topics)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        for subs in self._subs.values():
            if sub in subs:
                subs.remove(sub)

    async def publish(self, topic: str, event) -> None:
        self.published[topic] = self.published.get(topic, 0) + 1
        for sub in self._subs.get(topic, ()):
            q = sub.queue
            if q.full():
                if sub.policy == "block":
                    sub.stats.blocked += 1
                    await q.put(event)
                elif sub.policy == "drop_oldest":
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    sub.stats.dropped += 1
                    q.put_nowait(event)
                else:
                    sub.stats.dropped += 1
                    continue
            else:
                q.put_nowait(event)
            sub.stats.delivered += 1
            if q.qsize() > sub.stats.high_water:
                sub.stats.high_water = q.qsize()

    def publish_nowait(self, topic: str, event) -> bool:
        """Non-blocking publish for sync callers; a `block` subscriber with a
        full queue makes this return False (the event was not delivered to it)."""
        ok = True
        self.published[topic] = self.published.get(topic, 0) + 1
        for sub in self._subs.get(topic, ()):
            q = sub.queue
            if q.full():
                if sub.policy == "drop_oldest":
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    q.put_nowait(event)
                    sub.stats.delivered += 1
                sub.stats.dropped += 1
                ok = ok and sub.policy != "block"
                continue
            q.put_nowait(event)
            sub.stats.delivered += 1
            sub.stats.high_water = max(sub.stats.high_water, q.qsize())
        return ok

    def stats(self) -> list[dict]:
        out = []
        for topic, subs in self._subs.items():
            for s in subs:
                out.append({"topic": topic, "name": s.name, "policy": s.policy,
                            "depth": s.depth, "maxsize": s.queue.maxsize,
                            **vars(s.stats)})
        return out
