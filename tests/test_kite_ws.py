"""WS client tests: scripted fake connections, no network."""
import asyncio
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.kite.ws import KiteWS


def ltp_packet(token=1, price=10000):
    return struct.pack(">i i", token, price)


def frame(*payloads):
    out = struct.pack(">h", len(payloads))
    for p in payloads:
        out += struct.pack(">h", len(p)) + p
    return out


class FakeWS:
    def __init__(self, script):
        self.script = list(script)
        self.sent: list = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    async def send(self, data):
        import json
        self.sent.append(json.loads(data))

    async def recv(self):
        if self.closed:
            raise ConnectionError("closed")
        if not self.script:
            while not self.closed:
                await asyncio.sleep(0.01)
            raise ConnectionError("closed")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self):
        self.closed = True


class ScriptedFactory:
    def __init__(self, scripts):
        self.made = [FakeWS(s) for s in scripts]
        self._queue = list(self.made)
        self.urls: list = []

    def __call__(self, url, **kw):
        self.urls.append(url)
        return self._queue.pop(0)


class TestKiteWS(unittest.TestCase):
    def run_loop(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)

    def test_subscribe_tick_and_text(self):
        seen_ticks, seen_text, states = [], [], []
        factory = ScriptedFactory([[frame(ltp_packet(408065, 141295)),
                                    '{"type": "error", "data": "boom"}']])
        client = KiteWS("K", "T", on_ticks=lambda t, s: seen_ticks.extend(t),
                        on_text=lambda p: seen_text.append(p),
                        on_state=states.append,
                        ws_factory=factory)
        # drive one session manually then stop
        loop = asyncio.new_event_loop()
        loop.run_until_complete(client.subscribe([408065], "full"))
        task = loop.create_task(client.run())
        loop.run_until_complete(asyncio.sleep(0.3))
        loop.run_until_complete(client.close())
        loop.run_until_complete(task)
        loop.close()
        actions = [m["a"] for m in factory.made[0].sent]
        self.assertEqual(actions, ["subscribe", "mode"])
        self.assertEqual(factory.made[0].sent[0]["v"], [408065])
        self.assertEqual(factory.made[0].sent[1]["v"], ["full", [408065]])
        self.assertEqual(len(seen_ticks), 1)
        self.assertEqual(seen_ticks[0]["ltp"], 1412.95)
        self.assertEqual(seen_text[0]["type"], "error")
        self.assertIn("connected", states)
        self.assertEqual(states[-1], "closed")
        self.assertIn("api_key=K", factory.urls[0])
        self.assertIn("access_token=T", factory.urls[0])

    def test_reconnect_restores_intent(self):
        factory = ScriptedFactory([[ConnectionError("drop")], []])
        client = KiteWS("K", "T", ws_factory=factory)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(client.subscribe([1, 2], "quote"))
        task = loop.create_task(client.run())
        loop.run_until_complete(asyncio.sleep(2.8))  # backoff 1-2s, then reconnect
        loop.run_until_complete(client.close())
        loop.run_until_complete(task)
        loop.close()
        self.assertEqual(len(factory.urls), 2)
        second_sent = factory.made[1].sent
        self.assertEqual(second_sent[0], {"a": "subscribe", "v": [1, 2]})
        self.assertEqual(second_sent[1], {"a": "mode", "v": ["quote", [1, 2]]})
        self.assertGreaterEqual(client.counters["reconnects"], 1)
        self.assertGreaterEqual(client.counters["connects"], 1)

    def test_unsubscribe_updates_intent(self):
        factory = ScriptedFactory([[frame(ltp_packet())]])
        client = KiteWS("K", "T", ws_factory=factory)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(client.subscribe([1, 2]))
        loop.run_until_complete(client.unsubscribe([2]))
        self.assertEqual(client.subscribed(), {1: "quote"})
        task = loop.create_task(client.run())
        loop.run_until_complete(asyncio.sleep(0.2))
        loop.run_until_complete(client.close())
        loop.run_until_complete(task)
        loop.close()
        sent_sub = factory.made[0].sent[0]
        self.assertEqual(sent_sub, {"a": "subscribe", "v": [1]})

    def test_limit_guarded(self):
        client = KiteWS("K", "T")
        loop = asyncio.new_event_loop()
        with self.assertRaises(ValueError):
            loop.run_until_complete(client.subscribe(list(range(3001))))
        loop.close()

    def test_set_mode_only_known(self):
        client = KiteWS("K", "T")
        loop = asyncio.new_event_loop()
        loop.run_until_complete(client.subscribe([1]))
        loop.run_until_complete(client.set_mode([1, 999], "full"))
        self.assertEqual(client.subscribed(), {1: "full"})
        loop.close()


if __name__ == "__main__":
    unittest.main()
