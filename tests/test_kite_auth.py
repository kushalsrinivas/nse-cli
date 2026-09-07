"""Unit tests for Kite Phase 0: config + auth. Fully offline — the Kite
client class is faked, the config dir points at tmp, no network."""
import os
import sys
import unittest
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IST = ZoneInfo("Asia/Kolkata")


@contextmanager
def kite_env(key="K", secret="S", config_dir=None):
    env = {"KITE_API_KEY": key, "KITE_API_SECRET": secret}
    if config_dir is not None:
        env["KITE_CONFIG_DIR"] = config_dir
    old = {k: os.environ.pop(k, None) for k in
           ("KITE_API_KEY", "KITE_API_SECRET", "KITE_CONFIG_DIR")}
    os.environ.update({k: v for k, v in env.items() if v is not None})
    try:
        yield
    finally:
        for k in ("KITE_API_KEY", "KITE_API_SECRET", "KITE_CONFIG_DIR"):
            os.environ.pop(k, None)
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v


@contextmanager
def no_kite_env():
    old = {k: os.environ.pop(k, None) for k in
           ("KITE_API_KEY", "KITE_API_SECRET", "KITE_CONFIG_DIR")}
    try:
        yield
    finally:
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v


class FakeKiteConnect:
    seen: dict = {}

    def __init__(self, api_key):
        FakeKiteConnect.seen["api_key"] = api_key

    def generate_session(self, request_token, api_secret):
        FakeKiteConnect.seen["request_token"] = request_token
        FakeKiteConnect.seen["api_secret"] = api_secret
        return {"access_token": "tok123", "user_id": "AB1234",
                "login_time": "2026-09-06 10:00:00"}


class TestKiteConfig(unittest.TestCase):
    def test_missing_env_raises(self):
        from data.kite.config import KiteConfigError, credentials, has_credentials
        with no_kite_env():
            self.assertFalse(has_credentials())
            with self.assertRaises(KiteConfigError):
                credentials()

    def test_env_read(self):
        import tempfile
        from data.kite.config import credentials, has_credentials, session_path
        with tempfile.TemporaryDirectory() as tmp:
            with kite_env(key="KEY1", secret="SEC1", config_dir=tmp):
                self.assertTrue(has_credentials())
                c = credentials()
                self.assertEqual((c.api_key, c.api_secret), ("KEY1", "SEC1"))
                self.assertEqual(session_path(), Path(tmp) / "kite_session.json")


class TestKiteAuth(unittest.TestCase):
    def test_login_url_has_key_not_secret(self):
        from data.kite.auth import login_url
        url = login_url("MYKEY")
        self.assertIn("MYKEY", url)
        self.assertNotIn("SEC", url)
        self.assertTrue(url.startswith("https://kite.zerodha.com/connect/login"))

    def test_exchange_success(self):
        import tempfile
        from data.kite.auth import exchange_token
        with tempfile.TemporaryDirectory() as tmp:
            with kite_env(key="K", secret="S", config_dir=tmp):
                FakeKiteConnect.seen = {}
                s = exchange_token("reqtok", kite_cls=FakeKiteConnect)
                self.assertEqual(s["access_token"], "tok123")
                self.assertEqual(FakeKiteConnect.seen["api_key"], "K")
                self.assertEqual(FakeKiteConnect.seen["request_token"], "reqtok")
                # secret reaches the client (required for checksum) but is
                # never logged or persisted by us
                self.assertEqual(FakeKiteConnect.seen["api_secret"], "S")

    def test_exchange_failure_wraps(self):
        import tempfile
        from data.kite.auth import KiteAuthError, exchange_token

        class Boom:
            def __init__(self, api_key):
                pass

            def generate_session(self, request_token, api_secret):
                raise RuntimeError("nope")

        with tempfile.TemporaryDirectory() as tmp:
            with kite_env(config_dir=tmp):
                with self.assertRaises(KiteAuthError):
                    exchange_token("reqtok", kite_cls=Boom)

    def test_exchange_no_token_rejected(self):
        import tempfile
        from data.kite.auth import KiteAuthError, exchange_token

        class Empty:
            def __init__(self, api_key):
                pass

            def generate_session(self, request_token, api_secret):
                return {"user_id": "x"}

        with tempfile.TemporaryDirectory() as tmp:
            with kite_env(config_dir=tmp):
                with self.assertRaises(KiteAuthError):
                    exchange_token("reqtok", kite_cls=Empty)

    def test_save_load_roundtrip_0600(self):
        import stat
        import tempfile
        from data.kite import auth
        with tempfile.TemporaryDirectory() as tmp:
            with kite_env(config_dir=tmp):
                path = auth.save_session({"access_token": "t", "user_id": "u",
                                          "login_time": "2026-09-06 10:00:00"})
                self.assertTrue(path.exists())
                self.assertEqual(oct(path.stat().st_mode & 0o777), "0o600")
                loaded = auth.load_session()
                self.assertEqual(loaded["access_token"], "t")

    def test_load_corrupt_is_none(self):
        import tempfile
        from data.kite import auth
        from data.kite.config import session_path
        with tempfile.TemporaryDirectory() as tmp:
            with kite_env(config_dir=tmp):
                session_path().parent.mkdir(parents=True, exist_ok=True)
                session_path().write_text("{broken")
                self.assertIsNone(auth.load_session())

    def test_expiry_is_6am_next_day_ist(self):
        from data.kite.auth import session_expiry
        exp = session_expiry("2026-09-06 16:00:00")
        self.assertEqual(exp.strftime("%Y-%m-%d %H:%M"),
                         "2026-09-07 06:00")
        self.assertEqual(exp.tzinfo, IST)

    def test_validity_windows(self):
        from data.kite import auth
        s = {"access_token": "t", "login_time": "2026-09-06 10:00:00"}
        self.assertTrue(auth.session_valid(
            s, datetime(2026, 9, 6, 12, 0, tzinfo=IST)))
        self.assertFalse(auth.session_valid(
            s, datetime(2026, 9, 7, 7, 0, tzinfo=IST)))
        self.assertFalse(auth.session_valid(None))
        self.assertFalse(auth.session_valid({}))

    def test_status_never_leaks_tokens(self):
        import tempfile
        from data.kite import auth
        with tempfile.TemporaryDirectory() as tmp:
            with kite_env(config_dir=tmp):
                self.assertEqual(auth.status()["state"], "missing")
                auth.save_session({"access_token": "SECRET-TOK",
                                   "user_id": "AB1",
                                   "login_time": "2026-09-06 10:00:00"})
                st = auth.status()
                self.assertNotIn("SECRET-TOK", str(st))

    def test_clear(self):
        import tempfile
        from data.kite import auth
        with tempfile.TemporaryDirectory() as tmp:
            with kite_env(config_dir=tmp):
                self.assertFalse(auth.clear_session())
                auth.save_session({"access_token": "t"})
                self.assertTrue(auth.clear_session())
                self.assertIsNone(auth.load_session())


if __name__ == "__main__":
    unittest.main()
