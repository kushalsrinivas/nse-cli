#!/usr/bin/env bash
# One-shot Kite Connect login: open browser → catch redirect → store session.
#
# Prerequisite (one-time): on https://developers.kite.trade set your app's
# Redirect URL exactly to:
#   http://127.0.0.1:8000/
# (or whatever KITE_REDIRECT_HOST / KITE_REDIRECT_PORT you export).
#
# Usage:
#   ./scripts/kite-login.sh           # skip if session already valid
#   ./scripts/kite-login.sh --force   # re-login even if valid
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOST="${KITE_REDIRECT_HOST:-127.0.0.1}"
PORT="${KITE_REDIRECT_PORT:-8000}"
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force|-f) FORCE=1 ;;
    -h|--help)
      sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown arg: $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [[ -z "${KITE_API_KEY:-}" || -z "${KITE_API_SECRET:-}" ]]; then
  echo "missing KITE_API_KEY / KITE_API_SECRET — put them in .env or export them" >&2
  exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="python3"
fi
if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
  echo "python not found (tried \$PYTHON / .venv / python3)" >&2
  exit 1
fi

# Already logged in?
if [[ "$FORCE" -eq 0 ]]; then
  if status_out="$("$PYTHON" "$ROOT/model_cli.py" kite-login 2>&1)"; then
    if printf '%s\n' "$status_out" | grep -q 'kite session valid'; then
      echo "$status_out"
      echo "(re-run with --force to log in again)"
      exit 0
    fi
  fi
fi

TOKEN_FILE="$(mktemp -t kite_request_token.XXXXXX)"
SERVER_PID=""
cleanup() {
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true
  rm -f "$TOKEN_FILE"
}
trap cleanup EXIT

echo "listening for Kite redirect on http://${HOST}:${PORT}/"
echo "ensure developers.kite.trade Redirect URL is exactly: http://${HOST}:${PORT}/"

# Tiny one-shot callback server (writes request_token then exits).
"$PYTHON" - "$HOST" "$PORT" "$TOKEN_FILE" <<'PY' &
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

host, port_s, token_path = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])

DONE = """<!doctype html><html><body style="font-family:system-ui;padding:2rem">
<h1>Kite login captured</h1>
<p>You can close this tab and return to the terminal.</p>
</body></html>"""
FAIL = """<!doctype html><html><body style="font-family:system-ui;padding:2rem">
<h1>Login failed</h1>
<p>No request_token in the redirect. Try again from the terminal.</p>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        return

    def do_GET(self):
        qs = parse_qs(urlparse(self.path).query)
        token = (qs.get("request_token") or [None])[0]
        status = (qs.get("status") or [""])[0]
        ok = bool(token) and status.lower() in ("", "success")
        body = (DONE if ok else FAIL).encode()
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        if ok:
            token_path.write_text(token.strip() + "\n")
        # stop serving after first hit
        raise SystemExit(0 if ok else 1)


HTTPServer((host, port_s), Handler).serve_forever()
PY
SERVER_PID=$!

LOGIN_URL="https://kite.zerodha.com/connect/login?v=3&api_key=${KITE_API_KEY}"
echo "opening: $LOGIN_URL"
if command -v open >/dev/null 2>&1; then
  open "$LOGIN_URL"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$LOGIN_URL"
else
  echo "open this URL manually: $LOGIN_URL"
fi

echo "waiting for browser login (Ctrl-C to abort)..."
# Wait until the callback wrote the token (or the server died).
while [[ ! -s "$TOKEN_FILE" ]]; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID" || true
    echo "callback server exited without a token — check Redirect URL / login status" >&2
    exit 1
  fi
  sleep 0.2
done

# Server exits itself on success; reap it.
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""

REQUEST_TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
if [[ -z "$REQUEST_TOKEN" ]]; then
  echo "empty request_token" >&2
  exit 1
fi

echo "got request_token — exchanging..."
"$PYTHON" "$ROOT/model_cli.py" kite-login --request-token "$REQUEST_TOKEN"
