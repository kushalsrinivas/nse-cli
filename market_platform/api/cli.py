"""`platform_cli.py dashboard [--host 127.0.0.1] [--port 8765]` — read-only web UI."""

from __future__ import annotations


def cmd_dashboard(args) -> int:
    import uvicorn

    from market_platform.api.app import create_app
    from platform_cli import ROOT, console, dbs, load_cfg
    cfg = load_cfg(args)
    d = dbs(cfg)                       # creates/migrates the files if needed
    app_path, market_path = d.app_path, d.market_path
    d.close()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        console.print(f"[yellow]binding to {args.host}: the dashboard has no authentication — "
                      "put it behind your own access control[/]")
    console.print(f"dashboard on http://{args.host}:{args.port}  (read-only, paper)")
    uvicorn.run(create_app(cfg, app_path=app_path, market_path=market_path, root=ROOT),
                host=args.host, port=args.port, log_level="warning")
    return 0


def register(sub) -> dict:
    p = sub.add_parser("dashboard", help="read-only web dashboard (FastAPI)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    return {"dashboard": cmd_dashboard}
