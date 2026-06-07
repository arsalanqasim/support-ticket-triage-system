"""
Convenience launcher for the TriageIQ REST API.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --host 127.0.0.1 --port 8080 --reload
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn

from triage.config import API_HOST, API_PORT, LOG_LEVEL


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the TriageIQ FastAPI server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=API_HOST, help="Bind host.")
    parser.add_argument("--port", type=int, default=API_PORT, help="Bind port.")
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development.",
    )
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL.lower(),
        choices=["debug", "info", "warning", "error"],
        help="Uvicorn log level.",
    )
    args = parser.parse_args()

    print("\n🎯 TriageIQ API")
    print(f"   Docs  → http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/docs")
    print(f"   ReDoc → http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/redoc\n")

    uvicorn.run(
        "triage.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
