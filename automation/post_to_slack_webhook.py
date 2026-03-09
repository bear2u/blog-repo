#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def post(webhook_url: str, text: str, *, timeout_s: float) -> None:
    payload = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        resp.read()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Post a text file to Slack via Incoming Webhook.")
    parser.add_argument("--text-file", required=True, help="Path to a UTF-8 text file to post.")
    parser.add_argument(
        "--webhook-url",
        default="",
        help="Slack Incoming Webhook URL. Defaults to env SLACK_WEBHOOK_URL.",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout seconds (default: 10).")
    args = parser.parse_args(argv)

    webhook_url = args.webhook_url.strip() or os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not webhook_url:
        sys.stderr.write("ERROR: missing Slack webhook URL (set SLACK_WEBHOOK_URL or pass --webhook-url).\n")
        return 2

    text_path = Path(args.text_file).expanduser().resolve()
    text = text_path.read_text(encoding="utf-8")

    try:
        post(webhook_url, text, timeout_s=float(args.timeout))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        sys.stderr.write(f"ERROR: Slack webhook HTTP {e.code}: {body or e.reason}\n")
        return 2
    except Exception as e:
        sys.stderr.write(f"ERROR: Slack webhook failed: {type(e).__name__}: {e}\n")
        return 2

    sys.stdout.write("OK: posted to Slack webhook.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

