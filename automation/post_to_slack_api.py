#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


SLACK_API_BASE = "https://slack.com/api/"


def slack_api(token: str, method: str, params: dict[str, object]) -> dict[str, object]:
    data = urllib.parse.urlencode({k: str(v) for k, v in params.items() if v is not None}).encode("utf-8")
    req = urllib.request.Request(
        SLACK_API_BASE + method,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15.0) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        parsed = json.loads(raw)
    except Exception:
        raise RuntimeError(f"Slack API returned non-JSON for {method}: {raw[:200]}")
    return parsed


def choose_channel(token: str) -> str | None:
    resp = slack_api(
        token,
        "users.conversations",
        {
            "types": "public_channel,private_channel",
            "exclude_archived": "true",
            "limit": 200,
        },
    )
    if not resp.get("ok"):
        raise RuntimeError(f"Slack users.conversations failed: {resp.get('error')}")

    channels = resp.get("channels") or []
    if not isinstance(channels, list):
        return None

    candidates: list[dict[str, object]] = []
    preferred_names = {
        "trending",
        "trend",
        "trendradar",
        "blog",
        "blogs",
        "content",
        "notifications",
        "notify",
        "alerts",
        "automation",
    }
    for ch in channels:
        if not isinstance(ch, dict):
            continue
        name = str(ch.get("name") or "").lower().strip()
        if name in preferred_names:
            candidates.append(ch)

    pool = candidates or [ch for ch in channels if isinstance(ch, dict)]
    if len(pool) == 1:
        return str(pool[0].get("id") or "").strip() or None

    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Post a text file to Slack via Web API (chat.postMessage).")
    parser.add_argument("--text-file", required=True, help="Path to a UTF-8 text file to post.")
    parser.add_argument(
        "--channel",
        default="",
        help="Slack channel ID (e.g. C0123...). Defaults to env SLACK_CHANNEL_ID / SLACK_CHANNEL.",
    )
    parser.add_argument(
        "--token",
        default="",
        help="Slack bot token (xoxb-...). Defaults to env SLACK_BOT_TOKEN.",
    )
    args = parser.parse_args(argv)

    token = args.token.strip() or os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token:
        sys.stderr.write("ERROR: missing Slack bot token (set SLACK_BOT_TOKEN or pass --token).\n")
        return 2

    channel = (
        args.channel.strip()
        or os.environ.get("SLACK_CHANNEL_ID", "").strip()
        or os.environ.get("SLACK_CHANNEL", "").strip()
    )
    if not channel:
        try:
            channel = choose_channel(token) or ""
        except Exception as e:
            sys.stderr.write(f"ERROR: could not auto-select a Slack channel: {type(e).__name__}: {e}\n")
            return 2
        if not channel:
            sys.stderr.write(
                "ERROR: missing Slack channel. Pass --channel (channel ID) or set SLACK_CHANNEL_ID.\n"
            )
            return 2

    text_path = Path(args.text_file).expanduser().resolve()
    text = text_path.read_text(encoding="utf-8")

    try:
        resp = slack_api(token, "chat.postMessage", {"channel": channel, "text": text})
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        sys.stderr.write(f"ERROR: Slack API HTTP {e.code}: {body or e.reason}\n")
        return 2
    except Exception as e:
        sys.stderr.write(f"ERROR: Slack API request failed: {type(e).__name__}: {e}\n")
        return 2

    if not resp.get("ok"):
        sys.stderr.write(f"ERROR: Slack API error: {resp.get('error')}\n")
        return 2

    sys.stdout.write("OK: posted to Slack via chat.postMessage.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

