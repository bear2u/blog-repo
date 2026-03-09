#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import html
import json
import pathlib
import re
import signal
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request


TRENDING_BASE_URL = "https://github.com/trending"
USER_AGENT = "github-trending-to-blog/1.0 (+https://github.com/trending)"


@dataclasses.dataclass(frozen=True)
class TrendingRepo:
    rank: int
    full_name: str
    url: str
    description: str | None
    language: str | None
    stars_delta: int | None
    total_stars: int | None


def _fetch(url: str, timeout_s: float) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.1",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


def _strip_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_compact_int(value: str) -> int | None:
    v = (value or "").strip().lower().replace(",", "")
    if not v:
        return None

    m = re.fullmatch(r"(\d+(?:\.\d+)?)([km])?", v)
    if not m:
        return None

    num = float(m.group(1))
    suf = m.group(2) or ""
    if suf == "k":
        num *= 1_000
    elif suf == "m":
        num *= 1_000_000
    return int(round(num))


def _normalize_repo_path(href: str) -> str:
    href = (href or "").strip()
    href = href.split("#", 1)[0]
    href = href.split("?", 1)[0]
    return href.strip("/")


def _parse_article(article_html: str) -> tuple[str, dict[str, str]]:
    bits: dict[str, str] = {}

    m = re.search(r"<h2[^>]*>.*?<a[^>]*href=\"([^\"]+)\"", article_html, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return "", bits

    repo_path = _normalize_repo_path(m.group(1))
    if repo_path.count("/") != 1:
        return "", bits

    dm = re.search(
        r'<p[^>]*class="[^"]*\bcol-9\b[^"]*"[^>]*>(.*?)</p>',
        article_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if dm:
        bits["description"] = _strip_tags(dm.group(1))

    lm = re.search(r'itemprop="programmingLanguage"[^>]*>\s*([^<]+)\s*<', article_html, flags=re.IGNORECASE)
    if lm:
        bits["language"] = _strip_tags(lm.group(1))

    sm = re.search(
        rf'href="/{re.escape(repo_path)}/stargazers"[^>]*>(.*?)</a>',
        article_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if sm:
        bits["total_stars_raw"] = _strip_tags(sm.group(1))

    tm = re.search(
        r"([0-9][0-9,\.]*\s*[kKmM]?)\s+stars\s+(today|this\s+week|this\s+month)",
        article_html,
        flags=re.IGNORECASE,
    )
    if tm:
        bits["stars_delta_raw"] = _strip_tags(tm.group(1))

    return repo_path, bits


def fetch_trending(since: str, timeout_s: float) -> list[TrendingRepo]:
    url = f"{TRENDING_BASE_URL}?since={urllib.parse.quote(since)}"
    page = _fetch(url, timeout_s=timeout_s)

    articles = re.findall(
        r'<article[^>]*class="[^"]*\bBox-row\b[^"]*"[^>]*>(.*?)</article>',
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )

    repos: list[TrendingRepo] = []
    seen: set[str] = set()

    for article in articles:
        repo_path, bits = _parse_article(article)
        if not repo_path or repo_path in seen:
            continue
        seen.add(repo_path)

        repos.append(
            TrendingRepo(
                rank=len(repos) + 1,
                full_name=repo_path,
                url=f"https://github.com/{repo_path}",
                description=(bits.get("description") or "").strip() or None,
                language=(bits.get("language") or "").strip() or None,
                stars_delta=_parse_compact_int(bits.get("stars_delta_raw", "")),
                total_stars=_parse_compact_int(bits.get("total_stars_raw", "")),
            )
        )

    return repos


def _find_blog_root(start: pathlib.Path) -> pathlib.Path:
    for parent in [start, *start.parents]:
        if (parent / "_posts").is_dir():
            return parent
    return start


def _snapshot_hash(items: list[TrendingRepo], since: str) -> str:
    payload = {
        "since": since,
        "repos": [
            {
                "full_name": it.full_name,
                "stars_delta": it.stars_delta,
                "total_stars": it.total_stars,
                "language": it.language,
            }
            for it in items
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _extract_existing_hash(md: str) -> str | None:
    m = re.search(r"^snapshot_hash:\s*\"?([0-9a-f]{12,64})\"?\s*$", md, flags=re.MULTILINE | re.IGNORECASE)
    return m.group(1) if m else None


def _extract_existing_repos(md: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r"https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", md):
        out.append(m.group(1))
    deduped: list[str] = []
    seen: set[str] = set()
    for r in out:
        k = r.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)
    return deduped


def _render_post(date_str: str, since: str, items: list[TrendingRepo], generated_at_utc: dt.datetime) -> str:
    original_url = f"{TRENDING_BASE_URL}?since={since}"
    count = len(items)
    ts = generated_at_utc.replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    snap = _snapshot_hash(items=items, since=since)

    lines: list[str] = []
    lines.append("---")
    lines.append("layout: post")
    lines.append(f'title: "GitHub Trending 요약 ({date_str})"')
    lines.append(f"date: {date_str}")
    lines.append(f"permalink: /github-trending-{date_str}/")
    lines.append("author: GitHub Trending")
    lines.append("categories: [개발 트렌드, GitHub]")
    lines.append("tags: [GitHub, Trending, DevTrends]")
    lines.append(f'original_url: "{original_url}"')
    lines.append(f'snapshot_hash: "{snap}"')
    lines.append(f'excerpt: "{date_str} 기준 GitHub Trending({since}) 상위 레포지토리 {count}개를 빠르게 요약합니다."')
    lines.append("---")
    lines.append("")
    lines.append("## 스냅샷")
    lines.append("")
    lines.append(f"- 생성 시각(UTC): `{ts}`")
    lines.append(f"- 기준: GitHub Trending ({since})")
    lines.append("- 참고: Trending은 스냅샷/가변 지표라, 시간이 지나면 순위와 “stars today” 값이 달라질 수 있습니다.")
    lines.append("")
    lines.append("## 한 눈에 보기")
    lines.append("")
    stars_label = {
        "daily": "Stars today",
        "weekly": "Stars this week",
        "monthly": "Stars this month",
    }.get(since, "Stars")
    lines.append(f"| Rank | Repo | {stars_label} | Total stars | Lang | 한 줄 요약 |")
    lines.append("| ---: | --- | ---: | ---: | --- | --- |")

    for it in items:
        delta = f"+{it.stars_delta}" if it.stars_delta is not None else "-"
        total = f"{it.total_stars}" if it.total_stars is not None else "-"
        lang = it.language or "-"
        summary = (it.description or "-").replace("|", "\\|")
        lines.append(f"| {it.rank} | [{it.full_name}]({it.url}) | {delta} | {total} | {lang} | {summary} |")

    lines.append("")
    return "\n".join(lines)


def _render_console(items: list[TrendingRepo], since: str, generated_at_utc: dt.datetime, max_items: int) -> str:
    ts = generated_at_utc.replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    lines: list[str] = []
    lines.append(f"# GitHub Trending {min(max_items, len(items))}개 ({since})")
    lines.append("")
    lines.append(f"- 생성 시각(UTC): `{ts}`")
    lines.append("")
    for i, it in enumerate(items[:max_items], start=1):
        delta = f"+{it.stars_delta}" if it.stars_delta is not None else "-"
        lang = it.language or "-"
        desc = it.description or "-"
        lines.append(f"{i}. [{it.full_name}]({it.url}) (stars: {delta}, lang: {lang}) — {desc}")
    lines.append("")
    return "\n".join(lines)


def _run_git(blog_root: pathlib.Path, args: list[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(blog_root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )


def _current_branch(blog_root: pathlib.Path, timeout_s: float) -> str | None:
    res = _run_git(blog_root, ["branch", "--show-current"], timeout_s=timeout_s)
    if res.returncode != 0:
        return None
    name = (res.stdout or "").strip()
    return name or None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Job #4 helper: fetch GitHub Trending top N and auto-register a Jekyll post without duplicates."
    )
    parser.add_argument("--since", choices=["daily", "weekly", "monthly"], default="daily")
    parser.add_argument("--max", type=int, default=10, help="Maximum items to output/write (default: 10).")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout seconds (default: 15).")
    parser.add_argument(
        "--overall-timeout",
        type=float,
        default=120.0,
        help="Hard stop for the whole run in seconds (default: 120).",
    )
    parser.add_argument("--write", action="store_true", help="Write/update today's post in _posts/.")
    parser.add_argument("--force", action="store_true", help="Write even if snapshot is unchanged.")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of Markdown.")
    parser.add_argument("--blog-root", default="", help="Optional blog root path (must contain _posts/).")
    parser.add_argument("--commit", action="store_true", help="Commit the updated post file.")
    parser.add_argument("--push", action="store_true", help="Push after committing (implies --commit).")
    parser.add_argument("--remote", default="origin", help="Git remote name to push (default: origin).")
    parser.add_argument("--branch", default="", help="Git branch to push (default: current branch).")
    parser.add_argument("--git-timeout", type=float, default=60.0, help="Timeout for each git command in seconds.")
    args = parser.parse_args(argv)

    max_items = max(1, int(args.max))
    since = str(args.since)

    overall_timeout_s = float(args.overall_timeout)
    timer_set = False
    if overall_timeout_s > 0:
        def _alarm_handler(_signum: int, _frame: object) -> None:
            raise TimeoutError("overall timeout exceeded")

        try:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, overall_timeout_s)
            timer_set = True
        except Exception:
            timer_set = False

    try:
        generated_at_utc = dt.datetime.now(tz=dt.timezone.utc)
        date_str = generated_at_utc.date().isoformat()

        try:
            items = fetch_trending(since=since, timeout_s=float(args.timeout))
        except (urllib.error.URLError, TimeoutError) as e:
            sys.stderr.write(f"ERROR: failed to fetch trending: {e}\n")
            return 2
        except Exception as e:
            sys.stderr.write(f"ERROR: failed to fetch trending: {type(e).__name__}: {e}\n")
            return 2

        if not items:
            sys.stderr.write("ERROR: no trending items parsed (GitHub HTML may have changed).\n")
            return 2

        items = items[:max_items]

        if args.json:
            payload = {
                "generated_at_utc": generated_at_utc.isoformat(),
                "since": since,
                "items": [dataclasses.asdict(it) for it in items],
            }
            sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        else:
            sys.stdout.write(
                _render_console(items=items, since=since, generated_at_utc=generated_at_utc, max_items=max_items)
            )

        if not args.write:
            return 0

        if args.blog_root:
            blog_root = pathlib.Path(args.blog_root).expanduser().resolve()
        else:
            blog_root = _find_blog_root(pathlib.Path(__file__).resolve().parent)

        posts_dir = blog_root / "_posts"
        if not posts_dir.is_dir():
            sys.stderr.write(f"ERROR: _posts/ not found under: {blog_root}\n")
            return 2

        post_path = posts_dir / f"{date_str}-github-trending-{since}.md"
        new_hash = _snapshot_hash(items=items, since=since)

        if post_path.exists() and not args.force:
            existing = post_path.read_text(encoding="utf-8", errors="replace")
            existing_hash = _extract_existing_hash(existing)
            if existing_hash and existing_hash == new_hash:
                sys.stdout.write(f"SKIP: unchanged snapshot: {post_path}\n")
                return 0
            if not existing_hash:
                existing_repos = _extract_existing_repos(existing)
                if existing_repos and [it.full_name for it in items] == existing_repos[: len(items)]:
                    sys.stdout.write(f"SKIP: same repo list: {post_path}\n")
                    return 0

        md = _render_post(date_str=date_str, since=since, items=items, generated_at_utc=generated_at_utc)
        post_path.write_text(md, encoding="utf-8")
        sys.stdout.write(f"WROTE: {post_path}\n")

        do_commit = bool(args.commit or args.push)
        if not do_commit:
            return 0

        git_timeout_s = float(args.git_timeout)
        rel_post = str(post_path.relative_to(blog_root))

        add_res = _run_git(blog_root, ["add", "--", rel_post], timeout_s=git_timeout_s)
        if add_res.returncode != 0:
            sys.stderr.write(f"ERROR: git add failed:\n{add_res.stdout}\n")
            return 2

        msg = f"Update GitHub Trending ({since}) {date_str}"
        commit_res = _run_git(blog_root, ["commit", "-m", msg], timeout_s=git_timeout_s)
        if commit_res.returncode != 0:
            out = (commit_res.stdout or "").lower()
            if "nothing to commit" not in out and "no changes added" not in out:
                sys.stderr.write(f"ERROR: git commit failed:\n{commit_res.stdout}\n")
                return 2
            sys.stdout.write("SKIP: nothing to commit\n")
            return 0

        sys.stdout.write("OK: committed\n")

        if not args.push:
            return 0

        branch = (args.branch or "").strip() or _current_branch(blog_root, timeout_s=git_timeout_s) or "main"
        push_res = _run_git(blog_root, ["push", str(args.remote), branch], timeout_s=git_timeout_s)
        if push_res.returncode != 0:
            sys.stderr.write(f"ERROR: git push failed:\n{push_res.stdout}\n")
            return 2

        sys.stdout.write("OK: pushed\n")
        return 0
    except TimeoutError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2
    finally:
        if timer_set:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
