#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import textwrap
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True)
class TrendingItem:
    rank: int
    full_name: str
    url: str
    description: str
    language: str
    stars_today: int
    series: str
    already_registered: bool


GENERIC_SERIES = {
    "ui",
    "app",
    "apps",
    "cli",
    "api",
    "core",
    "server",
    "client",
    "tools",
    "tool",
    "sdk",
    "docs",
    "doc",
    "website",
    "site",
    "skills",
}


PROJECT_OVERRIDES: dict[str, dict[str, str]] = {
    "openai/skills": {
        "display_name": "OpenAI Skills (Codex)",
        "author": "OpenAI",
        "section": "AI 코딩 에이전트",
        "badge_class": "badge-coding",
        "badge_label": "AI 코딩",
        "icon": "fas fa-robot",
        "card_desc": "Codex용 Skills Catalog 기준으로 스킬 구조/작성/테스트/배포를 정리 (5챕터)",
    },
    "shareAI-lab/learn-claude-code": {
        "display_name": "learn-claude-code",
        "author": "shareAI-lab",
        "section": "AI 코딩 에이전트",
        "badge_class": "badge-coding",
        "badge_label": "AI 코딩",
        "icon": "fas fa-terminal",
        "card_desc": "Bash로 만드는 나노 Claude Code 스타일 에이전트: 0→1 구현 따라가기 (5챕터)",
    },
    "666ghj/MiroFish": {
        "display_name": "MiroFish",
        "author": "666ghj",
        "section": "AI 에이전트",
        "badge_class": "badge-agent",
        "badge_label": "AI 에이전트",
        "icon": "fas fa-robot",
        "card_desc": "간결한 범용 군집지능(스웜) 엔진으로 예측 문제를 실험/확장하는 프레임워크 (5챕터)",
    },
    "toeverything/AFFiNE": {
        "display_name": "AFFiNE",
        "author": "toeverything",
        "section": "개발 도구",
        "badge_class": "badge-tool",
        "badge_label": "개발 도구",
        "icon": "fas fa-tools",
        "card_desc": "Notion·Miro 대안 오픈소스 지식베이스: 프라이버시 우선, 커스터마이즈, 바로 사용 (5챕터)",
    },
    "GoogleCloudPlatform/generative-ai": {
        "display_name": "Google Cloud Generative AI",
        "author": "Google Cloud",
        "section": "LLM 학습",
        "badge_class": "badge-llm",
        "badge_label": "LLM",
        "icon": "fas fa-book",
        "card_desc": "Vertex AI(Gemini) 기반 생성형 AI 샘플 코드·노트북 모음 활용법 (5챕터)",
    },
    "shadcn-ui/ui": {
        "display_name": "shadcn/ui",
        "author": "shadcn-ui",
        "section": "개발 도구",
        "badge_class": "badge-tool",
        "badge_label": "UI",
        "icon": "fas fa-tools",
        "card_desc": "접근성 좋은 UI 컴포넌트 + 코드 배포 플랫폼: shadcn/ui 실전 사용 가이드 (5챕터)",
    },
    "pbakaus/impeccable": {
        "display_name": "Impeccable",
        "author": "pbakaus",
        "section": "개발 도구",
        "badge_class": "badge-tool",
        "badge_label": "디자인",
        "icon": "fas fa-tools",
        "card_desc": "AI가 더 좋은 UI를 만들도록 돕는 디자인 언어/규칙 세트 적용법 (5챕터)",
    },
    "virattt/ai-hedge-fund": {
        "display_name": "AI Hedge Fund",
        "author": "virattt",
        "section": "AI 에이전트",
        "badge_class": "badge-agent",
        "badge_label": "AI 에이전트",
        "icon": "fas fa-robot",
        "card_desc": "에이전트 팀으로 투자 아이디어→백테스트를 자동화하는 실험용 프로젝트 (5챕터)",
    },
    "Ed1s0nZ/CyberStrikeAI": {
        "display_name": "CyberStrikeAI",
        "author": "Ed1s0nZ",
        "section": "AI 에이전트",
        "badge_class": "badge-agent",
        "badge_label": "보안",
        "icon": "fas fa-shield-alt",
        "card_desc": "Go 기반 AI 네이티브 보안 테스트 플랫폼: 100+ 툴 오케스트레이션/역할 기반 테스트 (5챕터)",
    },
}


CHAPTERS: list[tuple[str, str]] = [
    ("intro", "소개 및 개요"),
    ("installation", "설치 및 빠른 시작"),
    ("architecture", "핵심 개념과 아키텍처"),
    ("usage", "실전 사용 패턴"),
    ("best-practices", "운영/확장/베스트 프랙티스"),
]


def slugify(value: str) -> str:
    v = re.sub(r"[^a-zA-Z0-9]+", "-", (value or "").strip().lower())
    v = re.sub(r"-+", "-", v).strip("-")
    return v


def choose_series(full_name: str) -> str:
    owner, repo = full_name.split("/", 1)
    owner_s = slugify(owner)
    repo_s = slugify(repo)
    if full_name == "openai/skills":
        return "openai-skills"
    if repo_s in GENERIC_SERIES or len(repo_s) <= 3:
        if repo_s == "ui":
            return owner_s
        return slugify(f"{owner_s}-{repo_s}")
    return repo_s


def parse_items(path: Path) -> list[TrendingItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[TrendingItem] = []
    for it in raw:
        out.append(
            TrendingItem(
                rank=int(it["rank"]),
                full_name=str(it["full_name"]),
                url=str(it["url"]),
                description=str(it.get("description") or "").strip(),
                language=str(it.get("language") or "").strip(),
                stars_today=int(it.get("stars_today") or 0),
                series=str(it.get("series") or choose_series(str(it["full_name"]))),
                already_registered=bool(it.get("already_registered")),
            )
        )
    out.sort(key=lambda x: x.rank)
    return out


def fetch_default_branch(session: requests.Session, full_name: str, timeout_s: float) -> str:
    owner, repo = full_name.split("/", 1)
    url = f"https://api.github.com/repos/{urllib.parse.quote(owner)}/{urllib.parse.quote(repo)}"
    r = session.get(url, timeout=timeout_s, headers={"Accept": "application/vnd.github+json"})
    if r.status_code != 200:
        return "main"
    data = r.json()
    return str(data.get("default_branch") or "main")


def fetch_readme(session: requests.Session, full_name: str, timeout_s: float) -> str:
    owner, repo = full_name.split("/", 1)
    branch = fetch_default_branch(session, full_name, timeout_s)
    candidates = [
        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/readme.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.rst",
        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.txt",
    ]
    for url in candidates:
        try:
            r = session.get(url, timeout=timeout_s)
        except Exception:
            continue
        if r.status_code == 200 and (r.text or "").strip():
            return r.text
    return ""


def extract_title_from_readme(readme: str) -> str | None:
    for line in (readme or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip() or None
        break
    return None


def extract_commands(readme: str, *, max_each: int = 2) -> tuple[list[str], list[str]]:
    install_patterns = [
        r"^\s*(pipx?\s+install\s+.+)$",
        r"^\s*(python\s+-m\s+pip\s+install\s+.+)$",
        r"^\s*(uv\s+pip\s+install\s+.+)$",
        r"^\s*(npm\s+install(?:\s+-g)?\s+.+)$",
        r"^\s*(pnpm\s+add(?:\s+-g)?\s+.+)$",
        r"^\s*(yarn\s+add(?:\s+global)?\s+.+)$",
        r"^\s*(bun\s+add(?:\s+-g)?\s+.+)$",
        r"^\s*(cargo\s+install\s+.+)$",
        r"^\s*(go\s+install\s+.+)$",
        r"^\s*(brew\s+install\s+.+)$",
    ]
    run_patterns = [
        r"^\s*(python\s+.+\.py(?:\s+.+)?)$",
        r"^\s*(python\s+-m\s+\S+(?:\s+.+)?)$",
        r"^\s*(node\s+.+)$",
        r"^\s*(npm\s+run\s+\S+.*)$",
        r"^\s*(pnpm\s+\S+.*)$",
        r"^\s*(yarn\s+\S+.*)$",
        r"^\s*(bun\s+\S+.*)$",
        r"^\s*(docker\s+compose\s+up.*)$",
        r"^\s*(make\s+\S+.*)$",
    ]

    install_cmds: list[str] = []
    run_cmds: list[str] = []

    for line in (readme or "").splitlines():
        if len(install_cmds) < max_each:
            for pat in install_patterns:
                m = re.match(pat, line.strip())
                if m:
                    cmd = m.group(1).strip()
                    if cmd not in install_cmds:
                        install_cmds.append(cmd)
                    break
        if len(run_cmds) < max_each:
            for pat in run_patterns:
                m = re.match(pat, line.strip())
                if m:
                    cmd = m.group(1).strip()
                    if cmd not in run_cmds:
                        run_cmds.append(cmd)
                    break
        if len(install_cmds) >= max_each and len(run_cmds) >= max_each:
            break

    return install_cmds, run_cmds


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fm_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_post(
    *,
    date_str: str,
    series: str,
    part: int,
    slug: str,
    project_name: str,
    chapter_title: str,
    author: str,
    categories: list[str],
    tags: list[str],
    original_url: str,
    excerpt: str,
    body: str,
) -> str:
    part_label = f"{part:02d}"
    lines: list[str] = []
    lines.append("---")
    lines.append("layout: post")
    lines.append(f'title: "{fm_quote(project_name)} 완벽 가이드 ({part_label}) - {fm_quote(chapter_title)}"')
    lines.append(f"date: {date_str}")
    lines.append(f"permalink: /{series}-guide-{part_label}-{slug}/")
    lines.append(f"author: {author}")
    lines.append(f"categories: [{', '.join(categories)}]")
    lines.append(f"tags: [{', '.join(tags)}]")
    lines.append(f'original_url: "{fm_quote(original_url)}"')
    lines.append(f'excerpt: "{fm_quote(excerpt)}"')
    lines.append("---")
    lines.append("")
    lines.append(body.rstrip())
    lines.append("")
    return "\n".join(lines)


def render_index(
    *,
    series: str,
    project_name: str,
    icon: str,
    one_liner: str,
    intro_paragraph: str,
) -> str:
    lines: list[str] = []
    lines.append("---")
    lines.append("layout: page")
    lines.append(f"title: {project_name} 가이드")
    lines.append(f"permalink: /{series}-guide/")
    lines.append(f"icon: {icon}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {project_name} 완벽 가이드")
    lines.append("")
    lines.append(f"> **{one_liner}**")
    lines.append("")
    lines.append(intro_paragraph.strip())
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 목차")
    lines.append("")
    lines.append("| # | 제목 | 내용 |")
    lines.append("|---|------|------|")
    for idx, (slug, title) in enumerate(CHAPTERS, start=1):
        part_label = f"{idx:02d}"
        lines.append(
            f'| {part_label} | [{title}](/blog-repo/{series}-guide-{part_label}-{slug}/) | {title} |'
        )
    lines.append("")
    lines.append("## 관련 링크")
    lines.append("")
    lines.append("- GitHub 저장소: (각 챕터 상단 `original_url` 참고)")
    lines.append("")
    return "\n".join(lines)


def render_card(*, series: str, badge_class: str, badge_label: str, title: str, desc: str) -> str:
    return textwrap.dedent(
        f"""\
          <div class="guide-card">
            <span class="badge {badge_class}">{badge_label}</span>
            <h3><a href="/blog-repo/{series}-guide/">{title}</a></h3>
            <p>{desc}</p>
          </div>
"""
    )


def insert_cards_into_guides(guides_path: Path, cards_by_section: dict[str, list[str]]) -> None:
    md = guides_path.read_text(encoding="utf-8")
    for section, cards in cards_by_section.items():
        if not cards:
            continue
        pattern = rf'(<h2 class="section-title">{re.escape(section)}</h2>\s*\n<div class="guide-grid">\s*\n)'
        m = re.search(pattern, md)
        if not m:
            raise SystemExit(f"ERROR: section not found in guides.md: {section}")
        insert_at = m.end(1)
        md = md[:insert_at] + "".join(cards) + md[insert_at:]
    guides_path.write_text(md, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Register GitHub Trending repos as new blog guide series.")
    parser.add_argument("--input", required=True, help="Path to enriched trending JSON.")
    parser.add_argument("--blog-root", default=".", help="Blog root (default: .). Must contain _posts/ and _tabs/.")
    parser.add_argument("--date", default="", help="Override date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print actions without writing files.")
    args = parser.parse_args()

    blog_root = Path(args.blog_root).resolve()
    posts_dir = blog_root / "_posts"
    tabs_dir = blog_root / "_tabs"
    guides_path = tabs_dir / "guides.md"
    if not posts_dir.is_dir():
        raise SystemExit(f"ERROR: _posts/ not found under {blog_root}")
    if not guides_path.is_file():
        raise SystemExit(f"ERROR: guides tab not found: {guides_path}")

    date_str = args.date.strip() or dt.date.today().isoformat()
    items = parse_items(Path(args.input).resolve())

    session = requests.Session()
    wrote_posts: list[Path] = []
    wrote_indexes: list[Path] = []
    cards_by_section: dict[str, list[str]] = {}

    for it in items:
        if it.already_registered:
            continue
        overrides = PROJECT_OVERRIDES.get(it.full_name, {})
        owner, repo = it.full_name.split("/", 1)
        series = it.series or choose_series(it.full_name)

        project_name = overrides.get("display_name") or repo
        author = overrides.get("author") or owner
        section = overrides.get("section") or "개발 도구"
        badge_class = overrides.get("badge_class") or "badge-tool"
        badge_label = overrides.get("badge_label") or "개발 도구"
        icon = overrides.get("icon") or "fas fa-book"
        card_desc = overrides.get("card_desc") or (it.description or "GitHub Trending 저장소") + " (5챕터)"

        readme = fetch_readme(session, it.full_name, timeout_s=float(args.timeout))
        readme_title = extract_title_from_readme(readme)
        if readme_title and len(readme_title) <= 60 and readme_title.lower() not in {"readme", "documentation"}:
            # Avoid titles like "ui" overriding "shadcn/ui"
            if project_name.lower() not in {"shadcn/ui", "openai skills (codex)"}:
                project_name = project_name

        install_cmds, run_cmds = extract_commands(readme)

        categories = [section, project_name.replace("/", "-")]
        base_tags = [project_name.replace("/", "-"), it.language or "GitHub", "GitHub Trending"]
        tags = base_tags[:]

        index_path = blog_root / f"{series}-guide.md"
        if not args.dry_run:
            idx_md = render_index(
                series=series,
                project_name=project_name,
                icon=icon,
                one_liner=(it.description or "GitHub Trending 프로젝트").strip(),
                intro_paragraph=f"**{project_name}**를 빠르게 훑고, 설치부터 활용/확장까지 핵심을 정리한 시리즈입니다.",
            )
            index_path.write_text(idx_md, encoding="utf-8")
            wrote_indexes.append(index_path)

        # guide card
        cards_by_section.setdefault(section, [])
        cards_by_section[section].append(
            render_card(
                series=series,
                badge_class=badge_class,
                badge_label=badge_label,
                title=project_name,
                desc=card_desc,
            )
        )

        # posts
        chapter_bodies: dict[int, str] = {}
        chapter_bodies[1] = textwrap.dedent(
            f"""\
            ## {project_name}란?

            GitHub Trending 기준으로 주목받는 **{it.full_name}**를 한국어로 정리합니다.

            - **한 줄 요약**: {it.description or '-'}
            - **언어**: {it.language or '-'}
            - **오늘 스타**: +{it.stars_today}
            - **원본**: {it.url}

            ---

            ## 이 가이드에서 다룰 것

            - 설치/실행 빠른 시작
            - 핵심 개념/구성요소
            - 자주 쓰는 사용 패턴
            - 운영/확장 시 체크리스트

            ---

            *다음 글에서는 설치 및 빠른 시작을 정리합니다.*
            """
        )

        install_block = "\n".join(f"- `{c}`" for c in install_cmds) if install_cmds else "- (README 기준 설치 명령을 확인하세요)"
        run_block = "\n".join(f"- `{c}`" for c in run_cmds) if run_cmds else "- (README 기준 실행/사용 예제를 확인하세요)"

        chapter_bodies[2] = textwrap.dedent(
            f"""\
            ## 요구사항 체크

            - OS: 프로젝트 문서(README) 기준
            - 런타임/툴체인: {it.language or '프로젝트 언어'} 생태계 표준 도구

            ---

            ## 설치

            {install_block}

            ---

            ## 실행/첫 사용

            {run_block}

            ---

            ## 팁

            - 설치/실행 단계에서 막히면, 우선 **README의 Quickstart/Usage 섹션**을 그대로 따라가세요.
            - 예제가 노트북 기반이면(예: Jupyter), 로컬 환경 대신 관리형 런타임(예: Colab/Vertex AI Workbench)도 고려하세요.

            ---

            *다음 글에서는 핵심 개념과 아키텍처를 정리합니다.*
            """
        )

        chapter_bodies[3] = textwrap.dedent(
            f"""\
            ## 핵심 개념(README 기반)

            {it.description or '프로젝트의 핵심 개념을 README를 기반으로 정리합니다.'}

            ---

            ## 구성요소 관점으로 보기

            아래는 “처음 접한 사람” 기준으로 빠르게 구조를 잡기 위한 관점입니다.

            1. **입력(Inputs)**: 데이터/프롬프트/설정 파일
            2. **코어(Core)**: 주요 알고리즘/엔진/워크플로우
            3. **인터페이스(Interface)**: CLI/API/노트북/UI
            4. **출력(Outputs)**: 결과물, 로그, 리포트

            ---

            ## 다음에 볼 것

            - README의 “Architecture/Design/How it works” 섹션
            - `docs/` 또는 `examples/` (있다면)
            - 설정 파일(예: `.env.example`, `config.*`) (있다면)

            ---

            *다음 글에서는 실전 사용 패턴을 정리합니다.*
            """
        )

        chapter_bodies[4] = textwrap.dedent(
            f"""\
            ## 실전 사용 패턴

            이 챕터는 “README의 예제”를 기반으로, 실제로 어디에 끼워 넣어 쓰는지에 초점을 둡니다.

            ---

            ## 패턴 1) 최소 실행 경로(MVP)

            1. 환경 준비 → 2. 예제 실행 → 3. 출력 확인 → 4. 파라미터 변경 → 5. 반복

            ---

            ## 패턴 2) 프로젝트에 통합

            - 리포지토리를 그대로 사용하기보다, 핵심 모듈/라이브러리만 가져와 **기존 코드베이스에 통합**하는 방식이 안정적일 때가 많습니다.
            - 외부 시스템 연동(클라우드, DB, 모델 제공자)이 있다면, 먼저 “인증/권한/비용”을 체크하세요.

            ---

            ## 체크리스트

            - 입력 데이터/설정은 재현 가능하게 버전 관리되는가?
            - 실행 결과를 비교할 수 있는 평가 지표가 있는가?
            - 실패 시 원인 파악이 가능한 로그가 남는가?

            ---

            *다음 글에서는 운영/확장 관점의 베스트 프랙티스를 정리합니다.*
            """
        )

        chapter_bodies[5] = textwrap.dedent(
            f"""\
            ## 운영/확장 체크리스트

            - **재현성**: 의존성 고정(락파일), 데이터 스냅샷, 실행 파라미터 기록
            - **관측성**: 로그/메트릭/트레이스(가능하면)로 실패 원인 추적
            - **보안**: 토큰/키는 `.env`/시크릿 관리로 분리, 결과물/로그에 민감정보가 섞이지 않게 필터
            - **비용**: API 호출/클라우드 런타임 비용을 측정하고 상한선을 둠

            ---

            ## 확장 아이디어

            - 예제(Example)부터 시작해, 작은 단위로 모듈화하여 확장하세요.
            - CLI/노트북이 있다면, 먼저 **자동화 가능한 인터페이스**(예: 스크립트/CI 잡)로 감싸면 운영이 쉬워집니다.

            ---

            ## 마무리

            이 시리즈는 GitHub Trending 스냅샷 기반 “빠른 온보딩”을 목표로 합니다. 더 깊은 내용은 원본 문서를 기준으로 업데이트하세요.
            """
        )

        for idx, (slug, chapter_title) in enumerate(CHAPTERS, start=1):
            part_label = f"{idx:02d}"
            post_name = f"{date_str}-{series}-guide-{part_label}-{slug}.md"
            post_path = posts_dir / post_name
            excerpt = (it.description or f"{project_name} 가이드").strip()
            if idx == 1:
                excerpt = f"{project_name} 프로젝트 소개와 핵심 포인트"
            elif idx == 2:
                excerpt = f"{project_name} 설치와 빠른 시작"
            elif idx == 3:
                excerpt = f"{project_name}의 핵심 개념과 구조"
            elif idx == 4:
                excerpt = f"{project_name} 실전 사용 패턴"
            elif idx == 5:
                excerpt = f"{project_name} 운영/확장 베스트 프랙티스"

            body = chapter_bodies.get(idx) or f"## {chapter_title}\n\n(작성 중)\n"
            md = render_post(
                date_str=date_str,
                series=series,
                part=idx,
                slug=slug,
                project_name=project_name,
                chapter_title=chapter_title,
                author=author,
                categories=categories,
                tags=tags,
                original_url=it.url,
                excerpt=excerpt,
                body=body,
            )
            if not args.dry_run:
                post_path.write_text(md, encoding="utf-8")
                wrote_posts.append(post_path)

    if not args.dry_run:
        insert_cards_into_guides(guides_path, cards_by_section)

    print(json.dumps(
        {
            "date": date_str,
            "wrote_posts": [str(p) for p in wrote_posts],
            "wrote_indexes": [str(p) for p in wrote_indexes],
            "updated_guides": str(guides_path) if not args.dry_run else None,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
