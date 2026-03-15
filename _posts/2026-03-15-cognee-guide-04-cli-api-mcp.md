---
layout: post
title: "Cognee 완벽 가이드 (04) - CLI/API/MCP 사용법"
date: 2026-03-15
permalink: /cognee-guide-04-cli-api-mcp/
author: topoteretes
categories: [GitHub Trending, cognee]
tags: [Trending, GitHub, cognee, CLI, FastAPI, MCP]
original_url: "https://github.com/topoteretes/cognee"
excerpt: "`pyproject.toml`의 scripts(`cognee-cli`)와 `cognee-mcp/README.md`의 Quick Start를 근거로, Cognee를 CLI/API/MCP 형태로 사용하는 방법을 정리합니다."
---

## 이 문서의 목적

- Cognee의 “인터페이스 3종(CLI / API / MCP)”을 각각 어떤 파일/명령으로 시작하는지 근거 기반으로 정리합니다.
- MCP 서버의 트랜스포트(stdio/sse/http) 및 Docker 실행 예시를 `cognee-mcp/README.md` 기준으로 정리합니다.

---

## 빠른 요약(근거)

- CLI 엔트리: `pyproject.toml`의 `[project.scripts]` → `cognee-cli = "cognee.cli._cognee:main"`
- CLI 기본 명령 예시: `cognee-cli add`, `cognee-cli cognify`, `cognee-cli search` (`README.md`)
- MCP 서버: `cognee-mcp/src/server.py`를 실행하며, `--transport {stdio,sse,http}` 옵션을 안내 (`cognee-mcp/README.md`)

---

## 1) CLI: `cognee-cli`

엔트리 근거: `pyproject.toml`

README 예시(근거): (`README.md`)

```bash
cognee-cli add \"Cognee turns documents into AI memory.\"
cognee-cli cognify
cognee-cli search \"What does Cognee do?\"
```

CLI 구현 위치 후보(근거): `cognee/cli/_cognee.py`, `cognee/cli/commands/*`

---

## 2) API(개념)

`pyproject.toml` 의존성에 `fastapi`가 포함되어 있고, `cognee/api/` 디렉토리가 존재합니다. (`pyproject.toml`, `cognee/api/`)

> 구체적인 실행 엔트리(모듈/명령)는 `cognee/api/` 내부 구현을 확인해 확정하는 것이 안전합니다.

---

## 3) MCP 서버: `cognee-mcp`

`cognee-mcp/README.md`의 Quick Start(근거):

```bash
cd cognee/cognee-mcp
pip install uv
uv sync --dev --all-extras --reinstall
source .venv/bin/activate
LLM_API_KEY=\"YOUR_OPENAI_API_KEY\"
python src/server.py
```

트랜스포트(근거): (`cognee-mcp/README.md`)

- stdio(기본): `python src/server.py`
- SSE: `python src/server.py --transport sse`
- HTTP: `python src/server.py --transport http --host 127.0.0.1 --port 8000 --path /mcp`

Docker 실행 예시도 `cognee-mcp/README.md`에 포함되어 있습니다.

---

## 근거(파일/경로)

- CLI scripts: `pyproject.toml`
- CLI 사용 예시: `README.md`
- CLI 구현: `cognee/cli/_cognee.py`, `cognee/cli/commands/*`
- MCP Quick Start/트랜스포트/Docker: `cognee-mcp/README.md`, `cognee-mcp/src/server.py`

---

## 주의사항/함정

- `cognee-mcp`는 `.env`/환경 변수(`LLM_API_KEY`) 설정이 선행되어야 하며, 설정 키/프로바이더는 환경에 따라 달라질 수 있습니다. (`cognee-mcp/README.md`, `.env.template`)
- Docker 실행은 “환경 변수 기반” 설정을 강조합니다(README). (`cognee-mcp/README.md`)

---

## TODO/확인 필요

- API 모드의 공식 엔트리포인트(실행 명령)와 인증 방식 확인 (`cognee/api/`)
- MCP 서버가 노출하는 tool 목록/스키마 확인 (`cognee-mcp/src/*`)

---

## 위키 링크

- `[[Cognee Guide - Index]]` → [가이드 목차](/blog-repo/cognee-guide/)
- `[[Cognee Guide - Advanced]]` → [05. 고급 설정·문제해결·자동화](/blog-repo/cognee-guide-05-advanced-troubleshooting-doc-automation/)

---

*다음 글에서는 optional dependencies와 Docker 배포 옵션을 정리하고, “살아있는지”를 점검하는 자동화 예시를 제공합니다.*

