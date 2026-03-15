---
layout: post
title: "Cognee 완벽 가이드 (05) - 고급 설정·문제해결·문서 점검 자동화"
date: 2026-03-15
permalink: /cognee-guide-05-advanced-troubleshooting-doc-automation/
author: topoteretes
categories: [GitHub Trending, cognee]
tags: [Trending, GitHub, cognee, Docker, Troubleshooting, Automation]
original_url: "https://github.com/topoteretes/cognee"
excerpt: "`pyproject.toml`의 optional dependencies와 `cognee-mcp/README.md`의 Docker/API 모드 설명을 근거로, 기능별 설치 전략·배포 옵션·점검 자동화 힌트를 정리합니다."
---

## 이 문서의 목적

- Cognee는 optional dependency 그룹이 매우 많으므로, “필요한 기능만 설치”하는 전략을 세울 수 있게 근거를 제공합니다. (`pyproject.toml`)
- MCP/Docker 운영 시 자주 부딪히는 포인트(환경 변수/트랜스포트)를 정리하고, 점검 자동화 예시를 제공합니다. (`cognee-mcp/README.md`)

---

## 빠른 요약(근거)

- optional dependency 그룹: `pyproject.toml`의 `[project.optional-dependencies]` (예: `postgres`, `neo4j`, `redis`, `monitoring`, `scraping`, `distributed` 등)
- MCP Docker 실행 시 “환경 변수 기반”을 강조하고, `TRANSPORT_MODE`/`EXTRAS` 등을 안내합니다. (`cognee-mcp/README.md`)

---

## 기능별 설치 전략(근거: optional deps)

`pyproject.toml`의 optional deps는 “기능 단위”로 묶여 있습니다. 예:

- `postgres` / `postgres-binary`
- `neo4j`, `neptune`
- `redis`
- `monitoring` (Sentry/Langfuse/OTEL 관련)
- `scraping` (playwright 등)
- `distributed` (modal)

근거: `pyproject.toml`

---

## MCP Docker 운영 포인트(근거)

`cognee-mcp/README.md`는 Docker 실행 시:

- HTTP transport 권장 예시
- `EXTRAS` 환경 변수로 optional deps를 런타임 설치하는 예시
- “Docker는 CLI 인자 대신 env vars”라는 주의

를 포함합니다. (`cognee-mcp/README.md`)

---

## 문서 점검 자동화(예시)

아래는 “MCP 서버가 떠 있는지”를 확인하는 아주 작은 점검 예시입니다.

> 주의: 실제 엔드포인트/포트는 `--transport http --port ... --path ...`에 따라 달라집니다. (`cognee-mcp/README.md`)

```bash
# 예: HTTP transport로 /mcp 경로에 띄운 경우
curl -sS http://127.0.0.1:8000/mcp | head -c 200 || true
```

또는 “CLI 엔트리 존재”를 최소로 확인:

```bash
python -c \"import cognee; print('ok')\"
```

---

## 근거(파일/경로)

- optional deps/CLI scripts/의존성: `pyproject.toml`
- MCP 서버 운영/Docker/API 모드: `cognee-mcp/README.md`
- MCP 서버 코드: `cognee-mcp/src/server.py`
- Docker compose: `docker-compose.yml`

---

## 주의사항/함정

- optional deps를 과도하게 설치하면 빌드/이미지 크기/보안 표면이 커질 수 있습니다. 필요한 기능부터 점진적으로 확장하는 전략이 안전합니다. (`pyproject.toml`)

---

## TODO/확인 필요

- `cognee/cli/_cognee.py` 기준으로 CLI 옵션/환경 변수 표 만들기
- FastAPI 서버를 실제로 띄우는 공식 명령/모듈 확인 후, “API 모드 점검” 절차 추가 (`cognee/api/`)

---

## 위키 링크

- `[[Cognee Guide - Index]]` → [가이드 목차](/blog-repo/cognee-guide/)
- `[[Cognee Guide - CLI/API/MCP]]` → [04. CLI/API/MCP 사용법](/blog-repo/cognee-guide-04-cli-api-mcp/)

---

*이 시리즈는 README/pyproject.toml 근거로 “표면과 구조”를 먼저 고정했습니다. 다음 확장 단계는 `cognee/` 구현을 근거로 파이프라인을 더 세밀하게 해설하는 것입니다.*

