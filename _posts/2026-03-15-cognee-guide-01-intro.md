---
layout: post
title: "Cognee 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-15
permalink: /cognee-guide-01-intro/
author: topoteretes
categories: [GitHub Trending, cognee]
tags: [Trending, GitHub, cognee, AgentMemory, KnowledgeGraph, VectorSearch]
original_url: "https://github.com/topoteretes/cognee"
excerpt: "Cognee의 목표(지식 엔진 기반 에이전트 메모리), 핵심 파이프라인(add→cognify→search), 제공 형태(라이브러리/CLI/API/MCP)를 레포 근거로 개관합니다."
---

## 이 문서의 목적

- Cognee가 제공하는 “에이전트 메모리”가 무엇인지와, 어떤 형태로 쓸 수 있는지(라이브러리/CLI/API/MCP)를 근거 기반으로 정리합니다. (`README.md`, `pyproject.toml`, `cognee-mcp/README.md`)
- 이후 챕터에서 따라갈 최소 흐름을 고정합니다: `add` → `cognify` → `search`. (`README.md`)

---

## 빠른 요약(README/pyproject.toml 기반)

- **핵심 파이프라인(README)**: 비정형 데이터 ingest → knowledge engine에 적재/학습 → 검색/질의 (`README.md`)
- **파이썬 버전(README)**: Python `3.10 ~ 3.13` (`README.md`)
- **CLI 엔트리(근거)**: `pyproject.toml`의 `[project.scripts]`에 `cognee-cli = "cognee.cli._cognee:main"` (`pyproject.toml`)
- **API 의존성(근거)**: `pyproject.toml` 의존성에 `fastapi`, `uvicorn`, `gunicorn` 등이 포함 (`pyproject.toml`)
- **MCP 서버(근거)**: `cognee-mcp/` 하위에 별도 README 및 실행/트랜스포트(stdio/sse/http) 안내 (`cognee-mcp/README.md`)

---

## 구성요소 지도(레포 구조 기반)

```text
cognee/
├── cognee/                  # 라이브러리 본체(파이프라인/모듈/CLI/API)
├── cognee/cli/              # CLI 구현 (pyproject.toml scripts)
├── cognee/api/              # FastAPI 기반 API(의존성 근거)
├── cognee-mcp/              # MCP 서버(별도 패키지/README)
├── cognee-frontend/         # 프론트엔드(별도)
├── examples/                # 예제/가이드
└── deployment/              # 배포(helm 등)
```

근거: 디렉토리 존재, `pyproject.toml`, `cognee-mcp/README.md`

---

## 상위 아키텍처(개념도)

README의 코드 예시와 CLI 명령을 기준으로, Cognee의 상위 흐름은 아래처럼 요약할 수 있습니다. (`README.md`)

```mermaid
flowchart LR
  D[Documents / Text / Files] --> A[cognee.add(...)]
  A --> P[cognee.cognify()]
  P --> S[(Knowledge Engine\nvector + graph)]
  Q[cognee.search(...)] --> S
  S --> R[Results]
```

---

## 근거(파일/경로)

- 프로젝트 소개/퀵스타트/CLI 명령: `README.md`
- 한국어 문서(레포 포함): `README_ko.md`
- CLI 엔트리/의존성/옵션 deps: `pyproject.toml`
- MCP 서버 안내: `cognee-mcp/README.md`

---

## 주의사항/함정

- LLM API 키/프로바이더 설정이 필요하며, README는 `.env.template` 활용을 안내합니다. (`README.md`, `.env.template`)
- optional dependency 그룹이 많아(예: postgres/neo4j/redis/monitoring 등) “내가 쓰는 기능에 필요한 것만” 설치 전략을 세우는 게 중요합니다. (`pyproject.toml`)

---

## TODO/확인 필요

- 기본 스토리지(로컬)에서 실제로 어떤 DB/파일 레이아웃이 생성되는지 `cognee/` 구현에서 확인
- `cognee.cognify()`가 수행하는 단계(청킹/임베딩/그래프 적재 등)를 코드로 분해해 문서화

---

## 위키 링크

- `[[Cognee Guide - Index]]` → [가이드 목차](/blog-repo/cognee-guide/)
- `[[Cognee Guide - Quickstart]]` → [02. 설치 및 빠른 시작](/blog-repo/cognee-guide-02-install-and-quickstart/)

---

*다음 글에서는 README의 `uv pip install cognee` 예시와 `cognee-cli` 명령을 따라, 최소 파이프라인을 실제로 돌리는 방법을 정리합니다.*

