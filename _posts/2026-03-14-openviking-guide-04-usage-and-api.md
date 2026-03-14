---
layout: post
title: "OpenViking 완벽 가이드 (04) - 사용법 & API: Resources/Filesystem/Search/Sessions"
date: 2026-03-14
permalink: /openviking-guide-04-usage-and-api/
author: volcengine
categories: [AI 에이전트, OpenViking]
tags: [Trending, GitHub, OpenViking, API, CLI, HTTP]
original_url: "https://github.com/volcengine/OpenViking"
excerpt: "OpenViking의 API 표면을 문서(docs/en/api/*)와 구현(openviking/service/*, openviking_cli/client/*) 기준으로 연결해봅니다."
---

## 이 문서의 목적

- OpenViking을 “라이브러리/서버/CLI”로 사용할 때의 인터페이스를 한 장에 정리합니다.
- 문서 API 항목을 코드 파일로 추적할 때의 출발점을 제공합니다.

---

## 빠른 요약 (Docs 기반)

OpenViking의 API 문서는 크게 아래로 나뉩니다.

- 개요: `docs/en/api/01-overview.md`
- 리소스: `docs/en/api/02-resources.md`
- 파일시스템: `docs/en/api/03-filesystem.md`
- 스킬: `docs/en/api/04-skills.md`
- 세션: `docs/en/api/05-sessions.md`
- 검색: `docs/en/api/06-retrieval.md`
- 시스템/관리: `docs/en/api/07-system.md`, `docs/en/api/08-admin.md`

---

## 1) Python SDK(로컬/서버 공통 관점)

문서 퀵스타트에 등장하는 핵심 작업은 아래와 같은 동작군으로 묶을 수 있습니다. (`docs/en/getting-started/02-quickstart.md`)

- 리소스 추가/삭제/상태: `add_resource`, `wait_processed` 등
- FS 탐색/읽기: `ls`, `glob`, `read`
- 요약/개요: `abstract`, `overview`
- 검색: `find`

코드 탐색 출발점:

- 클라이언트: `openviking/client.py`, `openviking/sync_client.py`, `openviking/async_client.py`
- HTTP 클라이언트(서버 모드): `openviking_cli/client/sync_http.py`, `openviking_cli/client/http.py`

---

## 2) HTTP 서버(엔드포인트 예시)

서버 퀵스타트 문서에는 아래와 같은 엔드포인트 예시가 등장합니다. (`docs/en/getting-started/03-quickstart-server.md`)

- 상태 확인: `GET /health`
- 리소스 추가: `POST /api/v1/resources`
- FS ls: `GET /api/v1/fs/ls?uri=...`
- 검색: `POST /api/v1/search/find`

> 정확한 스키마/파라미터는 `docs/en/api/*`를 1차 기준으로 보고, 실제 구현은 서버 라우팅 코드(서버 부트스트랩/서비스 레이어)에서 확인하세요.

---

## 3) CLI(서버 관측/리소스 추가/검색)

서버 모드에서 CLI는 “관측/추가/탐색/검색” 루프를 빠르게 돌리기 위한 도구로 소개됩니다. (`docs/en/getting-started/03-quickstart-server.md`)

코드 근거(진입점):

- Python 래퍼: `openviking_cli/rust_cli.py`
- Rust CLI: `crates/ov_cli/src/main.rs`
- (문서 기준) 예시 커맨드: `openviking observer system`, `openviking add-resource ...`, `openviking find ...`

---

## 근거(파일/경로)

- API 문서: `docs/en/api/*`
- 서버 퀵스타트: `docs/en/getting-started/03-quickstart-server.md`
- 클라이언트: `openviking/client.py`, `openviking/sync_client.py`, `openviking/async_client.py`
- HTTP 클라이언트: `openviking_cli/client/*`
- 서버 부트스트랩: `openviking_cli/server_bootstrap.py`
- CLI: `openviking_cli/rust_cli.py`, `crates/ov_cli/*`

---

## 주의사항/함정

- 문서의 엔드포인트 경로와 실제 서비스 구현이 변경될 수 있습니다. 운영 중이라면, 릴리스/체인지로그(`docs/en/about/02-changelog.md`)도 함께 확인하세요.
- “서버 모드 + CLI”는 설정 파일이 2개로 나뉩니다: 모델 설정(`~/.openviking/ov.conf`)과 CLI 서버 연결 설정(`~/.openviking/ovcli.conf`). (`docs/en/getting-started/02-quickstart.md`, `docs/en/getting-started/03-quickstart-server.md`)

---

## TODO/확인 필요

- `openviking_cli/client/*`가 어떤 엔드포인트를 호출하는지(경로/메서드) 표로 정리하기
- `docs/en/api/04-skills.md`(skills)와 `examples/skills/*` 연결(실제 파일/예제) 강화

---

## 위키 링크

- `[[OpenViking Guide - Index]]` → [가이드 목차](/blog-repo/openviking-guide/)
- `[[OpenViking Guide - Ops]]` → [05. 운영/확장/트러블슈팅](/blog-repo/openviking-guide-05-ops-extensions-troubleshooting/)

---

*다음 글에서는 배포/모니터링/인증/MCP 연동과 예제 디렉토리를 중심으로 운영 관점을 정리합니다.*

