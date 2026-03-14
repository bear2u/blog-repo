---
layout: post
title: "OpenViking 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-14
permalink: /openviking-guide-01-intro/
author: volcengine
categories: [AI 에이전트, OpenViking]
tags: [Trending, GitHub, OpenViking, ContextDB, Agent Memory]
original_url: "https://github.com/volcengine/OpenViking"
excerpt: "OpenViking의 목표(컨텍스트 DB), 파일시스템 패러다임, L0/L1/L2 계층 로딩 개념을 레포/문서 구조 기반으로 정리합니다."
---

## OpenViking이란?

GitHub Trending(daily, 2026-03-14 기준) 상위에 오른 **volcengine/OpenViking**를 한국어로 정리합니다.

- **한 줄 요약(README 기반)**: AI 에이전트를 위한 “컨텍스트 데이터베이스”로, 메모리/리소스/스킬을 파일 시스템 패러다임으로 통합 관리합니다. (`README.md`)
- **문서(개념)**: 아키텍처/컨텍스트 타입/레이어/검색/세션이 `docs/en/concepts/*`로 분리되어 있습니다.

---

## 이 문서의 목적

- OpenViking의 “무엇(문제/해결)”과 “어디에 무엇이 있는지(레포 지도)”를 빠르게 잡습니다.
- 다음 챕터(설치/퀵스타트)로 이어질 수 있도록, 실행 주체(Python SDK/서버/CLI)와 문서 축을 정리합니다.

---

## 빠른 요약 (README/Docs 기반)

- OpenViking은 기존 RAG의 “평면 벡터 저장” 중심 접근 대신, **디렉토리/계층 기반 컨텍스트 조직**을 강조합니다. (`README.md`, `docs/en/concepts/03-context-layers.md`)
- L0/L1/L2 같은 계층 로딩으로 “필요한 것만 로딩”하는 비용 최적화를 이야기합니다. (`README.md`, `docs/en/concepts/03-context-layers.md`)
- 서버 모드에서는 `openviking-server`로 HTTP 서버를 띄우고, 클라이언트(SDK/CLI/curl)가 API에 연결합니다. (`docs/en/getting-started/03-quickstart-server.md`, `openviking_cli/server_bootstrap.py`)

---

## 레포 구조(상위)

```text
OpenViking/
  README.md
  pyproject.toml
  openviking/              # Python SDK/서비스 로직
  openviking_cli/          # CLI/HTTP 클라이언트/서버 bootstrap
  crates/ov_cli/           # Rust CLI (bin: ov)
  docs/                    # 개념/가이드/API 문서
  examples/                # MCP/플러그인/클라우드/서버-클라이언트 예제
  third_party/             # AGFS/leveldb 등 서브모듈/벤더
  tests/                   # 테스트/통합/벤치
  bot/                     # vikingbot 관련 코드/문서(별도 파트)
```

---

## (개략) 구성요소 지도

```mermaid
flowchart LR
  Agent[Agent / App] -->|SDK(Local)| Local[openviking/*\nLocal client]
  Agent -->|HTTP| Server[openviking-server\nopenviking_cli/server_bootstrap.py]
  Agent -->|CLI| CLI[openviking/ov\nopenviking_cli/rust_cli.py\ncrates/ov_cli]
  Server --> Svc[openviking/service/*]
  Local --> Svc
  Svc --> Parse[openviking/parse/*]
  Svc --> Retrieve[openviking/retrieve/*]
  Svc --> FS[openviking/service/fs_service.py]
  Svc --> Storage[openviking/service/resource_service.py]
  Svc --> Native[openviking/pyagfs + third_party/agfs]
```

> 위 도식은 “문서 읽기/코드 탐색의 출발점”을 위한 지도입니다. 실제 호출 관계는 각 모듈의 진입점(예: `openviking_cli/server_bootstrap.py`, `openviking/service/core.py`) 기준으로 확인하는 것이 안전합니다.

---

## 근거(파일/경로)

- 프로젝트/문제정의/퀵스타트: `README.md`, `docs/en/getting-started/01-introduction.md`
- 컨셉(아키텍처/레이어/검색/세션): `docs/en/concepts/*`
- 패키징/엔트리포인트: `pyproject.toml` (`[project.scripts]`)
- 서버 부트스트랩: `openviking_cli/server_bootstrap.py`
- 서비스/FS/리소스: `openviking/service/*`
- 파서/트리 빌딩: `openviking/parse/*`
- 검색/리트리버: `openviking/retrieve/*`
- Rust CLI: `crates/ov_cli/*`

---

## 주의사항/함정

- OpenViking은 모델(VLM/Embedding) 설정이 전제라서, 설정 파일(`~/.openviking/ov.conf`) 준비가 핵심 병목이 됩니다. (`docs/en/getting-started/02-quickstart.md`)
- 멀티언어(파이썬 + Rust CLI + 네이티브 확장/서드파티)가 섞여 있어, “어떤 실행 모드”를 선택했는지 먼저 구분해야 탐색이 쉬워집니다. (`pyproject.toml`, `crates/ov_cli/*`, `src/CMakeLists.txt`)

---

## TODO/확인 필요

- `openviking`(Python SDK)에서 “로컬 모드”와 “서버 모드”의 내부 분기/공통 인터페이스를 `openviking/client/*` 기준으로 더 정확히 확인하기
- `third_party/agfs` 및 `openviking/pyagfs/*`의 책임 경계(파일시스템 패러다임 구현 범위) 명확화

---

## 위키 링크

- `[[OpenViking Guide - Index]]` → [가이드 목차](/blog-repo/openviking-guide/)
- `[[OpenViking Guide - Install & Quickstart]]` → [02. 설치 및 빠른 시작](/blog-repo/openviking-guide-02-install-and-quickstart/)

---

*다음 글에서는 `docs/en/getting-started/*`를 기준으로 설치/설정/첫 실행 루트를 정리합니다.*

