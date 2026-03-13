---
layout: post
title: "Hindsight 완벽 가이드 (02) - 구성요소 맵"
date: 2026-03-13
permalink: /hindsight-guide-02-components/
author: vectorize-io
categories: [AI 에이전트, hindsight]
tags: [Trending, GitHub, hindsight, Memory, API, CLI, GitHub Trending]
original_url: "https://github.com/vectorize-io/hindsight"
excerpt: "레포 디렉토리 구조(hindsight-api/cli/clients/control-plane 등)와 README의 통합 설명을 근거로 구성요소를 지도 형태로 정리합니다."
---

## 이 문서의 목적

- Hindsight를 “어떤 서비스/패키지로 나뉘는지” 먼저 파악해, 필요한 부분만 빠르게 들어갈 수 있게 합니다.
- API/CLI/클라이언트/배포 구성(helm/docker)을 한 번에 연결합니다.

---

## 빠른 요약

레포에는 구성요소가 명시적으로 분리된 디렉토리로 존재합니다.

- `hindsight-api/`: API 서버(파이썬 프로젝트로 보임; 별도 `pyproject.toml` 존재)
- `hindsight-cli/`: CLI (`Cargo.toml` 존재 → Rust)
- `hindsight-clients/`: SDK/클라이언트 묶음
- `hindsight-control-plane/`: 컨트롤 플레인(별도 `package.json`)
- `helm/`: Helm 차트(쿠버네티스 배포)
- `docker/docker-compose/`: 외부 Postgres 포함 docker compose 실행 경로(README에 안내)
- `hindsight-docs/`: 문서 사이트(정적 리소스/이미지 포함)

근거:
- 레포 디렉토리/파일 존재
- `README.md`의 “API/SDK/CLI” 언급

---

## 1) “사용자 관점”으로 본 흐름

Hindsight README는 두 가지 통합 방식(고수준)을 제시합니다. (`README.md`)

1) **LLM Wrapper**: 기존 LLM 클라이언트를 wrapper로 교체해 자동 retain/recall(2줄 통합)
2) **API/SDK 직접 통합**: retain/recall/reflect를 필요할 때 호출

이 둘은 내부적으로 같은 서버(또는 동일 기능 집합)로 수렴합니다.

---

## 2) 런타임 구성(포트/서비스)

README Quick Start 기준:

- API: `http://localhost:8888`
- UI: `http://localhost:9999`

근거:
- `README.md` Quick Start 섹션

---

## 3) 아키텍처 경계(개략)

```mermaid
flowchart TB
  subgraph Clients[Integration Surface]
    SDK[Python/NPM client]
    CLI[hindsight-cli]
    Wrapper[LLM Wrapper]
  end

  subgraph Runtime[Hindsight Runtime]
    API[hindsight-api (8888)]
    UI[UI (9999)]
    DB[(PostgreSQL / embedded pg0)]
  end

  SDK --> API
  CLI --> API
  Wrapper --> API
  UI --> API
  API --> DB
```

---

## 주의사항/함정

- 레포가 모노레포 성격이라 “어디를 수정해야 기능이 바뀌는지”가 처음엔 헷갈릴 수 있습니다. 보통은 `hindsight-api/`(서버)와 `hindsight-clients/`(SDK)부터 잡는 게 안전합니다.
- `docker/docker-compose/` 경로는 “external PostgreSQL” 실행을 위한 별도 루트로 README에 언급됩니다.

---

## TODO / 확인 필요

- UI 구현의 정확한 위치(코드 디렉토리)와 API 라우트 구조는 `hindsight-api/` 내부를 읽고 “엔드포인트/스키마”로 확정하는 것이 좋습니다(이 챕터는 컴포넌트 맵 중심).

---

## 위키 링크

- `[[Hindsight Guide - Index]]` → [가이드 목차](/blog-repo/hindsight-guide/)
- `[[Hindsight Guide - Quickstart]]` → [03. 빠른 시작(로컬)](/blog-repo/hindsight-guide-03-quickstart/)
- `[[Hindsight Guide - Memory Design]]` → [04. 메모리 설계/데이터 흐름](/blog-repo/hindsight-guide-04-memory-design/)

---

*다음 글에서는 README의 Docker quick start를 근거로 가장 빠르게 띄우는 방법을 정리합니다.*

