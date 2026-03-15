---
layout: post
title: "Project N.O.M.A.D. 완벽 가이드 (03) - 아키텍처(컨테이너 오케스트레이션)"
date: 2026-03-15
permalink: /project-nomad-guide-03-architecture/
author: Crosstalk-Solutions
categories: [GitHub Trending, project-nomad]
tags: [Trending, GitHub, project-nomad, Architecture, AdonisJS, Inertia, Docker]
original_url: "https://github.com/Crosstalk-Solutions/project-nomad"
excerpt: "`admin/`의 라우트/API 그룹과 의존성(dockerode, bullmq 등)을 근거로, Command Center가 Docker 기반 도구 묶음을 오케스트레이션하는 구조를 그립니다."
---

## 이 문서의 목적

- README가 말하는 “Command Center + API가 컨테이너화된 도구를 오케스트레이션”을 코드 근거로 분해합니다. (`README.md`, `admin/start/routes.ts`, `admin/package.json`)
- 운영 관점에서 “어떤 기능이 어떤 API 그룹”으로 노출되는지 큰 틀을 잡습니다. (`admin/start/routes.ts`)

---

## 빠른 요약(코드 구조 기반)

- Command Center(웹 UI + API)는 `admin/`에 있고, 라우트는 `admin/start/routes.ts`가 한 곳에서 모읍니다.
- `admin/package.json` 의존성에 `dockerode`, `bullmq`, `@adonisjs/*`, `@inertiajs/react`, `react`, `vite` 등이 포함되어, “서버(AdonisJS) + UI(React/Inertia) + Docker 제어 + 백그라운드 작업(큐)” 결합 구조임을 시사합니다. (`admin/package.json`)
- 기능은 API 그룹으로 분리되어 있습니다: `/api/system`, `/api/ollama`, `/api/rag`, `/api/zim`, `/api/maps`, `/api/benchmark`, `/api/chat/sessions` 등. (`admin/start/routes.ts`)

---

## 레포 구조(상위)

```text
project-nomad/
├── install/                  # 설치/업데이트 스크립트/사이드카
├── admin/                    # Command Center (AdonisJS + Inertia/React + API)
│   ├── start/routes.ts        # HTTP 라우트(페이지 + API)
│   ├── app/controllers/       # API 컨트롤러
│   ├── app/jobs/              # 큐/백그라운드 작업(추정: bullmq)
│   ├── database/              # DB/마이그레이션(루시드)
│   └── resources/             # 프론트엔드 리소스(React)
└── collections/              # 큐레이션 콘텐츠 컬렉션
```

근거: 디렉토리 존재 및 라우트 파일(`admin/start/routes.ts`), 의존성(`admin/package.json`)

---

## 기능→API 그룹 매핑(라우트 근거)

`admin/start/routes.ts` 기준으로, N.O.M.A.D.의 주요 기능은 대략 아래처럼 그룹화되어 있습니다.

- **System/Services/Update**: `/api/system/*`
  - 예: 서비스 목록/설치/업데이트/강제 재설치 등 (`/api/system/services/*`)
- **Ollama/모델 관리**: `/api/ollama/*`
  - 예: `/api/ollama/models`, `/api/ollama/installed-models`
- **RAG(문서 업로드/동기화/잡 상태)**: `/api/rag/*`
  - 예: `/api/rag/upload`, `/api/rag/job-status`
- **오프라인 위키(ZIM)**: `/api/zim/*`
  - 예: `/api/zim/wikipedia/select`
- **Maps(지역/스타일/컬렉션 다운로드)**: `/api/maps/*`
- **Benchmark**: `/api/benchmark/*`
- **Chat 세션/메시지**: `/api/chat/sessions/*`

근거: `admin/start/routes.ts`

---

## 컴포넌트/데이터 흐름(개념도)

아래는 라우트 그룹과 “Docker 오케스트레이션” 설명(README)을 합친 개념 흐름입니다. (`README.md`, `admin/start/routes.ts`)

```mermaid
flowchart TB
  U[Browser] -->|:8080| UI[Inertia/React Pages\n/admin/routes: /, /chat, /maps, ...]
  UI --> API[HTTP API\n/admin/start/routes.ts]

  API --> SYS[SystemController\n/api/system/*]
  API --> RAG[RagController\n/api/rag/*]
  API --> OLL[OllamaController\n/api/ollama/*]
  API --> ZIM[ZimController\n/api/zim/*]
  API --> MAP[MapsController\n/api/maps/*]
  API --> BEN[BenchmarkController\n/api/benchmark/*]

  SYS --> DOCKER[(Docker Engine)]
  DOCKER --> SVC[Service Containers\n(Kiwix/Kolibri/Ollama/Qdrant/...)]
```

주의: 각 컨트롤러 내부 구현(예: dockerode 호출 위치)은 이 챕터에서는 라우트/의존성 근거로만 설명하고, 세부는 다음 챕터에서 필요한 범위로 좁혀 확인합니다.

---

## 근거(파일/경로)

- 전체 개요/오케스트레이션 설명: `README.md`
- HTTP 라우팅(기능 그룹 근거): `admin/start/routes.ts`
- 기술 스택/의존성 근거: `admin/package.json`

---

## 주의사항/함정

- “설치 스크립트가 무엇을 세팅하는가”는 실제 스크립트(`install/install_nomad.sh`)를 읽어야 정확합니다(이 챕터는 라우트 중심 개관). (`install/install_nomad.sh`)
- `/api/system/services/*`는 강력한 제어 권한을 의미할 수 있어, 외부 노출을 피하는 README 권고와 함께 봐야 합니다. (`README.md`, `admin/start/routes.ts`)

---

## TODO/확인 필요

- `admin/app/controllers/*`에서 dockerode 사용 지점을 확인하고 “서비스 설치/업데이트”가 어떤 명령/이미지로 매핑되는지 문서화
- `collections/`의 포맷(매니페스트)과 `/api/manifests/refresh`의 입력/출력 관계를 추적 (`admin/start/routes.ts`)

---

## 위키 링크

- `[[Project NOMAD Guide - Index]]` → [가이드 목차](/blog-repo/project-nomad-guide/)
- `[[Project NOMAD Guide - API & Ops]]` → [04. API/운영](/blog-repo/project-nomad-guide-04-api-and-operations/)

---

*다음 글에서는 `admin/start/routes.ts`의 API 그룹을 운영 시나리오(서비스 설치/업데이트, RAG 업로드, 모델 관리)로 엮어, 실제로 어떤 호출 순서로 쓰는지 예시를 제공합니다.*

