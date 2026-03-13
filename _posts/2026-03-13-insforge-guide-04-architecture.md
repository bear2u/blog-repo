---
layout: post
title: "InsForge 완벽 가이드 (04) - 구성요소/아키텍처"
date: 2026-03-13
permalink: /insforge-guide-04-architecture/
author: InsForge
categories: [개발 도구, insforge]
tags: [Trending, GitHub, insforge, Architecture, PostgREST, Deno, GitHub Trending]
original_url: "https://github.com/InsForge/InsForge"
excerpt: "docker-compose*.yml과 레포 디렉토리(backend/frontend/auth/functions)를 근거로 InsForge 구성요소 경계를 정리합니다."
---

## 이 문서의 목적

- InsForge를 “모노레포 구성요소 + 런타임 서비스”로 나눠서 이해할 수 있게 구조도를 제공합니다.
- 어떤 기능(인증/DB/스토리지/함수/로그)이 어떤 컨테이너/디렉토리에 대응하는지 연결합니다.

---

## 빠른 요약

레포 구조(디렉토리 기준):

- `backend/`: 백엔드 API(Express 기반, 라우트/서비스/프로바이더)
- `frontend/`: 대시보드(UI)
- `auth/`: 인증 관련 프론트 앱(별도 포트로 노출)
- `functions/`: Deno 기반 서버리스 함수(런타임은 `deno` 서비스)
- `shared-schemas/`, `ui/`: 공유 스키마/컴포넌트

런타임(Compose 기준):

- `postgres` + `postgrest` + `insforge` + `deno` (+ `vector`)

근거:
- 레포 디렉토리 구조
- `docker-compose.yml`, `docker-compose.prod.yml`

---

## 1) 데이터 계층: Postgres + PostgREST

`docker-compose.yml`/`docker-compose.prod.yml`은:

- `postgres` 컨테이너를 기동하고
- `postgrest`가 `PGRST_DB_URI`로 Postgres에 연결하여 HTTP API를 제공합니다.

근거:
- `docker-compose.yml`, `docker-compose.prod.yml`

---

## 2) 백엔드/대시보드/인증 앱

`insforge` 서비스는 다음 포트를 노출합니다.

- `7130`: Backend API
- `7131`: Dashboard
- `7132`: Auth 앱

근거:
- `docker-compose.yml`, `docker-compose.prod.yml`

개발 compose에서는 소스 디렉토리/노드모듈을 다수 볼륨으로 마운트하여 “컨테이너 내부에서 dev 서버 실행” 형태를 취합니다. (`docker-compose.yml`)

---

## 3) Functions: Deno 런타임

`docker-compose.prod.yml`에는 `deno` 서비스가 있으며:

- `functions/server.ts`를 실행하고
- `functions/worker-template.js`와 함께 Deno 서브호스팅 기반 워커 형태를 암시합니다.

근거:
- `docker-compose.prod.yml`
- `functions/server.ts`, `functions/worker-template.js` 존재

---

## 4) 관측/로그: Vector

`docker-compose.prod.yml`은 `vector` 컨테이너를 포함하며:

- `deploy/docker-init/logs/vector.yml`을 설정으로 사용하고
- 도커 소켓(`/var/run/docker.sock`) 및 공유 로그 볼륨을 마운트합니다.

근거:
- `docker-compose.prod.yml`

---

## 아키텍처 다이어그램(컨테이너 경계)

```mermaid
flowchart LR
  subgraph Core[Core Services]
    PG[(postgres)]
    PGR[postgrest]
    API[insforge (7130)]
    UI[dashboard (7131)]
    AUTH[auth app (7132)]
  end

  subgraph Fn[Functions Runtime]
    DENO[deno (7133)]
  end

  subgraph Obs[Observability]
    VEC[vector]
  end

  API --> PG
  PGR --> PG
  API --> PGR
  UI --> API
  AUTH --> API
  API --> DENO
  VEC --> API
```

---

## 주의사항/함정

- `docker-compose.yml`의 `insforge` 서비스는 `npm install` + 마이그레이션 실행(`cd backend && npm run migrate:up`)이 포함된 긴 `command`를 갖고 있어, 초기 기동 시간이 걸릴 수 있습니다. (근거: `docker-compose.yml`)
- 개발/운영 compose는 볼륨 마운트/빌드 타깃이 다르므로, 운영에 개발 compose를 그대로 쓰는 것은 피하는 것이 좋습니다. (근거: 두 compose 파일 비교)

---

## TODO / 확인 필요

- “모델 게이트웨이(OpenAI compatible API)” 등 README에 나열된 기능 묶음이 백엔드 코드에서 어떤 라우트/서비스로 구현되는지(예: `backend/src/api/routes/*`)를 기능별로 매핑하면, 이 챕터가 더 ‘API 레퍼런스’에 가까워집니다.

---

## 위키 링크

- `[[InsForge Guide - Index]]` → [가이드 목차](/blog-repo/insforge-guide/)
- `[[InsForge Guide - Docker]]` → [02. 설치 및 실행(Docker)](/blog-repo/insforge-guide-02-docker/)
- `[[InsForge Guide - Ops]]` → [05. 운영/보안/트러블슈팅](/blog-repo/insforge-guide-05-ops-and-troubleshooting/)

---

*다음 글에서는 .env/포트/스토리지/마이그레이션을 기준으로 운영 체크리스트를 만듭니다.*

