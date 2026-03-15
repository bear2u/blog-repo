---
layout: post
title: "Project N.O.M.A.D. 완벽 가이드 (04) - Command Center API/운영(업데이트·서비스 관리)"
date: 2026-03-15
permalink: /project-nomad-guide-04-api-and-operations/
author: Crosstalk-Solutions
categories: [GitHub Trending, project-nomad]
tags: [Trending, GitHub, project-nomad, API, Operations, Docker]
original_url: "https://github.com/Crosstalk-Solutions/project-nomad"
excerpt: "`admin/start/routes.ts`를 근거로 핵심 API 그룹을 운영 시나리오로 재구성하고, 헬스체크/서비스 관리/모델 관리/RAG 업로드 흐름을 예시로 정리합니다."
---

## 이 문서의 목적

- “어떤 운영을 할 때 어떤 API를 호출하는가”를 라우트 정의로 역추적해 정리합니다. (`admin/start/routes.ts`)
- UI를 쓰는 것이 기본이겠지만, 문제해결/자동화에는 **API 표면**을 알아두는 것이 유용합니다.

---

## 빠른 요약(라우트 기반)

- 최소 생존 확인: `GET /api/health` (`admin/start/routes.ts`)
- 시스템/서비스 제어: `/api/system/services/*` (서비스 설치/업데이트/강제 재설치 등) (`admin/start/routes.ts`)
- Ollama 모델 관리: `/api/ollama/*` (모델 목록/다운로드/삭제) (`admin/start/routes.ts`)
- 문서 기반 RAG: `/api/rag/*` (upload/sync/job-status) (`admin/start/routes.ts`)
- 오프라인 위키(ZIM): `/api/zim/*` (`admin/start/routes.ts`)

---

## 1) 헬스체크

```bash
curl -sS http://localhost:8080/api/health
```

코드 근거: `admin/start/routes.ts`

---

## 2) 시스템 정보/인터넷 상태

`/api/system` 그룹에는 “기기 상태/연결 상태”로 보이는 엔드포인트가 있습니다. (`admin/start/routes.ts`)

예시:

```bash
curl -sS http://localhost:8080/api/system/info
curl -sS http://localhost:8080/api/system/internet-status
```

---

## 3) 서비스 목록/설치/업데이트(핵심 운영 축)

`admin/start/routes.ts`에는 서비스 관리 API가 다수 정의되어 있습니다.

대표 엔드포인트(정의 근거):
- `GET /api/system/services`
- `POST /api/system/services/install`
- `POST /api/system/services/update`
- `POST /api/system/services/force-reinstall`
- `POST /api/system/services/check-updates`

주의: 실제 요청 바디/파라미터는 컨트롤러 구현을 확인해야 정확합니다(이 문서는 라우트 표면만 보장). (`admin/app/controllers/*`)

---

## 4) Ollama: 모델 목록/다운로드/삭제

`/api/ollama` 그룹은 모델 설치/삭제/조회 기능을 제공합니다. (`admin/start/routes.ts`)

대표 엔드포인트:
- `GET /api/ollama/models`
- `POST /api/ollama/models` (다운로드 디스패치)
- `DELETE /api/ollama/models`
- `GET /api/ollama/installed-models`

---

## 5) RAG: 업로드→상태 확인→동기화

`/api/rag` 그룹은 업로드/파일 관리/잡 상태 확인/스캔 동기화로 분리되어 있습니다. (`admin/start/routes.ts`)

대표 엔드포인트:
- `POST /api/rag/upload`
- `GET /api/rag/job-status`
- `POST /api/rag/sync`
- `GET /api/rag/files`
- `DELETE /api/rag/files`

---

## 근거(파일/경로)

- API 그룹/엔드포인트: `admin/start/routes.ts`
- 프로젝트의 “오케스트레이션” 목표: `README.md`

---

## 주의사항/함정

- 이 챕터는 “라우트가 존재한다”까지를 근거로 합니다. 실제 운영 자동화(예: 서비스 설치 요청)는 컨트롤러가 기대하는 입력을 반드시 확인해야 합니다. (`admin/app/controllers/*`)
- 서비스 제어 API는 외부 노출 시 위험이 커질 수 있으므로, README의 네트워크 노출 경고를 함께 고려해야 합니다. (`README.md`)

---

## TODO/확인 필요

- 컨트롤러별 입력 스키마(요청 바디/쿼리)를 추출해 “curl 예제(완전판)” 만들기:
  - `admin/app/controllers/system_controller.ts`
  - `admin/app/controllers/ollama_controller.ts`
  - `admin/app/controllers/rag_controller.ts`
- 큐 워커/백그라운드 잡이 실제로 어떤 작업을 비동기화하는지 정리(다운로드/모델/벤치마크 등): `admin/package.json`의 `work:*` 스크립트 + `admin/app/jobs/*`

---

## 위키 링크

- `[[Project NOMAD Guide - Index]]` → [가이드 목차](/blog-repo/project-nomad-guide/)
- `[[Project NOMAD Guide - Best Practices]]` → [05. 모범사례·문제해결·자동화](/blog-repo/project-nomad-guide-05-best-practices-and-doc-automation/)

---

*다음 글에서는 운영 체크리스트와 문제해결 포인트를 정리하고, `/api/health` 및 Docker 상태를 주기적으로 점검하는 자동화 예시를 제공합니다.*

