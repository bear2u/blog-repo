---
layout: post
title: "Project N.O.M.A.D. 완벽 가이드 (05) - 모범사례·문제해결·문서 점검 자동화"
date: 2026-03-15
permalink: /project-nomad-guide-05-best-practices-and-doc-automation/
author: Crosstalk-Solutions
categories: [GitHub Trending, project-nomad]
tags: [Trending, GitHub, project-nomad, Troubleshooting, BestPractices, Automation]
original_url: "https://github.com/Crosstalk-Solutions/project-nomad"
excerpt: "README의 운영 전제(오프라인/LAN, 인터넷 노출 금지)를 바탕으로, 최소 점검/트러블슈팅 체크리스트와 API+Docker 기반 점검 자동화(크론) 예시를 제공합니다."
---

## 이 문서의 목적

- 운영 시 “문제인지/정상인지”를 빠르게 가르는 최소 체크리스트를 제공합니다.
- 반복 점검을 자동화하는 예시(스크립트/크론)를 제공해, 장기 운영 시 회복력을 높입니다.

---

## 빠른 요약(README/라우트 기반)

- README는 **인터넷 직접 노출을 권장하지 않음**을 명확히 경고합니다. (`README.md`)
- 최소 확인은 `GET /api/health`로 시작할 수 있습니다. (`admin/start/routes.ts`)
- 서비스 관리는 `/api/system/services/*` 아래에 집중되어 있습니다. (`admin/start/routes.ts`)

---

## 운영 모범사례(README 기반)

- **노출 범위 최소화**: 기본은 로컬/내부망(LAN)에서만 접근(README 경고 준수). (`README.md`)
- **업데이트는 통제된 창에서**: 오프라인-우선이라도 초기 설치/업데이트에는 네트워크가 필요할 수 있으므로(README), “업데이트 윈도우”를 정해 수행.
- **헬스체크 기준을 문서화**: `/api/health` 응답, 핵심 도구 컨테이너 상태를 “정상 기준”으로 고정.

---

## 트러블슈팅 체크리스트(최소)

1) UI 접속 불가
- `http://localhost:8080`이 열리는지(README 기준 포트) (`README.md`)
- `/api/health`가 `ok`를 주는지 (`admin/start/routes.ts`)

2) 기능별 문제
- RAG: `/api/rag/*` 그룹의 잡 상태/파일 목록이 정상인지 (`admin/start/routes.ts`)
- 모델: `/api/ollama/*`에서 설치 모델/다운로드 요청이 되는지 (`admin/start/routes.ts`)

3) “서비스 설치/업데이트” 실패
- `/api/system/services/*` 호출 경로 확인 (`admin/start/routes.ts`)
- 실제 입력 스키마는 컨트롤러 구현 확인 필요 (`admin/app/controllers/*`)

---

## 문서 점검 자동화(예시)

아래는 “N.O.M.A.D.가 살아있는지”를 주기적으로 확인하는 매우 작은 스크립트 예시입니다.

> 주의: 설치 환경(시스템 사용자, Docker 권한, 실제 서비스 구성)에 따라 수정이 필요합니다. 이 예시는 `/api/health` 라우트 존재(`admin/start/routes.ts`)만을 근거로 합니다.

`/usr/local/bin/nomad-healthcheck.sh` 예시:

```bash
#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8080}"

echo "[nomad] health: ${BASE_URL}/api/health"
curl -fsS "${BASE_URL}/api/health" | head -c 200
echo

if command -v docker >/dev/null 2>&1; then
  echo "[nomad] docker ps (top 10)"
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | head -n 11
fi
```

크론(예시, 5분마다):

```cron
*/5 * * * * BASE_URL=http://localhost:8080 /usr/local/bin/nomad-healthcheck.sh >> /var/log/nomad-healthcheck.log 2>&1
```

---

## 근거(파일/경로)

- 네트워크/운영 전제(오프라인-우선, 인터넷 노출 경고): `README.md`
- 헬스체크 엔드포인트: `admin/start/routes.ts` (`/api/health`)
- 서비스/시스템 API 그룹: `admin/start/routes.ts` (`/api/system/*`)

---

## 주의사항/함정

- 자동화에서 `/api/system/services/*` 같은 “변경” API를 다룰 경우, 인증/권한/입력 스키마를 반드시 확인해야 합니다(문서만 보고 호출 금지). (`admin/app/controllers/*`)
- Docker 명령 자동화는 호스트 권한/그룹 설정에 민감합니다.

---

## TODO/확인 필요

- 인증/권한 모델(로그인/세션/CSRF 등)을 확인하고 “API를 안전하게 호출하는 방법”을 문서화 (`admin/`의 auth/session 관련 설정)
- 서비스 목록/상태를 주는 API가 있는지 확인하고(예: `GET /api/system/services`) “정상 기준”을 표준화 (`admin/start/routes.ts`)

---

## 위키 링크

- `[[Project NOMAD Guide - Index]]` → [가이드 목차](/blog-repo/project-nomad-guide/)
- `[[Project NOMAD Guide - Architecture]]` → [03. 아키텍처](/blog-repo/project-nomad-guide-03-architecture/)

---

*이 시리즈는 “README+라우트 근거”로 시작해, 이후에는 컨트롤러/서비스 구현을 근거로 더 정밀한 운영 가이드를 보강할 수 있습니다.*

