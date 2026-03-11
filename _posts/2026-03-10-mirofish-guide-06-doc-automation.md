---
layout: post
title: "MiroFish 완벽 가이드 (06) - 문서 점검 자동화"
date: 2026-03-10
permalink: /mirofish-guide-06-doc-automation/
author: 666ghj
categories: [AI 에이전트, MiroFish]
tags: [Trending, GitHub, MiroFish, Documentation, Automation, Checklist]
original_url: "https://github.com/666ghj/MiroFish"
excerpt: "가이드/온보딩 문서가 코드 변경에 뒤처지지 않도록, ‘깨지기 쉬운 계약(계약=routes/env/경로)’을 자동 점검하는 방법."
---

## 이 문서의 목적

- 코드가 바뀌어도 가이드가 “조용히” 깨지지 않게, **점검할 항목을 계약(Contract)**으로 정의합니다.
- 사람이 매번 눈으로 확인하는 대신, 최소한의 스크립트로 **변경 감지 + 경고**를 만들 수 있게 합니다.

---

## 빠른 요약

MiroFish 가이드에서 특히 깨지기 쉬운 계약은 3가지입니다.

1) **API 라우트/엔드포인트** (예: `/api/graph/build` 등)  
2) **필수 환경 변수** (예: `LLM_API_KEY`, `ZEP_API_KEY`)  
3) **상태/데이터 경로** (예: `backend/uploads/projects/...`, `backend/logs/...`)  

근거:
- 라우트: `backend/app/api/*.py`, `backend/app/__init__.py`
- 필수 키: `backend/app/config.py`, `backend/run.py`, `.env.example`
- 상태/로그 경로: `backend/app/models/project.py`, `backend/app/services/simulation_manager.py`, `backend/app/utils/logger.py`

---

## 체크리스트(변경 시 반드시 재검토)

### A) 실행/설치 계약

- `package.json`의 스크립트(`setup:all`, `dev`, `backend`, `frontend`)가 바뀌었는가?
- `.env.example`의 키 이름/의미가 바뀌었는가?
- `backend/app/config.py`에서 `Config.validate()`의 필수 키 조건이 바뀌었는가?

### B) API 계약

- 그래프:
  - `/api/graph/ontology/generate` (multipart, `files`, `simulation_requirement`)
  - `/api/graph/build` (JSON, `project_id`)
  - `/api/graph/task/<task_id>` (폴링)
- 시뮬레이션:
  - `/api/simulation/create`
  - `/api/simulation/prepare` + `/prepare/status`
  - `/api/simulation/start`, `/stop`
- 리포트:
  - `/api/report/generate` + `/generate/status`

이 라우트들은 `backend/app/api/graph.py`, `backend/app/api/simulation.py`, `backend/app/api/report.py`에서 정의됩니다.

### C) 데이터/로그 계약

- `backend/uploads/` 아래 상태 파일 경로가 바뀌었는가?
- 로그 파일 경로(`backend/logs/YYYY-MM-DD.log`)가 바뀌었는가?
- Docker 볼륨 매핑(`docker-compose.yml`)이 바뀌었는가?

---

## 자동 점검 스크립트(예시)

아래는 “가이드가 의존하는 계약이 깨졌는지”를 빠르게 감지하는 최소 스크립트 예시입니다.

> 주의: 예시 스크립트는 **실행 환경/비밀키**를 다루지 않으며, “정적 점검(파일/문자열)” 위주입니다.

### 1) 정적 점검: 파일/키/라우트 존재 확인

```bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

required_files=(
  "README.md"
  "package.json"
  ".env.example"
  "backend/run.py"
  "backend/app/config.py"
  "backend/app/api/graph.py"
  "backend/app/api/simulation.py"
  "backend/app/api/report.py"
)

for f in "${required_files[@]}"; do
  test -f "$f" || { echo "MISSING: $f"; exit 1; }
done

# 필수 env 키(.env.example에 포함되는지)
grep -q "^LLM_API_KEY=" .env.example || { echo "MISSING env key: LLM_API_KEY"; exit 1; }
grep -q "^ZEP_API_KEY=" .env.example || { echo "MISSING env key: ZEP_API_KEY"; exit 1; }

# 라우트 문자열(간단 감지)
grep -q "@graph_bp.route('/build'" backend/app/api/graph.py || { echo "MISSING route: /api/graph/build"; exit 1; }
grep -q "@simulation_bp.route('/prepare'" backend/app/api/simulation.py || { echo "MISSING route: /api/simulation/prepare"; exit 1; }
grep -q "@report_bp.route('/generate'" backend/app/api/report.py || { echo "MISSING route: /api/report/generate"; exit 1; }

echo "OK: contracts look present"
```

### 2) 런타임 점검(선택): 헬스 체크

백엔드를 띄울 수 있는 환경이라면, 최소한 `/health` 응답만 확인해도 “서버 기동/라우팅”의 큰 문제를 잡을 수 있습니다.

```bash
curl -fsS http://localhost:5001/health | cat
```

---

## 확장 포인트

- “라우트 계약”을 더 엄밀하게 하고 싶으면:
  - `rg "@.*_bp\\.route\\(" backend/app/api/*.py` 결과를 표준 목록으로 스냅샷하고, diff로 변경 감지
- “가이드 링크”까지 자동 점검하고 싶으면:
  - Jekyll 빌드(`bundle exec jekyll build`)가 가능한 환경에서 빌드 실패/경고를 CI에 포함
- “비용/품질” 자동 점검을 하고 싶으면:
  - `REPORT_AGENT_*`, `OASIS_DEFAULT_MAX_ROUNDS` 같은 파라미터를 환경별로 템플릿화(`.env.example` → 배포용 Secret)

---

## 근거(파일/경로)

- 라우트 정의: `backend/app/api/graph.py`, `backend/app/api/simulation.py`, `backend/app/api/report.py`
- 필수 키/검증: `backend/app/config.py`, `backend/run.py`, `.env.example`
- 로그: `backend/app/utils/logger.py`
- 상태 파일: `backend/app/models/project.py`, `backend/app/services/simulation_manager.py`

---

## TODO/확인 필요

- 프로젝트 CI가 실제로 어떤 형태로 운영되는지(테스트/빌드/배포)는 `.github/` 구성과 함께 별도 점검이 필요합니다.

---

## 위키 링크

- [[MiroFish Guide - Intro]] / [소개 및 개요](/blog-repo/mirofish-guide-01-intro/)
- [[MiroFish Guide - Installation]] / [설치 및 빠른 시작](/blog-repo/mirofish-guide-02-installation/)
- [[MiroFish Guide - Architecture]] / [핵심 개념과 아키텍처](/blog-repo/mirofish-guide-03-architecture/)
- [[MiroFish Guide - Usage]] / [실전 사용 패턴](/blog-repo/mirofish-guide-04-usage/)
- [[MiroFish Guide - Best Practices]] / [운영/확장/베스트 프랙티스](/blog-repo/mirofish-guide-05-best-practices/)

