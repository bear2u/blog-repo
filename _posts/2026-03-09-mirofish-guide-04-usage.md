---
layout: post
title: "MiroFish 완벽 가이드 (04) - 실전 사용 패턴"
date: 2026-03-10
permalink: /mirofish-guide-04-usage/
author: 666ghj
categories: [AI 에이전트, MiroFish]
tags: [Trending, GitHub, MiroFish, Workflow, API, Graph, Simulation, Report]
original_url: "https://github.com/666ghj/MiroFish"
excerpt: "문서 업로드→본체→그래프→시뮬레이션→리포트까지, UI/HTTP API 기준의 ‘한 번 돌려보기’ 체크리스트."
---

## 이 문서의 목적

- “실제로 한 번 돌려보기”를 목표로 **UI 경로**와 **API 경로(curl)**를 정리합니다.
- 실패가 잦은 지점을 미리 체크(키/상태 파일/태스크 폴링)해서 디버깅 시간을 줄입니다.

---

## 빠른 요약

- UI는 `Step 1~5` 워크플로우로 설계돼 있습니다(`frontend/src/views/MainView.vue`의 `currentStep`).
- 그래프 빌드는 **비동기 태스크**이며 `GET /api/graph/task/<task_id>`로 폴링합니다(`backend/app/api/graph.py`).
- 시뮬레이션 준비/리포트 생성도 **비동기 태스크**입니다(`backend/app/api/simulation.py`, `backend/app/api/report.py`).

---

## 경로 A) UI로 한 번 돌려보기(권장)

UI는 크게 5단계를 가정합니다(표기/라벨은 중국어 UI 기준).

1) **Step 1: 图谱构建**
   - 문서 업로드 + `simulation_requirement` 입력
   - Ontology 생성 → Graph build 태스크 시작
2) **Step 2: 环境搭建**
   - (프로젝트/그래프 상태를 기반으로) 시뮬레이션 준비로 진행
3) **Step 3: 开始模拟**
4) **Step 4: 报告生成**
5) **Step 5: 深度互动**

근거:
- 단계 정의: `frontend/src/views/MainView.vue`
- Step 컴포넌트: `frontend/src/components/Step1GraphBuild.vue` 등

---

## 경로 B) API로 한 번 돌려보기(curl)

아래는 백엔드(기본 `http://localhost:5001`)를 직접 호출하는 “최소 플로우”입니다.

### 0) 전제: 서버 실행

```bash
npm run dev
# 또는 backend만: npm run backend
```

헬스 체크:

```bash
curl -sS http://localhost:5001/health | cat
```

### 1) Ontology 생성(문서 업로드)

`/api/graph/ontology/generate`는 `multipart/form-data`이며, 최소 `files`와 `simulation_requirement`가 필요합니다(`backend/app/api/graph.py`).

```bash
curl -sS -X POST http://localhost:5001/api/graph/ontology/generate \\\n  -F 'files=@./README.md' \\\n  -F 'simulation_requirement=이 문서를 바탕으로 사회적 반응과 정보 확산을 시뮬레이션하고 싶다' \\\n  -F 'project_name=MiroFish Demo' \\\n  | cat
```

성공 시 `project_id`가 반환됩니다.

### 2) 그래프 빌드(비동기 태스크)

```bash
curl -sS -X POST http://localhost:5001/api/graph/build \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"project_id\":\"<project_id>\",\"chunk_size\":500,\"chunk_overlap\":50}' \\\n  | cat
```

응답의 `task_id`로 상태를 폴링합니다.

```bash
curl -sS http://localhost:5001/api/graph/task/<task_id> | cat
```

태스크가 완료되면 결과에 `graph_id`가 포함됩니다(저장은 `ProjectManager`가 담당).

### 3) 시뮬레이션 생성

```bash
curl -sS -X POST http://localhost:5001/api/simulation/create \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"project_id\":\"<project_id>\",\"enable_twitter\":true,\"enable_reddit\":true}' \\\n  | cat
```

성공 시 `simulation_id`를 얻습니다.

### 4) 시뮬레이션 준비(비동기 태스크)

`/api/simulation/prepare`는 그래프 엔티티를 읽고, 프로필/설정 파일을 생성합니다(`backend/app/api/simulation.py`, `backend/app/services/simulation_manager.py`).

```bash
curl -sS -X POST http://localhost:5001/api/simulation/prepare \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"simulation_id\":\"<simulation_id>\",\"use_llm_for_profiles\":true,\"parallel_profile_count\":5}' \\\n  | cat
```

상태 조회:

```bash
curl -sS -X POST http://localhost:5001/api/simulation/prepare/status \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"task_id\":\"<task_id>\"}' \\\n  | cat
```

### 5) 시뮬레이션 시작/중지

시작/중지 엔드포인트는 `backend/app/api/simulation.py`의 `/start`, `/stop`에 정의돼 있습니다.

```bash
curl -sS -X POST http://localhost:5001/api/simulation/start \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"simulation_id\":\"<simulation_id>\"}' \\\n  | cat
```

진행 상태(런타임):

```bash
curl -sS http://localhost:5001/api/simulation/<simulation_id>/run-status | cat
```

### 6) 리포트 생성(비동기 태스크)

```bash
curl -sS -X POST http://localhost:5001/api/report/generate \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"simulation_id\":\"<simulation_id>\",\"force_regenerate\":false}' \\\n  | cat
```

상태 조회:

```bash
curl -sS -X POST http://localhost:5001/api/report/generate/status \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"task_id\":\"<task_id>\"}' \\\n  | cat
```

---

## 체크리스트(디버깅 우선순위)

1) `/health`가 응답하는가? (`backend/app/__init__.py`)
2) `.env`에 `LLM_API_KEY`, `ZEP_API_KEY`가 있는가? (`backend/app/config.py`, `backend/run.py`)
3) 그래프 빌드 태스크가 `failed`라면, `task.error`(traceback)를 먼저 본다. (`backend/app/models/task.py`)
4) 시뮬레이션 준비가 재실행될 때는 `force_regenerate` 동작과 “이미 준비됨” 감지를 확인한다. (`backend/app/api/simulation.py`의 `_check_simulation_prepared`)
5) 파일 영속 경로(`backend/uploads/...`)의 용량/권한 문제는 없는가? (`docker-compose.yml` 볼륨 매핑)

---

## 근거(파일/경로)

- UI 단계 흐름: `frontend/src/views/MainView.vue`
- 그래프 API: `backend/app/api/graph.py`
- 시뮬레이션 API: `backend/app/api/simulation.py`
- 리포트 API: `backend/app/api/report.py`
- 시뮬레이션 실행기: `backend/app/services/simulation_runner.py`

---

## 주의사항/함정

- 샘플로 `README.md`를 업로드해도 동작은 하지만, 실제론 PDF/리포트/기사/소설 등 “시드 문서”의 품질이 결과를 좌우합니다. 업로드 크기 제한은 `Config.MAX_CONTENT_LENGTH`(50MB)입니다(`backend/app/config.py`).
- 그래프/시뮬레이션/리포트는 비동기 태스크가 많아서, “버튼을 눌렀는데 아무 일도 안 일어남”처럼 보일 수 있습니다. 반드시 태스크 상태를 폴링하세요.

---

## TODO/확인 필요

- 실제로 어떤 “OASIS 스크립트”가 실행되는지(프로세스 인자/파일 포맷/출력 파일)는 `SimulationRunner`와 `backend/scripts/*`를 기준으로 추가 정리가 필요합니다.

---

## 위키 링크

- [[MiroFish Guide - Intro]] / [소개 및 개요](/blog-repo/mirofish-guide-01-intro/)
- [[MiroFish Guide - Installation]] / [설치 및 빠른 시작](/blog-repo/mirofish-guide-02-installation/)
- [[MiroFish Guide - Architecture]] / [핵심 개념과 아키텍처](/blog-repo/mirofish-guide-03-architecture/)
- [[MiroFish Guide - Best Practices]] / [운영/확장/베스트 프랙티스](/blog-repo/mirofish-guide-05-best-practices/)
- [[MiroFish Guide - Doc Automation]] / [문서 점검 자동화](/blog-repo/mirofish-guide-06-doc-automation/)

다음 글에서는 운영 관점(로그/데이터/성능/재실행)에서 자주 부딪히는 포인트를 정리합니다.
