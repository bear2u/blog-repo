---
layout: post
title: "MiroFish 완벽 가이드 (05) - 운영/확장/베스트 프랙티스"
date: 2026-03-10
permalink: /mirofish-guide-05-best-practices/
author: 666ghj
categories: [AI 에이전트, MiroFish]
tags: [MiroFish, Operations, Logging, Docker, Zep, LLM]
original_url: "https://github.com/666ghj/MiroFish"
excerpt: "백엔드 상태 파일/로그/프로세스(시뮬레이션) 관점에서 운영 체크리스트와 튜닝 포인트를 정리합니다."
---

## 이 문서의 목적

- MiroFish를 “데모 한 번”이 아니라 **반복 실행/디버깅 가능한 시스템**으로 운영하기 위한 기준점을 제공합니다.
- 코드에서 실제로 사용하는 디렉토리/로그/상태 파일을 근거로, 장애 대응 순서를 정리합니다.

---

## 빠른 요약

- 운영 관점의 핵심은 `backend/uploads/` 아래에 남는 **프로젝트/시뮬레이션 상태 파일**과 `backend/logs/`의 **로그 파일**을 이해하는 것입니다(`backend/app/models/project.py`, `backend/app/services/simulation_manager.py`, `backend/app/utils/logger.py`).
- “무언가 멈춘 것처럼 보일 때”는 **비동기 태스크(TaskManager)** 진행률/에러를 먼저 확인합니다(`backend/app/models/task.py`, `backend/app/api/graph.py`, `backend/app/api/simulation.py`, `backend/app/api/report.py`).

---

## 운영 체크리스트(실전)

### 1) 상태/데이터 경로를 먼저 고정

MiroFish는 DB가 아니라 **파일 기반**으로 상태를 저장하는 부분이 많습니다.

- 프로젝트: `backend/uploads/projects/<project_id>/project.json` (`ProjectManager`)
- 업로드 파일: `backend/uploads/projects/<project_id>/files/*`
- 추출 텍스트: `backend/uploads/projects/<project_id>/extracted_text.txt`
- 시뮬레이션: `backend/uploads/simulations/<simulation_id>/state.json` (`SimulationManager`)

Docker 실행 시에는 `docker-compose.yml`이 `./backend/uploads:/app/backend/uploads`를 볼륨으로 매핑하므로, 호스트에서 `backend/uploads`를 백업/정리 대상으로 삼으면 됩니다.

### 2) 로그는 “파일 로그”가 기준

`setup_logger()`는 다음을 동시에 설정합니다(`backend/app/utils/logger.py`).

- 콘솔: `INFO` 이상
- 파일: `DEBUG` 포함 상세 로그, `backend/logs/YYYY-MM-DD.log`로 RotatingFileHandler

운영 중 디버깅은 “콘솔이 조용해도 파일에는 남아있는지”를 먼저 확인하는 편이 빠릅니다.

### 3) 키/비용: 최소 2개의 외부 API 호출이 항상 있다

필수 키:

- LLM: `LLM_API_KEY` (`backend/app/config.py`, `backend/app/utils/llm_client.py`)
- Zep: `ZEP_API_KEY` (`backend/app/config.py`, `backend/app/services/graph_builder.py`)

권장 운영 습관:

- `.env`는 git에 커밋하지 않고(`.gitignore`), 배포 환경에선 Secret manager로 주입.
- “비용 폭주” 리스크는 시뮬레이션 라운드 수/리포트 툴 호출 제한으로 먼저 제어합니다(아래 튜닝 포인트 참고).

### 4) 비동기 태스크 실패 시 “traceback” 우선

- 그래프 빌드: `GET /api/graph/task/<task_id>` (`backend/app/api/graph.py`)
- 시뮬레이션 준비: `/api/simulation/prepare/status` (`backend/app/api/simulation.py`)
- 리포트 생성: `/api/report/generate/status` (`backend/app/api/report.py`)

Task payload에는 `error`(traceback)가 포함될 수 있으니, UI가 아닌 API로도 확인하세요(`backend/app/models/task.py`).

---

## 튜닝 포인트(코드에 있는 것만)

### 1) 업로드/텍스트 처리

- 업로드 크기 제한: `Config.MAX_CONTENT_LENGTH` (50MB) (`backend/app/config.py`)
- 텍스트 청크: `Config.DEFAULT_CHUNK_SIZE`/`DEFAULT_CHUNK_OVERLAP` (`backend/app/config.py`)
- 그래프 빌드 요청에서 `chunk_size`, `chunk_overlap` 오버라이드 가능 (`backend/app/api/graph.py`)

### 2) 시뮬레이션 라운드/리소스

- 기본 라운드 수: `Config.OASIS_DEFAULT_MAX_ROUNDS` (`backend/app/config.py`)
- 실행기는 별도 프로세스를 띄우고 모니터링 스레드를 운영합니다(`backend/app/services/simulation_runner.py`).
- 서버 종료 시 정리: `SimulationRunner.register_cleanup()`가 앱 초기화 때 호출됩니다(`backend/app/__init__.py`).

### 3) 리포트(LLM 호출량 제한)

`Config`에 리포트 관련 파라미터가 있습니다(`backend/app/config.py`).

- `REPORT_AGENT_MAX_TOOL_CALLS`
- `REPORT_AGENT_MAX_REFLECTION_ROUNDS`
- `REPORT_AGENT_TEMPERATURE`

값을 낮추면 비용/시간을 줄일 수 있지만, 리포트 품질도 함께 내려갈 수 있습니다(트레이드오프).

---

## 확장 포인트(어디를 고치면 무엇이 바뀌는가)

- “어떤 본체(ontology)를 만들까?”: `backend/app/services/ontology_generator.py`의 시스템 프롬프트/검증 로직
- “그래프 빌드 방식”: `backend/app/services/graph_builder.py` (청크/배치, Zep SDK 호출)
- “엔티티→프로필 변환”: `backend/app/services/oasis_profile_generator.py`
- “시뮬레이션 실행/IPC”: `backend/app/services/simulation_runner.py`, `backend/app/services/simulation_ipc.py`
- “리포트 생성/툴”: `backend/app/services/report_agent.py`, `backend/app/api/report.py`

---

## 근거(파일/경로)

- 상태/영속: `backend/app/models/project.py`, `backend/app/services/simulation_manager.py`
- 로그: `backend/app/utils/logger.py`
- 설정/튜닝 파라미터: `backend/app/config.py`
- 실행기 정리 훅: `backend/app/__init__.py`, `backend/app/services/simulation_runner.py`

---

## 주의사항/함정

- `backend/uploads`가 커지면(문서/시뮬레이션 로그) 성능/디스크 이슈가 바로 납니다. Docker 환경이라면 볼륨이 어디에 붙는지(호스트 경로)를 항상 확인하세요.
- UI에서 “진행 중”처럼 보여도 실제로는 태스크가 실패했을 수 있습니다. 운영에서는 태스크 상태 API를 우선 기준으로 삼는 편이 안정적입니다.

---

## TODO/확인 필요

- 인증/인가(사용자 분리, 키 보호) 설계는 이 레포의 API 레이어에서 명시적으로 보이지 않습니다. 외부 공개 배포 시 보안 모델을 별도로 설계해야 합니다.
- CI/CD 파이프라인(테스트/배포 자동화)은 `.github/`를 포함해 추가 점검이 필요합니다(이 시리즈 범위 밖).

---

## 위키 링크

- [[MiroFish Guide - Intro]] / [소개 및 개요](/blog-repo/mirofish-guide-01-intro/)
- [[MiroFish Guide - Installation]] / [설치 및 빠른 시작](/blog-repo/mirofish-guide-02-installation/)
- [[MiroFish Guide - Architecture]] / [핵심 개념과 아키텍처](/blog-repo/mirofish-guide-03-architecture/)
- [[MiroFish Guide - Usage]] / [실전 사용 패턴](/blog-repo/mirofish-guide-04-usage/)
- [[MiroFish Guide - Doc Automation]] / [문서 점검 자동화](/blog-repo/mirofish-guide-06-doc-automation/)

다음 글(보너스)에서는 이 시리즈 자체를 유지보수하기 위한 **문서 점검 자동화**를 정리합니다.
