---
layout: post
title: "MiroFish 완벽 가이드 (02) - 설치 및 빠른 시작"
date: 2026-03-10
permalink: /mirofish-guide-02-installation/
author: 666ghj
categories: [AI 에이전트, MiroFish]
tags: [MiroFish, Setup, Node.js, Python, uv, Docker]
original_url: "https://github.com/666ghj/MiroFish"
excerpt: ".env 구성 → 의존성 설치 → 프론트/백엔드 동시 실행까지, README와 스크립트 근거로 정리합니다."
---

## 이 문서의 목적

- MiroFish를 “한 번 실행 가능한 상태”로 만드는 최소 경로를 제공합니다.
- 로컬 실행(개발 모드)과 Docker 실행을 각각 정리합니다.

---

## 빠른 요약

- 필수 준비물: **Node.js 18+**, **Python 3.11~3.12**, **uv**, 그리고 **LLM_API_KEY / ZEP_API_KEY**(`README.md`, `package.json`, `backend/app/config.py`, `.env.example`)
- 가장 빠른 로컬 실행: `cp .env.example .env` → `npm run setup:all` → `npm run dev`
- 기본 포트: 프론트 `http://localhost:3000`, 백엔드 `http://localhost:5001` (`README.md`)

---

## 0) 저장소 준비

```bash
git clone https://github.com/666ghj/MiroFish.git
cd MiroFish
```

---

## 1) 환경 변수(.env)

루트의 `.env`는 백엔드에서 직접 로드됩니다(`backend/app/config.py`).

```bash
cp .env.example .env
```

필수 키는 아래 2개입니다.

| 키 | 용도 | 근거 |
|---|---|---|
| `LLM_API_KEY` | LLM 호출(ontology/프로필/리포트 등) | `backend/app/config.py`, `backend/app/utils/llm_client.py` |
| `ZEP_API_KEY` | Zep Cloud 그래프/엔티티/메모리 | `backend/app/config.py`, `backend/app/services/graph_builder.py` |

추가로 `.env.example`에는 “가속 LLM”용 선택 키도 있습니다(병렬 시뮬레이션에서 provider를 분리해 병목을 줄이는 의도). 해당 키를 실제로 쓰는 코드는 `backend/scripts/run_parallel_simulation.py`에서 확인할 수 있습니다.

---

## 2) 의존성 설치(로컬)

루트 `package.json`의 스크립트를 그대로 사용합니다.

```bash
# 루트 + frontend node_modules
npm run setup

# backend: uv로 Python 의존성 동기화
npm run setup:backend

# 또는 한번에
npm run setup:all
```

---

## 3) 개발 모드 실행(로컬)

```bash
npm run dev
```

`concurrently`로 프론트/백엔드를 동시에 실행합니다(`package.json`).

- 프론트: `frontend`의 `vite --host` (`frontend/package.json`)
- 백엔드: `backend/uv run python run.py` (`package.json`, `backend/run.py`)

헬스 체크:

- 백엔드: `GET http://localhost:5001/health` (`backend/app/__init__.py`)

---

## 4) Docker로 실행

### 4.1 docker-compose (권장)

```bash
cp .env.example .env
docker compose up -d
```

`docker-compose.yml`은 기본적으로 다음을 수행합니다.

- 이미지: `ghcr.io/666ghj/mirofish:latest`
- 포트: `3000:3000`, `5001:5001`
- 볼륨: `./backend/uploads:/app/backend/uploads` (업로드/시뮬레이션 데이터 영속)

### 4.2 Dockerfile (구조 이해)

`Dockerfile`은 한 컨테이너에 Python+Node+uv를 넣고, `npm run dev`로 프론트/백엔드를 함께 띄웁니다.

---

## 근거(파일/경로)

- 요구사항/포트/빠른 시작: `README.md`
- 설치/실행 스크립트: `package.json`, `frontend/package.json`
- 설정 로드/필수 키 검증: `backend/app/config.py`, `backend/run.py`
- Docker 실행: `docker-compose.yml`, `Dockerfile`

---

## 주의사항/함정

- `.env` 위치: 백엔드는 `backend/.env`가 아니라 **루트의 `.env`**를 찾습니다(`backend/app/config.py`가 `../../.env`를 로드).
- 키가 없으면 백엔드가 `Config.validate()`에서 종료합니다(`backend/run.py`).
- Docker 실행 시 `backend/uploads` 볼륨이 붙기 때문에, 로컬 파일 권한/용량 문제가 나면 먼저 이 경로를 확인하세요.

---

## TODO/확인 필요

- “프로덕션” 모드로 프론트를 빌드하고 정적 서빙하는 경로는 `frontend`/`backend`에서 별도 문서가 없어 보입니다(현재 Docker도 dev 모드로 실행). 운영 배포 시 목적에 맞는 실행 모드(예: `vite build` + 정적 호스팅)를 별도로 정해야 합니다.

---

## 위키 링크

- [[MiroFish Guide - Intro]] / [소개 및 개요](/blog-repo/mirofish-guide-01-intro/)
- [[MiroFish Guide - Architecture]] / [핵심 개념과 아키텍처](/blog-repo/mirofish-guide-03-architecture/)
- [[MiroFish Guide - Usage]] / [실전 사용 패턴](/blog-repo/mirofish-guide-04-usage/)
- [[MiroFish Guide - Best Practices]] / [운영/확장/베스트 프랙티스](/blog-repo/mirofish-guide-05-best-practices/)

다음 글에서는 **프론트/백엔드/API/상태 파일** 관점에서 전체 구조를 그려봅니다.
