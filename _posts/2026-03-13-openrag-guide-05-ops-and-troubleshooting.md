---
layout: post
title: "openrag 완벽 가이드 (05) - 운영/확장/트러블슈팅"
date: 2026-03-13
permalink: /openrag-guide-05-ops-and-troubleshooting/
author: langflow-ai
categories: [AI 에이전트, openrag]
tags: [Trending, GitHub, openrag, Troubleshooting, OpenSearch, Langflow, GitHub Trending]
original_url: "https://github.com/langflow-ai/openrag"
excerpt: "Makefile(health/logs/clean), .env.example, docker-compose.yml을 근거로 운영 체크리스트와 대표 장애 포인트를 정리합니다."
---

## 이 문서의 목적

- OpenRAG를 운영할 때 가장 흔한 문제(환경 변수 누락/서비스 기동 순서/인덱스 초기화/시간초과)를 체크리스트로 정리합니다.
- “무엇을 어디서 확인할지”를 Makefile/compose/.env를 근거로 연결합니다.

---

## 빠른 요약

- 상태/로그: `make status`, `make logs` (`Makefile`)
- 헬스 체크: `make health` (`Makefile`)
- 정리/초기화: `make clean`, `make factory-reset` (`Makefile`)
- 가장 흔한 설정 누락: `OPENSEARCH_PASSWORD`, LLM 키(`OPENAI_API_KEY` 등), `LANGFLOW_SECRET_KEY` (`.env.example`)

---

## 1) 1차 점검: 컨테이너가 떴는가?

```bash
make status
make logs
```

근거:
- `Makefile`의 유틸리티/헬프 섹션

---

## 2) OpenSearch 초기화/인증 이슈

### 증상

- OpenSearch가 뜨긴 했는데, 백엔드에서 인증 오류/접속 실패가 난다.

### 확인 포인트(파일 기준)

- `OPENSEARCH_PASSWORD`가 `.env`에 설정돼 있는가? (`.env.example`)
- `docker-compose.yml`의 `opensearch`가 `OPENSEARCH_INITIAL_ADMIN_PASSWORD`로 비밀번호를 받는 구조인가? (Yes)
- `OPENSEARCH_HOST/PORT/USERNAME`가 백엔드/랭플로우에 전달되는가? (Yes)

근거:
- `.env.example`
- `docker-compose.yml`

---

## 3) Langflow/인제스트 타임아웃

`.env.example`에는 대형 문서(300+ pages)에서 ingest가 30분 이상 걸릴 수 있으며, `LANGFLOW_TIMEOUT`, `INGESTION_TIMEOUT` 등을 늘리라는 주석이 포함됩니다.

근거:
- `.env.example`

운영 팁:

- 대용량 PDF ingest가 끊기면, 타임아웃 값을 조정하고(또는 문서를 분할) 재시도하는 것이 안전합니다.

---

## 4) 포트 충돌

`.env.example`에는 포트 설정이 노출되어 있습니다.

- `FRONTEND_PORT`(기본 3000)
- `LANGFLOW_PORT`(기본 7860)

근거:
- `.env.example`

---

## 5) 인덱스/데이터 초기화 루틴

Makefile에는 “완전 초기화(factory reset)” 및 OpenSearch 데이터 정리 루틴이 존재합니다.

- `make factory-reset` (컨테이너/볼륨/데이터 정리 + 이미지 제거 등)

근거:
- `Makefile`

---

## 6) 확장 포인트(코드/디렉토리 기준)

- 커넥터: `src/connectors/*` (google_drive/onedrive/sharepoint)
- API 확장: `src/api/v1/*`
- UI 확장: `frontend/app/*`, `frontend/components/*`
- 배포/운영: `kubernetes/`, `docker-compose*.yml`

근거:
- 레포 디렉토리 구성

---

## TODO / 확인 필요

- “Docling serve” 연동의 실제 동작/포트(기본 5001 추정)는 `.env.example` 주석과 `scripts/docling_ctl.py`를 읽고 확정하는 것이 좋습니다(이 문서는 운영 체크리스트 중심).

---

## 위키 링크

- `[[OpenRAG Guide - Index]]` → [가이드 목차](/blog-repo/openrag-guide/)
- `[[OpenRAG Guide - Quickstart]]` → [02. Quickstart로 실행](/blog-repo/openrag-guide-02-quickstart/)
- `[[OpenRAG Guide - Docker]]` → [03. Docker로 실행](/blog-repo/openrag-guide-03-docker/)
- `[[OpenRAG Guide - Architecture]]` → [04. 구성요소/아키텍처](/blog-repo/openrag-guide-04-architecture/)

