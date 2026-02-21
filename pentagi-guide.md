---
layout: page
title: PentAGI 가이드
permalink: /pentagi-guide/
icon: fas fa-robot
---

# PentAGI 가이드

> **Docker 샌드박스 + 멀티 에이전트로 운영하는 자동화 보안 테스트 플랫폼**

**PentAGI**는 Docker로 격리된 실행 환경에서 보안 테스트 워크플로우를 자동화하는 멀티 에이전트 시스템입니다.  
이 시리즈는 **합법적/승인된 환경(자체 실습 랩·사내 승인된 진단·계약 기반 테스트)**에서 PentAGI를 **설치·구성·운영·개발**하는 데 필요한 구조를 한국어로 정리합니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/pentagi-guide-01-intro/) | PentAGI가 해결하는 문제, 기능, 시리즈 범위 |
| 02 | [Quick Start](/blog-repo/pentagi-guide-02-quick-start/) | 설치(인스톨러/수동), 첫 실행, 기본 접속 |
| 03 | [환경설정](/blog-repo/pentagi-guide-03-configuration/) | `.env` 핵심 변수, LLM/검색/SSL 설정 |
| 04 | [Docker 격리 & 배포](/blog-repo/pentagi-guide-04-docker-and-deployment/) | Compose 스택, 네트워크/볼륨, 워커 노드 개념 |
| 05 | [백엔드 구조](/blog-repo/pentagi-guide-05-backend-architecture/) | Go 서버, Gin 라우팅, DB/GraphQL 구성 |
| 06 | [프론트엔드 구조](/blog-repo/pentagi-guide-06-frontend-architecture/) | React/Vite, Apollo, 라우팅·상태·UI |
| 07 | [에이전트/프롬프트](/blog-repo/pentagi-guide-07-agents-and-prompts/) | 멀티 에이전트 역할, 템플릿, 컨텍스트 제어 |
| 08 | [툴 실행 & 검색](/blog-repo/pentagi-guide-08-tools-and-search/) | Docker 실행기, 브라우저/검색 툴, 로깅 |
| 09 | [메모리 & Graphiti](/blog-repo/pentagi-guide-09-memory-and-graphiti/) | pgvector, 임베딩, 요약기, 지식 그래프 |
| 10 | [관측성/테스트/빌드](/blog-repo/pentagi-guide-10-ops-testing-build/) | Langfuse·OTEL, ctester/ftester, Dockerfile 빌드 |

---

## 주요 특징(요약)

- **격리 실행**: 모든 작업을 Docker 기반 샌드박스에서 수행
- **멀티 에이전트**: 역할 분리(리서치/코딩/실행/보고 등)로 워크플로우를 분업
- **검색/브라우저 내장**: 웹 스크래퍼 + 다양한 외부 검색 엔진 연동
- **영속 로그/메모리**: PostgreSQL + pgvector에 실행/결과를 저장
- **관측성**: Langfuse, OpenTelemetry, Grafana 스택과 연동 가능
- **선택적 지식 그래프**: Graphiti + Neo4j로 관계/컨텍스트를 구조화

---

## 빠른 시작(최소)

```bash
# 1) 작업 디렉토리
mkdir -p pentagi && cd pentagi

# 2) .env 준비(샘플에서 시작)
curl -o .env https://raw.githubusercontent.com/vxcontrol/pentagi/master/.env.example

# 3) Compose 실행(기본 스택)
curl -O https://raw.githubusercontent.com/vxcontrol/pentagi/master/docker-compose.yml
docker compose up -d
```

기본 UI는 보통 `https://localhost:8443`에서 접근합니다(환경에 따라 포트/SSL은 변경 가능).

---

## 아키텍처 개요

```mermaid
flowchart TB
  UI[Frontend (React)] --> API[Backend (Go + GraphQL/REST)]
  API --> DB[(PostgreSQL + pgvector)]
  API --> DOCKER[Docker Engine]
  API --> SCRAPER[Scraper (Isolated Browser)]
  API --> SEARCH[Search APIs]
  API --> OBS[Langfuse / OTEL (Optional)]
  API --> KG[Graphiti + Neo4j (Optional)]
```

---

## 기술 스택

| 기술 | 용도 |
|------|------|
| Go + Gin | 백엔드 HTTP API |
| GraphQL | UI 실시간 데이터/구독(Subscriptions) |
| PostgreSQL + pgvector | 영속 저장 + 임베딩 검색 |
| React + Vite + Apollo | 프론트엔드 SPA |
| Docker Compose | 로컬/서버 배포 |
| Langfuse / OpenTelemetry / Grafana | LLM·시스템 관측성(선택) |
| Graphiti + Neo4j | 지식 그래프(선택) |

---

## 관련 링크

- GitHub 저장소: `vxcontrol/pentagi`
- 공식 문서: `docs.pentagi.com`
- 커뮤니티: Discord / Telegram(README 참고)

