---
layout: post
title: "WrenAI 완벽 가이드 (1) - 소개 및 개요"
date: 2025-02-05
permalink: /wrenai-guide-01-intro/
author: Canner
categories: [AI 에이전트, WrenAI]
tags: [WrenAI, Text-to-SQL, GenBI, AI Agent, LLM]
original_url: "https://github.com/Canner/WrenAI"
excerpt: "자연어를 SQL과 차트로 변환하는 오픈소스 GenBI 에이전트 WrenAI를 소개합니다."
---

## WrenAI란?

**WrenAI**는 오픈소스 **GenBI(Generative BI) 에이전트**로서, 자연어 질문을 SQL 쿼리와 차트로 변환하는 엔터프라이즈급 AI 솔루션입니다.

```
┌─────────────────────────────────────────────┐
│              WrenAI의 핵심 가치              │
├─────────────────────────────────────────────┤
│  • 자연어 → SQL 변환 (Text-to-SQL)          │
│  • 자동 차트 생성 (Text-to-Chart)           │
│  • 의미론적 계층(MDL) 기반 정확성           │
│  • RAG 기반 컨텍스트 검색                   │
│  • 12개 이상의 데이터소스 지원              │
└─────────────────────────────────────────────┘
```

---

## 주요 특징

| 특징 | 설명 |
|------|------|
| **Text-to-SQL** | 자연어 질문을 정확한 SQL 쿼리로 변환 |
| **Text-to-Chart** | 쿼리 결과를 자동으로 시각화 |
| **MDL (Metadata Definition Language)** | 비즈니스 로직을 정의하는 의미론적 계층 |
| **RAG 기반** | 벡터 검색으로 관련 스키마/예제 자동 검색 |
| **다중 데이터소스** | PostgreSQL, BigQuery, Snowflake 등 12개+ 지원 |
| **셀프 호스팅** | Docker로 쉽게 로컬/클라우드 배포 |

---

## 왜 WrenAI인가?

### 기존 BI 도구의 한계

```
┌─────────────────────────────────────────────┐
│          기존 BI 도구의 문제점              │
├─────────────────────────────────────────────┤
│  ❌ SQL 전문 지식 필요                      │
│  ❌ 복잡한 조인/집계 작성 어려움            │
│  ❌ 비개발자의 데이터 접근성 낮음           │
│  ❌ 반복적인 쿼리 작성 시간 낭비            │
└─────────────────────────────────────────────┘
```

### WrenAI의 해결책

```
┌─────────────────────────────────────────────┐
│           WrenAI가 제공하는 가치            │
├─────────────────────────────────────────────┤
│  ✅ "지난 분기 매출은?" → SQL 자동 생성     │
│  ✅ 비즈니스 용어 → 기술 쿼리 자동 매핑     │
│  ✅ 누구나 데이터 분석 가능                 │
│  ✅ 반복 쿼리 자동화로 생산성 향상          │
└─────────────────────────────────────────────┘
```

---

## 지원 데이터소스

| 카테고리 | 데이터소스 |
|----------|-----------|
| **클라우드 DW** | BigQuery, Snowflake, Redshift, Databricks |
| **RDBMS** | PostgreSQL, MySQL, SQL Server, Oracle |
| **분석 DB** | ClickHouse, Trino, Athena |
| **로컬** | DuckDB |

---

## 빠른 시작

```bash
# Docker Compose로 설치
git clone https://github.com/Canner/WrenAI.git
cd WrenAI/docker

# 환경 설정
cp .env.example .env.local
cp config.example.yaml config.yaml

# config.yaml에서 LLM API 키 설정 후
docker compose --env-file .env.local up -d

# 접속
# UI: http://localhost:3000
```

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────┐
│                    사용자 (브라우저)                     │
│                    http://localhost:3000                │
└─────────────────────────┬───────────────────────────────┘
                          │ GraphQL
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Wren UI (Next.js)                     │
│              Apollo Server + React + Ant Design          │
└────────────┬────────────────────────────┬───────────────┘
             │ REST API                    │
             ▼                             ▼
┌────────────────────────┐    ┌────────────────────────────┐
│   Wren AI Service      │    │    SQLite/PostgreSQL       │
│   (FastAPI + Python)   │    │    (메타데이터 저장)        │
│                        │    └────────────────────────────┘
│  • SQL Generation      │
│  • Chart Generation    │
│  • Intent Classification│
└────────┬───────────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
┌───────┐ ┌───────┐ ┌────────┐ ┌────────┐
│Qdrant │ │ Wren  │ │ Ibis   │ │  LLM   │
│벡터DB │ │Engine │ │Server  │ │Provider│
│:6333  │ │:8080  │ │:8000   │ │(Cloud) │
└───────┘ └───────┘ └────────┘ └────────┘
```

---

## 핵심 구성요소

| 구성요소 | 역할 | 기술 |
|---------|------|------|
| **Wren UI** | 웹 인터페이스, GraphQL API | Next.js 14, Apollo, TypeScript |
| **Wren AI Service** | AI 처리, RAG 파이프라인 | FastAPI, Haystack, Python 3.12 |
| **Wren Engine** | SQL 검증 및 실행 | Java 기반 SQL 엔진 |
| **Qdrant** | 벡터 검색, 스키마 색인 | 벡터 데이터베이스 |
| **Ibis Server** | 데이터소스 추상화 | Python Ibis 라이브러리 |

---

## 사용 흐름

```
1. 사용자: "지난 분기 매출은?"
              ↓
2. WrenAI: 의도 분류 (Intent Classification)
              ↓
3. WrenAI: 관련 스키마 검색 (RAG)
              ↓
4. WrenAI: SQL 생성 (LLM)
              ↓
5. WrenAI: SQL 검증 (Engine)
              ↓
6. WrenAI: 결과 반환 + 차트 생성
              ↓
7. 사용자: 결과 확인 및 피드백
```

---

## 이 가이드에서 다루는 내용

1. **소개 및 개요** (현재 글)
2. **설치 및 환경 설정** - Docker, 설정 파일
3. **아키텍처 심층 분석** - 서비스 구조
4. **MDL (Metadata Definition Language)** - 스키마 정의
5. **RAG 파이프라인** - 검색 및 생성
6. **LLM 연동** - OpenAI, Azure 등 설정
7. **프론트엔드 구조** - Next.js, GraphQL
8. **백엔드 API** - FastAPI 엔드포인트
9. **배포 가이드** - Docker, Kubernetes
10. **확장 및 커스터마이징** - 파이프라인 수정

---

## 라이선스

- **Wren AI Service**: AGPL-3.0
- **Wren UI**: AGPL-3.0
- **Wren Engine**: Apache-2.0

---

*다음 글에서는 WrenAI의 설치 및 환경 설정 방법을 살펴봅니다.*
