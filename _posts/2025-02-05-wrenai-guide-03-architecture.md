---
layout: post
title: "WrenAI 완벽 가이드 (3) - 아키텍처 심층 분석"
date: 2025-02-05
permalink: /wrenai-guide-03-architecture/
author: Canner
categories: [AI 에이전트, WrenAI]
tags: [WrenAI, Architecture, System Design, Microservices]
original_url: "https://github.com/Canner/WrenAI"
excerpt: "WrenAI의 마이크로서비스 아키텍처와 각 구성요소의 역할을 상세히 분석합니다."
---

## 전체 아키텍처

WrenAI는 마이크로서비스 아키텍처로 구성되어 있습니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                         사용자 (브라우저)                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP/WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Wren UI (:3000)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  Next.js     │  │ Apollo Server│  │  SQLite/PostgreSQL    │  │
│  │  Pages       │  │ (GraphQL)    │  │  (메타데이터)          │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
└───────────┬─────────────────────────────────────┬───────────────┘
            │ REST API                            │ REST API
            ▼                                     ▼
┌───────────────────────────┐         ┌───────────────────────────┐
│   Wren AI Service (:5555) │         │    Wren Engine (:8080)    │
│  ┌─────────────────────┐  │         │  ┌─────────────────────┐  │
│  │  FastAPI Routers    │  │         │  │  SQL Parser         │  │
│  │  - /v1/asks         │  │         │  │  SQL Validator      │  │
│  │  - /v1/charts       │  │         │  │  Query Executor     │  │
│  │  - /v1/semantics    │  │         │  └─────────────────────┘  │
│  └─────────────────────┘  │         └───────────────────────────┘
│  ┌─────────────────────┐  │
│  │  RAG Pipelines      │  │         ┌───────────────────────────┐
│  │  - Indexing         │◀─┼────────▶│      Qdrant (:6333)       │
│  │  - Retrieval        │  │         │  ┌─────────────────────┐  │
│  │  - Generation       │  │         │  │  Vector Collections │  │
│  └─────────────────────┘  │         │  │  - table_schema     │  │
│  ┌─────────────────────┐  │         │  │  - historical_qa    │  │
│  │  LLM Provider       │──┼────┐    │  │  - sql_samples      │  │
│  └─────────────────────┘  │    │    │  └─────────────────────┘  │
└───────────────────────────┘    │    └───────────────────────────┘
                                 │
                                 ▼
                    ┌───────────────────────┐
                    │    LLM APIs (Cloud)    │
                    │  - OpenAI             │
                    │  - Azure OpenAI       │
                    │  - Google Gemini      │
                    └───────────────────────┘
```

---

## 핵심 컴포넌트 상세

### 1. Wren UI (프론트엔드)

```
wren-ui/
├── src/
│   ├── pages/                  # Next.js 페이지 라우터
│   │   ├── index.tsx           # 홈
│   │   ├── modeling/           # 모델링 페이지
│   │   ├── setup/              # 설정 마법사
│   │   └── api/                # API 라우트 (GraphQL)
│   │
│   ├── apollo/
│   │   ├── server/             # GraphQL 서버
│   │   │   ├── resolvers/      # Query/Mutation 핸들러
│   │   │   ├── services/       # 비즈니스 로직
│   │   │   └── repositories/   # 데이터 접근 계층
│   │   └── client/             # GraphQL 클라이언트
│   │
│   ├── components/             # React 컴포넌트
│   │   ├── sidebar/
│   │   ├── chart/
│   │   └── diagram/
│   │
│   └── hooks/                  # Custom React Hooks
│
├── migrations/                 # Knex DB 마이그레이션
└── e2e/                       # Playwright 테스트
```

**기술 스택:**
- Next.js 14.2 (프레임워크)
- TypeScript 5.2
- Apollo Server/Client 3.x (GraphQL)
- Ant Design 4.20 (UI)
- React Flow 11.10 (다이어그램)
- Vega-Lite 6.2 (차트)
- Knex 3.1 (쿼리 빌더)

---

### 2. Wren AI Service (백엔드)

```
wren-ai-service/
├── src/
│   ├── web/
│   │   └── v1/
│   │       ├── routers/        # FastAPI 엔드포인트
│   │       │   ├── ask.py
│   │       │   ├── chart.py
│   │       │   └── semantics.py
│   │       └── services/       # 비즈니스 로직
│   │           ├── ask.py
│   │           ├── chart.py
│   │           └── indexing.py
│   │
│   ├── pipelines/              # RAG 파이프라인
│   │   ├── indexing/           # 색인화
│   │   │   ├── db_schema.py
│   │   │   └── historical_question.py
│   │   ├── retrieval/          # 검색
│   │   │   └── db_schema.py
│   │   └── generation/         # 생성
│   │       ├── sql_generation.py
│   │       └── chart_generation.py
│   │
│   ├── providers/              # 외부 서비스 연동
│   │   ├── llm/               # LLM (OpenAI, Azure 등)
│   │   ├── embedder/          # 임베딩 모델
│   │   ├── document_store/    # Qdrant
│   │   └── engine/            # Wren Engine
│   │
│   ├── core/                  # 추상화 계층
│   └── config.py              # 설정 관리
│
├── eval/                      # 평가 프레임워크
└── tests/                     # 테스트
```

**기술 스택:**
- Python 3.12
- FastAPI 0.121 (웹 프레임워크)
- Haystack 2.7 (RAG 프레임워크)
- Hamilton 1.69 (파이프라인 오케스트레이션)
- LiteLLM 1.75 (LLM 통합)
- Qdrant Client 1.11

---

### 3. Qdrant (벡터 데이터베이스)

**컬렉션 구조:**

```
Qdrant Collections
├── table_schema_{project_id}
│   └── 문서: {
│       "type": "TABLE_SCHEMA",
│       "name": "customers",
│       "columns": [...],
│       "content": "CREATE TABLE ..."
│     }
│
├── historical_questions_{project_id}
│   └── 문서: {
│       "question": "지난 분기 매출은?",
│       "sql": "SELECT SUM(revenue)...",
│       "timestamp": "2024-02-05"
│     }
│
├── sql_samples_{project_id}
│   └── 문서: { "question", "sql" }
│
└── instructions_{project_id}
    └── 문서: { "title", "description" }

임베딩 모델: text-embedding-3-large (차원: 3072)
유사도 측정: Cosine
```

---

## 요청 처리 흐름

### Ask 쿼리 흐름

```
1. 사용자 입력
   └─▶ "지난 분기 매출은?"

2. Wren UI (GraphQL)
   └─▶ mutation { ask(question: "...") }
   └─▶ askingService.ask()
   └─▶ POST /v1/asks → AI Service

3. Wren AI Service
   ┌─▶ Intent Classification
   │   └─▶ "TEXT_TO_SQL" 판단
   │
   ├─▶ Context Retrieval (RAG)
   │   ├─▶ DB Schema 검색 (Qdrant)
   │   ├─▶ Historical Questions 검색
   │   ├─▶ SQL Samples 검색
   │   └─▶ Instructions 검색
   │
   ├─▶ SQL Generation
   │   └─▶ LLM 호출 (프롬프트 + 컨텍스트)
   │
   └─▶ SQL Validation
       └─▶ Wren Engine으로 검증
       └─▶ 오류 시 Correction (최대 3회)

4. 응답 반환
   └─▶ {
         "sql": "SELECT SUM(revenue)...",
         "reasoning": "분기별 매출을 집계...",
         "status": "finished"
       }

5. 차트 생성 (선택)
   └─▶ POST /v1/charts
   └─▶ Vega-Lite 스펙 생성
```

---

## 서비스 통신

### 내부 통신 매트릭스

| From | To | Protocol | Purpose |
|------|-----|----------|---------|
| UI | AI Service | REST | SQL 생성, 차트 생성 |
| UI | Engine | REST | MDL 배포, SQL 실행 |
| AI Service | Qdrant | gRPC | 벡터 검색/저장 |
| AI Service | Engine | REST | SQL 검증 |
| AI Service | LLM | REST | 텍스트 생성 |

### Docker 네트워크

```yaml
# docker-compose.yaml
networks:
  wren:
    driver: bridge

services:
  wren-ui:
    networks: [wren]
    environment:
      - WREN_AI_ENDPOINT=http://wren-ai-service:5555
      - WREN_ENGINE_ENDPOINT=http://wren-engine:8080

  wren-ai-service:
    networks: [wren]
    depends_on:
      - qdrant
```

---

## 확장성 고려사항

### 수평 확장

```
┌─────────────────────────────────────────┐
│           Load Balancer                  │
└─────────────┬───────────────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│ UI-1  │ │ UI-2  │ │ UI-3  │   (Stateless)
└───────┘ └───────┘ └───────┘
    │         │         │
    └─────────┼─────────┘
              ▼
┌─────────────────────────────────────────┐
│     Shared PostgreSQL (메타데이터)       │
└─────────────────────────────────────────┘
```

### 성능 특성

| 작업 | 평균 시간 | 병목 |
|------|----------|------|
| SQL 생성 | 3-10초 | LLM API |
| 차트 생성 | 2-5초 | LLM API |
| 벡터 검색 | <500ms | Qdrant |
| SQL 검증 | <1초 | Engine |

---

*다음 글에서는 MDL(Metadata Definition Language)을 상세히 살펴봅니다.*
