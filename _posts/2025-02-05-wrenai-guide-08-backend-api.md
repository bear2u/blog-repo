---
layout: post
title: "WrenAI 완벽 가이드 (8) - 백엔드 API"
date: 2025-02-05
permalink: /wrenai-guide-08-backend-api/
author: Canner
categories: [AI 에이전트, WrenAI]
tags: [WrenAI, FastAPI, REST API, Backend, Python]
original_url: "https://github.com/Canner/WrenAI"
excerpt: "WrenAI AI Service의 FastAPI 백엔드 API 구조와 엔드포인트를 분석합니다."
---

## 백엔드 API 개요

WrenAI AI Service는 **FastAPI**로 구축된 REST API를 제공합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                 Wren AI Service API                          │
├─────────────────────────────────────────────────────────────┤
│  📍 기본 URL: http://localhost:5555                         │
│  📖 문서: http://localhost:5555/docs (Swagger)              │
│  📘 ReDoc: http://localhost:5555/redoc                      │
└─────────────────────────────────────────────────────────────┘
```

---

## API 엔드포인트 목록

### Ask (질문) API

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/v1/asks` | 질문 제출 |
| PATCH | `/v1/asks/{query_id}` | 질문 중단 |
| GET | `/v1/asks/{query_id}/result` | 결과 조회 (폴링) |
| GET | `/v1/asks/{query_id}/streaming-result` | 결과 스트리밍 |

### Chart (차트) API

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/v1/charts` | 차트 생성 |
| PATCH | `/v1/charts/{query_id}` | 차트 조정 |
| GET | `/v1/charts/{query_id}/result` | 차트 결과 조회 |

### Semantics (스키마) API

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/v1/semantics/preparations` | MDL 색인화 |
| PATCH | `/v1/semantics/preparations/{id}` | MDL 업데이트 |
| DELETE | `/v1/semantics/preparations/{id}` | MDL 삭제 |
| GET | `/v1/semantics/descriptions` | 스키마 설명 조회 |

### Knowledge (지식) API

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/v1/sql_pairs` | SQL 쌍 추가 |
| GET | `/v1/sql_pairs` | SQL 쌍 조회 |
| PATCH | `/v1/sql_pairs/{id}` | SQL 쌍 수정 |
| DELETE | `/v1/sql_pairs/{id}` | SQL 쌍 삭제 |
| POST | `/v1/instructions` | 지시사항 추가 |
| GET | `/v1/instructions` | 지시사항 조회 |
| PATCH | `/v1/instructions/{id}` | 지시사항 수정 |
| DELETE | `/v1/instructions/{id}` | 지시사항 삭제 |

### Recommendations (추천) API

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/v1/question_recommendations` | 질문 추천 |
| POST | `/v1/relationship_recommendations` | 관계 추천 |

### Feedback (피드백) API

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/v1/feedbacks` | 피드백 제출 |

---

## Ask API 상세

### 질문 제출

```bash
POST /v1/asks

# Request
{
  "query": "지난 분기 매출은 얼마인가요?",
  "project_id": "project-uuid",
  "configurations": {
    "language": "ko",
    "timezone": "Asia/Seoul"
  }
}

# Response
{
  "query_id": "query-uuid"
}
```

### 결과 조회 (폴링)

```bash
GET /v1/asks/{query_id}/result

# Response (진행 중)
{
  "status": "understanding",
  "type": null,
  "response": null,
  "error": null
}

# Response (완료)
{
  "status": "finished",
  "type": "TEXT_TO_SQL",
  "response": [
    {
      "sql": "SELECT SUM(amount) as total_revenue FROM orders WHERE quarter = 'Q4'",
      "summary": "지난 분기(Q4)의 총 매출을 조회합니다.",
      "type": "llm"
    }
  ],
  "error": null
}

# Response (실패)
{
  "status": "failed",
  "type": null,
  "response": null,
  "error": {
    "code": "GENERAL_ERROR",
    "message": "Failed to generate SQL"
  }
}
```

### 상태 값

| 상태 | 설명 |
|------|------|
| `understanding` | 질문 분석 중 |
| `searching` | 컨텍스트 검색 중 |
| `generating` | SQL 생성 중 |
| `finished` | 완료 |
| `failed` | 실패 |
| `stopped` | 중단됨 |

---

## Chart API 상세

### 차트 생성

```bash
POST /v1/charts

# Request
{
  "query": "월별 매출 추이를 차트로 보여줘",
  "sql": "SELECT month, SUM(amount) as revenue FROM orders GROUP BY month",
  "data": [
    {"month": "2024-01", "revenue": 1000000},
    {"month": "2024-02", "revenue": 1200000}
  ],
  "project_id": "project-uuid"
}

# Response
{
  "query_id": "chart-query-uuid"
}
```

### 차트 결과 조회

```bash
GET /v1/charts/{query_id}/result

# Response
{
  "status": "finished",
  "response": {
    "chart_type": "line",
    "chart_schema": {
      "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
      "mark": "line",
      "encoding": {
        "x": {"field": "month", "type": "temporal"},
        "y": {"field": "revenue", "type": "quantitative"}
      }
    },
    "reasoning": "월별 매출 추이를 보여주기 위해 라인 차트를 선택했습니다."
  }
}
```

---

## Semantics API 상세

### MDL 색인화

```bash
POST /v1/semantics/preparations

# Request
{
  "mdl": {
    "catalog": "ecommerce",
    "schema": "public",
    "models": [
      {
        "name": "orders",
        "columns": [
          {"name": "id", "type": "int"},
          {"name": "amount", "type": "decimal"}
        ]
      }
    ]
  },
  "project_id": "project-uuid"
}

# Response
{
  "id": "preparation-uuid"
}
```

### 색인화 상태 조회

```bash
GET /v1/semantics/preparations/{id}/status

# Response
{
  "status": "finished"
}
```

---

## 코드 구조

### 라우터

```python
# src/web/v1/routers/ask.py

from fastapi import APIRouter, Depends
from src.web.v1.services import AskService

router = APIRouter(prefix="/v1/asks", tags=["Ask"])

@router.post("")
async def ask(
    request: AskRequest,
    service: AskService = Depends(get_ask_service)
):
    query_id = await service.ask(
        query=request.query,
        project_id=request.project_id,
        configurations=request.configurations
    )
    return {"query_id": query_id}

@router.get("/{query_id}/result")
async def get_result(
    query_id: str,
    service: AskService = Depends(get_ask_service)
):
    result = await service.get_result(query_id)
    return result

@router.patch("/{query_id}")
async def stop_ask(
    query_id: str,
    request: StopAskRequest,
    service: AskService = Depends(get_ask_service)
):
    await service.stop(query_id, request.status)
    return {"status": "stopped"}
```

### 서비스

```python
# src/web/v1/services/ask.py

from src.pipelines import (
    IntentClassificationPipeline,
    SQLGenerationPipeline,
    SQLCorrectionPipeline
)

class AskService:
    def __init__(
        self,
        intent_pipeline: IntentClassificationPipeline,
        sql_pipeline: SQLGenerationPipeline,
        correction_pipeline: SQLCorrectionPipeline,
        cache: QueryCache
    ):
        self.intent_pipeline = intent_pipeline
        self.sql_pipeline = sql_pipeline
        self.correction_pipeline = correction_pipeline
        self.cache = cache

    async def ask(
        self,
        query: str,
        project_id: str,
        configurations: dict
    ) -> str:
        # 1. 캐시 확인
        cached = await self.cache.get(query, project_id)
        if cached:
            return cached.query_id

        # 2. 의도 분류
        intent = await self.intent_pipeline.run(query)

        if intent.type == "TEXT_TO_SQL":
            # 3. SQL 생성
            result = await self.sql_pipeline.run(
                query=query,
                project_id=project_id
            )

            # 4. SQL 검증 및 수정
            if result.needs_correction:
                result = await self.correction_pipeline.run(
                    sql=result.sql,
                    error=result.error,
                    max_retries=3
                )

        # 5. 결과 저장
        query_id = await self.cache.set(query, project_id, result)

        return query_id

    async def get_result(self, query_id: str) -> AskResult:
        return await self.cache.get_result(query_id)
```

---

## 요청/응답 모델

```python
# src/web/v1/models.py

from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class AskStatus(str, Enum):
    UNDERSTANDING = "understanding"
    SEARCHING = "searching"
    GENERATING = "generating"
    FINISHED = "finished"
    FAILED = "failed"
    STOPPED = "stopped"

class AskRequest(BaseModel):
    query: str
    project_id: str
    configurations: Optional[dict] = None

class AskResponse(BaseModel):
    query_id: str

class AskResultResponse(BaseModel):
    status: AskStatus
    type: Optional[str] = None
    response: Optional[List[dict]] = None
    error: Optional[dict] = None

class ChartRequest(BaseModel):
    query: str
    sql: str
    data: List[dict]
    project_id: str

class ChartResultResponse(BaseModel):
    status: AskStatus
    response: Optional[dict] = None
    error: Optional[dict] = None
```

---

## 에러 코드

| 코드 | 설명 |
|------|------|
| `GENERAL_ERROR` | 일반 오류 |
| `NO_RELEVANT_DATA` | 관련 데이터 없음 |
| `NO_RELEVANT_SQL` | 관련 SQL 없음 |
| `MISLEADING_QUERY` | 모호한 질문 |
| `SQL_GENERATION_FAILED` | SQL 생성 실패 |
| `SQL_VALIDATION_FAILED` | SQL 검증 실패 |

---

## Python 클라이언트 예제

```python
import httpx
import asyncio

class WrenAIClient:
    def __init__(self, base_url: str = "http://localhost:5555"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()

    async def ask(self, query: str, project_id: str) -> dict:
        # 1. 질문 제출
        response = await self.client.post(
            f"{self.base_url}/v1/asks",
            json={
                "query": query,
                "project_id": project_id
            }
        )
        query_id = response.json()["query_id"]

        # 2. 결과 폴링
        while True:
            result = await self.client.get(
                f"{self.base_url}/v1/asks/{query_id}/result"
            )
            data = result.json()

            if data["status"] in ["finished", "failed", "stopped"]:
                return data

            await asyncio.sleep(1)

# 사용 예시
async def main():
    client = WrenAIClient()
    result = await client.ask(
        query="지난 분기 매출은?",
        project_id="my-project"
    )
    print(result)

asyncio.run(main())
```

---

*다음 글에서는 배포 가이드를 살펴봅니다.*
