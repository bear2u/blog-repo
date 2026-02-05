---
layout: post
title: "WrenAI 완벽 가이드 (5) - RAG 파이프라인"
date: 2025-02-05
permalink: /wrenai-guide-05-rag-pipeline/
author: Canner
categories: [AI 에이전트, WrenAI]
tags: [WrenAI, RAG, Pipeline, Vector Search, Haystack]
original_url: "https://github.com/Canner/WrenAI"
excerpt: "WrenAI의 RAG(Retrieval-Augmented Generation) 파이프라인 구조와 작동 원리를 분석합니다."
---

## RAG 파이프라인 개요

WrenAI는 **RAG(Retrieval-Augmented Generation)** 패턴을 사용하여 정확한 SQL을 생성합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG 파이프라인 흐름                      │
└─────────────────────────────────────────────────────────────┘

1. Indexing (색인화)
   DB 스키마, 예제, 지시사항 → 벡터 임베딩 → Qdrant 저장

2. Retrieval (검색)
   사용자 질문 → 임베딩 → 유사 문서 검색

3. Generation (생성)
   질문 + 검색된 컨텍스트 → LLM → SQL 생성
```

---

## 파이프라인 종류

### Indexing 파이프라인

| 파이프라인 | 역할 | 입력 |
|-----------|------|------|
| `db_schema_indexing` | DB 스키마 색인 | 테이블, 컬럼 정보 |
| `historical_question_indexing` | 과거 질문 색인 | Q&A 히스토리 |
| `table_description_indexing` | 테이블 설명 색인 | 사용자 정의 설명 |
| `sql_pairs_indexing` | SQL 예제 색인 | 질문-SQL 쌍 |
| `instructions_indexing` | 지시사항 색인 | 사용자 지시사항 |

### Retrieval 파이프라인

| 파이프라인 | 역할 | 출력 |
|-----------|------|------|
| `db_schema_retrieval` | 관련 스키마 검색 | 테이블/컬럼 정보 |
| `historical_question_retrieval` | 유사 질문 검색 | 과거 Q&A |
| `sql_pairs_retrieval` | SQL 예제 검색 | 참고 SQL |
| `instructions_retrieval` | 지시사항 검색 | 관련 지시사항 |

### Generation 파이프라인

| 파이프라인 | 역할 | 출력 |
|-----------|------|------|
| `intent_classification` | 의도 분류 | TEXT_TO_SQL / DATA_ASSISTANCE |
| `sql_generation` | SQL 생성 | SQL 쿼리 |
| `sql_correction` | SQL 수정 | 수정된 SQL |
| `chart_generation` | 차트 생성 | Vega-Lite 스펙 |

---

## 상세 흐름

### 1. Schema Indexing

```python
# 스키마 색인화 프로세스
def index_db_schema(mdl: dict):
    documents = []

    for model in mdl["models"]:
        # 테이블 문서 생성
        table_doc = {
            "type": "TABLE_SCHEMA",
            "name": model["name"],
            "columns": model["columns"],
            "content": generate_ddl(model),  # CREATE TABLE 문
            "metadata": model.get("properties", {})
        }
        documents.append(table_doc)

        # 컬럼별 문서 생성
        for column in model["columns"]:
            col_doc = {
                "type": "COLUMN",
                "table": model["name"],
                "name": column["name"],
                "type": column["type"],
                "content": f"{model['name']}.{column['name']}"
            }
            documents.append(col_doc)

    # 임베딩 및 저장
    embeddings = embedder.embed(documents)
    qdrant.upsert(collection, documents, embeddings)
```

### 2. Context Retrieval

```python
# 컨텍스트 검색 프로세스
def retrieve_context(question: str, project_id: int):
    # 질문 임베딩
    query_embedding = embedder.embed(question)

    # 1. 스키마 검색
    schema_docs = qdrant.search(
        collection=f"table_schema_{project_id}",
        query_vector=query_embedding,
        limit=10,
        score_threshold=0.7
    )

    # 2. 과거 질문 검색
    similar_questions = qdrant.search(
        collection=f"historical_questions_{project_id}",
        query_vector=query_embedding,
        limit=5,
        score_threshold=0.9
    )

    # 3. SQL 예제 검색
    sql_samples = qdrant.search(
        collection=f"sql_samples_{project_id}",
        query_vector=query_embedding,
        limit=10,
        score_threshold=0.7
    )

    # 4. 지시사항 검색
    instructions = qdrant.search(
        collection=f"instructions_{project_id}",
        query_vector=query_embedding,
        limit=5,
        score_threshold=0.7
    )

    return {
        "schema": schema_docs,
        "similar_questions": similar_questions,
        "sql_samples": sql_samples,
        "instructions": instructions
    }
```

### 3. SQL Generation

```python
# SQL 생성 프로세스
def generate_sql(question: str, context: dict):
    prompt = f"""
### DATABASE SCHEMA ###
{format_schema(context["schema"])}

### SIMILAR QUESTIONS ###
{format_similar_questions(context["similar_questions"])}

### SQL EXAMPLES ###
{format_sql_samples(context["sql_samples"])}

### USER INSTRUCTIONS ###
{format_instructions(context["instructions"])}

### QUESTION ###
{question}

### TASK ###
Generate a SQL query that answers the question.
Think step by step:
1. Identify required tables
2. Determine necessary columns
3. Define join conditions
4. Apply filters and aggregations
5. Write the final SQL

### SQL ###
"""

    response = llm.generate(
        prompt=prompt,
        system_prompt=SQL_GENERATION_SYSTEM_PROMPT,
        temperature=0,
        max_tokens=4096
    )

    return extract_sql(response)
```

---

## 파이프라인 설정

### config.yaml 파이프라인 섹션

```yaml
type: pipeline
pipes:
  # SQL 생성 파이프라인
  - name: sql_generation
    llm: litellm_llm.default
    engine: wren_ui
    document_store: qdrant

  # SQL 수정 파이프라인
  - name: sql_correction
    llm: litellm_llm.default
    engine: wren_ui
    document_store: qdrant

  # 차트 생성 파이프라인
  - name: chart_generation
    llm: litellm_llm.default
    document_store: qdrant

  # 의도 분류 파이프라인
  - name: intent_classification
    llm: litellm_llm.default

settings:
  # 검색 설정
  table_retrieval_size: 10
  table_column_retrieval_size: 100
  historical_question_retrieval_similarity_threshold: 0.9
  sql_pairs_similarity_threshold: 0.7
  sql_pairs_retrieval_max_size: 10
  instructions_similarity_threshold: 0.7

  # 생성 설정
  allow_intent_classification: true
  allow_sql_generation_reasoning: true
  allow_sql_functions_retrieval: true
  max_sql_correction_retries: 3
```

---

## Qdrant 컬렉션 구조

### 컬렉션 생성

```python
# 컬렉션 설정
collections = {
    "table_schema": {
        "size": 3072,  # embedding dimension
        "distance": "Cosine"
    },
    "historical_questions": {
        "size": 3072,
        "distance": "Cosine"
    },
    "sql_samples": {
        "size": 3072,
        "distance": "Cosine"
    },
    "instructions": {
        "size": 3072,
        "distance": "Cosine"
    }
}
```

### 문서 구조

```json
// Table Schema 문서
{
  "id": "uuid",
  "content": "CREATE TABLE customers (id INT, name VARCHAR, ...)",
  "metadata": {
    "type": "TABLE_SCHEMA",
    "table_name": "customers",
    "column_count": 5,
    "description": "고객 정보"
  },
  "vector": [0.1, 0.2, ...] // 3072 dimensions
}

// Historical Question 문서
{
  "id": "uuid",
  "content": "지난 분기 매출은 얼마인가요?",
  "metadata": {
    "sql": "SELECT SUM(amount) FROM orders WHERE quarter = 'Q4'",
    "timestamp": "2024-02-05",
    "success": true
  },
  "vector": [...]
}
```

---

## 파이프라인 최적화

### 1. 검색 임계값 조정

```yaml
settings:
  # 높은 임계값 = 더 정확한 결과만
  historical_question_retrieval_similarity_threshold: 0.9

  # 낮은 임계값 = 더 많은 결과
  sql_pairs_similarity_threshold: 0.7
```

### 2. 검색 결과 수 조정

```yaml
settings:
  table_retrieval_size: 10           # 테이블 10개
  table_column_retrieval_size: 100   # 컬럼 100개
  sql_pairs_retrieval_max_size: 10   # SQL 예제 10개
```

### 3. 컬럼 프루닝

```yaml
settings:
  enable_column_pruning: true  # 불필요한 컬럼 제거
```

---

## 디버깅

### 파이프라인 로그

```bash
# AI Service 로그 확인
docker compose logs -f wren-ai-service

# 로그 레벨 조정
settings:
  logging_level: DEBUG
```

### LangFuse 추적

```yaml
settings:
  langfuse_enable: true
  langfuse_host: https://cloud.langfuse.com
```

```bash
# 환경변수
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-...
```

---

*다음 글에서는 LLM 연동 방법을 상세히 살펴봅니다.*
