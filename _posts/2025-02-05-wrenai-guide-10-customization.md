---
layout: post
title: "WrenAI 완벽 가이드 (10) - 확장 및 커스터마이징"
date: 2025-02-05
permalink: /wrenai-guide-10-customization/
author: Canner
categories: [AI 에이전트, WrenAI]
tags: [WrenAI, Customization, Extension, Pipeline, Plugin]
original_url: "https://github.com/Canner/WrenAI"
excerpt: "WrenAI의 파이프라인과 기능을 커스터마이징하고 확장하는 방법을 안내합니다."
---

## 커스터마이징 개요

WrenAI는 다양한 수준의 커스터마이징을 지원합니다:

```
┌─────────────────────────────────────────────────────────────┐
│                  커스터마이징 레벨                           │
├─────────────────────────────────────────────────────────────┤
│  🔧 설정 레벨: config.yaml 수정                             │
│  📝 프롬프트 레벨: 시스템 프롬프트 수정                     │
│  🔌 파이프라인 레벨: 파이프라인 추가/수정                   │
│  🛠️ 코드 레벨: 소스 코드 직접 수정                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 설정 커스터마이징

### 검색 파라미터 조정

```yaml
# config.yaml
settings:
  # 테이블 검색 설정
  table_retrieval_size: 15              # 기본 10 → 15
  table_column_retrieval_size: 150      # 기본 100 → 150

  # 유사도 임계값
  historical_question_retrieval_similarity_threshold: 0.85  # 기본 0.9 → 0.85
  sql_pairs_similarity_threshold: 0.65   # 기본 0.7 → 0.65

  # SQL 수정 재시도
  max_sql_correction_retries: 5          # 기본 3 → 5
```

### 기능 토글

```yaml
settings:
  # 의도 분류 비활성화 (항상 TEXT_TO_SQL)
  allow_intent_classification: false

  # SQL 추론 단계 비활성화 (더 빠른 응답)
  allow_sql_generation_reasoning: false

  # SQL 함수 검색 활성화
  allow_sql_functions_retrieval: true

  # SQL 진단 활성화
  allow_sql_diagnosis: true

  # 컬럼 프루닝 활성화
  enable_column_pruning: true
```

---

## 프롬프트 커스터마이징

### SQL 생성 프롬프트

```python
# src/pipelines/generation/sql_generation.py

SQL_GENERATION_SYSTEM_PROMPT = """
You are a SQL expert assistant.
Generate accurate SQL queries based on the user's question and provided schema.

Rules:
1. Use only the tables and columns provided in the schema
2. Always use proper JOIN conditions
3. Use explicit column aliases for calculated fields
4. Handle NULL values appropriately
5. Use appropriate aggregation functions

Language: Generate SQL comments in {language}
"""

# 커스터마이징 예시: 특정 DB 방언 추가
SQL_GENERATION_SYSTEM_PROMPT_SNOWFLAKE = SQL_GENERATION_SYSTEM_PROMPT + """
Additional Snowflake Rules:
- Use ILIKE for case-insensitive matching
- Use :: for type casting (e.g., column::VARCHAR)
- Use FLATTEN for nested JSON
"""
```

### 차트 생성 프롬프트

```python
# src/pipelines/generation/chart_generation.py

CHART_GENERATION_SYSTEM_PROMPT = """
You are a data visualization expert.
Generate Vega-Lite specifications for the given data.

Guidelines:
1. Choose appropriate chart type based on data characteristics
2. Use clear labels and titles
3. Apply color schemes appropriately
4. Consider accessibility (colorblind-friendly palettes)

Output: JSON Vega-Lite specification only
"""
```

---

## 파이프라인 커스터마이징

### 커스텀 파이프라인 생성

```python
# src/pipelines/generation/custom_sql_generation.py

from src.core.pipeline import BasicPipeline
from src.providers.llm import LLMProvider
from src.providers.document_store import DocumentStoreProvider

class CustomSQLGenerationPipeline(BasicPipeline):
    def __init__(
        self,
        llm_provider: LLMProvider,
        document_store: DocumentStoreProvider,
        custom_rules: list[str] = None
    ):
        self.llm = llm_provider
        self.store = document_store
        self.custom_rules = custom_rules or []

    async def run(
        self,
        query: str,
        project_id: str,
        **kwargs
    ) -> dict:
        # 1. 컨텍스트 검색
        context = await self.retrieve_context(query, project_id)

        # 2. 커스텀 규칙 추가
        rules = self.format_custom_rules()

        # 3. 프롬프트 구성
        prompt = self.build_prompt(query, context, rules)

        # 4. LLM 호출
        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.get_system_prompt()
        )

        # 5. 후처리
        sql = self.extract_sql(response)
        sql = self.apply_custom_transforms(sql)

        return {
            "sql": sql,
            "reasoning": response,
            "context": context
        }

    def format_custom_rules(self) -> str:
        if not self.custom_rules:
            return ""
        rules = "\n".join(f"- {rule}" for rule in self.custom_rules)
        return f"\n### CUSTOM RULES ###\n{rules}\n"

    def apply_custom_transforms(self, sql: str) -> str:
        # 커스텀 SQL 변환 로직
        # 예: 특정 테이블명 변환, 스키마 프리픽스 추가 등
        return sql
```

### 파이프라인 등록

```python
# src/globals.py

from src.pipelines.generation.custom_sql_generation import CustomSQLGenerationPipeline

class ServiceContainer:
    def __init__(self, config: Config):
        # ... 기존 초기화 코드 ...

        # 커스텀 파이프라인 등록
        self.custom_sql_pipeline = CustomSQLGenerationPipeline(
            llm_provider=self.llm_provider,
            document_store=self.document_store,
            custom_rules=[
                "Always use LIMIT 1000 for safety",
                "Prefer window functions over subqueries",
                "Add execution hints for large tables"
            ]
        )
```

---

## 커스텀 LLM 제공자

```python
# src/providers/llm/custom_llm.py

from src.core.provider import LLMProvider

class CustomLLMProvider(LLMProvider):
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        **kwargs
    ) -> str:
        # 커스텀 LLM API 호출
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.endpoint,
                json={
                    "prompt": prompt,
                    "system": system_prompt,
                    "max_tokens": kwargs.get("max_tokens", 4096),
                    "temperature": kwargs.get("temperature", 0)
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}"
                }
            )
            return response.json()["text"]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # 커스텀 임베딩 API 호출
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoint}/embeddings",
                json={"texts": texts},
                headers={
                    "Authorization": f"Bearer {self.api_key}"
                }
            )
            return response.json()["embeddings"]
```

### 설정에서 사용

```yaml
# config.yaml
type: llm
provider: custom_llm
models:
  - alias: default
    api_key: ${CUSTOM_LLM_API_KEY}
    endpoint: https://my-llm-api.com/v1/generate
```

---

## 커스텀 데이터소스

```python
# src/providers/engine/custom_engine.py

from src.core.provider import EngineProvider

class CustomEngineProvider(EngineProvider):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    async def validate_sql(self, sql: str) -> dict:
        # SQL 검증 로직
        try:
            # 파싱만 수행 (실행하지 않음)
            parsed = sqlparse.parse(sql)
            return {"valid": True, "error": None}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    async def execute_sql(self, sql: str) -> dict:
        # SQL 실행 및 결과 반환
        async with self.get_connection() as conn:
            result = await conn.execute(sql)
            columns = [col.name for col in result.description]
            rows = await result.fetchall()
            return {
                "columns": columns,
                "data": [dict(zip(columns, row)) for row in rows]
            }

    async def get_schema(self) -> dict:
        # 스키마 조회
        async with self.get_connection() as conn:
            # 데이터베이스별 스키마 조회 쿼리
            result = await conn.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
            """)
            return self.format_schema(await result.fetchall())
```

---

## UI 커스터마이징

### 테마 수정

```less
// src/styles/variables.less

@primary-color: #1890ff;    // 메인 색상
@link-color: #1890ff;       // 링크 색상
@success-color: #52c41a;    // 성공 색상
@warning-color: #faad14;    // 경고 색상
@error-color: #f5222d;      // 오류 색상

@font-size-base: 14px;      // 기본 폰트 크기
@border-radius-base: 4px;   // 기본 테두리 반경
```

### 커스텀 컴포넌트

```tsx
// src/components/custom/CustomAskInput.tsx

import { Input, Button } from 'antd';
import { useAsk } from '@/hooks/useAsk';

export function CustomAskInput({ projectId }: { projectId: number }) {
  const [question, setQuestion] = useState('');
  const { ask, loading, result } = useAsk();

  const handleAsk = async () => {
    if (!question.trim()) return;
    await ask(projectId, question);
  };

  return (
    <div className="custom-ask-input">
      <Input.TextArea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="데이터에 대해 질문하세요..."
        autoSize={{ minRows: 2, maxRows: 6 }}
      />
      <Button
        type="primary"
        onClick={handleAsk}
        loading={loading}
      >
        질문하기
      </Button>
      {result && (
        <div className="result">
          <pre>{result.sql}</pre>
        </div>
      )}
    </div>
  );
}
```

---

## 플러그인 시스템

### 플러그인 인터페이스

```python
# src/plugins/base.py

from abc import ABC, abstractmethod

class WrenAIPlugin(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def on_before_ask(self, query: str, context: dict) -> dict:
        """질문 전처리"""
        pass

    @abstractmethod
    async def on_after_ask(self, query: str, result: dict) -> dict:
        """결과 후처리"""
        pass

class LoggingPlugin(WrenAIPlugin):
    def name(self) -> str:
        return "logging"

    async def on_before_ask(self, query: str, context: dict) -> dict:
        logger.info(f"Query: {query}")
        return context

    async def on_after_ask(self, query: str, result: dict) -> dict:
        logger.info(f"Result: {result['status']}")
        return result
```

---

## 모니터링 확장

### 커스텀 메트릭

```python
# src/monitoring/metrics.py

from prometheus_client import Counter, Histogram

# 커스텀 메트릭
ask_requests_total = Counter(
    'wrenai_ask_requests_total',
    'Total number of ask requests',
    ['project_id', 'status']
)

sql_generation_duration = Histogram(
    'wrenai_sql_generation_duration_seconds',
    'Time spent generating SQL',
    ['model']
)

# 사용
@sql_generation_duration.labels(model='gpt-4o-mini').time()
async def generate_sql(query: str) -> str:
    # SQL 생성 로직
    pass
```

---

## 베스트 프랙티스

### 1. 점진적 커스터마이징

```
1단계: 설정 파일로 조정
   ↓
2단계: 프롬프트 수정
   ↓
3단계: 파이프라인 확장
   ↓
4단계: 코드 수정 (최후의 수단)
```

### 2. 테스트 작성

```python
# tests/test_custom_pipeline.py

import pytest
from src.pipelines.generation.custom_sql_generation import CustomSQLGenerationPipeline

@pytest.fixture
def pipeline():
    return CustomSQLGenerationPipeline(
        llm_provider=MockLLMProvider(),
        document_store=MockDocumentStore(),
        custom_rules=["Always use LIMIT"]
    )

async def test_custom_rules_applied(pipeline):
    result = await pipeline.run(
        query="Show all customers",
        project_id="test"
    )
    assert "LIMIT" in result["sql"]
```

### 3. 버전 관리

```bash
# 커스터마이징 브랜치 생성
git checkout -b custom/my-company

# 업스트림 변경 병합
git fetch upstream
git merge upstream/main
```

---

## 요약

WrenAI는 다양한 수준의 커스터마이징을 지원합니다:

| 레벨 | 방법 | 난이도 |
|------|------|--------|
| 설정 | config.yaml 수정 | 쉬움 |
| 프롬프트 | 시스템 프롬프트 수정 | 보통 |
| 파이프라인 | Python 클래스 확장 | 보통 |
| 코드 | 소스 직접 수정 | 어려움 |

---

*이것으로 WrenAI 완벽 가이드 시리즈를 마칩니다. 자연어를 SQL로 변환하는 GenBI의 세계를 탐험해보세요!*
