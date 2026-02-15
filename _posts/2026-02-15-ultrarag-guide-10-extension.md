---
layout: post
title: "UltraRAG 완벽 가이드 (10) - 확장 및 커스터마이징"
date: 2026-02-15
permalink: /ultrarag-guide-10-extension/
author: UltraRAG Team
categories: [AI 에이전트, RAG]
tags: [RAG, MCP, LLM, UltraRAG, Extension, Custom, Plugin]
original_url: "https://github.com/OpenBMB/UltraRAG"
excerpt: "UltraRAG의 확장과 커스터마이징: 커스텀 Server, Plugin, 기존 코드 통합 방법을详细介绍합니다."
---

## 확장 개요

UltraRAG는 **모듈식 설계**로 인해 다양한 확장이 가능합니다:

- **커스텀 Server**: 새로운 기능 추가
- **Plugin**: 외부 플러그인 연동
- **코드 통합**: 기존 Python 코드와 통합
- **커스텀 컴포넌트**:独自の 검색기/생성기 개발

---

## 커스텀 Server 개발

### Server 구조

```python
# servers/custom/my_server.py
from mcp.server import Server
from pydantic import AnyUrl
import asyncio

# Server 초기화
app = Server("my-custom-server")

# Tool 정의
@app.list_tools()
async def list_tools():
    return [
        {
            "name": "my_custom_function",
            "description": "사용자 정의 함수",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"},
                    "param2": {"type": "number"}
                },
                "required": ["param1"]
            }
        }
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "my_custom_function":
        # 로직 구현
        result = do_something(arguments["param1"], arguments.get("param2"))
        return result
    raise ValueError(f"Unknown tool: {name}")
```

### Server 등록

```yaml
# my_pipeline.yaml
servers:
  custom: servers/custom

pipeline:
  - custom.my_custom_function:
      param1: "value1"
      param2: 42
```

---

## 커스텀 Retriever

###独自の 임베딩 모델 지원

```python
from ultrarag.retriever import BaseRetriever

class MyEmbeddingRetriever(BaseRetriever):
    name = "my-embedding"

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = load_my_model(model_path)
        self.device = device

    def encode(self, texts: list[str]):
        return self.model.encode(texts)

    def search(self, query: str, top_k: int = 10):
        query_embedding = self.encode([query])[0]
        results = self.vector_db.search(query_embedding, top_k)
        return results
```

### 사용

```yaml
pipeline:
  - retriever.retriever_init:
      method: "custom"
      retriever_class: "my_module.MyEmbeddingRetriever"
      model_path: "./models/my-embedding.pt"
```

---

## 커스텀 Generation

### 자체 LLM 연동

```python
from ultrarag.generation import BaseGenerator

class MyLLMGenerator(BaseGenerator):
    name = "my-llm"

    def __init__(self, model: str, api_url: str):
        self.model = model
        self.api_url = api_url

    def generate(self, prompt: str, **kwargs):
        response = call_my_api(
            model=self.model,
            prompt=prompt,
            **kwargs
        )
        return response.text
```

---

## 기존 코드 통합

### Python 코드에서 UltraRAG 사용

```python
from ultrarag import UltraRAGClient

# 클라이언트 초기화
client = UltraRAGClient(
    config_path="./configs/my_rag.yaml",
    components={
        "my_retriever": MyCustomRetriever(),
        "my_generator": MyCustomGenerator()
    }
)

# 검색
query = "질문"
docs = client.retrieve(query, top_k=5)

# 생성
answer = client.generate(
    query=query,
    context=docs,
    prompt_template="custom_template"
)
```

### API 서버로 노출

```python
from ultrarag.api import create_app

app = create_app(
    config_path="./configs/rag.yaml",
    components={...}
)

# Flask 앱 실행
app.run(host="0.0.0.0", port=5000)
```

---

## Plugin 시스템

### Plugin 개발

```python
# plugins/my_plugin.py
from ultrarag.plugin import BasePlugin

class MyPlugin(BasePlugin):
    name = "my_plugin"
    version = "1.0.0"

    def on_retrieve(self, query, results):
        # 후처리
        return self.filter_results(results)

    def on_generate(self, prompt, response):
        # 후처리
        return self.format_response(response)

    def install(self, app):
        # 설치 시 실행
        app.register_hook("retrieve", self.on_retrieve)
        app.register_hook("generate", self.on_generate)
```

### Plugin 사용

```yaml
pipeline:
  - retriever.retriever_search
  - plugins.my_plugin.on_retrieve  # 후처리 적용
  - generation.generate
```

---

## Workflow 확장

### 커스텀 조건 분기

```python
# servers/custom/condition.py
from ultrarag.workflow import BranchCondition

class MyCondition(BranchCondition):
    def evaluate(self, context: dict) -> str:
        query = context.get("query", "")

        if "비교" in query:
            return "compare"
        elif "분석" in query:
            return "analyze"
        else:
            return "default"
```

### 사용

```yaml
pipeline:
  - custom.classify_query

  - branch:
      conditions:
        - if: "branch == 'compare'"
          then:
            - retriever.search_comparison
            - generation.compare

        - if: "branch == 'analyze'"
          then:
            - retriever.search_analysis
            - generation.analyze
```

---

## 환경 변수 및 설정

### .env 파일

```bash
# .env
OPENAI_API_KEY=sk-...
MILVUS_HOST=localhost
MILVUS_PORT=19530
LOG_LEVEL=INFO
```

### 설정 파일

```yaml
# config.yaml
environment:
  log_level: "DEBUG"
  cache_enabled: true
  cache_ttl: 3600

retriever:
  default_top_k: 10
  timeout: 30

generation:
  default_temperature: 0.7
  default_max_tokens: 2048
```

---

## 커뮤니티 Plugin

| Plugin | 설명 |
|--------|------|
| `ultrarag-plugin-slack` | Slack 연동 |
| `ultrarag-plugin-discord` | Discord 연동 |
| `ultrarag-plugin-notion` | Notion 연동 |
| `ultrarag-plugin-arxiv` | ArXiv 검색 |

---

## 기여 방법

### Pull Request 프로세스

1. **Fork** 저장소
2. **Issue** 생성 (버그 리포트 또는 기능 요청)
3. **Branch** 생성: `feature/my-feature`
4. **개발** 및 **테스트**
5. **PR** 생성

### 코딩 컨벤션

```python
# 테스트 작성
def test_my_function():
    # Given
    input_data = "test"

    # When
    result = my_function(input_data)

    # Then
    assert result == expected
```

---

## 결론

UltraRAG는 **MCP 기반 모듈식 설계**로 인해 다음과 같은 강점을 갖습니다:

- **쉬운 확장**: 새로운 기능 추가 용이
- **유연성**: 다양한 사용 시나리오 지원
- **커스터마이징**: 특정 요구에 맞게 조정 가능
- **커뮤니티**: 활발한 오픈소스 생태계

더 자세한 내용은 공식 문서를 참고하세요:

- [GitHub](https://github.com/OpenBMB/UltraRAG)
- [문서](https://ultrarag.openbmb.cn)
- [Discord](https://discord.gg/yRFFjjJnnS)

---

*UltraRAG 완벽 가이드 시리즈를 읽어주셔서 감사합니다!*
