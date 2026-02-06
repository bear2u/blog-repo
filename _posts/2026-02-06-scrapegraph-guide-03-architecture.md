---
layout: post
title: "ScrapeGraphAI 완벽 가이드 (3) - 아키텍처 분석"
date: 2026-02-06
permalink: /scrapegraph-guide-03-architecture/
author: ScrapeGraphAI Team
categories: [AI 도구, 웹 스크래핑]
tags: [ScrapeGraphAI, Architecture, Graph Design, Langchain, Pipeline]
original_url: "https://github.com/ScrapeGraphAI/Scrapegraph-ai"
excerpt: "ScrapeGraphAI의 그래프 기반 아키텍처와 핵심 컴포넌트를 이해합니다."
---

## 아키텍처 개요

ScrapeGraphAI는 **그래프 기반 파이프라인** 아키텍처를 사용합니다. 전통적인 선형 스크래핑 스크립트와 달리, 노드(Node)와 엣지(Edge)로 구성된 방향성 비순환 그래프(DAG)를 통해 작업을 수행합니다.

### 왜 그래프 기반인가?

- **모듈화**: 각 노드는 독립적인 작업 수행 (fetch, parse, extract 등)
- **재사용성**: 노드를 조합하여 다양한 파이프라인 구성
- **확장성**: 새로운 노드를 추가하여 기능 확장
- **가시성**: 워크플로우를 시각적으로 이해 가능

## 핵심 컴포넌트

### 1. 그래프 계층 구조

```
AbstractGraph (추상 클래스)
    ├── BaseGraph (기본 그래프 로직)
    └── Specialized Graphs (특화 그래프들)
            ├── SmartScraperGraph
            ├── SearchGraph
            ├── DocumentScraperGraph
            └── ... (20개 이상)
```

### 2. 주요 디렉토리 구조

```
scrapegraphai/
├── graphs/          # 그래프 파이프라인 구현
│   ├── abstract_graph.py
│   ├── base_graph.py
│   ├── smart_scraper_graph.py
│   └── ...
├── nodes/           # 개별 작업 노드
│   ├── fetch_node.py
│   ├── parse_node.py
│   ├── rag_node.py
│   └── ...
├── models/          # LLM 모델 래퍼
├── prompts/         # 프롬프트 템플릿
├── helpers/         # 유틸리티 함수
├── utils/           # 공통 도구
├── docloaders/      # 문서 로더
├── integrations/    # 외부 통합
└── telemetry/       # 분석 데이터
```

## 그래프 실행 흐름

### SmartScraperGraph 예시

```python
from scrapegraphai.graphs import SmartScraperGraph

smart_scraper = SmartScraperGraph(
    prompt="Extract product names",
    source="https://example.com",
    config={"llm": {"model": "ollama/llama3.2"}}
)

result = smart_scraper.run()
```

내부 실행 과정:

```
1. FetchNode
   - 웹사이트 접속
   - HTML 다운로드
   - JavaScript 렌더링

2. ParseNode
   - HTML 파싱
   - 불필요한 태그 제거
   - 텍스트 정제

3. RAGNode (Retrieval-Augmented Generation)
   - 프롬프트와 콘텐츠를 LLM에 전달
   - 관련 정보 추출
   - 구조화된 데이터 생성

4. OutputNode
   - JSON 포맷팅
   - 결과 검증
   - 반환
```

## 노드 시스템

### 기본 노드 타입

| 노드 | 역할 |
|-----|------|
| **FetchNode** | 웹페이지 다운로드 (Playwright/HTTP) |
| **ParseNode** | HTML 파싱 및 정제 |
| **RAGNode** | LLM 기반 정보 추출 |
| **GenerateAnswerNode** | 최종 답변 생성 |
| **SearchInternetNode** | 검색 엔진 쿼리 |
| **ImageToTextNode** | OCR 처리 |

### 커스텀 노드 생성

직접 노드를 만들어 파이프라인을 확장할 수 있습니다:

```python
from scrapegraphai.nodes import BaseNode

class CustomValidationNode(BaseNode):
    """추출된 데이터 검증 노드"""

    def execute(self, state):
        data = state.get("data", {})

        # 검증 로직
        if not data.get("email"):
            raise ValueError("Email is required")

        # 다음 노드로 전달
        state["validated"] = True
        return state
```

## 그래프 타입별 아키텍처

### 1. SmartScraperGraph (단일 페이지)

```
[FetchNode] → [ParseNode] → [RAGNode] → [GenerateAnswerNode]
```

가장 단순한 파이프라인으로, 한 페이지에서 정보를 추출합니다.

### 2. SearchGraph (검색 + 스크래핑)

```
[SearchInternetNode] → [FetchNode] → [ParseNode] → [RAGNode] → [GenerateAnswerNode]
```

검색 엔진에서 상위 N개 결과를 찾아 스크래핑합니다.

### 3. SmartScraperMultiGraph (병렬 처리)

```
                    ┌─ [Fetch+Parse+RAG] (URL 1)
[Split URLs] ───────┼─ [Fetch+Parse+RAG] (URL 2)
                    └─ [Fetch+Parse+RAG] (URL 3)
                            │
                    [Aggregate Results]
```

여러 URL을 동시에 스크래핑하여 성능을 향상시킵니다.

### 4. ScriptCreatorGraph (코드 생성)

```
[FetchNode] → [ParseNode] → [RAGNode] → [GenerateScriptNode]
```

스크래핑 결과를 보고 Python 스크립트를 자동으로 생성합니다.

## 설정 시스템

### 그래프 설정 구조

```python
graph_config = {
    # LLM 설정
    "llm": {
        "model": "ollama/llama3.2",
        "temperature": 0.0,
        "model_tokens": 8192,
    },

    # 브라우저 설정
    "headless": True,
    "browser_type": "chromium",  # chromium, firefox, webkit

    # 프록시 설정
    "proxy": {
        "server": "http://proxy.com:8080",
        "username": "user",
        "password": "pass"
    },

    # 타임아웃
    "loader_kwargs": {
        "timeout": 30000,
        "wait_until": "networkidle",
    },

    # 디버깅
    "verbose": True,
    "burr_kwargs": {
        "app_instance_id": "my-scraper",
    }
}
```

## LLM 모델 시스템

### 지원 모델 제공자

ScrapeGraphAI는 Langchain을 기반으로 다양한 LLM을 지원합니다:

```python
# OpenAI
{"llm": {"model": "openai/gpt-4o-mini", "api_key": "..."}}

# Anthropic Claude
{"llm": {"model": "anthropic/claude-3-sonnet", "api_key": "..."}}

# Google Gemini
{"llm": {"model": "gemini/gemini-pro", "api_key": "..."}}

# Ollama (로컬)
{"llm": {"model": "ollama/llama3.2"}}

# Groq
{"llm": {"model": "groq/mixtral-8x7b", "api_key": "..."}}

# Azure OpenAI
{"llm": {
    "model": "azure/gpt-4",
    "api_key": "...",
    "azure_endpoint": "https://..."
}}
```

### 모델 토큰 관리

```python
graph_config = {
    "llm": {
        "model": "ollama/llama3.2",
        "model_tokens": 8192,  # 컨텍스트 윈도우 크기
    },
}
```

큰 웹페이지를 처리할 때는 토큰 제한을 고려해야 합니다. ScrapeGraphAI는 자동으로 청크로 나눕니다.

## 프롬프트 시스템

### 내장 프롬프트 템플릿

ScrapeGraphAI는 각 노드별로 최적화된 프롬프트를 사용합니다:

```python
# scrapegraphai/prompts/
├── extract_prompt.py      # 정보 추출 프롬프트
├── generate_prompt.py     # 답변 생성 프롬프트
├── search_prompt.py       # 검색 쿼리 생성
└── ...
```

### 커스텀 프롬프트

사용자 정의 프롬프트를 지정할 수 있습니다:

```python
from scrapegraphai.graphs import SmartScraperGraph

custom_prompt = """
Given the following HTML content:
{html_content}

Extract the following information in JSON format:
- Company name
- Founded year
- CEO name
"""

smart_scraper = SmartScraperGraph(
    prompt=custom_prompt,
    source="https://example.com",
    config={"llm": {"model": "ollama/llama3.2"}}
)
```

## Playwright 통합

### 브라우저 자동화

ScrapeGraphAI는 Playwright를 사용하여 동적 웹사이트를 처리합니다:

```python
graph_config = {
    "llm": {"model": "ollama/llama3.2"},

    # Playwright 옵션
    "loader_kwargs": {
        "headless": False,              # 브라우저 표시
        "timeout": 30000,                # 30초 타임아웃
        "wait_until": "networkidle",     # 네트워크 유휴 대기
        "user_agent": "Custom Agent",    # User-Agent 설정
    }
}
```

### JavaScript 렌더링

```python
# JavaScript가 렌더링될 때까지 대기
graph_config = {
    "llm": {"model": "ollama/llama3.2"},
    "loader_kwargs": {
        "wait_until": "domcontentloaded",  # DOM 로드 완료
        "wait_for_selector": "#product",    # 특정 요소 대기
    }
}
```

## 에러 핸들링

### 재시도 로직

```python
graph_config = {
    "llm": {"model": "ollama/llama3.2"},
    "max_retries": 3,           # 실패 시 재시도 횟수
    "retry_delay": 2,           # 재시도 간격 (초)
}
```

### 로깅 및 디버깅

```python
import logging

# 상세 로그 활성화
logging.basicConfig(level=logging.DEBUG)

graph_config = {
    "llm": {"model": "ollama/llama3.2"},
    "verbose": True,  # 각 노드의 실행 정보 출력
}
```

## 성능 최적화

### 캐싱

```python
# HTML 캐싱으로 중복 요청 방지
graph_config = {
    "llm": {"model": "ollama/llama3.2"},
    "cache_path": "./cache",  # 캐시 디렉토리
}
```

### 병렬 처리

```python
from scrapegraphai.graphs import SmartScraperMultiGraph

# 여러 URL 동시 스크래핑
multi_scraper = SmartScraperMultiGraph(
    prompt="Extract product info",
    source=["https://site1.com", "https://site2.com", "https://site3.com"],
    config={"llm": {"model": "ollama/llama3.2"}}
)

results = multi_scraper.run()  # 병렬 실행
```

## 다음 단계

다음 챕터에서는 가장 많이 사용되는 **SmartScraperGraph**를 심층 분석합니다.

---

## 시리즈 네비게이션

- **이전**: [(2) 설치 및 빠른 시작]({{ site.baseurl }}/scrapegraph-guide-02-installation/)
- **현재**: (3) 아키텍처 분석
- **다음**: [(4) SmartScraper 그래프]({{ site.baseurl }}/scrapegraph-guide-04-smartscraper/)

[📚 전체 목차로 돌아가기]({{ site.baseurl }}/scrapegraph-guide/)
