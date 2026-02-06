---
layout: post
title: "ScrapeGraphAI 완벽 가이드 (8) - LLM 모델 연동"
date: 2026-02-06
permalink: /scrapegraph-guide-08-llm-integration/
author: ScrapeGraphAI Team
categories: [AI 도구, 웹 스크래핑]
tags: [ScrapeGraphAI, LLM, OpenAI, Anthropic, Ollama, Gemini, Groq]
original_url: "https://github.com/ScrapeGraphAI/Scrapegraph-ai"
excerpt: "다양한 LLM 제공자와의 연동 방법을 상세히 알아봅니다."
---

## LLM 통합 개요

ScrapeGraphAI는 **Langchain**을 기반으로 하여 20개 이상의 LLM 제공자를 지원합니다. 모델 설정만 바꾸면 쉽게 전환할 수 있습니다.

## 지원 LLM 제공자

| 제공자 | 모델 예시 | 비용 | 특징 |
|--------|----------|------|------|
| **OpenAI** | GPT-4o, GPT-4o-mini | 유료 | 높은 정확도, 안정성 |
| **Anthropic** | Claude 3.5 Sonnet | 유료 | 긴 컨텍스트, 안전성 |
| **Google** | Gemini Pro, Gemini Flash | 유료/무료 | 멀티모달, 빠른 속도 |
| **Groq** | Llama 3, Mixtral | 무료 | 초고속 추론 |
| **Ollama** | Llama 3.2, Mistral | 무료 | 로컬 실행, 프라이버시 |
| **Azure OpenAI** | GPT-4, GPT-3.5 | 유료 | 엔터프라이즈, 컴플라이언스 |

## OpenAI 연동

### 기본 설정

```python
from scrapegraphai.graphs import SmartScraperGraph

config = {
    "llm": {
        "api_key": "sk-proj-...",
        "model": "openai/gpt-4o-mini",
    }
}

scraper = SmartScraperGraph(
    prompt="Extract product information",
    source="https://example.com",
    config=config
)

result = scraper.run()
```

### 환경 변수 사용

```bash
export OPENAI_API_KEY="sk-proj-..."
```

```python
import os

config = {
    "llm": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "openai/gpt-4o-mini",
    }
}
```

### 고급 설정

```python
config = {
    "llm": {
        "api_key": "sk-proj-...",
        "model": "openai/gpt-4o-mini",
        "temperature": 0.0,        # 일관성 (0.0 ~ 2.0)
        "max_tokens": 4096,        # 최대 출력 토큰
        "top_p": 1.0,              # 샘플링 확률
        "frequency_penalty": 0.0,  # 반복 억제
        "presence_penalty": 0.0,   # 다양성 증가
    }
}
```

### 모델 선택 가이드

```python
# 빠르고 저렴 (추천)
{"llm": {"model": "openai/gpt-4o-mini"}}  # $0.15 / 1M tokens

# 균형잡힌 성능
{"llm": {"model": "openai/gpt-4o"}}       # $2.50 / 1M tokens

# 최고 성능
{"llm": {"model": "openai/o1"}}           # $15 / 1M tokens
```

## Anthropic Claude 연동

### 기본 설정

```python
config = {
    "llm": {
        "api_key": "sk-ant-...",
        "model": "anthropic/claude-3-5-sonnet-20241022",
    }
}
```

### Claude 모델 비교

```python
# Claude 3.5 Sonnet (추천)
{"llm": {"model": "anthropic/claude-3-5-sonnet-20241022"}}  # $3 / 1M tokens

# Claude 3 Opus (최고 성능)
{"llm": {"model": "anthropic/claude-3-opus-20240229"}}      # $15 / 1M tokens

# Claude 3 Haiku (빠르고 저렴)
{"llm": {"model": "anthropic/claude-3-haiku-20240307"}}     # $0.25 / 1M tokens
```

### 긴 컨텍스트 활용

```python
config = {
    "llm": {
        "api_key": "sk-ant-...",
        "model": "anthropic/claude-3-5-sonnet-20241022",
        "max_tokens": 8192,  # Claude는 200K 토큰 지원
    }
}

# 매우 긴 웹페이지도 처리 가능
scraper = SmartScraperGraph(
    prompt="Summarize this entire documentation",
    source="https://docs.example.com/full-guide",  # 긴 문서
    config=config
)
```

## Google Gemini 연동

### 기본 설정

```python
config = {
    "llm": {
        "api_key": "AIza...",
        "model": "gemini/gemini-1.5-pro",
    }
}
```

### Gemini 모델

```python
# Gemini 1.5 Pro (추천)
{"llm": {"model": "gemini/gemini-1.5-pro"}}    # $1.25 / 1M tokens

# Gemini 1.5 Flash (빠름)
{"llm": {"model": "gemini/gemini-1.5-flash"}}  # $0.075 / 1M tokens

# Gemini 2.0 Flash (최신)
{"llm": {"model": "gemini/gemini-2.0-flash"}}  # 실험적
```

### 무료 티어

```python
# Gemini API는 무료 티어 제공
config = {
    "llm": {
        "api_key": "AIza...",
        "model": "gemini/gemini-1.5-flash",  # 무료로 사용 가능
    }
}
```

## Groq 연동 (초고속)

Groq는 **전용 LPU 하드웨어**로 초고속 추론을 제공합니다.

### 기본 설정

```python
config = {
    "llm": {
        "api_key": "gsk_...",
        "model": "groq/llama-3.1-70b-versatile",
    }
}
```

### Groq 모델

```python
# Llama 3.1 70B (추천)
{"llm": {"model": "groq/llama-3.1-70b-versatile"}}

# Llama 3.1 8B (빠름)
{"llm": {"model": "groq/llama-3.1-8b-instant"}}

# Mixtral 8x7B
{"llm": {"model": "groq/mixtral-8x7b-32768"}}

# Gemma 2 9B
{"llm": {"model": "groq/gemma2-9b-it"}}
```

### 속도 비교

```python
import time

# Groq (초고속)
start = time.time()
groq_scraper = SmartScraperGraph(
    prompt="Extract data",
    source="https://example.com",
    config={"llm": {"model": "groq/llama-3.1-8b-instant", "api_key": "gsk_..."}}
)
result = groq_scraper.run()
print(f"Groq: {time.time() - start:.2f}s")  # ~2초

# OpenAI (일반 속도)
start = time.time()
openai_scraper = SmartScraperGraph(
    prompt="Extract data",
    source="https://example.com",
    config={"llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}}
)
result = openai_scraper.run()
print(f"OpenAI: {time.time() - start:.2f}s")  # ~5초
```

## Ollama 연동 (로컬)

### Ollama 설치 및 실행

```bash
# Ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# 모델 다운로드
ollama pull llama3.2      # 3B (가벼움)
ollama pull mistral       # 7B (균형)
ollama pull llama3.1      # 8B (고성능)

# 서버 실행
ollama serve
```

### 기본 설정

```python
config = {
    "llm": {
        "model": "ollama/llama3.2",
        # API 키 불필요
    }
}
```

### 커스텀 Ollama 서버

```python
config = {
    "llm": {
        "model": "ollama/llama3.2",
        "base_url": "http://localhost:11434",  # 기본값
    }
}

# 원격 Ollama 서버
config = {
    "llm": {
        "model": "ollama/llama3.1",
        "base_url": "http://remote-server:11434",
    }
}
```

### Ollama 모델 추천

```python
# 빠른 스크래핑 (3B)
{"llm": {"model": "ollama/llama3.2"}}

# 균형 (7B)
{"llm": {"model": "ollama/mistral"}}

# 고성능 (8B)
{"llm": {"model": "ollama/llama3.1"}}

# 코드 특화 (7B)
{"llm": {"model": "ollama/codellama"}}
```

## Azure OpenAI 연동

엔터프라이즈 환경에서 사용하는 Azure OpenAI 서비스:

### 기본 설정

```python
config = {
    "llm": {
        "api_key": "YOUR_AZURE_API_KEY",
        "model": "azure/gpt-4",
        "azure_endpoint": "https://your-resource.openai.azure.com/",
        "api_version": "2024-02-15-preview",
        "azure_deployment": "gpt-4-deployment-name",
    }
}
```

### 엔터프라이즈 기능

```python
config = {
    "llm": {
        "api_key": "...",
        "model": "azure/gpt-4",
        "azure_endpoint": "...",
        "azure_ad_token": "...",      # Azure AD 인증
        "organization": "org-123",    # 조직 ID
    }
}
```

## AWS Bedrock 연동

### 기본 설정

```python
config = {
    "llm": {
        "model": "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        "aws_access_key_id": "AKIA...",
        "aws_secret_access_key": "...",
        "region_name": "us-east-1",
    }
}
```

## 모델 성능 비교

### 정확도 테스트

```python
models = [
    ("openai/gpt-4o-mini", "sk-..."),
    ("anthropic/claude-3-haiku", "sk-ant-..."),
    ("groq/llama-3.1-8b-instant", "gsk_..."),
    ("ollama/llama3.2", None),
]

test_url = "https://example.com/complex-page"
test_prompt = "Extract company info: name, industry, employees, revenue"

for model, api_key in models:
    config = {"llm": {"model": model}}
    if api_key:
        config["llm"]["api_key"] = api_key

    scraper = SmartScraperGraph(
        prompt=test_prompt,
        source=test_url,
        config=config
    )

    result = scraper.run()
    print(f"{model}: {result}")
```

## 비용 최적화 전략

### 1. 적절한 모델 선택

```python
# 간단한 작업: 저렴한 모델
simple_config = {"llm": {"model": "openai/gpt-4o-mini"}}  # $0.15 / 1M

# 복잡한 작업: 고성능 모델
complex_config = {"llm": {"model": "openai/gpt-4o"}}      # $2.50 / 1M
```

### 2. 로컬 모델 활용

```python
# 무료: Ollama
free_config = {"llm": {"model": "ollama/llama3.2"}}

# 거의 무료: Groq (무료 티어)
fast_free_config = {
    "llm": {
        "model": "groq/llama-3.1-8b-instant",
        "api_key": "gsk_..."
    }
}
```

### 3. 캐싱

```python
config = {
    "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."},
    "cache_path": "./cache",  # HTML 캐싱
}
```

## 에러 핸들링

### Rate Limit 처리

```python
from time import sleep

def scrape_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            scraper = SmartScraperGraph(
                prompt="Extract data",
                source=url,
                config={
                    "llm": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-..."
                    }
                }
            )
            return scraper.run()
        except Exception as e:
            if "rate_limit" in str(e).lower():
                wait_time = 2 ** attempt  # 지수 백오프
                print(f"Rate limit hit, waiting {wait_time}s...")
                sleep(wait_time)
            else:
                raise
```

### API 키 로테이션

```python
api_keys = [
    "sk-proj-key1...",
    "sk-proj-key2...",
    "sk-proj-key3..."
]

current_key = 0

def get_config():
    global current_key
    config = {
        "llm": {
            "model": "openai/gpt-4o-mini",
            "api_key": api_keys[current_key]
        }
    }
    current_key = (current_key + 1) % len(api_keys)
    return config
```

## 다음 단계

다음 챕터에서는 **통합 및 확장** (API/SDK, Langchain, n8n 등)을 다룹니다.

---

## 시리즈 네비게이션

- **이전**: [(7) 고급 그래프]({{ site.baseurl }}/scrapegraph-guide-07-advanced/)
- **현재**: (8) LLM 모델 연동
- **다음**: [(9) 통합 및 확장]({{ site.baseurl }}/scrapegraph-guide-09-integrations/)

[📚 전체 목차로 돌아가기]({{ site.baseurl }}/scrapegraph-guide/)
