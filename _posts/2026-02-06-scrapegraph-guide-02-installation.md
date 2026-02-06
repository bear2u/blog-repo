---
layout: post
title: "ScrapeGraphAI 완벽 가이드 (2) - 설치 및 빠른 시작"
date: 2026-02-06
permalink: /scrapegraph-guide-02-installation/
author: ScrapeGraphAI Team
categories: [AI 도구, 웹 스크래핑]
tags: [ScrapeGraphAI, Installation, Quickstart, Python, Ollama]
original_url: "https://github.com/ScrapeGraphAI/Scrapegraph-ai"
excerpt: "ScrapeGraphAI 설치부터 첫 스크래핑까지 단계별로 알아봅니다."
---

## 설치 전 준비사항

### 시스템 요구사항

- **Python**: 3.10 이상, 4.0 미만
- **운영체제**: Linux, macOS, Windows
- **메모리**: 최소 8GB RAM (로컬 LLM 사용 시 16GB 권장)

### 가상 환경 설정 (권장)

라이브러리 충돌을 방지하기 위해 가상 환경을 사용하는 것이 좋습니다:

```bash
# venv 사용
python -m venv scrapegraph-env
source scrapegraph-env/bin/activate  # Windows: scrapegraph-env\Scripts\activate

# 또는 conda 사용
conda create -n scrapegraph python=3.11
conda activate scrapegraph
```

## ScrapeGraphAI 설치

### PyPI를 통한 설치

```bash
pip install scrapegraphai
```

### Playwright 설치 (필수)

웹사이트 콘텐츠를 가져오기 위해 Playwright가 필요합니다:

```bash
playwright install
```

이 명령은 Chromium, Firefox, WebKit 브라우저를 자동으로 다운로드합니다.

### 선택적 의존성

필요에 따라 추가 기능을 설치할 수 있습니다:

```bash
# NVIDIA GPU 지원
pip install scrapegraphai[nvidia]

# OCR 기능 (이미지에서 텍스트 추출)
pip install scrapegraphai[ocr]

# Burr 워크플로우 시각화
pip install scrapegraphai[burr]
```

## LLM 설정

ScrapeGraphAI는 다양한 LLM 제공자를 지원합니다. 여기서는 가장 일반적인 두 가지 방법을 소개합니다.

### 방법 1: Ollama (로컬 LLM - 무료)

#### Ollama 설치

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: https://ollama.com/download에서 설치 프로그램 다운로드
```

#### 모델 다운로드

```bash
# Llama 3.2 (3B 파라미터 - 가벼움)
ollama pull llama3.2

# Mistral (7B 파라미터 - 균형잡힌 성능)
ollama pull mistral

# Llama 3.1 (8B 파라미터 - 높은 성능)
ollama pull llama3.1
```

#### 설정 예시

```python
graph_config = {
    "llm": {
        "model": "ollama/llama3.2",
        "model_tokens": 8192,
        "format": "json",
    },
    "verbose": True,
    "headless": False,
}
```

### 방법 2: OpenAI API (클라우드)

#### API 키 설정

```bash
export OPENAI_API_KEY="sk-..."
```

또는 `.env` 파일 사용:

```bash
# .env
OPENAI_API_KEY=sk-...
```

#### 설정 예시

```python
graph_config = {
    "llm": {
        "api_key": "YOUR_OPENAI_API_KEY",
        "model": "openai/gpt-4o-mini",
    },
    "verbose": True,
    "headless": False,
}
```

## 첫 번째 스크래핑

이제 실제로 웹사이트를 스크래핑해봅시다!

### 예제 1: 기본 정보 추출

```python
from scrapegraphai.graphs import SmartScraperGraph

# 그래프 설정
graph_config = {
    "llm": {
        "model": "ollama/llama3.2",
        "model_tokens": 8192,
        "format": "json",
    },
    "verbose": True,
    "headless": False,
}

# SmartScraper 인스턴스 생성
smart_scraper = SmartScraperGraph(
    prompt="Extract the company name, description, and email contact",
    source="https://scrapegraphai.com/",
    config=graph_config
)

# 실행
result = smart_scraper.run()

import json
print(json.dumps(result, indent=4))
```

**출력:**
```json
{
    "company_name": "ScrapeGraphAI",
    "description": "AI-powered web scraping platform using LLMs",
    "email_contact": "contact@scrapegraphai.com"
}
```

### 예제 2: 뉴스 헤드라인 추출

```python
from scrapegraphai.graphs import SmartScraperGraph

graph_config = {
    "llm": {
        "model": "ollama/llama3.2",
    },
}

scraper = SmartScraperGraph(
    prompt="Extract all news article titles and their URLs",
    source="https://news.ycombinator.com/",
    config=graph_config
)

result = scraper.run()
print(result)
```

### 예제 3: 로컬 HTML 파일 스크래핑

웹사이트뿐만 아니라 로컬 파일도 스크래핑할 수 있습니다:

```python
from scrapegraphai.graphs import SmartScraperGraph

graph_config = {
    "llm": {"model": "ollama/llama3.2"},
}

scraper = SmartScraperGraph(
    prompt="Extract all product names and prices",
    source="/path/to/local/products.html",
    config=graph_config
)

result = scraper.run()
```

## 설정 옵션 상세

### 주요 설정 파라미터

```python
graph_config = {
    # LLM 설정
    "llm": {
        "model": "ollama/llama3.2",      # 사용할 모델
        "api_key": "...",                 # API 키 (필요시)
        "model_tokens": 8192,             # 최대 토큰 수
        "temperature": 0.7,               # 창의성 (0.0~1.0)
        "format": "json",                 # 출력 포맷
    },

    # 브라우저 설정
    "headless": True,                     # 헤드리스 모드 (UI 없음)
    "verbose": False,                     # 디버그 로그 출력

    # 스크래핑 설정
    "user_agent": "custom-agent",         # User-Agent 커스터마이징
    "proxy": "http://proxy.com:8080",    # 프록시 설정
}
```

### 브라우저 옵션

```python
from scrapegraphai.graphs import SmartScraperGraph

# 브라우저 표시 (디버깅용)
config = {
    "llm": {"model": "ollama/llama3.2"},
    "headless": False,  # 브라우저 창 표시
}

# 타임아웃 설정
config = {
    "llm": {"model": "ollama/llama3.2"},
    "loader_kwargs": {
        "timeout": 30000,  # 30초
    }
}
```

## 일반적인 설치 문제 해결

### 문제 1: Playwright 설치 실패

```bash
# 권한 문제 해결
playwright install --with-deps

# 특정 브라우저만 설치
playwright install chromium
```

### 문제 2: Ollama 연결 오류

```bash
# Ollama 서비스 실행 확인
ollama serve

# 모델 목록 확인
ollama list
```

### 문제 3: 메모리 부족

로컬 LLM 사용 시 메모리가 부족하면:

```python
# 더 작은 모델 사용
graph_config = {
    "llm": {
        "model": "ollama/llama3.2",  # 3B 파라미터 (가벼움)
    },
}
```

또는 클라우드 API로 전환:

```python
graph_config = {
    "llm": {
        "api_key": "YOUR_API_KEY",
        "model": "openai/gpt-4o-mini",
    },
}
```

## 환경 변수 관리

`.env` 파일로 API 키를 안전하게 관리하세요:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
```

Python 코드에서 로드:

```python
from dotenv import load_dotenv
import os

load_dotenv()

graph_config = {
    "llm": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "openai/gpt-4o-mini",
    },
}
```

## 텔레메트리 (선택 사항)

ScrapeGraphAI는 익명 사용 통계를 수집합니다. 원치 않으면:

```bash
export SCRAPEGRAPHAI_TELEMETRY_ENABLED=false
```

## 다음 단계

이제 ScrapeGraphAI를 설치하고 첫 스크래핑을 성공했습니다! 다음 챕터에서는 아키텍처와 그래프 기반 설계를 자세히 알아봅니다.

---

## 시리즈 네비게이션

- **이전**: [(1) 소개 및 개요]({{ site.baseurl }}/scrapegraph-guide-01-intro/)
- **현재**: (2) 설치 및 빠른 시작
- **다음**: [(3) 아키텍처 분석]({{ site.baseurl }}/scrapegraph-guide-03-architecture/)

[📚 전체 목차로 돌아가기]({{ site.baseurl }}/scrapegraph-guide/)
