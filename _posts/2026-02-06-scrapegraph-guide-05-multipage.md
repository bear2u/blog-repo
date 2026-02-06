---
layout: post
title: "ScrapeGraphAI 완벽 가이드 (5) - 멀티페이지 스크래핑"
date: 2026-02-06
permalink: /scrapegraph-guide-05-multipage/
author: ScrapeGraphAI Team
categories: [AI 도구, 웹 스크래핑]
tags: [ScrapeGraphAI, Multi-page, SearchGraph, Parallel Processing]
original_url: "https://github.com/ScrapeGraphAI/Scrapegraph-ai"
excerpt: "여러 페이지를 동시에 스크래핑하는 고급 기법을 배웁니다."
---

## 멀티페이지 스크래핑이란?

단일 페이지가 아닌 **여러 웹페이지**를 동시에 또는 순차적으로 스크래핑하는 기법입니다. ScrapeGraphAI는 이를 위한 여러 전문화된 그래프를 제공합니다.

## SmartScraperMultiGraph

### 기본 사용법

```python
from scrapegraphai.graphs import SmartScraperMultiGraph

multi_scraper = SmartScraperMultiGraph(
    prompt="Extract company name, industry, and employee count",
    source=[
        "https://company1.com/about",
        "https://company2.com/about",
        "https://company3.com/about"
    ],
    config={
        "llm": {"model": "ollama/llama3.2"}
    }
)

results = multi_scraper.run()
```

### 출력 형식

```json
[
    {
        "company_name": "Company 1",
        "industry": "Technology",
        "employee_count": 500
    },
    {
        "company_name": "Company 2",
        "industry": "Healthcare",
        "employee_count": 1200
    },
    {
        "company_name": "Company 3",
        "industry": "Finance",
        "employee_count": 800
    }
]
```

### 병렬 처리

SmartScraperMultiGraph는 자동으로 병렬 처리를 수행하여 속도를 크게 향상시킵니다:

```python
import time

urls = [f"https://example.com/product/{i}" for i in range(1, 51)]

start = time.time()
multi_scraper = SmartScraperMultiGraph(
    prompt="Extract product name and price",
    source=urls,  # 50개 URL
    config={
        "llm": {"model": "ollama/llama3.2"},
        "max_workers": 5  # 동시 5개 처리
    }
)
results = multi_scraper.run()
elapsed = time.time() - start

print(f"Scraped {len(results)} pages in {elapsed:.2f} seconds")
```

## SearchGraph

**SearchGraph**는 검색 엔진을 통해 관련 페이지를 찾고 스크래핑합니다.

### 기본 예제

```python
from scrapegraphai.graphs import SearchGraph

search_scraper = SearchGraph(
    prompt="What are the latest AI trends in 2024?",
    config={
        "llm": {"model": "ollama/llama3.2"},
        "max_results": 5,  # 상위 5개 검색 결과
    }
)

answer = search_scraper.run()
print(answer)
```

### 작동 방식

```
1. 검색 엔진에 쿼리 실행 (DuckDuckGo)
2. 상위 N개 결과 URL 수집
3. 각 페이지 스크래핑
4. LLM이 모든 정보를 종합하여 답변 생성
```

### 심화 활용

```python
search_scraper = SearchGraph(
    prompt="Compare pricing of top 3 project management tools",
    config={
        "llm": {
            "model": "openai/gpt-4o-mini",
            "api_key": "sk-..."
        },
        "max_results": 10,
        "search_engine": "duckduckgo",  # 기본값
    }
)

comparison = search_scraper.run()
```

## DepthSearchGraph

웹사이트를 **깊이 우선 탐색**하여 링크를 따라가며 스크래핑합니다.

### 사용 사례

- 문서 사이트 전체 크롤링
- 블로그 아카이브 수집
- 제품 카탈로그 전체 스크래핑

### 예제

```python
from scrapegraphai.graphs import DepthSearchGraph

depth_scraper = DepthSearchGraph(
    prompt="Extract all article titles and summaries",
    source="https://blog.example.com",
    config={
        "llm": {"model": "ollama/llama3.2"},
        "max_depth": 2,        # 최대 깊이
        "max_pages": 20,       # 최대 페이지 수
        "same_domain": True,   # 같은 도메인만
    }
)

articles = depth_scraper.run()
```

### 실행 흐름

```
시작 페이지: https://blog.example.com

Depth 1:
├── /article-1
├── /article-2
└── /article-3

Depth 2:
    ├── /article-1/comments
    ├── /article-2/related
    └── /article-3/author
```

## SmartScraperMultiConcatGraph

여러 페이지의 콘텐츠를 **하나로 합쳐서** 분석합니다.

### 사용 시나리오

- 여러 페이지에 걸친 긴 문서
- 시리즈 블로그 포스트
- 페이지네이션된 콘텐츠

### 예제

```python
from scrapegraphai.graphs import SmartScraperMultiConcatGraph

concat_scraper = SmartScraperMultiConcatGraph(
    prompt="Summarize the entire tutorial series",
    source=[
        "https://tutorial.com/part-1",
        "https://tutorial.com/part-2",
        "https://tutorial.com/part-3",
        "https://tutorial.com/part-4"
    ],
    config={
        "llm": {"model": "ollama/llama3.1"}
    }
)

summary = concat_scraper.run()
```

## SearchLinkGraph

검색 결과의 **URL 목록**만 수집합니다 (스크래핑 없음).

### 사용 사례

```python
from scrapegraphai.graphs import SearchLinkGraph

link_collector = SearchLinkGraph(
    prompt="Find official documentation for Python web frameworks",
    config={
        "llm": {"model": "ollama/llama3.2"},
        "max_results": 10
    }
)

urls = link_collector.run()
print(urls)
```

**출력:**
```python
[
    "https://docs.djangoproject.com/",
    "https://flask.palletsprojects.com/",
    "https://fastapi.tiangolo.com/",
    ...
]
```

## 실전 활용 사례

### 사례 1: 경쟁사 가격 비교

```python
from scrapegraphai.graphs import SmartScraperMultiGraph

competitors = [
    "https://competitor1.com/pricing",
    "https://competitor2.com/pricing",
    "https://competitor3.com/pricing"
]

price_scraper = SmartScraperMultiGraph(
    prompt="""
    Extract:
    - Company name
    - Plan names
    - Prices (monthly and annual if available)
    - Features for each plan
    """,
    source=competitors,
    config={
        "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}
    }
)

pricing_data = price_scraper.run()

# 분석
import pandas as pd
df = pd.DataFrame(pricing_data)
df.to_csv("competitor_pricing.csv", index=False)
```

### 사례 2: 뉴스 모니터링

```python
from scrapegraphai.graphs import SearchGraph

news_monitor = SearchGraph(
    prompt="What are the latest developments in quantum computing this week?",
    config={
        "llm": {"model": "ollama/llama3.1"},
        "max_results": 15
    }
)

news_summary = news_monitor.run()
print(news_summary)
```

### 사례 3: 리드 제너레이션

```python
from scrapegraphai.graphs import SearchLinkGraph, SmartScraperMultiGraph

# 1단계: 리드 URL 수집
lead_finder = SearchLinkGraph(
    prompt="Find SaaS companies in healthcare industry",
    config={
        "llm": {"model": "ollama/llama3.2"},
        "max_results": 50
    }
)

company_urls = lead_finder.run()

# 2단계: 각 회사 정보 스크래핑
lead_scraper = SmartScraperMultiGraph(
    prompt="""
    Extract:
    - Company name
    - Website
    - Industry
    - Contact email
    - LinkedIn URL
    """,
    source=company_urls,
    config={
        "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."},
        "max_workers": 10
    }
)

leads = lead_scraper.run()
```

### 사례 4: 콘텐츠 집계

```python
from scrapegraphai.graphs import SmartScraperMultiGraph

# 여러 블로그의 최신 글 수집
blog_urls = [
    "https://blog1.com",
    "https://blog2.com",
    "https://blog3.com"
]

content_scraper = SmartScraperMultiGraph(
    prompt="""
    Extract the 5 most recent blog posts:
    - Title
    - Author
    - Published date
    - Summary (2 sentences)
    - URL
    """,
    source=blog_urls,
    config={
        "llm": {"model": "ollama/mistral"}
    }
)

all_posts = content_scraper.run()

# 날짜순 정렬
sorted_posts = sorted(
    [post for blog in all_posts for post in blog["posts"]],
    key=lambda x: x["published_date"],
    reverse=True
)
```

## 성능 최적화

### 동시 처리 수 조정

```python
config = {
    "llm": {"model": "ollama/llama3.2"},
    "max_workers": 10,  # CPU 코어 수에 맞게 조정
}
```

### 타임아웃 설정

```python
config = {
    "llm": {"model": "ollama/llama3.2"},
    "loader_kwargs": {
        "timeout": 30000,  # 느린 사이트는 건너뛰기
    }
}
```

### 에러 무시

```python
config = {
    "llm": {"model": "ollama/llama3.2"},
    "continue_on_error": True,  # 일부 실패해도 계속 진행
}
```

## 프록시 로테이션

대량 스크래핑 시 프록시를 사용하세요:

```python
proxies = [
    "http://proxy1.com:8080",
    "http://proxy2.com:8080",
    "http://proxy3.com:8080"
]

config = {
    "llm": {"model": "ollama/llama3.2"},
    "proxy": {
        "server": proxies[0],  # 순환하여 사용
    }
}
```

## 모니터링 및 로깅

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='scraping.log'
)

multi_scraper = SmartScraperMultiGraph(
    prompt="Extract data",
    source=urls,
    config={
        "llm": {"model": "ollama/llama3.2"},
        "verbose": True
    }
)

results = multi_scraper.run()
```

## 다음 단계

다음 챕터에서는 JSON, CSV, XML 등 **다양한 데이터 포맷**을 다루는 방법을 배웁니다.

---

## 시리즈 네비게이션

- **이전**: [(4) SmartScraper 그래프]({{ site.baseurl }}/scrapegraph-guide-04-smartscraper/)
- **현재**: (5) 멀티페이지 스크래핑
- **다음**: [(6) 다양한 데이터 포맷]({{ site.baseurl }}/scrapegraph-guide-06-formats/)

[📚 전체 목차로 돌아가기]({{ site.baseurl }}/scrapegraph-guide/)
