---
layout: page
title: ScrapeGraphAI 가이드
permalink: /scrapegraph-guide/
icon: fas fa-spider
---

# ScrapeGraphAI 완벽 가이드

> **LLM 기반 차세대 웹 스크래핑 라이브러리**

ScrapeGraphAI는 LLM(대규모 언어 모델)과 그래프 로직을 활용하여 자연어 프롬프트만으로 웹사이트 및 문서에서 데이터를 추출하는 Python 라이브러리입니다. "You Only Scrape Once" - 웹사이트 구조가 변경되어도 코드 수정 없이 자동으로 적응합니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요]({{ site.baseurl }}/scrapegraph-guide-01-intro/) | ScrapeGraphAI란?, 주요 특징, LLM 기반 스크래핑 |
| 02 | [설치 및 빠른 시작]({{ site.baseurl }}/scrapegraph-guide-02-installation/) | 설치, 첫 스크래핑, LLM 설정 |
| 03 | [아키텍처 분석]({{ site.baseurl }}/scrapegraph-guide-03-architecture/) | 그래프 기반 설계, 노드 시스템, 실행 흐름 |
| 04 | [SmartScraper 그래프]({{ site.baseurl }}/scrapegraph-guide-04-smartscraper/) | 단일 페이지 스크래핑, 프롬프트 작성, 실전 사례 |
| 05 | [멀티페이지 스크래핑]({{ site.baseurl }}/scrapegraph-guide-05-multipage/) | SearchGraph, Multi 그래프, 병렬 처리 |
| 06 | [다양한 데이터 포맷]({{ site.baseurl }}/scrapegraph-guide-06-formats/) | JSON, CSV, XML, PDF, Document 스크래핑 |
| 07 | [고급 그래프]({{ site.baseurl }}/scrapegraph-guide-07-advanced/) | CodeGenerator, ScriptCreator, SpeechGraph |
| 08 | [LLM 모델 연동]({{ site.baseurl }}/scrapegraph-guide-08-llm-integration/) | OpenAI, Claude, Gemini, Groq, Ollama |
| 09 | [통합 및 확장]({{ site.baseurl }}/scrapegraph-guide-09-integrations/) | API/SDK, Langchain, n8n, Zapier, CrewAI |
| 10 | [실전 활용 및 팁]({{ site.baseurl }}/scrapegraph-guide-10-tips/) | 프로덕션 배포, 최적화, 문제 해결 |

---

## 핵심 특징

### 🤖 LLM 기반 인텔리전트 스크래핑

```python
from scrapegraphai.graphs import SmartScraperGraph

scraper = SmartScraperGraph(
    prompt="Extract company name, founders, and social links",
    source="https://scrapegraphai.com/",
    config={"llm": {"model": "ollama/llama3.2"}}
)

result = scraper.run()
```

### 🔄 자동 적응

웹사이트 구조가 변경되어도 LLM이 자동으로 새로운 구조를 이해하고 데이터를 추출합니다. CSS 셀렉터 수정 불필요!

### 📊 다양한 그래프 타입

- **SmartScraperGraph**: 단일 페이지 스크래핑
- **SearchGraph**: 검색 엔진 + 스크래핑
- **Multi 그래프**: 병렬 처리로 여러 페이지 동시 스크래핑
- **CodeGenerator**: Python 스크립트 자동 생성
- **SpeechGraph**: 스크래핑 + 음성 파일 변환

### 🌐 폭넓은 LLM 지원

| 제공자 | 모델 | 비용 |
|--------|------|------|
| **Ollama** | Llama 3.2, Mistral | 무료 (로컬) |
| **OpenAI** | GPT-4o, GPT-4o-mini | 유료 |
| **Anthropic** | Claude 3.5 Sonnet | 유료 |
| **Groq** | Llama 3.1, Mixtral | 무료 (초고속) |
| **Google** | Gemini Pro, Flash | 유료/무료 |

### 🔌 강력한 통합

- **LLM 프레임워크**: Langchain, LlamaIndex, CrewAI, Agno
- **노코드**: n8n, Zapier, Pipedream, Bubble
- **API/SDK**: Python SDK, Node.js SDK

---

## 빠른 시작

### 1. 설치

```bash
pip install scrapegraphai
playwright install
```

### 2. 첫 스크래핑

```python
from scrapegraphai.graphs import SmartScraperGraph

# Ollama (로컬 LLM) 사용
scraper = SmartScraperGraph(
    prompt="Extract all product names and prices",
    source="https://example.com/products",
    config={
        "llm": {"model": "ollama/llama3.2"}
    }
)

result = scraper.run()
print(result)
```

### 3. OpenAI 사용

```python
scraper = SmartScraperGraph(
    prompt="Extract article title, author, and publish date",
    source="https://blog.example.com/article",
    config={
        "llm": {
            "api_key": "sk-proj-...",
            "model": "openai/gpt-4o-mini"
        }
    }
)

result = scraper.run()
```

---

## 실전 활용 사례

### 🏢 경쟁사 분석

```python
from scrapegraphai.graphs import SmartScraperMultiGraph

competitors = [
    "https://competitor1.com/pricing",
    "https://competitor2.com/pricing",
    "https://competitor3.com/pricing"
]

scraper = SmartScraperMultiGraph(
    prompt="Extract pricing plans and features",
    source=competitors,
    config={"llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}}
)

pricing_data = scraper.run()
```

### 📰 뉴스 모니터링

```python
from scrapegraphai.graphs import SearchGraph

news_monitor = SearchGraph(
    prompt="What are the latest AI developments this week?",
    config={
        "llm": {"model": "groq/llama-3.1-70b-versatile", "api_key": "gsk_..."},
        "max_results": 10
    }
)

summary = news_monitor.run()
```

### 🎯 리드 생성

```python
from scrapegraphai.graphs import SearchGraph, SmartScraperMultiGraph

# 1. 회사 찾기
lead_finder = SearchGraph(
    prompt="Find SaaS companies in healthcare",
    config={"llm": {"model": "ollama/llama3.2"}, "max_results": 50}
)

company_urls = lead_finder.run()

# 2. 정보 수집
info_scraper = SmartScraperMultiGraph(
    prompt="Extract company name, email, and LinkedIn",
    source=company_urls,
    config={"llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}}
)

leads = info_scraper.run()
```

---

## 벤치마크

Firecrawl 벤치마크에 따르면, **ScrapeGraphAI는 시장 최고의 페처 성능**을 보여줍니다:

- ✅ 높은 정확도
- ⚡ 빠른 처리 속도
- 🎯 안정적인 구조화된 데이터 추출

---

## 프로젝트 통계

- **PyPI 다운로드**: 1M+ downloads
- **GitHub Stars**: 18k+ stars
- **최신 버전**: v1.73.0
- **라이선스**: MIT

---

## 공식 링크

- [GitHub 저장소](https://github.com/ScrapeGraphAI/Scrapegraph-ai)
- [공식 문서](https://scrapegraph-ai.readthedocs.io/)
- [API 문서](https://docs.scrapegraphai.com/)
- [Discord 커뮤니티](https://discord.gg/gkxQDAjfeX)
- [API 대시보드](https://dashboard.scrapegraphai.com/)

---

## 왜 ScrapeGraphAI인가?

| 기존 스크래핑 | ScrapeGraphAI |
|--------------|---------------|
| CSS 셀렉터 작성 | 자연어 프롬프트 |
| 웹사이트 변경 시 코드 수정 | 자동 적응 |
| 사이트마다 새 스크립트 | 범용 솔루션 |
| 정적 콘텐츠만 | 동적 콘텐츠 지원 |
| 수작업 데이터 정제 | 구조화된 출력 |

---

## 시작하기

[🚀 Chapter 1: 소개 및 개요부터 시작하기]({{ site.baseurl }}/scrapegraph-guide-01-intro/)
