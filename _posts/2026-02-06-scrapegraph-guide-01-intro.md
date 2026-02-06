---
layout: post
title: "ScrapeGraphAI 완벽 가이드 (1) - 소개 및 개요"
date: 2026-02-06
permalink: /scrapegraph-guide-01-intro/
author: ScrapeGraphAI Team
categories: [AI 도구, 웹 스크래핑]
tags: [ScrapeGraphAI, Web Scraping, LLM, Langchain, AI Agent]
original_url: "https://github.com/ScrapeGraphAI/Scrapegraph-ai"
excerpt: "LLM을 활용한 차세대 웹 스크래핑 라이브러리 ScrapeGraphAI를 소개합니다."
---

## ScrapeGraphAI란?

ScrapeGraphAI는 **LLM(대규모 언어 모델)과 그래프 로직을 활용한 Python 웹 스크래핑 라이브러리**입니다. 기존 스크래핑 도구와 달리, **자연어 프롬프트만으로** 웹사이트나 로컬 문서(XML, HTML, JSON, Markdown 등)에서 원하는 정보를 추출할 수 있습니다.

### You Only Scrape Once

ScrapeGraphAI의 슬로건은 **"You Only Scrape Once"**입니다. 웹사이트 구조가 변경되어도 코드 수정 없이 LLM이 자동으로 적응합니다.

```python
from scrapegraphai.graphs import SmartScraperGraph

# 단 5줄의 코드로 스크래핑 완료!
smart_scraper_graph = SmartScraperGraph(
    prompt="Extract company description, founders and social media links",
    source="https://scrapegraphai.com/",
    config={"llm": {"model": "ollama/llama3.2"}}
)

result = smart_scraper_graph.run()
```

## 왜 ScrapeGraphAI인가?

### 기존 스크래핑 도구의 한계

전통적인 웹 스크래핑은 다음과 같은 문제점이 있습니다:

- **유지보수 부담**: 웹사이트 구조 변경 시 CSS 셀렉터 수정 필요
- **복잡한 로직**: XPath, BeautifulSoup 등 복잡한 파싱 코드 작성
- **동적 콘텐츠**: JavaScript 렌더링 처리의 어려움
- **재사용성 낮음**: 사이트마다 새로운 스크립트 작성

### ScrapeGraphAI의 해결책

- **자연어 기반**: "회사 소개와 소셜 미디어 링크를 추출해줘" 같은 프롬프트로 작동
- **자동 적응**: LLM이 웹사이트 구조를 이해하고 필요한 정보 추출
- **멀티모달**: 텍스트, 이미지, PDF 등 다양한 형식 지원
- **그래프 기반**: 복잡한 스크래핑 파이프라인을 그래프로 구성

## 핵심 특징

### 1. LLM 기반 인텔리전트 스크래핑

ScrapeGraphAI는 다양한 LLM을 지원합니다:

- **로컬 모델**: Ollama (Llama 3.2, Mistral 등)
- **클라우드 API**: OpenAI GPT-4, Anthropic Claude, Google Gemini
- **엔터프라이즈**: Azure OpenAI, AWS Bedrock

### 2. 다양한 그래프 파이프라인

| 그래프 타입 | 설명 |
|------------|------|
| **SmartScraperGraph** | 단일 페이지 스크래핑 (가장 기본) |
| **SearchGraph** | 검색 결과 상위 N개 페이지 스크래핑 |
| **SpeechGraph** | 스크래핑 + 음성 파일 생성 |
| **ScriptCreatorGraph** | 스크래핑 Python 스크립트 자동 생성 |
| **Multi 버전** | 병렬 처리로 여러 페이지 동시 스크래핑 |

### 3. 폭넓은 통합 지원

- **LLM 프레임워크**: Langchain, LlamaIndex, CrewAI, Agno
- **노코드 플랫폼**: n8n, Zapier, Pipedream, Bubble
- **API/SDK**: Python SDK, Node.js SDK 제공

## 실제 사용 예시

```python
from scrapegraphai.graphs import SmartScraperGraph

graph_config = {
    "llm": {
        "model": "ollama/llama3.2",
        "model_tokens": 8192,
        "format": "json",
    },
    "verbose": True,
    "headless": False,
}

smart_scraper_graph = SmartScraperGraph(
    prompt="Extract useful information from the webpage",
    source="https://scrapegraphai.com/",
    config=graph_config
)

result = smart_scraper_graph.run()
print(result)
```

**출력 예시:**
```json
{
    "description": "ScrapeGraphAI transforms websites into clean, organized data for AI agents",
    "founders": [
        {"name": "Marco Vinciguerra", "role": "Founder & Software Engineer"}
    ],
    "social_media_links": {
        "github": "https://github.com/ScrapeGraphAI/Scrapegraph-ai"
    }
}
```

## 벤치마크 성능

Firecrawl 벤치마크에 따르면, ScrapeGraphAI는 **시장 최고의 페처(fetcher) 성능**을 자랑합니다.

주요 장점:
- 높은 정확도
- 빠른 처리 속도
- 안정적인 구조화된 데이터 추출

## 프로젝트 통계

- **PyPI 다운로드**: 100만+ 다운로드
- **GitHub Stars**: 18k+ stars
- **버전**: v1.73.0 (활발한 업데이트)
- **라이선스**: MIT

## 누가 사용해야 하나?

ScrapeGraphAI는 다음과 같은 경우에 적합합니다:

- **데이터 엔지니어**: 대규모 웹 데이터 수집 및 정제
- **AI 개발자**: RAG 시스템을 위한 데이터 소싱
- **리서처**: 웹 기반 연구 데이터 자동 수집
- **스타트업**: 빠른 프로토타이핑과 MVP 개발

## 다음 단계

다음 챕터에서는 ScrapeGraphAI를 설치하고 첫 스크래핑을 실행하는 방법을 다룹니다.

---

## 시리즈 네비게이션

- **현재**: (1) 소개 및 개요
- **다음**: [(2) 설치 및 빠른 시작]({{ site.baseurl }}/scrapegraph-guide-02-installation/)

[📚 전체 목차로 돌아가기]({{ site.baseurl }}/scrapegraph-guide/)
