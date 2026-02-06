---
layout: post
title: "ScrapeGraphAI 완벽 가이드 (9) - 통합 및 확장"
date: 2026-02-06
permalink: /scrapegraph-guide-09-integrations/
author: ScrapeGraphAI Team
categories: [AI 도구, 웹 스크래핑]
tags: [ScrapeGraphAI, Integration, API, SDK, Langchain, n8n, Zapier]
original_url: "https://github.com/ScrapeGraphAI/Scrapegraph-ai"
excerpt: "ScrapeGraphAI를 다양한 프레임워크 및 플랫폼과 통합하는 방법을 배웁니다."
---

## 통합 개요

ScrapeGraphAI는 다음과 같은 통합을 제공합니다:

- **API/SDK**: Python SDK, Node.js SDK
- **LLM 프레임워크**: Langchain, LlamaIndex, CrewAI, Agno
- **노코드 플랫폼**: n8n, Zapier, Pipedream, Bubble
- **MCP 서버**: Claude Desktop 통합

## 공식 API 사용

### API 가입

1. [ScrapeGraphAI Dashboard](https://dashboard.scrapegraphai.com/login) 접속
2. 계정 생성 및 API 키 발급
3. 크레딧 구매 (무료 티어 제공)

### Python SDK

#### 설치

```bash
pip install scrapegraph-py
```

#### 기본 사용법

```python
from scrapegraph_py import Client

client = Client(api_key="sgai-...")

# SmartScraper 실행
result = client.smartscraper(
    website_url="https://example.com",
    user_prompt="Extract product names and prices"
)

print(result)
```

#### 고급 사용

```python
# 여러 URL 스크래핑
results = client.smartscraper(
    website_url=[
        "https://site1.com",
        "https://site2.com",
        "https://site3.com"
    ],
    user_prompt="Extract company info",
    output_schema={
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "industry": {"type": "string"}
        }
    }
)
```

### Node.js SDK

#### 설치

```bash
npm install scrapegraph-js
```

#### 사용 예제

```javascript
const { Client } = require('scrapegraph-js');

const client = new Client({ apiKey: 'sgai-...' });

async function scrape() {
    const result = await client.smartscraper({
        websiteUrl: 'https://example.com',
        userPrompt: 'Extract all article titles'
    });

    console.log(result);
}

scrape();
```

### REST API

#### cURL 예제

```bash
curl -X POST https://api.scrapegraphai.com/v1/smartscraper \
  -H "Authorization: Bearer sgai-..." \
  -H "Content-Type: application/json" \
  -d '{
    "website_url": "https://example.com",
    "user_prompt": "Extract product information"
  }'
```

#### Python requests

```python
import requests

response = requests.post(
    "https://api.scrapegraphai.com/v1/smartscraper",
    headers={
        "Authorization": "Bearer sgai-...",
        "Content-Type": "application/json"
    },
    json={
        "website_url": "https://example.com",
        "user_prompt": "Extract data"
    }
)

result = response.json()
print(result)
```

## Langchain 통합

### Langchain Tool로 사용

```python
from langchain.tools import Tool
from scrapegraphai.graphs import SmartScraperGraph

def scrape_website(url: str) -> dict:
    """웹사이트 스크래핑 함수"""
    scraper = SmartScraperGraph(
        prompt="Extract main content",
        source=url,
        config={"llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}}
    )
    return scraper.run()

# Langchain Tool 생성
scraping_tool = Tool(
    name="WebScraper",
    func=scrape_website,
    description="Scrapes a website and extracts structured information"
)
```

### Agent와 함께 사용

```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", api_key="sk-...")

# Agent 생성
agent = initialize_agent(
    tools=[scraping_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 실행
response = agent.run("Scrape https://example.com and summarize the content")
print(response)
```

## LlamaIndex 통합

### Reader로 사용

```python
from llama_index.core import SimpleDirectoryReader
from scrapegraphai.graphs import SmartScraperGraph

class ScrapeGraphReader:
    def __init__(self, config):
        self.config = config

    def load_data(self, urls, prompt):
        """URLs에서 데이터 로드"""
        documents = []

        for url in urls:
            scraper = SmartScraperGraph(
                prompt=prompt,
                source=url,
                config=self.config
            )
            result = scraper.run()
            documents.append(result)

        return documents

# 사용
reader = ScrapeGraphReader(
    config={"llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}}
)

docs = reader.load_data(
    urls=["https://site1.com", "https://site2.com"],
    prompt="Extract article content"
)
```

## CrewAI 통합

### CrewAI Tool로 사용

```python
from crewai import Agent, Task, Crew
from crewai_tools import tool
from scrapegraphai.graphs import SmartScraperGraph

@tool("Web Scraping Tool")
def scrape_tool(url: str, prompt: str) -> dict:
    """Scrapes a website using ScrapeGraphAI"""
    scraper = SmartScraperGraph(
        prompt=prompt,
        source=url,
        config={"llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}}
    )
    return scraper.run()

# Agent 생성
researcher = Agent(
    role="Research Analyst",
    goal="Gather competitive intelligence",
    tools=[scrape_tool],
    verbose=True
)

# Task 정의
task = Task(
    description="Scrape competitor websites and extract pricing info",
    agent=researcher
)

# Crew 실행
crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

## Agno 통합

```python
from agno import Agent
from scrapegraphai.graphs import SmartScraperGraph

def scrape_data(url: str) -> dict:
    scraper = SmartScraperGraph(
        prompt="Extract product data",
        source=url,
        config={"llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}}
    )
    return scraper.run()

# Agno Agent
agent = Agent(
    name="Data Collector",
    tools=[scrape_data],
    model="openai/gpt-4o-mini"
)

response = agent.run("Collect product information from https://example.com")
```

## n8n 통합

### n8n 노드 사용

1. n8n에서 "HTTP Request" 노드 추가
2. URL: `https://api.scrapegraphai.com/v1/smartscraper`
3. Method: POST
4. Headers:
   - `Authorization: Bearer sgai-...`
   - `Content-Type: application/json`
5. Body:
   ```json
   {
     "website_url": "https://example.com",
     "user_prompt": "Extract data"
   }
   ```

### 워크플로우 예제

```
[Trigger] → [HTTP Request (ScrapeGraphAI)] → [Process Data] → [Save to Database]
```

## Zapier 통합

### Zapier App

1. [Zapier](https://zapier.com/apps/scrapegraphai/integrations) 접속
2. "Create Zap" 클릭
3. Trigger 선택 (예: Google Sheets 새 행)
4. Action: ScrapeGraphAI - Scrape Website
5. 설정:
   - API Key: `sgai-...`
   - Website URL: 트리거의 URL 컬럼
   - User Prompt: "Extract product info"

### 사용 사례

- **리드 생성**: Google Sheets → ScrapeGraphAI → CRM
- **가격 모니터링**: Scheduler → ScrapeGraphAI → Slack
- **콘텐츠 수집**: RSS Feed → ScrapeGraphAI → Notion

## Pipedream 통합

```javascript
// Pipedream Step
export default defineComponent({
  async run({ steps, $ }) {
    const response = await require("@pipedream/platform").axios($, {
      method: "POST",
      url: "https://api.scrapegraphai.com/v1/smartscraper",
      headers: {
        Authorization: `Bearer ${this.scrapegraphai.$auth.api_key}`,
      },
      data: {
        website_url: "https://example.com",
        user_prompt: "Extract data"
      }
    });

    return response;
  }
});
```

## Bubble 통합

### API Connector 설정

1. Bubble 앱에서 Plugins → API Connector
2. Add API:
   - Name: ScrapeGraphAI
   - Authentication: Private key in header
   - Key: Authorization
   - Value: Bearer sgai-...
3. Add Call:
   - Name: smartscraper
   - Type: POST
   - URL: `https://api.scrapegraphai.com/v1/smartscraper`
   - Body:
     ```json
     {
       "website_url": "<url>",
       "user_prompt": "<prompt>"
     }
     ```

## MCP Server (Claude Desktop)

### 설치

```bash
npx @smithery/cli install @ScrapeGraphAI/scrapegraph-mcp --client claude
```

### 사용

Claude Desktop에서 바로 사용:

```
User: Scrape https://example.com and extract product names

Claude: [Uses ScrapeGraph MCP Server]
```

## Dify 통합

Dify는 LLM 앱 개발 플랫폼입니다.

### Tool 추가

1. Dify Studio → Tools
2. Add Custom Tool
3. 설정:
   - Name: ScrapeGraphAI
   - Method: POST
   - URL: `https://api.scrapegraphai.com/v1/smartscraper`
   - Headers: Authorization, Content-Type
   - Body: JSON schema

## 실전 활용 사례

### 사례 1: 자동 리드 생성 파이프라인

```
Google Sheets (기업 목록)
    ↓
Zapier Trigger (새 행 추가)
    ↓
ScrapeGraphAI (회사 정보 수집)
    ↓
Airtable (리드 저장)
    ↓
Slack (알림)
```

### 사례 2: 경쟁사 모니터링

```
n8n Schedule (매일 오전 9시)
    ↓
ScrapeGraphAI (경쟁사 가격 스크래핑)
    ↓
Supabase (데이터 저장)
    ↓
Grafana (대시보드 업데이트)
    ↓
Discord (변경사항 알림)
```

### 사례 3: 콘텐츠 집계 봇

```python
# CrewAI + ScrapeGraphAI
from crewai import Agent, Task, Crew

# Content Collector Agent
collector = Agent(
    role="Content Collector",
    goal="Collect tech news from top sources",
    tools=[scrape_tool]
)

# Content Summarizer Agent
summarizer = Agent(
    role="Content Summarizer",
    goal="Create concise summaries"
)

# Tasks
collect_task = Task(
    description="Scrape tech news from 10 sources",
    agent=collector
)

summarize_task = Task(
    description="Summarize collected articles",
    agent=summarizer
)

# Run
crew = Crew(
    agents=[collector, summarizer],
    tasks=[collect_task, summarize_task]
)

result = crew.kickoff()
```

## 비용 및 제한사항

### API 가격

- **Free**: 100 크레딧 (테스트용)
- **Starter**: $29/월 (1,000 크레딧)
- **Professional**: $99/월 (5,000 크레딧)
- **Enterprise**: 커스텀 가격

### Rate Limits

- Free: 10 requests/min
- Starter: 30 requests/min
- Professional: 100 requests/min

## 다음 단계

마지막 챕터에서는 **실전 활용 및 팁**을 다룹니다.

---

## 시리즈 네비게이션

- **이전**: [(8) LLM 모델 연동]({{ site.baseurl }}/scrapegraph-guide-08-llm-integration/)
- **현재**: (9) 통합 및 확장
- **다음**: [(10) 실전 활용 및 팁]({{ site.baseurl }}/scrapegraph-guide-10-tips/)

[📚 전체 목차로 돌아가기]({{ site.baseurl }}/scrapegraph-guide/)
