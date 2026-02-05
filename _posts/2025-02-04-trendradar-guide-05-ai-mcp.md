---
layout: post
title: "TrendRadar 완벽 가이드 (5) - AI 분석 & MCP"
date: 2025-02-04
permalink: /trendradar-guide-05-ai-mcp/
author: sansan0
categories: [AI]
tags: [TrendRadar, AI, MCP, OpenAI, Claude, LLM]
original_url: "https://github.com/sansan0/TrendRadar"
excerpt: "TrendRadar의 AI 분석 기능과 MCP 서버 통합을 분석합니다."
---

## AI 분석 개요

TrendRadar는 **LLM을 활용한 뉴스 분석 기능**을 제공합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Analysis Pipeline                          │
│                                                                  │
│   Raw News ──▶ Summarize ──▶ Translate ──▶ Analyze ──▶ Push    │
│                    │             │            │                  │
│                    ▼             ▼            ▼                  │
│               요약 생성     다국어 번역   트렌드 분석            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 지원 AI 프로바이더

| 프로바이더 | 모델 | 특징 |
|-----------|------|------|
| **OpenAI** | GPT-4o, GPT-4o-mini | 가장 널리 사용 |
| **Anthropic** | Claude 3.5 Sonnet | 긴 컨텍스트 |
| **DeepSeek** | DeepSeek-V3 | 저렴한 비용 |
| **Local** | Ollama | 무료, 프라이버시 |

---

## AI 설정

```yaml
# config/config.yaml

ai:
  enabled: true
  provider: openai  # openai, anthropic, deepseek, local

  openai:
    api_key: ${OPENAI_API_KEY}
    model: gpt-4o-mini
    base_url: https://api.openai.com/v1  # 선택적

  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-3-5-sonnet-20241022

  deepseek:
    api_key: ${DEEPSEEK_API_KEY}
    model: deepseek-chat

  local:
    base_url: http://localhost:11434/v1
    model: llama3.2

  # 분석 설정
  analysis:
    summarize: true
    translate: false
    translate_to: ko
    max_tokens: 500
```

---

## AI 분석 구현

### 베이스 클래스

```python
# trendradar/ai/base.py

from abc import ABC, abstractmethod

class BaseAIProvider(ABC):
    def __init__(self, context: Context):
        self.context = context

    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """텍스트 생성"""
        pass

    async def summarize(self, text: str) -> str:
        """텍스트 요약"""
        prompt = f"""다음 뉴스 기사를 2-3문장으로 요약해주세요.
핵심 내용만 간결하게 작성하세요.

기사:
{text}

요약:"""

        return await self.complete(prompt)

    async def translate(self, text: str, target_lang: str) -> str:
        """텍스트 번역"""
        prompt = f"""다음 텍스트를 {target_lang}로 번역해주세요.

원문:
{text}

번역:"""

        return await self.complete(prompt)
```

### OpenAI 프로바이더

```python
# trendradar/ai/openai.py

class OpenAIProvider(BaseAIProvider):
    """OpenAI API 프로바이더"""

    def __init__(self, context: Context):
        super().__init__(context)
        cfg = context.config.ai.openai
        self.api_key = cfg.api_key
        self.model = cfg.model
        self.base_url = cfg.base_url or "https://api.openai.com/v1"

    async def complete(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.context.config.ai.analysis.max_tokens,
        }

        async with self.context.http_client.post(
            url, headers=headers, json=payload
        ) as resp:
            data = await resp.json()
            return data["choices"][0]["message"]["content"]
```

### AI 분석 관리자

```python
# trendradar/ai/analyzer.py

class AIAnalyzer:
    """AI 분석 관리자"""

    PROVIDERS = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'deepseek': DeepSeekProvider,
        'local': LocalProvider,
    }

    def __init__(self, context: Context):
        self.context = context
        self.config = context.config.ai

        provider_cls = self.PROVIDERS.get(self.config.provider)
        if not provider_cls:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        self.provider = provider_cls(context)

    async def analyze(self, items: List[NewsItem]) -> List[NewsItem]:
        """뉴스 항목 분석"""
        if not self.config.enabled:
            return items

        tasks = [self._analyze_item(item) for item in items]
        return await asyncio.gather(*tasks)

    async def _analyze_item(self, item: NewsItem) -> NewsItem:
        # 요약
        if self.config.analysis.summarize and item.content:
            item.summary = await self.provider.summarize(item.content)

        # 번역
        if self.config.analysis.translate:
            text = item.summary or item.title
            item.summary = await self.provider.translate(
                text,
                self.config.analysis.translate_to
            )

        return item
```

---

## MCP 서버

TrendRadar는 **Model Context Protocol (MCP)** 서버를 제공하여 AI 에이전트와 통합할 수 있습니다.

### MCP 서버 구조

```
mcp_server/
├── __init__.py
├── server.py        # MCP 서버 메인
├── tools/           # MCP 도구
│   ├── news.py      # 뉴스 조회 도구
│   └── search.py    # 검색 도구
├── services/        # 서비스 레이어
└── utils/           # 유틸리티
```

### MCP 서버 구현

```python
# mcp_server/server.py

from mcp.server import Server
from mcp.types import Tool, TextContent

class TrendRadarMCPServer:
    def __init__(self):
        self.server = Server("trendradar")
        self._register_tools()

    def _register_tools(self):
        @self.server.tool()
        async def get_trending_news(
            category: str = "all",
            limit: int = 10
        ) -> list:
            """Get trending news from various sources"""
            news = await self._fetch_news(category, limit)
            return [
                {
                    "title": item.title,
                    "url": item.url,
                    "source": item.source,
                    "summary": item.summary,
                }
                for item in news
            ]

        @self.server.tool()
        async def search_news(
            query: str,
            category: str = "all",
            limit: int = 10
        ) -> list:
            """Search news by keyword"""
            news = await self._search_news(query, category, limit)
            return [
                {
                    "title": item.title,
                    "url": item.url,
                    "source": item.source,
                    "relevance": item.relevance,
                }
                for item in news
            ]

        @self.server.tool()
        async def get_news_summary(url: str) -> str:
            """Get AI summary of a specific news article"""
            content = await self._fetch_article(url)
            summary = await self._summarize(content)
            return summary

    async def run(self):
        async with stdio_server() as streams:
            await self.server.run(
                streams[0],
                streams[1],
                self.server.create_initialization_options()
            )
```

### Claude Desktop 설정

```json
// ~/.config/claude-desktop/config.json

{
  "mcpServers": {
    "trendradar": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "/path/to/TrendRadar"
    }
  }
}
```

---

## MCP 도구 목록

### get_trending_news

```typescript
{
  name: "get_trending_news",
  description: "Get trending news from various sources",
  inputSchema: {
    type: "object",
    properties: {
      category: {
        type: "string",
        enum: ["all", "tech", "finance", "world", "china"],
        default: "all"
      },
      limit: {
        type: "integer",
        default: 10,
        maximum: 50
      }
    }
  }
}
```

### search_news

```typescript
{
  name: "search_news",
  description: "Search news by keyword",
  inputSchema: {
    type: "object",
    required: ["query"],
    properties: {
      query: {
        type: "string",
        description: "Search keyword"
      },
      category: {
        type: "string",
        default: "all"
      },
      limit: {
        type: "integer",
        default: 10
      }
    }
  }
}
```

### get_news_summary

```typescript
{
  name: "get_news_summary",
  description: "Get AI summary of a specific news article",
  inputSchema: {
    type: "object",
    required: ["url"],
    properties: {
      url: {
        type: "string",
        description: "News article URL"
      }
    }
  }
}
```

---

## 사용 예시

### Claude와 대화

```
User: 오늘 기술 뉴스 중 AI 관련 소식 알려줘

Claude: [get_trending_news 도구 호출]
        category: "tech", limit: 20

        오늘의 주요 AI 관련 뉴스입니다:

        1. OpenAI, GPT-5 개발 착수 발표
           - 출처: TechCrunch
           - 요약: OpenAI가 차세대 모델 개발을 공식화...

        2. Google, Gemini 2.0 업데이트 출시
           - 출처: The Verge
           - 요약: 멀티모달 성능 50% 향상...
```

---

## 비용 최적화

### 토큰 사용량 관리

```python
# 캐싱으로 중복 요청 방지
class CachedAIProvider:
    def __init__(self, provider: BaseAIProvider, ttl: int = 3600):
        self.provider = provider
        self.cache = TTLCache(maxsize=1000, ttl=ttl)

    async def complete(self, prompt: str) -> str:
        cache_key = hashlib.md5(prompt.encode()).hexdigest()

        if cache_key in self.cache:
            return self.cache[cache_key]

        result = await self.provider.complete(prompt)
        self.cache[cache_key] = result

        return result
```

### 배치 처리

```python
# 여러 뉴스를 한 번에 요약
async def batch_summarize(self, items: List[NewsItem]) -> List[str]:
    combined = "\n---\n".join([
        f"[{i+1}] {item.title}\n{item.content}"
        for i, item in enumerate(items)
    ])

    prompt = f"""다음 뉴스들을 각각 1문장으로 요약해주세요.
[번호] 요약 형식으로 작성하세요.

{combined}"""

    result = await self.provider.complete(prompt)
    return self._parse_batch_result(result)
```

---

*다음 글에서는 배포 및 활용 방법을 살펴봅니다.*
