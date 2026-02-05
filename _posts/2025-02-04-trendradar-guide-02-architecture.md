---
layout: post
title: "TrendRadar 완벽 가이드 (2) - 아키텍처"
date: 2025-02-04
permalink: /trendradar-guide-02-architecture/
author: sansan0
categories: [개발 도구, TrendRadar]
tags: [TrendRadar, Architecture, Python, Async]
original_url: "https://github.com/sansan0/TrendRadar"
excerpt: "TrendRadar의 모듈 구조와 데이터 흐름을 분석합니다."
---

## 전체 아키텍처

TrendRadar는 **모듈화된 파이프라인 아키텍처**를 사용합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                   TrendRadar Pipeline                            │
│                                                                  │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐        │
│   │ Sources │──▶│  Core   │──▶│   AI    │──▶│  Push   │        │
│   │ Module  │   │ Engine  │   │ Module  │   │ Module  │        │
│   └─────────┘   └─────────┘   └─────────┘   └─────────┘        │
│        │             │             │             │               │
│        │             │             │             │               │
│        ▼             ▼             ▼             ▼               │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐        │
│   │ NewsNow │   │ Context │   │ OpenAI  │   │Telegram │        │
│   │   RSS   │   │ Storage │   │ Claude  │   │ WeChat  │        │
│   │ Custom  │   │ Cache   │   │DeepSeek │   │ Slack   │        │
│   └─────────┘   └─────────┘   └─────────┘   └─────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 핵심 모듈

### 1. 진입점 (\_\_main\_\_.py)

```python
# trendradar/__main__.py

class TrendRadarApp:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.context = Context(self.config)

        # 모듈 초기화
        self.crawler_manager = CrawlerManager(self.context)
        self.ai_analyzer = AIAnalyzer(self.context)
        self.notifier_manager = NotifierManager(self.context)

    async def run(self):
        """메인 실행 루프"""
        while True:
            # 1. 뉴스 수집
            news_items = await self.crawler_manager.fetch_all()

            # 2. 새로운 뉴스 필터링
            new_items = self.context.storage.filter_new(news_items)

            if new_items:
                # 3. AI 분석 (옵션)
                if self.config.ai.enabled:
                    new_items = await self.ai_analyzer.analyze(new_items)

                # 4. 알림 전송
                await self.notifier_manager.notify_all(new_items)

                # 5. 저장
                self.context.storage.save(new_items)

            # 6. 대기
            await asyncio.sleep(self.config.schedule.interval)
```

### 2. 컨텍스트 관리 (context.py)

```python
# trendradar/context.py

class Context:
    """전역 컨텍스트 - 모든 모듈이 공유하는 상태"""

    def __init__(self, config: Config):
        self.config = config
        self.storage = Storage(config.storage)
        self.http_client = aiohttp.ClientSession()
        self.logger = self._setup_logger()

    async def close(self):
        """리소스 정리"""
        await self.http_client.close()
        self.storage.close()
```

---

## 모듈별 상세 구조

### Crawler 모듈

```
trendradar/crawler/
├── __init__.py
├── base.py           # 크롤러 베이스 클래스
├── newsnow.py        # NewsNow API 크롤러
├── rss.py            # RSS 피드 크롤러
└── manager.py        # 크롤러 관리자
```

```python
# trendradar/crawler/base.py

from abc import ABC, abstractmethod

class BaseCrawler(ABC):
    def __init__(self, context: Context):
        self.context = context
        self.config = context.config

    @abstractmethod
    async def fetch(self) -> List[NewsItem]:
        """뉴스 항목 수집"""
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """소스 이름 반환"""
        pass
```

### Notification 모듈

```
trendradar/notification/
├── __init__.py
├── base.py           # 알림 베이스 클래스
├── telegram.py       # Telegram 봇
├── wechat.py         # WeChat (기업/개인)
├── dingtalk.py       # DingTalk
├── feishu.py         # Feishu (Lark)
├── slack.py          # Slack
├── email.py          # Email (SMTP)
├── ntfy.py           # ntfy
├── bark.py           # Bark
├── webhook.py        # 커스텀 Webhook
└── manager.py        # 알림 관리자
```

```python
# trendradar/notification/base.py

class BaseNotifier(ABC):
    def __init__(self, context: Context):
        self.context = context

    @abstractmethod
    async def send(self, news_items: List[NewsItem]) -> bool:
        """알림 전송"""
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """활성화 여부"""
        pass
```

### AI 모듈

```
trendradar/ai/
├── __init__.py
├── base.py           # AI 분석 베이스
├── openai.py         # OpenAI (GPT)
├── anthropic.py      # Anthropic (Claude)
├── deepseek.py       # DeepSeek
├── local.py          # 로컬 LLM
└── analyzer.py       # AI 분석 관리자
```

---

## 데이터 모델

### NewsItem

```python
# trendradar/core/models.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

@dataclass
class NewsItem:
    id: str                          # 고유 ID
    title: str                       # 제목
    url: str                         # 원본 URL
    source: str                      # 소스 (newsnow, rss 등)
    category: str                    # 카테고리
    published_at: datetime           # 게시 시간

    # 선택적 필드
    content: Optional[str] = None    # 본문
    summary: Optional[str] = None    # AI 요약
    image_url: Optional[str] = None  # 이미지
    tags: List[str] = None           # 태그

    # AI 분석 결과
    ai_analysis: Optional[dict] = None
```

### Config

```python
@dataclass
class Config:
    sources: List[SourceConfig]
    ai: AIConfig
    notifications: NotificationConfig
    schedule: ScheduleConfig
    storage: StorageConfig
```

---

## 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Flow                                   │
│                                                                  │
│   1. FETCH                                                       │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                    │
│   │ NewsNow │───▶│         │◀───│   RSS   │                    │
│   │   API   │    │ Crawler │    │  Feeds  │                    │
│   └─────────┘    │ Manager │    └─────────┘                    │
│                  └────┬────┘                                     │
│                       │                                          │
│   2. FILTER          ▼                                          │
│                  ┌─────────┐                                    │
│                  │ Storage │ ──▶ 이미 본 뉴스 제외              │
│                  │  Check  │                                    │
│                  └────┬────┘                                     │
│                       │                                          │
│   3. ANALYZE         ▼                                          │
│                  ┌─────────┐                                    │
│                  │   AI    │ ──▶ 요약, 번역, 분석               │
│                  │ Module  │                                    │
│                  └────┬────┘                                     │
│                       │                                          │
│   4. NOTIFY          ▼                                          │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                    │
│   │Telegram │◀───│Notifier │───▶│  Slack  │                    │
│   └─────────┘    │ Manager │    └─────────┘                    │
│                  └────┬────┘                                     │
│                       │                                          │
│   5. SAVE            ▼                                          │
│                  ┌─────────┐                                    │
│                  │ Storage │ ──▶ 다음 실행 시 중복 방지         │
│                  │  Save   │                                    │
│                  └─────────┘                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 비동기 처리

TrendRadar는 `asyncio`를 사용한 비동기 처리로 효율성을 극대화합니다.

```python
# 병렬 크롤링
async def fetch_all(self) -> List[NewsItem]:
    tasks = [
        crawler.fetch()
        for crawler in self.crawlers
        if crawler.is_enabled()
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_items = []
    for result in results:
        if isinstance(result, Exception):
            self.logger.error(f"Crawler error: {result}")
        else:
            all_items.extend(result)

    return all_items

# 병렬 알림 전송
async def notify_all(self, items: List[NewsItem]):
    tasks = [
        notifier.send(items)
        for notifier in self.notifiers
        if notifier.is_enabled()
    ]

    await asyncio.gather(*tasks, return_exceptions=True)
```

---

## 설정 시스템

### 환경 변수 지원

```python
# config 로드 시 환경 변수 치환
def load_config(path: str) -> Config:
    with open(path) as f:
        content = f.read()

    # ${VAR_NAME} 패턴 치환
    pattern = r'\$\{(\w+)\}'

    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, '')

    content = re.sub(pattern, replacer, content)

    return yaml.safe_load(content)
```

### 설정 검증

```python
def validate_config(config: Config):
    # 필수 설정 검증
    if not config.sources:
        raise ConfigError("At least one source required")

    # 알림 채널 검증
    enabled_notifiers = [
        n for n in config.notifications
        if n.enabled
    ]

    if not enabled_notifiers:
        raise ConfigError("At least one notifier required")
```

---

*다음 글에서는 크롤러와 데이터 소스를 살펴봅니다.*
