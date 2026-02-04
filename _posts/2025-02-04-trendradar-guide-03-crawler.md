---
layout: post
title: "TrendRadar 완벽 가이드 (3) - 크롤러 & 데이터 소스"
date: 2025-02-04
permalink: /trendradar-guide-03-crawler/
author: sansan0
category: AI
tags: [TrendRadar, Crawler, RSS, NewsNow, Web Scraping]
series: trendradar-guide
part: 3
original_url: "https://github.com/sansan0/TrendRadar"
excerpt: "TrendRadar의 뉴스 수집 시스템과 다양한 데이터 소스를 분석합니다."
---

## 크롤러 개요

TrendRadar는 여러 소스에서 뉴스를 수집하는 **플러그인 기반 크롤러 시스템**을 사용합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Crawler Architecture                          │
│                                                                  │
│   ┌─────────────────────────────────────────────────┐           │
│   │              CrawlerManager                      │           │
│   │                                                  │           │
│   │   ┌──────────┐ ┌──────────┐ ┌──────────┐       │           │
│   │   │ NewsNow  │ │   RSS    │ │  Custom  │       │           │
│   │   │ Crawler  │ │ Crawler  │ │ Crawler  │       │           │
│   │   └──────────┘ └──────────┘ └──────────┘       │           │
│   └─────────────────────────────────────────────────┘           │
│                          │                                       │
│                          ▼                                       │
│                  List[NewsItem]                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## NewsNow 크롤러

NewsNow는 TrendRadar의 **주요 데이터 소스**입니다. [newsnow](https://github.com/ourongxing/newsnow) 프로젝트의 API를 사용합니다.

### 지원 카테고리

| 카테고리 | 설명 | 소스 |
|----------|------|------|
| `tech` | 기술 뉴스 | Hacker News, TechCrunch 등 |
| `finance` | 금융/경제 | Bloomberg, Reuters 등 |
| `world` | 세계 뉴스 | 주요 뉴스 미디어 |
| `china` | 중국 뉴스 | 웨이보, 즈후 등 |
| `weibo` | 웨이보 핫이슈 | 웨이보 트렌딩 |
| `zhihu` | 즈후 인기글 | 즈후 핫 토픽 |
| `bilibili` | 빌리빌리 | 빌리빌리 인기 영상 |
| `github` | GitHub 트렌딩 | GitHub Trending |

### 구현

```python
# trendradar/crawler/newsnow.py

class NewsNowCrawler(BaseCrawler):
    """NewsNow API 크롤러"""

    BASE_URL = "https://newsnow.busiyi.world/api"

    def __init__(self, context: Context, category: str):
        super().__init__(context)
        self.category = category

    async def fetch(self) -> List[NewsItem]:
        url = f"{self.BASE_URL}/{self.category}"

        async with self.context.http_client.get(url) as response:
            if response.status != 200:
                raise CrawlerError(f"NewsNow API error: {response.status}")

            data = await response.json()

        return [self._parse_item(item) for item in data.get('items', [])]

    def _parse_item(self, raw: dict) -> NewsItem:
        return NewsItem(
            id=self._generate_id(raw['url']),
            title=raw['title'],
            url=raw['url'],
            source='newsnow',
            category=self.category,
            published_at=datetime.fromisoformat(raw.get('pubDate', '')),
            content=raw.get('description'),
        )

    def _generate_id(self, url: str) -> str:
        """URL 기반 고유 ID 생성"""
        return hashlib.md5(url.encode()).hexdigest()[:12]

    def get_source_name(self) -> str:
        return f"newsnow-{self.category}"
```

---

## RSS 크롤러

커스텀 RSS 피드를 구독할 수 있습니다.

### 설정

```yaml
# config/config.yaml

sources:
  - type: rss
    enabled: true
    feeds:
      - name: "Hacker News"
        url: "https://news.ycombinator.com/rss"
        category: tech

      - name: "TechCrunch"
        url: "https://techcrunch.com/feed/"
        category: tech

      - name: "AI News"
        url: "https://www.artificialintelligence-news.com/feed/"
        category: ai
```

### 구현

```python
# trendradar/crawler/rss.py

import feedparser

class RSSCrawler(BaseCrawler):
    """RSS 피드 크롤러"""

    def __init__(self, context: Context, feed_config: dict):
        super().__init__(context)
        self.name = feed_config['name']
        self.url = feed_config['url']
        self.category = feed_config.get('category', 'general')

    async def fetch(self) -> List[NewsItem]:
        # feedparser는 동기이므로 executor 사용
        loop = asyncio.get_event_loop()
        feed = await loop.run_in_executor(
            None,
            feedparser.parse,
            self.url
        )

        if feed.bozo:
            raise CrawlerError(f"RSS parse error: {feed.bozo_exception}")

        return [self._parse_entry(entry) for entry in feed.entries[:50]]

    def _parse_entry(self, entry) -> NewsItem:
        # 발행 시간 파싱
        published = entry.get('published_parsed') or entry.get('updated_parsed')
        if published:
            published_at = datetime(*published[:6])
        else:
            published_at = datetime.now()

        return NewsItem(
            id=self._generate_id(entry.link),
            title=entry.title,
            url=entry.link,
            source=self.name,
            category=self.category,
            published_at=published_at,
            content=entry.get('summary', ''),
        )

    def get_source_name(self) -> str:
        return f"rss-{self.name}"
```

---

## 크롤러 관리자

모든 크롤러를 관리하고 병렬 실행합니다.

```python
# trendradar/crawler/manager.py

class CrawlerManager:
    """크롤러 관리자"""

    def __init__(self, context: Context):
        self.context = context
        self.crawlers: List[BaseCrawler] = []
        self._init_crawlers()

    def _init_crawlers(self):
        """설정에 따라 크롤러 초기화"""
        for source in self.context.config.sources:
            if not source.enabled:
                continue

            if source.type == 'newsnow':
                for category in source.categories:
                    self.crawlers.append(
                        NewsNowCrawler(self.context, category)
                    )

            elif source.type == 'rss':
                for feed in source.feeds:
                    self.crawlers.append(
                        RSSCrawler(self.context, feed)
                    )

    async def fetch_all(self) -> List[NewsItem]:
        """모든 크롤러에서 뉴스 수집"""
        tasks = [crawler.fetch() for crawler in self.crawlers]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_items = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.context.logger.error(
                    f"Crawler {self.crawlers[i].get_source_name()} failed: {result}"
                )
            else:
                all_items.extend(result)

        # 중복 제거 (URL 기준)
        seen_urls = set()
        unique_items = []
        for item in all_items:
            if item.url not in seen_urls:
                seen_urls.add(item.url)
                unique_items.append(item)

        # 최신순 정렬
        unique_items.sort(key=lambda x: x.published_at, reverse=True)

        return unique_items
```

---

## 필터링 & 중복 제거

### 키워드 필터링

```yaml
# config/config.yaml

filters:
  # 포함할 키워드 (하나라도 매칭 시 포함)
  include:
    - AI
    - LLM
    - Claude
    - GPT

  # 제외할 키워드
  exclude:
    - 광고
    - sponsored
```

```python
# trendradar/core/filter.py

class NewsFilter:
    def __init__(self, config: FilterConfig):
        self.include_keywords = config.include or []
        self.exclude_keywords = config.exclude or []

    def filter(self, items: List[NewsItem]) -> List[NewsItem]:
        result = []

        for item in items:
            text = f"{item.title} {item.content or ''}"

            # 제외 키워드 체크
            if any(kw.lower() in text.lower() for kw in self.exclude_keywords):
                continue

            # 포함 키워드 체크 (비어있으면 모두 포함)
            if self.include_keywords:
                if not any(kw.lower() in text.lower() for kw in self.include_keywords):
                    continue

            result.append(item)

        return result
```

### 중복 감지

```python
# trendradar/storage/dedup.py

class DedupStorage:
    """중복 감지를 위한 저장소"""

    def __init__(self, db_path: str, max_age_days: int = 7):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        self.max_age_days = max_age_days

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS seen_items (
                id TEXT PRIMARY KEY,
                url TEXT,
                seen_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def filter_new(self, items: List[NewsItem]) -> List[NewsItem]:
        """이미 본 항목 제외"""
        new_items = []

        for item in items:
            cursor = self.conn.execute(
                "SELECT 1 FROM seen_items WHERE id = ?",
                (item.id,)
            )
            if not cursor.fetchone():
                new_items.append(item)

        return new_items

    def mark_seen(self, items: List[NewsItem]):
        """항목을 본 것으로 표시"""
        for item in items:
            self.conn.execute(
                "INSERT OR IGNORE INTO seen_items (id, url) VALUES (?, ?)",
                (item.id, item.url)
            )
        self.conn.commit()

    def cleanup_old(self):
        """오래된 항목 정리"""
        self.conn.execute("""
            DELETE FROM seen_items
            WHERE seen_at < datetime('now', ?)
        """, (f'-{self.max_age_days} days',))
        self.conn.commit()
```

---

## 커스텀 크롤러 추가

새로운 데이터 소스를 추가하려면:

```python
# trendradar/crawler/custom.py

class CustomCrawler(BaseCrawler):
    """커스텀 크롤러 예시"""

    def __init__(self, context: Context, api_url: str):
        super().__init__(context)
        self.api_url = api_url

    async def fetch(self) -> List[NewsItem]:
        async with self.context.http_client.get(self.api_url) as response:
            data = await response.json()

        return [self._parse_item(item) for item in data]

    def _parse_item(self, raw: dict) -> NewsItem:
        return NewsItem(
            id=raw['id'],
            title=raw['title'],
            url=raw['link'],
            source='custom',
            category='custom',
            published_at=datetime.fromisoformat(raw['date']),
        )

    def get_source_name(self) -> str:
        return "custom"
```

---

## 레이트 리밋

API 과부하를 방지하기 위한 레이트 리밋:

```python
# trendradar/utils/ratelimit.py

class RateLimiter:
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []

    async def acquire(self):
        now = time.time()

        # 1분 이내 호출 기록만 유지
        self.calls = [t for t in self.calls if now - t < 60]

        if len(self.calls) >= self.calls_per_minute:
            # 대기 필요
            sleep_time = 60 - (now - self.calls[0])
            await asyncio.sleep(sleep_time)

        self.calls.append(time.time())
```

---

*다음 글에서는 알림 시스템을 살펴봅니다.*
