---
layout: post
title: "ScrapeGraphAI 완벽 가이드 (4) - SmartScraper 그래프"
date: 2026-02-06
permalink: /scrapegraph-guide-04-smartscraper/
author: ScrapeGraphAI Team
categories: [AI 도구, 웹 스크래핑]
tags: [ScrapeGraphAI, SmartScraper, Single Page, LLM Scraping]
original_url: "https://github.com/ScrapeGraphAI/Scrapegraph-ai"
excerpt: "가장 기본이 되는 SmartScraperGraph의 모든 기능과 활용법을 알아봅니다."
---

## SmartScraperGraph란?

**SmartScraperGraph**는 ScrapeGraphAI의 가장 기본적이고 강력한 그래프입니다. **단일 웹페이지**에서 사용자가 원하는 정보를 자연어 프롬프트만으로 추출합니다.

### 핵심 특징

- **단순성**: 프롬프트, 소스, 설정만 있으면 OK
- **유연성**: 모든 종류의 웹사이트 지원
- **정확성**: LLM이 컨텍스트를 이해하고 추출
- **구조화**: JSON 형식으로 결과 반환

## 기본 사용법

### 최소 구성 예제

```python
from scrapegraphai.graphs import SmartScraperGraph

smart_scraper = SmartScraperGraph(
    prompt="Extract all product names",
    source="https://example.com/products",
    config={
        "llm": {"model": "ollama/llama3.2"}
    }
)

result = smart_scraper.run()
print(result)
```

### 결과 예시

```json
{
    "products": [
        "iPhone 15 Pro",
        "MacBook Air M3",
        "AirPods Pro"
    ]
}
```

## 프롬프트 작성 가이드

### 좋은 프롬프트의 조건

1. **구체적**: "모든 정보"보다 "제품명, 가격, 재고 상태"
2. **구조화**: 원하는 출력 형식 명시
3. **간결함**: 불필요한 설명 제거

### 프롬프트 예시

#### ❌ 나쁜 예

```python
prompt = "이 웹사이트에서 정보를 추출해줘"
```

너무 모호하여 LLM이 무엇을 추출할지 알 수 없습니다.

#### ✅ 좋은 예

```python
prompt = """
Extract the following information:
- Product name
- Price (in USD)
- Availability (in stock / out of stock)
- Rating (1-5 stars)

Return as a list of products.
"""
```

### 복잡한 프롬프트

```python
prompt = """
From the article, extract:
1. Title
2. Author name and bio
3. Publication date (format: YYYY-MM-DD)
4. Main content (summary in 2-3 sentences)
5. Tags or categories
6. Number of comments

Return as JSON with these exact keys: title, author, date, summary, tags, comments.
"""
```

## 다양한 소스 타입

### 1. 웹 URL

```python
smart_scraper = SmartScraperGraph(
    prompt="Extract company info",
    source="https://scrapegraphai.com",
    config={"llm": {"model": "ollama/llama3.2"}}
)
```

### 2. 로컬 HTML 파일

```python
smart_scraper = SmartScraperGraph(
    prompt="Extract table data",
    source="/path/to/local/file.html",
    config={"llm": {"model": "ollama/llama3.2"}}
)
```

### 3. HTML 문자열

```python
html_content = """
<html>
    <body>
        <h1>Products</h1>
        <div class="product">
            <span class="name">Laptop</span>
            <span class="price">$999</span>
        </div>
    </body>
</html>
"""

smart_scraper = SmartScraperGraph(
    prompt="Extract product name and price",
    source=html_content,
    config={"llm": {"model": "ollama/llama3.2"}}
)
```

## 실전 활용 사례

### 사례 1: 뉴스 기사 스크래핑

```python
from scrapegraphai.graphs import SmartScraperGraph

news_scraper = SmartScraperGraph(
    prompt="""
    Extract:
    - Headline
    - Author
    - Published date
    - Article body (first 3 paragraphs)
    - Image URL (if available)
    """,
    source="https://techcrunch.com/2024/01/01/some-article",
    config={
        "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."},
        "headless": True,
    }
)

article = news_scraper.run()
```

**출력:**
```json
{
    "headline": "AI Startup Raises $100M",
    "author": "John Doe",
    "published_date": "2024-01-01",
    "article_body": "An AI startup...",
    "image_url": "https://..."
}
```

### 사례 2: 전자상거래 제품 정보

```python
product_scraper = SmartScraperGraph(
    prompt="""
    Extract product information:
    - Name
    - Brand
    - Price (current and original if discounted)
    - Rating (average score)
    - Number of reviews
    - Main features (as a list)
    - Availability
    """,
    source="https://amazon.com/product/B08XYZ",
    config={
        "llm": {"model": "ollama/llama3.1"},
        "loader_kwargs": {
            "wait_until": "networkidle",
            "timeout": 45000,
        }
    }
)

product_info = product_scraper.run()
```

### 사례 3: 채용 공고 스크래핑

```python
job_scraper = SmartScraperGraph(
    prompt="""
    Extract job posting details:
    - Job title
    - Company name
    - Location (city, state, remote option)
    - Salary range (if mentioned)
    - Required skills (as array)
    - Experience level (entry/mid/senior)
    - Application deadline
    """,
    source="https://jobs.example.com/posting/123",
    config={"llm": {"model": "ollama/mistral"}}
)

job_details = job_scraper.run()
```

### 사례 4: SNS 프로필 정보

```python
profile_scraper = SmartScraperGraph(
    prompt="""
    Extract profile information:
    - Username
    - Display name
    - Bio/description
    - Follower count
    - Following count
    - Profile image URL
    - Website link (if any)
    """,
    source="https://twitter.com/username",
    config={
        "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."},
        "headless": False,  # 동적 콘텐츠 로드 확인
    }
)

profile = profile_scraper.run()
```

## 고급 설정

### 타임아웃 및 대기 조건

```python
config = {
    "llm": {"model": "ollama/llama3.2"},
    "loader_kwargs": {
        "timeout": 60000,  # 60초 타임아웃
        "wait_until": "load",  # load, domcontentloaded, networkidle
        "wait_for_selector": ".product-list",  # 특정 요소 대기
    }
}
```

### User-Agent 커스터마이징

```python
config = {
    "llm": {"model": "ollama/llama3.2"},
    "loader_kwargs": {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
}
```

### JavaScript 실행

```python
config = {
    "llm": {"model": "ollama/llama3.2"},
    "loader_kwargs": {
        "execute_script": """
            // 더보기 버튼 클릭
            document.querySelector('.load-more').click();
        """
    }
}
```

## 에러 핸들링

### Try-Except 패턴

```python
from scrapegraphai.graphs import SmartScraperGraph

try:
    scraper = SmartScraperGraph(
        prompt="Extract data",
        source="https://example.com",
        config={"llm": {"model": "ollama/llama3.2"}}
    )
    result = scraper.run()
except TimeoutError:
    print("Page load timeout")
except ValueError as e:
    print(f"Invalid configuration: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 재시도 로직

```python
import time

def scrape_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            scraper = SmartScraperGraph(
                prompt="Extract info",
                source=url,
                config={"llm": {"model": "ollama/llama3.2"}}
            )
            return scraper.run()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # 지수 백오프
```

## 결과 후처리

### JSON 검증

```python
import json

result = scraper.run()

# 필수 필드 확인
required_fields = ["title", "price", "stock"]
for field in required_fields:
    if field not in result:
        raise ValueError(f"Missing field: {field}")

# 저장
with open("output.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
```

### 데이터 정제

```python
result = scraper.run()

# 가격 정제
if "price" in result:
    price_str = result["price"]
    result["price"] = float(price_str.replace("$", "").replace(",", ""))

# 날짜 파싱
from datetime import datetime
if "date" in result:
    result["date"] = datetime.strptime(result["date"], "%Y-%m-%d")
```

## 성능 최적화

### 1. 헤드리스 모드 사용

```python
config = {
    "llm": {"model": "ollama/llama3.2"},
    "headless": True,  # 브라우저 UI 없이 실행 (빠름)
}
```

### 2. 작은 LLM 모델 선택

```python
# 빠르지만 정확도 낮음
config = {"llm": {"model": "ollama/llama3.2"}}  # 3B 파라미터

# 느리지만 정확도 높음
config = {"llm": {"model": "openai/gpt-4o"}}
```

### 3. 캐싱 활용

```python
config = {
    "llm": {"model": "ollama/llama3.2"},
    "cache_path": "./cache",  # HTML 캐싱
}
```

## SmartScraperLiteGraph

더 빠른 처리를 위한 경량 버전도 제공됩니다:

```python
from scrapegraphai.graphs import SmartScraperLiteGraph

# 일부 기능 제한, 성능 최적화
lite_scraper = SmartScraperLiteGraph(
    prompt="Extract product names",
    source="https://example.com",
    config={"llm": {"model": "ollama/llama3.2"}}
)

result = lite_scraper.run()
```

## 다음 단계

다음 챕터에서는 여러 페이지를 동시에 스크래핑하는 **멀티페이지 그래프**를 다룹니다.

---

## 시리즈 네비게이션

- **이전**: [(3) 아키텍처 분석]({{ site.baseurl }}/scrapegraph-guide-03-architecture/)
- **현재**: (4) SmartScraper 그래프
- **다음**: [(5) 멀티페이지 스크래핑]({{ site.baseurl }}/scrapegraph-guide-05-multipage/)

[📚 전체 목차로 돌아가기]({{ site.baseurl }}/scrapegraph-guide/)
