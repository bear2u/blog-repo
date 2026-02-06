---
layout: post
title: "ScrapeGraphAI 완벽 가이드 (10) - 실전 활용 및 팁"
date: 2026-02-06
permalink: /scrapegraph-guide-10-tips/
author: ScrapeGraphAI Team
categories: [AI 도구, 웹 스크래핑]
tags: [ScrapeGraphAI, Best Practices, Tips, Optimization, Troubleshooting]
original_url: "https://github.com/ScrapeGraphAI/Scrapegraph-ai"
excerpt: "프로덕션 환경에서 ScrapeGraphAI를 효과적으로 사용하기 위한 팁과 노하우를 공유합니다."
---

## 프로덕션 체크리스트

### 1. 에러 핸들링

```python
from scrapegraphai.graphs import SmartScraperGraph
import logging

logging.basicConfig(level=logging.ERROR, filename='scraping_errors.log')

def safe_scrape(url, prompt, config, max_retries=3):
    """안전한 스크래핑 래퍼"""
    for attempt in range(max_retries):
        try:
            scraper = SmartScraperGraph(
                prompt=prompt,
                source=url,
                config=config
            )
            result = scraper.run()

            # 결과 검증
            if not result or len(result) == 0:
                raise ValueError("Empty result")

            return result

        except TimeoutError:
            logging.error(f"Timeout on {url}, attempt {attempt + 1}")
            if attempt == max_retries - 1:
                return {"error": "timeout", "url": url}

        except Exception as e:
            logging.error(f"Error on {url}: {e}")
            if attempt == max_retries - 1:
                return {"error": str(e), "url": url}

        time.sleep(2 ** attempt)  # 지수 백오프

    return None
```

### 2. 프록시 로테이션

```python
import random

PROXY_LIST = [
    "http://proxy1.com:8080",
    "http://proxy2.com:8080",
    "http://proxy3.com:8080",
    "http://proxy4.com:8080"
]

def get_random_proxy():
    return random.choice(PROXY_LIST)

def scrape_with_proxy(url, prompt):
    config = {
        "llm": {"model": "ollama/llama3.2"},
        "proxy": {"server": get_random_proxy()}
    }

    scraper = SmartScraperGraph(
        prompt=prompt,
        source=url,
        config=config
    )

    return scraper.run()
```

### 3. User-Agent 로테이션

```python
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
]

def scrape_with_random_ua(url, prompt):
    config = {
        "llm": {"model": "ollama/llama3.2"},
        "loader_kwargs": {
            "user_agent": random.choice(USER_AGENTS)
        }
    }

    scraper = SmartScraperGraph(
        prompt=prompt,
        source=url,
        config=config
    )

    return scraper.run()
```

## 성능 최적화

### 1. 병렬 처리

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from scrapegraphai.graphs import SmartScraperGraph

urls = [f"https://example.com/page/{i}" for i in range(1, 101)]

def scrape_url(url):
    scraper = SmartScraperGraph(
        prompt="Extract data",
        source=url,
        config={"llm": {"model": "ollama/llama3.2"}}
    )
    return scraper.run()

# 10개 동시 실행
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(scrape_url, url): url for url in urls}

    results = []
    for future in as_completed(futures):
        url = futures[future]
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            print(f"Error on {url}: {e}")
```

### 2. 캐싱 전략

```python
import hashlib
import json
import os

CACHE_DIR = "./cache"

def get_cache_key(url, prompt):
    """URL과 프롬프트로 캐시 키 생성"""
    data = f"{url}:{prompt}"
    return hashlib.md5(data.encode()).hexdigest()

def get_cached_result(url, prompt):
    """캐시된 결과 가져오기"""
    cache_key = get_cache_key(url, prompt)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")

    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None

def save_to_cache(url, prompt, result):
    """결과를 캐시에 저장"""
    cache_key = get_cache_key(url, prompt)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(result, f)

def smart_scrape(url, prompt, config, use_cache=True):
    """캐시를 활용한 스크래핑"""
    if use_cache:
        cached = get_cached_result(url, prompt)
        if cached:
            print(f"Cache hit for {url}")
            return cached

    # 캐시 미스: 실제 스크래핑
    scraper = SmartScraperGraph(
        prompt=prompt,
        source=url,
        config=config
    )
    result = scraper.run()

    if use_cache:
        save_to_cache(url, prompt, result)

    return result
```

### 3. 배치 처리

```python
def batch_scrape(urls, prompt, batch_size=10):
    """배치 단위로 스크래핑"""
    results = []

    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{len(urls)//batch_size + 1}")

        batch_results = []
        for url in batch:
            try:
                result = smart_scrape(url, prompt, config)
                batch_results.append(result)
            except Exception as e:
                print(f"Error: {e}")
                batch_results.append({"error": str(e)})

        results.extend(batch_results)

        # 배치 간 대기 (Rate limit 방지)
        time.sleep(2)

    return results
```

## 모니터링 및 로깅

### 1. 구조화된 로깅

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, log_file="scraping.log"):
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_scrape(self, url, prompt, result, duration):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "url": url,
            "prompt": prompt,
            "success": "error" not in result,
            "duration_seconds": duration,
            "result_size": len(json.dumps(result))
        }
        self.logger.info(json.dumps(log_entry))

# 사용
logger = StructuredLogger()

start = time.time()
result = smart_scrape(url, prompt, config)
duration = time.time() - start

logger.log_scrape(url, prompt, result, duration)
```

### 2. 메트릭 수집

```python
from collections import defaultdict

class ScrapeMetrics:
    def __init__(self):
        self.metrics = defaultdict(int)
        self.durations = []

    def record_success(self, duration):
        self.metrics["success"] += 1
        self.durations.append(duration)

    def record_failure(self, error_type):
        self.metrics[f"error_{error_type}"] += 1

    def get_summary(self):
        total = sum(self.metrics.values())
        success_rate = (self.metrics["success"] / total * 100) if total > 0 else 0
        avg_duration = sum(self.durations) / len(self.durations) if self.durations else 0

        return {
            "total_requests": total,
            "success_rate": f"{success_rate:.2f}%",
            "avg_duration": f"{avg_duration:.2f}s",
            "errors": {k: v for k, v in self.metrics.items() if k.startswith("error_")}
        }

# 사용
metrics = ScrapeMetrics()

for url in urls:
    start = time.time()
    try:
        result = smart_scrape(url, prompt, config)
        duration = time.time() - start
        metrics.record_success(duration)
    except TimeoutError:
        metrics.record_failure("timeout")
    except Exception as e:
        metrics.record_failure("other")

print(metrics.get_summary())
```

## 프롬프트 엔지니어링 팁

### 1. 구체적인 출력 구조 지정

```python
prompt = """
Extract product information in the following JSON format:
{
    "name": "product name",
    "price": {
        "amount": 99.99,
        "currency": "USD"
    },
    "rating": {
        "average": 4.5,
        "count": 120
    },
    "availability": "in_stock" or "out_of_stock",
    "features": ["feature1", "feature2"]
}

Return ONLY valid JSON, no additional text.
"""
```

### 2. Few-Shot 예제 제공

```python
prompt = """
Extract article metadata.

Example 1:
Input: <article><h1>AI News</h1><p>By John Doe</p></article>
Output: {"title": "AI News", "author": "John Doe"}

Example 2:
Input: <article><h1>Tech Update</h1><p>By Jane Smith</p></article>
Output: {"title": "Tech Update", "author": "Jane Smith"}

Now extract from the following HTML:
"""
```

### 3. 조건부 추출

```python
prompt = """
Extract product information:
- If price includes discount, extract both original and discounted price
- If product is unavailable, set availability to "out_of_stock"
- If rating is not shown, omit the rating field
- Extract up to 5 main features
"""
```

## 데이터 품질 검증

### 1. 스키마 검증

```python
from jsonschema import validate, ValidationError

PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "number"},
        "stock": {"type": "string", "enum": ["in_stock", "out_of_stock"]}
    },
    "required": ["name", "price"]
}

def validate_result(result):
    try:
        validate(instance=result, schema=PRODUCT_SCHEMA)
        return True
    except ValidationError as e:
        print(f"Validation error: {e.message}")
        return False

# 사용
result = smart_scrape(url, prompt, config)
if validate_result(result):
    print("Valid data")
else:
    print("Invalid data, re-scraping...")
```

### 2. 필드 검증

```python
def validate_product_data(data):
    """제품 데이터 검증"""
    errors = []

    # 필수 필드 확인
    if "name" not in data or not data["name"]:
        errors.append("Missing or empty name")

    # 가격 범위 확인
    if "price" in data:
        try:
            price = float(data["price"])
            if price < 0 or price > 100000:
                errors.append(f"Unrealistic price: ${price}")
        except ValueError:
            errors.append("Invalid price format")

    # 이메일 형식 확인
    if "email" in data:
        if "@" not in data["email"]:
            errors.append("Invalid email format")

    return len(errors) == 0, errors

# 사용
result = smart_scrape(url, prompt, config)
valid, errors = validate_product_data(result)

if not valid:
    print(f"Validation errors: {errors}")
```

## 비용 최적화

### 1. 적응형 모델 선택

```python
def adaptive_scrape(url, prompt, complexity="auto"):
    """페이지 복잡도에 따라 모델 선택"""

    if complexity == "auto":
        # 간단한 휴리스틱으로 복잡도 판단
        if len(prompt) < 50 and "simple" in url:
            complexity = "simple"
        else:
            complexity = "complex"

    if complexity == "simple":
        # 저렴한 모델
        config = {"llm": {"model": "ollama/llama3.2"}}
    else:
        # 고성능 모델
        config = {"llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}}

    scraper = SmartScraperGraph(
        prompt=prompt,
        source=url,
        config=config
    )

    return scraper.run()
```

### 2. 토큰 사용량 모니터링

```python
from tiktoken import encoding_for_model

def estimate_cost(prompt, result, model="gpt-4o-mini"):
    """비용 추정"""
    enc = encoding_for_model(model)

    prompt_tokens = len(enc.encode(prompt))
    output_tokens = len(enc.encode(json.dumps(result)))

    # GPT-4o-mini 가격
    input_cost = prompt_tokens / 1_000_000 * 0.15
    output_cost = output_tokens / 1_000_000 * 0.60

    total_cost = input_cost + output_cost

    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_cost": f"${total_cost:.6f}"
    }
```

## 일반적인 문제 해결

### 1. JavaScript 렌더링 문제

```python
config = {
    "llm": {"model": "ollama/llama3.2"},
    "loader_kwargs": {
        "wait_until": "networkidle",  # 네트워크 유휴 대기
        "timeout": 60000,              # 충분한 타임아웃
        "wait_for_selector": ".content"  # 특정 요소 대기
    }
}
```

### 2. CAPTCHA 우회

```python
# Undetected Playwright 사용
config = {
    "llm": {"model": "ollama/llama3.2"},
    "use_undetected_playwright": True,  # Anti-bot 우회
}
```

### 3. Rate Limit 처리

```python
import time
from ratelimit import limits, sleep_and_retry

CALLS = 10
PERIOD = 60  # seconds

@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
def rate_limited_scrape(url, prompt, config):
    scraper = SmartScraperGraph(
        prompt=prompt,
        source=url,
        config=config
    )
    return scraper.run()
```

## 보안 고려사항

### 1. API 키 보호

```python
# ❌ 나쁜 예: 코드에 하드코딩
config = {"llm": {"api_key": "sk-proj-abc123..."}}

# ✅ 좋은 예: 환경 변수 사용
import os
config = {"llm": {"api_key": os.getenv("OPENAI_API_KEY")}}
```

### 2. 민감 데이터 마스킹

```python
import re

def mask_sensitive_data(result):
    """민감 데이터 마스킹"""
    result_str = json.dumps(result)

    # 이메일 마스킹
    result_str = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        '***@***.***',
        result_str
    )

    # 전화번호 마스킹
    result_str = re.sub(
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        '***-***-****',
        result_str
    )

    return json.loads(result_str)
```

## 결론

ScrapeGraphAI는 강력하고 유연한 웹 스크래핑 도구입니다. 이 가이드에서 다룬 내용을 요약하면:

1. **LLM 기반**: 자연어 프롬프트로 스크래핑
2. **다양한 그래프**: 단일/멀티 페이지, 검색, 문서 등
3. **멀티 포맷**: JSON, CSV, XML, PDF 지원
4. **고급 기능**: 코드 생성, 스크립트 생성, 음성 변환
5. **폭넓은 통합**: API, SDK, Langchain, n8n, Zapier

### 추가 리소스

- [공식 문서](https://scrapegraph-ai.readthedocs.io/)
- [GitHub 저장소](https://github.com/ScrapeGraphAI/Scrapegraph-ai)
- [Discord 커뮤니티](https://discord.gg/gkxQDAjfeX)
- [API 대시보드](https://dashboard.scrapegraphai.com/)

---

## 시리즈 네비게이션

- **이전**: [(9) 통합 및 확장]({{ site.baseurl }}/scrapegraph-guide-09-integrations/)
- **현재**: (10) 실전 활용 및 팁

[📚 전체 목차로 돌아가기]({{ site.baseurl }}/scrapegraph-guide/)

---

이것으로 **ScrapeGraphAI 완벽 가이드** 시리즈를 마칩니다. 여러분의 프로젝트에 ScrapeGraphAI를 성공적으로 적용하시길 바랍니다!
