---
layout: post
title: "ScrapeGraphAI 완벽 가이드 (7) - 고급 그래프"
date: 2026-02-06
permalink: /scrapegraph-guide-07-advanced/
author: ScrapeGraphAI Team
categories: [AI 도구, 웹 스크래핑]
tags: [ScrapeGraphAI, CodeGenerator, ScriptCreator, SpeechGraph, Advanced]
original_url: "https://github.com/ScrapeGraphAI/Scrapegraph-ai"
excerpt: "코드 생성, 스크립트 생성, 음성 변환 등 고급 기능을 탐구합니다."
---

## 고급 그래프 소개

ScrapeGraphAI는 단순 스크래핑을 넘어 **코드 자동 생성**, **스크립트 생성**, **음성 변환** 등의 고급 기능을 제공합니다.

## ScriptCreatorGraph

웹페이지를 분석하여 **스크래핑 Python 스크립트를 자동으로 생성**합니다.

### 기본 사용법

```python
from scrapegraphai.graphs import ScriptCreatorGraph

script_generator = ScriptCreatorGraph(
    prompt="Extract all product names and prices",
    source="https://example.com/products",
    config={
        "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}
    }
)

python_script = script_generator.run()
print(python_script)
```

### 생성된 스크립트 예시

```python
import requests
from bs4 import BeautifulSoup

url = "https://example.com/products"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

products = []
for item in soup.select('.product-item'):
    name = item.select_one('.product-name').text.strip()
    price = item.select_one('.product-price').text.strip()
    products.append({
        'name': name,
        'price': price
    })

print(products)
```

### 스크립트 저장 및 실행

```python
script = script_generator.run()

# 파일로 저장
with open("scraper.py", "w") as f:
    f.write(script)

# 바로 실행
exec(script)
```

### 멀티 페이지 스크립트 생성

```python
from scrapegraphai.graphs import ScriptCreatorMultiGraph

multi_script_gen = ScriptCreatorMultiGraph(
    prompt="Extract article titles and URLs",
    source=[
        "https://news1.com",
        "https://news2.com",
        "https://news3.com"
    ],
    config={
        "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}
    }
)

scripts = multi_script_gen.run()

# 각 사이트별 스크립트 생성됨
for i, script in enumerate(scripts):
    with open(f"scraper_{i+1}.py", "w") as f:
        f.write(script)
```

## CodeGeneratorGraph

**BeautifulSoup, Selenium, Playwright 등 다양한 라이브러리**를 사용하는 코드를 생성합니다.

### 사용 예제

```python
from scrapegraphai.graphs import CodeGeneratorGraph

code_gen = CodeGeneratorGraph(
    prompt="""
    Generate a Python script that:
    1. Uses Selenium to navigate to a login page
    2. Fills in username and password
    3. Clicks the login button
    4. Extracts user profile information
    5. Saves to JSON file
    """,
    source="https://example.com/login",
    config={
        "llm": {"model": "openai/gpt-4o", "api_key": "sk-..."}
    }
)

selenium_script = code_gen.run()
```

### 생성된 코드 (Selenium)

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json

# 드라이버 초기화
driver = webdriver.Chrome()

try:
    # 로그인 페이지 접속
    driver.get("https://example.com/login")

    # 로그인 정보 입력
    username_field = driver.find_element(By.ID, "username")
    password_field = driver.find_element(By.ID, "password")

    username_field.send_keys("your_username")
    password_field.send_keys("your_password")

    # 로그인 버튼 클릭
    login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
    login_button.click()

    # 프로필 페이지 대기
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "profile"))
    )

    # 프로필 정보 추출
    profile_data = {
        "name": driver.find_element(By.CLASS_NAME, "user-name").text,
        "email": driver.find_element(By.CLASS_NAME, "user-email").text,
    }

    # JSON 저장
    with open("profile.json", "w") as f:
        json.dump(profile_data, f, indent=2)

finally:
    driver.quit()
```

## SpeechGraph

스크래핑한 콘텐츠를 **음성 파일(MP3)**로 변환합니다.

### 기본 사용법

```python
from scrapegraphai.graphs import SpeechGraph

speech_gen = SpeechGraph(
    prompt="Summarize this article in 2-3 sentences",
    source="https://blog.example.com/article",
    config={
        "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."},
        "tts_model": {
            "provider": "openai",
            "model": "tts-1",
            "voice": "alloy"
        }
    }
)

result = speech_gen.run()
```

### 결과 구조

```python
{
    "summary": "This article discusses...",
    "audio_file": "/tmp/speech_output.mp3"
}
```

### 음성 옵션 설정

```python
config = {
    "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."},
    "tts_model": {
        "provider": "openai",
        "model": "tts-1-hd",  # 고품질 음성
        "voice": "nova",       # alloy, echo, fable, onyx, nova, shimmer
        "speed": 1.0           # 속도 (0.25 ~ 4.0)
    },
    "output_path": "./audio/summary.mp3"
}
```

### 실용 사례: 뉴스 팟캐스트

```python
from scrapegraphai.graphs import SpeechGraph

news_urls = [
    "https://news.com/tech-news-1",
    "https://news.com/tech-news-2",
    "https://news.com/tech-news-3"
]

for i, url in enumerate(news_urls):
    speech_gen = SpeechGraph(
        prompt="Create a 30-second news summary",
        source=url,
        config={
            "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."},
            "tts_model": {
                "provider": "openai",
                "model": "tts-1",
                "voice": "alloy"
            },
            "output_path": f"./podcast/news_{i+1}.mp3"
        }
    )
    result = speech_gen.run()
    print(f"Generated: news_{i+1}.mp3")
```

## ScreenshotScraperGraph

웹페이지의 **스크린샷을 캡처**하고 분석합니다.

### 기본 사용

```python
from scrapegraphai.graphs import ScreenshotScraperGraph

screenshot_scraper = ScreenshotScraperGraph(
    prompt="Describe the layout and main elements of this page",
    source="https://example.com",
    config={
        "llm": {"model": "openai/gpt-4o", "api_key": "sk-..."},
        "screenshot_path": "./screenshots/example.png"
    }
)

analysis = screenshot_scraper.run()
```

### Vision 모델 활용

```python
config = {
    "llm": {
        "model": "openai/gpt-4o",  # Vision 지원 모델
        "api_key": "sk-..."
    },
    "screenshot_options": {
        "full_page": True,        # 전체 페이지 캡처
        "width": 1920,
        "height": 1080
    }
}
```

## 커스텀 그래프 생성

직접 그래프를 만들어 워크플로우를 정의할 수 있습니다.

### 기본 구조

```python
from scrapegraphai.graphs import BaseGraph
from scrapegraphai.nodes import FetchNode, ParseNode, RAGNode

class CustomNewsGraph(BaseGraph):
    def __init__(self, prompt, source, config):
        super().__init__(prompt, source, config)

    def _create_graph(self):
        """그래프 구조 정의"""
        # 노드 생성
        fetch_node = FetchNode(
            "fetch",
            input="url",
            output=["html"],
            config=self.config
        )

        parse_node = ParseNode(
            "parse",
            input="html",
            output=["cleaned_html"],
            config=self.config
        )

        rag_node = RAGNode(
            "rag",
            input="cleaned_html | prompt",
            output=["answer"],
            config=self.config
        )

        # 노드 연결
        return [fetch_node, parse_node, rag_node]
```

### 사용

```python
custom_scraper = CustomNewsGraph(
    prompt="Extract news headlines",
    source="https://news.example.com",
    config={"llm": {"model": "ollama/llama3.2"}}
)

result = custom_scraper.run()
```

## 실전 활용 사례

### 사례 1: 자동화된 스크래퍼 생성

```python
from scrapegraphai.graphs import ScriptCreatorGraph

# 고객사 웹사이트 분석 후 맞춤 스크래퍼 생성
websites = [
    "https://client1.com/products",
    "https://client2.com/catalog",
    "https://client3.com/items"
]

for i, site in enumerate(websites):
    script_gen = ScriptCreatorGraph(
        prompt="Extract product name, price, and stock status",
        source=site,
        config={
            "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}
        }
    )

    script = script_gen.run()

    # 고객별 스크래퍼 제공
    with open(f"client_{i+1}_scraper.py", "w") as f:
        f.write(script)
```

### 사례 2: 오디오 뉴스 브리핑

```python
from scrapegraphai.graphs import SearchGraph, SpeechGraph

# 1단계: 최신 뉴스 검색 및 요약
search_scraper = SearchGraph(
    prompt="What are the top 3 tech news today?",
    config={
        "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."},
        "max_results": 5
    }
)

news_summary = search_scraper.run()

# 2단계: 음성으로 변환
speech_gen = SpeechGraph(
    prompt="Create a 2-minute audio briefing",
    source=news_summary,
    config={
        "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."},
        "tts_model": {
            "provider": "openai",
            "model": "tts-1",
            "voice": "nova"
        },
        "output_path": "./daily_briefing.mp3"
    }
)

audio_briefing = speech_gen.run()
print(f"Audio saved to: {audio_briefing['audio_file']}")
```

### 사례 3: 경쟁사 모니터링 자동화

```python
from scrapegraphai.graphs import ScriptCreatorMultiGraph
import schedule
import time

# 경쟁사 웹사이트 모니터링 스크립트 생성
competitors = [
    "https://competitor1.com/pricing",
    "https://competitor2.com/features",
    "https://competitor3.com/updates"
]

script_gen = ScriptCreatorMultiGraph(
    prompt="Extract pricing, new features, and recent updates",
    source=competitors,
    config={
        "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}
    }
)

monitoring_scripts = script_gen.run()

# 생성된 스크립트를 정기 실행
def run_monitoring():
    for i, script in enumerate(monitoring_scripts):
        exec(script)

# 매일 오전 9시 실행
schedule.every().day.at("09:00").do(run_monitoring)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## 성능 및 비용 고려사항

### Vision 모델 비용

```python
# 스크린샷 분석은 Vision 모델 필요 (비용 높음)
config = {
    "llm": {
        "model": "openai/gpt-4o",  # $2.50 / 1M tokens (input)
        "api_key": "sk-..."
    }
}

# 텍스트 추출 후 분석 (비용 절감)
config = {
    "llm": {
        "model": "openai/gpt-4o-mini",  # $0.15 / 1M tokens
        "api_key": "sk-..."
    }
}
```

### TTS 비용

```python
# OpenAI TTS 가격: $15 / 1M characters
# 대안: 로컬 TTS 라이브러리 (무료)
```

## 다음 단계

다음 챕터에서는 **LLM 모델 연동**을 심층적으로 다룹니다.

---

## 시리즈 네비게이션

- **이전**: [(6) 다양한 데이터 포맷]({{ site.baseurl }}/scrapegraph-guide-06-formats/)
- **현재**: (7) 고급 그래프
- **다음**: [(8) LLM 모델 연동]({{ site.baseurl }}/scrapegraph-guide-08-llm-integration/)

[📚 전체 목차로 돌아가기]({{ site.baseurl }}/scrapegraph-guide/)
