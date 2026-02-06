---
layout: post
title: "ScrapeGraphAI 완벽 가이드 (6) - 다양한 데이터 포맷"
date: 2026-02-06
permalink: /scrapegraph-guide-06-formats/
author: ScrapeGraphAI Team
categories: [AI 도구, 웹 스크래핑]
tags: [ScrapeGraphAI, JSON, CSV, XML, PDF, Document]
original_url: "https://github.com/ScrapeGraphAI/Scrapegraph-ai"
excerpt: "JSON, CSV, XML, PDF 등 다양한 형식의 데이터를 스크래핑하는 방법을 배웁니다."
---

## 지원 데이터 포맷

ScrapeGraphAI는 웹페이지(HTML)뿐만 아니라 다양한 문서 형식을 지원합니다:

- **JSON**: API 응답, 설정 파일
- **CSV**: 표 형식 데이터
- **XML**: RSS 피드, 구조화된 문서
- **PDF**: 보고서, 논문
- **Markdown**: 문서, README 파일
- **Office 문서**: DOCX, XLSX (DocumentScraper 사용)

## JSONScraperGraph

### 기본 사용법

```python
from scrapegraphai.graphs import JSONScraperGraph

json_scraper = JSONScraperGraph(
    prompt="Extract all user names and their email addresses",
    source="https://api.example.com/users",
    config={
        "llm": {"model": "ollama/llama3.2"}
    }
)

result = json_scraper.run()
```

### 로컬 JSON 파일

```python
json_scraper = JSONScraperGraph(
    prompt="Extract product categories and their counts",
    source="/path/to/data.json",
    config={
        "llm": {"model": "ollama/llama3.2"}
    }
)

categories = json_scraper.run()
```

### JSON 문자열 직접 파싱

```python
json_data = '''
{
    "users": [
        {"name": "Alice", "age": 30, "city": "Seoul"},
        {"name": "Bob", "age": 25, "city": "Busan"}
    ]
}
'''

json_scraper = JSONScraperGraph(
    prompt="Create a summary of users by city",
    source=json_data,
    config={
        "llm": {"model": "ollama/llama3.2"}
    }
)

summary = json_scraper.run()
```

### 복잡한 JSON 구조

```python
json_scraper = JSONScraperGraph(
    prompt="""
    From the API response:
    1. Extract all products with price > $100
    2. Group by category
    3. Calculate average price per category
    4. Return as structured JSON
    """,
    source="https://api.store.com/products?limit=1000",
    config={
        "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}
    }
)

analysis = json_scraper.run()
```

## CSVScraperGraph

### 기본 CSV 스크래핑

```python
from scrapegraphai.graphs import CSVScraperGraph

csv_scraper = CSVScraperGraph(
    prompt="Find all employees in the Engineering department with salary > 100000",
    source="/path/to/employees.csv",
    config={
        "llm": {"model": "ollama/llama3.2"}
    }
)

engineers = csv_scraper.run()
```

### CSV URL 스크래핑

```python
csv_scraper = CSVScraperGraph(
    prompt="Calculate the average stock price and identify the highest/lowest days",
    source="https://data.example.com/stock_prices.csv",
    config={
        "llm": {"model": "ollama/mistral"}
    }
)

stock_analysis = csv_scraper.run()
```

### 멀티 CSV 처리

```python
from scrapegraphai.graphs import CSVScraperMultiGraph

csv_files = [
    "/data/sales_2023_q1.csv",
    "/data/sales_2023_q2.csv",
    "/data/sales_2023_q3.csv",
    "/data/sales_2023_q4.csv"
]

multi_csv_scraper = CSVScraperMultiGraph(
    prompt="Calculate total sales and identify top 5 products across all quarters",
    source=csv_files,
    config={
        "llm": {"model": "ollama/llama3.1"}
    }
)

yearly_report = multi_csv_scraper.run()
```

## XMLScraperGraph

### RSS 피드 스크래핑

```python
from scrapegraphai.graphs import XMLScraperGraph

rss_scraper = XMLScraperGraph(
    prompt="Extract all article titles, publication dates, and URLs from the RSS feed",
    source="https://blog.example.com/rss.xml",
    config={
        "llm": {"model": "ollama/llama3.2"}
    }
)

articles = rss_scraper.run()
```

### 구조화된 XML 데이터

```python
xml_content = '''
<catalog>
    <book category="fiction">
        <title>The Great Gatsby</title>
        <author>F. Scott Fitzgerald</author>
        <price>10.99</price>
    </book>
    <book category="non-fiction">
        <title>Sapiens</title>
        <author>Yuval Noah Harari</author>
        <price>15.99</price>
    </book>
</catalog>
'''

xml_scraper = XMLScraperGraph(
    prompt="Extract all fiction books with their authors and prices",
    source=xml_content,
    config={
        "llm": {"model": "ollama/llama3.2"}
    }
)

fiction_books = xml_scraper.run()
```

### Sitemap 분석

```python
sitemap_scraper = XMLScraperGraph(
    prompt="Extract all page URLs and their last modification dates",
    source="https://example.com/sitemap.xml",
    config={
        "llm": {"model": "ollama/llama3.2"}
    }
)

pages = sitemap_scraper.run()
```

## DocumentScraperGraph

### PDF 스크래핑

```python
from scrapegraphai.graphs import DocumentScraperGraph

pdf_scraper = DocumentScraperGraph(
    prompt="""
    From the research paper:
    - Extract the title
    - List all authors
    - Summarize the abstract
    - Extract the main findings
    """,
    source="/path/to/research_paper.pdf",
    config={
        "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}
    }
)

paper_summary = pdf_scraper.run()
```

### DOCX 문서

```python
docx_scraper = DocumentScraperGraph(
    prompt="Extract all section headings and create a table of contents",
    source="/path/to/report.docx",
    config={
        "llm": {"model": "ollama/llama3.1"}
    }
)

toc = docx_scraper.run()
```

### Markdown 문서

```python
md_scraper = DocumentScraperGraph(
    prompt="Extract all code blocks with their language tags and create a summary",
    source="/path/to/tutorial.md",
    config={
        "llm": {"model": "ollama/llama3.2"}
    }
)

code_summary = md_scraper.run()
```

### 멀티 문서 처리

```python
from scrapegraphai.graphs import DocumentScraperMultiGraph

documents = [
    "/docs/chapter1.pdf",
    "/docs/chapter2.pdf",
    "/docs/chapter3.pdf"
]

multi_doc_scraper = DocumentScraperMultiGraph(
    prompt="Create a comprehensive summary of all chapters",
    source=documents,
    config={
        "llm": {"model": "openai/gpt-4o", "api_key": "sk-..."}
    }
)

book_summary = multi_doc_scraper.run()
```

## 실전 활용 사례

### 사례 1: API 데이터 통합

```python
from scrapegraphai.graphs import JSONScraperGraph

# GitHub API에서 인기 레포지토리 분석
github_scraper = JSONScraperGraph(
    prompt="""
    Extract:
    - Repository names
    - Star counts
    - Main programming language
    - Last update date

    Sort by star count (highest first)
    """,
    source="https://api.github.com/search/repositories?q=language:python&sort=stars",
    config={
        "llm": {"model": "ollama/llama3.2"}
    }
)

trending_repos = github_scraper.run()
```

### 사례 2: 재무 데이터 분석

```python
from scrapegraphai.graphs import CSVScraperGraph

financial_scraper = CSVScraperGraph(
    prompt="""
    From the financial data:
    1. Calculate quarterly revenue growth
    2. Identify the most profitable product lines
    3. Find any unusual spending patterns
    4. Provide year-over-year comparison
    """,
    source="/data/financials_2024.csv",
    config={
        "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}
    }
)

financial_insights = financial_scraper.run()
```

### 사례 3: 뉴스 피드 집계

```python
from scrapegraphai.graphs import XMLScraperMultiGraph

rss_feeds = [
    "https://news-site-1.com/rss",
    "https://news-site-2.com/rss",
    "https://news-site-3.com/rss"
]

news_aggregator = XMLScraperMultiGraph(
    prompt="""
    Collect all news articles from the past 24 hours about AI.
    For each article provide:
    - Title
    - Source
    - Summary (1 sentence)
    - URL
    """,
    source=rss_feeds,
    config={
        "llm": {"model": "ollama/mistral"}
    }
)

ai_news = news_aggregator.run()
```

### 사례 4: 연구 논문 리뷰

```python
from scrapegraphai.graphs import DocumentScraperMultiGraph
import glob

# 디렉토리의 모든 PDF 논문 분석
papers = glob.glob("/research/papers/*.pdf")

research_scraper = DocumentScraperMultiGraph(
    prompt="""
    For each paper:
    - Title
    - Authors
    - Publication year
    - Main contribution (2 sentences)
    - Methodology used
    - Key findings
    """,
    source=papers,
    config={
        "llm": {"model": "openai/gpt-4o", "api_key": "sk-..."}
    }
)

literature_review = research_scraper.run()

# CSV로 저장
import pandas as pd
df = pd.DataFrame(literature_review)
df.to_csv("literature_review.csv", index=False)
```

## OmniScraperGraph

**모든 형식**을 자동으로 감지하여 처리하는 범용 스크래퍼:

```python
from scrapegraphai.graphs import OmniScraperGraph

omni_scraper = OmniScraperGraph(
    prompt="Extract key information",
    source=[
        "https://example.com/page.html",    # HTML
        "/data/report.pdf",                  # PDF
        "https://api.example.com/data",      # JSON
        "/data/stats.csv"                    # CSV
    ],
    config={
        "llm": {"model": "openai/gpt-4o-mini", "api_key": "sk-..."}
    }
)

all_data = omni_scraper.run()
```

## 데이터 변환 파이프라인

### JSON → CSV

```python
from scrapegraphai.graphs import JSONScraperGraph
import pandas as pd

# JSON 데이터 스크래핑
json_scraper = JSONScraperGraph(
    prompt="Extract all user records with name, email, and signup_date",
    source="https://api.example.com/users",
    config={"llm": {"model": "ollama/llama3.2"}}
)

users = json_scraper.run()

# CSV로 변환
df = pd.DataFrame(users["users"])
df.to_csv("users.csv", index=False)
```

### CSV → JSON

```python
from scrapegraphai.graphs import CSVScraperGraph
import json

csv_scraper = CSVScraperGraph(
    prompt="Convert all rows to JSON format with proper data types",
    source="/data/products.csv",
    config={"llm": {"model": "ollama/llama3.2"}}
)

products = csv_scraper.run()

with open("products.json", "w") as f:
    json.dump(products, f, indent=2)
```

## 성능 고려사항

### 대용량 파일 처리

```python
# 큰 CSV 파일은 청크로 나누기
config = {
    "llm": {"model": "ollama/llama3.2"},
    "chunk_size": 1000,  # 1000 rows at a time
}
```

### PDF OCR 활성화

```python
# 이미지 기반 PDF 처리
config = {
    "llm": {"model": "ollama/llama3.2"},
    "use_ocr": True,  # OCR 의존성 필요
}
```

## 다음 단계

다음 챕터에서는 **고급 그래프**(CodeGenerator, ScriptCreator, SpeechGraph)를 다룹니다.

---

## 시리즈 네비게이션

- **이전**: [(5) 멀티페이지 스크래핑]({{ site.baseurl }}/scrapegraph-guide-05-multipage/)
- **현재**: (6) 다양한 데이터 포맷
- **다음**: [(7) 고급 그래프]({{ site.baseurl }}/scrapegraph-guide-07-advanced/)

[📚 전체 목차로 돌아가기]({{ site.baseurl }}/scrapegraph-guide/)
