---
layout: page
title: LLM Reader 가이드
permalink: /llm-reader-guide/
icon: fas fa-globe
---

# LLM Reader 완벽 가이드

> **웹페이지 HTML을 LLM 입력에 맞게 정규화하는 Python 전처리 도구 분석**

**LLM Reader (`m92vyas/llm-reader`)**는 웹페이지 HTML을 수집한 뒤, 이미지/링크/표를 LLM 추출 작업에 맞게 재구성해 텍스트 입력으로 변환하는 경량 오픈소스 라이브러리입니다.

- 원문 저장소: https://github.com/m92vyas/llm-reader
- 관련 문서: https://github.com/m92vyas/llm-reader/wiki/Documentation

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개/포지셔닝](/blog-repo/llm-reader-guide-01-intro/) | 어떤 문제를 해결하는지, Firecrawl/Reader API 대안 관점 |
| 02 | [설치/빠른 시작](/blog-repo/llm-reader-guide-02-installation-and-quickstart/) | Python/Playwright 설치와 첫 실행 |
| 03 | [엔드투엔드 파이프라인](/blog-repo/llm-reader-guide-03-end-to-end-pipeline/) | URL -> HTML -> LLM 텍스트 흐름 |
| 04 | [Playwright 수집기](/blog-repo/llm-reader-guide-04-playwright-html-fetcher/) | `get_page_source` 내부 동작과 수집 안정성 |
| 05 | [HTML 정리/태그 제어](/blog-repo/llm-reader-guide-05-html-cleaning-and-tag-control/) | script/style 제거와 커스텀 태그 전략 |
| 06 | [이미지/링크 정규화](/blog-repo/llm-reader-guide-06-image-link-normalization/) | URL 절대경로화와 토큰 절약 옵션 |
| 07 | [표 변환 로직](/blog-repo/llm-reader-guide-07-table-markdown-conversion/) | HTML table -> Markdown 변환 경로 |
| 08 | [래퍼 함수/동시성 패턴](/blog-repo/llm-reader-guide-08-wrapper-and-concurrency-patterns/) | `url_to_llm_text` 사용법과 배치 처리 |
| 09 | [LLM 추출 연동](/blog-repo/llm-reader-guide-09-llm-extraction-integration/) | 프롬프트 설계, 길이 제한, 비용 최적화 |
| 10 | [제한사항/트러블슈팅](/blog-repo/llm-reader-guide-10-limitations-and-troubleshooting/) | 운영 시 주의점, 확장 포인트 |

---

## 핵심 특징

- **HTML 수집 + 전처리 분리**: `get_page_source`와 `get_processed_text`를 분리해 구성 가능
- **링크/이미지 보존 제어**: 작업 목적에 맞게 토큰 사용량을 직접 조정
- **표 구조 보존 시도**: 표를 Markdown 형태로 바꿔 LLM 추출 안정성 향상
- **비동기 호출 친화적**: `async/await` 기반으로 동시 URL 처리에 적합
- **오픈소스 기반 비용 절감**: HTML 소스만 확보하면 후처리는 로컬에서 무료 수행

---

## 빠른 시작

```bash
pip install git+https://github.com/m92vyas/llm-reader.git
playwright install
playwright install-deps
```

```python
from url_to_llm_text.get_html_text import get_page_source
from url_to_llm_text.get_llm_input_text import get_processed_text

url = "https://example.com"
page_source = await get_page_source(url)
llm_text = await get_processed_text(page_source, url)
```

다음 글부터 내부 모듈을 코드 기준으로 하나씩 해설합니다.
