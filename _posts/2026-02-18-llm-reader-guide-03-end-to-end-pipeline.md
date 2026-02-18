---
layout: post
title: "LLM Reader 가이드 (03) - 엔드투엔드 파이프라인: URL에서 LLM 입력까지"
date: 2026-02-18
permalink: /llm-reader-guide-03-end-to-end-pipeline/
author: m92vyas
categories: ['개발 도구', '웹 스크래핑']
tags: [llm-reader, Pipeline, LLM Input, HTML, Data Extraction]
original_url: "https://github.com/m92vyas/llm-reader"
excerpt: "URL 수집부터 전처리 옵션 적용, LLM 프롬프트 결합까지 실제 파이프라인 흐름을 정리합니다."
---

## 전체 흐름

```text
URL
 -> Playwright(get_page_source)
 -> BeautifulSoup 정리(get_processed_text)
 -> 링크/이미지/표 정규화
 -> 평문(LLM 입력)
 -> 프롬프트 템플릿 결합
 -> 모델 호출
```

---

## 파이프라인 분리의 장점

- HTML 수집 방식 교체가 쉬움
- 전처리 규칙을 태스크별로 다르게 적용 가능
- 모델 교체(OpenAI/Anthropic/Groq 등)와 독립적

즉 스크래핑과 추출 프롬프트를 느슨하게 결합할 수 있습니다.

---

## 실전 호출 패턴

```python
page_source = await get_page_source(url, wait=2.0)
llm_text = await get_processed_text(
    page_source,
    url,
    keep_images=True,
    keep_webpage_links=True,
    remove_script_tag=True,
    remove_style_tag=True,
)

prompt = f"""아래 웹페이지에서 상품명/가격/링크를 JSON으로 추출해줘.\n\n{llm_text[:40000]}"""
```

핵심은 모델 컨텍스트 길이에 맞춰 잘라 넣는 규칙을 별도 함수로 관리하는 것입니다.

---

## 품질과 비용 균형

- 링크가 필요 없으면 `keep_webpage_links=False`
- 이미지 URL이 중요하지 않으면 `keep_images=False`
- 추출 모드가 단순 본문 중심이면 `extract=True` 실험

다음 장에서는 HTML을 가져오는 `get_page_source` 내부 동작을 봅니다.
