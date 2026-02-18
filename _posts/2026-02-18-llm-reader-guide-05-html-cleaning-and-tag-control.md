---
layout: post
title: "LLM Reader 가이드 (05) - HTML 정리/태그 제어: 노이즈를 줄여 추출 정확도 높이기"
date: 2026-02-18
permalink: /llm-reader-guide-05-html-cleaning-and-tag-control/
author: m92vyas
categories: ['개발 도구', '웹 스크래핑']
tags: [BeautifulSoup, HTML Cleaning, llm-reader, Parser, Extraction]
original_url: "https://github.com/m92vyas/llm-reader"
excerpt: "get_processed_text의 태그 제거 로직과 파서/옵션 조합으로 전처리 품질을 조정하는 방법을 설명합니다."
---

## 처리 진입점

핵심 함수는 `get_processed_text(page_source, base_url, ...)`입니다.

초기 단계에서 BeautifulSoup 파서를 적용하고, 실패 시 예외를 흡수하도록 되어 있습니다.

---

## 태그 제거 로직

다음 옵션이 기본으로 켜져 있습니다.

- `remove_script_tag=True`
- `remove_style_tag=True`

또한 `remove_tags` 리스트를 추가로 받아 커스텀 태그를 제거할 수 있습니다.

```python
llm_text = await get_processed_text(
    html,
    url,
    remove_tags=["noscript", "iframe", "footer"],
)
```

---

## `extract` 모드의 의미

- `extract=False`(기본): 본문 구조를 최대한 보존하면서 텍스트 변환
- `extract=True`: `soup.get_text()` 중심으로 단순 추출

테이블/링크를 많이 다루는 작업은 기본 모드가, 단순 본문 요약은 extract 모드가 유리할 때가 많습니다.

---

## 파서 선택

기본 `html_parser='lxml'`이지만 환경에 따라 파서를 바꿔 비교할 수 있습니다.

- 문법 복원이 중요한 페이지: `lxml`
- 의존성 단순화가 필요할 때: 기본 파서(성능/정확도는 별도 검증)

다음 장에서는 이미지/링크 URL 정규화 로직을 다룹니다.
