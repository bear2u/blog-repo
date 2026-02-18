---
layout: post
title: "LLM Reader 가이드 (06) - 이미지/링크 정규화: URL 보존과 토큰 절감의 균형"
date: 2026-02-18
permalink: /llm-reader-guide-06-image-link-normalization/
author: m92vyas
categories: ['개발 도구', '웹 스크래핑']
tags: [URL Normalization, Image Link, Anchor Tag, llm-reader, Tokens]
original_url: "https://github.com/m92vyas/llm-reader"
excerpt: "img/a 태그를 절대 URL 텍스트로 바꾸는 방식과 토큰 비용을 줄이기 위한 옵션 선택 기준을 정리합니다."
---

## 이미지 처리 방식

코드는 `img` 태그를 순회하면서 다음 규칙을 적용합니다.

- `keep_images=False`면 이미지 제거
- `True`면 `src`를 `urljoin(base_url, src)`로 절대경로화
- 기본적으로 `.svg`, `.gif`는 제거 대상

이 방식은 LLM이 실제 리소스 링크를 잃지 않게 하면서도 불필요한 이미지를 줄이는 데 초점을 둡니다.

---

## 링크 처리 방식

`a[href]` 태그는 아래 형태로 치환됩니다.

```text
링크텍스트: 절대URL
```

링크 URL이 추출 결과의 근거가 되는 작업(상품 링크, 문서 참조 링크)에 특히 유용합니다.

---

## 토큰 비용 줄이는 선택

- 링크가 필요 없는 분류/감성 작업: `keep_webpage_links=False`
- 이미지가 필요 없는 텍스트 추출: `keep_images=False`
- 특정 확장자만 제외: `remove_image_types=[".webp", ".ico"]`

작업 목적별 프리셋 함수를 만들어 두면 반복 비용을 줄일 수 있습니다.

---

## 주의점

`srcset`, lazy-load 속성(`data-src`) 같은 변형은 기본 구현에 포함되지 않습니다.

복잡한 상거래 사이트를 다룰 때는 이 지점을 후속 확장 포인트로 보는 것이 좋습니다.

다음 장에서는 표(Table) 변환 로직을 분석합니다.
