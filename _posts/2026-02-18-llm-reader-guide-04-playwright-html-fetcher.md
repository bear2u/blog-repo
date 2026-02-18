---
layout: post
title: "LLM Reader 가이드 (04) - Playwright 수집기: get_page_source 내부 동작"
date: 2026-02-18
permalink: /llm-reader-guide-04-playwright-html-fetcher/
author: m92vyas
categories: ['개발 도구', '웹 스크래핑']
tags: [Playwright, Browser Automation, llm-reader, Headless, HTML]
original_url: "https://github.com/m92vyas/llm-reader"
excerpt: "get_page_source 함수의 브라우저 실행 방식, 대기 전략, 커스터마이징 포인트를 해설합니다."
---

## 함수 시그니처

`get_page_source(url, wait=1.5, user_agent=...)` 형태로 간단합니다.

내부적으로는 다음 순서로 처리됩니다.

1. `async_playwright().start()`
2. Chromium headless launch
3. 사용자 지정 user-agent로 context 생성
4. `page.goto(url)` 후 load state 대기
5. `PageDown` 입력 + 추가 대기
6. `page.content()` 반환

---

## 기본 안정화 옵션

코드에는 아래 플래그가 포함됩니다.

- `--disable-dev-shm-usage`
- `--no-sandbox`
- `--disable-blink-features=AutomationControlled`
- `--disable-infobars`

컨테이너/CI에서 브라우저 실행 실패를 줄이기 위한 선택입니다.

---

## 개선해서 쓰기 좋은 지점

- `goto`에 `timeout`/`wait_until` 명시
- 무한 스크롤 페이지 대응(반복 scroll)
- 네트워크 실패 시 재시도(backoff)
- 브라우저 재사용으로 배치 처리 성능 향상

현재 구현은 "간단하고 빠른 기본값"에 집중한 형태입니다.

---

## 실패 처리

예외 시 빈 문자열을 반환하므로, 호출 측에서 아래를 강제하는 것이 안전합니다.

- 빈 HTML 감지
- 재시도 횟수 제한
- 실패 URL 별도 큐 적재

다음 장에서는 HTML 정리 단계 옵션을 자세히 봅니다.
