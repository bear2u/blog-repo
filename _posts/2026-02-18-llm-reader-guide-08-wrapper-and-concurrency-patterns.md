---
layout: post
title: "LLM Reader 가이드 (08) - 래퍼 함수/동시성 패턴: url_to_llm_text 실전 사용"
date: 2026-02-18
permalink: /llm-reader-guide-08-wrapper-and-concurrency-patterns/
author: m92vyas
categories: ['개발 도구', '웹 스크래핑']
tags: [Asyncio, Concurrency, llm-reader, Wrapper, Batch]
original_url: "https://github.com/m92vyas/llm-reader"
excerpt: "url_to_llm_text 래퍼 함수와 다중 URL 배치 처리 시 필요한 동시성/안정성 패턴을 정리합니다."
---

## 래퍼 함수의 역할

`url_to_llm_text(url, ...)`는 내부에서 다음을 순서대로 수행합니다.

1. `get_page_source`
2. `get_processed_text`
3. 최종 문자열 반환

간단한 작업에는 진입점이 하나라서 사용성이 좋습니다.

---

## 배치 처리 예시

```python
import asyncio
from url_to_llm_text.get_llm_ready_text import url_to_llm_text

async def run_batch(urls):
    sem = asyncio.Semaphore(5)

    async def worker(u):
        async with sem:
            return u, await url_to_llm_text(u, wait=2.0)

    return await asyncio.gather(*(worker(u) for u in urls), return_exceptions=True)
```

동시성 제한(`Semaphore`) 없이 대량 요청을 보내면 브라우저 자원 고갈이 쉽게 발생합니다.

---

## 실전에서 추가할 요소

- 요청별 타임아웃
- 실패 재시도/백오프
- 결과 캐시(중복 URL 방지)
- 메트릭(성공률, 평균 처리 시간, 토큰 길이)

---

## 함수 경계 설계 팁

수집(`get_page_source`)과 정제(`get_processed_text`)를 분리해 호출하면,
사이트별 수집 전략만 별도로 바꿔도 전체 파이프라인 재사용이 쉽습니다.

다음 장에서는 실제 LLM 추출 프롬프트와 결합하는 방법을 다룹니다.
