---
layout: post
title: "LLM Reader 가이드 (02) - 설치/빠른 시작: Python + Playwright 환경 구성"
date: 2026-02-18
permalink: /llm-reader-guide-02-installation-and-quickstart/
author: m92vyas
categories: ['개발 도구', '웹 스크래핑']
tags: [llm-reader, Playwright, Python, Setup, Asyncio]
original_url: "https://github.com/m92vyas/llm-reader"
excerpt: "llm-reader 설치, Playwright 의존성 준비, 가장 짧은 실행 예제로 동작을 확인합니다."
---

## 기본 설치

```bash
python -m venv .venv
source .venv/bin/activate

pip install git+https://github.com/m92vyas/llm-reader.git
playwright install
playwright install-deps
```

저장소의 `requirements.txt` 기준 핵심 의존성은 다음입니다.

- `beautifulsoup4`
- `inscriptis`
- `minify_html`
- `pytest-playwright`

---

## 최소 실행 예제

```python
import asyncio
from url_to_llm_text.get_html_text import get_page_source
from url_to_llm_text.get_llm_input_text import get_processed_text

async def main(url: str):
    page_source = await get_page_source(url)
    llm_text = await get_processed_text(page_source, url)
    print(llm_text[:2000])

asyncio.run(main("https://example.com"))
```

---

## 설치 시 자주 막히는 지점

1. `playwright install` 누락
2. 리눅스 서버에서 `playwright install-deps` 누락
3. 이벤트 루프 충돌(Jupyter/웹 프레임워크 내 중복 `asyncio.run`)

운영 코드에서는 실행 환경별로 이벤트 루프 정책을 명확히 분리하는 것이 안전합니다.

---

## 빠른 검증 체크리스트

- 빈 문자열이 아닌 HTML이 반환되는지
- 링크가 절대경로로 출력되는지
- 이미지/표가 목적에 맞게 포함되는지

다음 장에서는 URL부터 최종 텍스트까지 전체 흐름을 한 번에 봅니다.
