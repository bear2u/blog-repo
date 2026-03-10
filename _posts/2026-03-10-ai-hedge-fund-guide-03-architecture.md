---
layout: post
title: "ai-hedge-fund 완벽 가이드 (03) - 핵심 개념과 아키텍처"
date: 2026-03-10
permalink: /ai-hedge-fund-guide-03-architecture/
author: virattt
categories: [개발 도구, ai-hedge-fund]
tags: [Trending, GitHub, ai-hedge-fund]
original_url: "https://github.com/virattt/ai-hedge-fund"
excerpt: "agents/graph/tools 구조로 전체 흐름을 잡습니다."
---

## 핵심 엔트리 포인트

- CLI 실행: `src/main.py`
- 에이전트 구현: `src/agents/`
- LLM/유틸: `src/llm/`, `src/utils/`
- 상태/흐름: `src/graph/`

---

## 큰 흐름(개념도)

```mermaid
flowchart TD
  A[티커/기간 입력] --> B[src/main.py]
  B --> C[분석 에이전트들\n(valuation/sentiment/fundamentals/technicals)]
  C --> D[Risk Manager]
  D --> E[Portfolio Manager]
  E --> F[결과 출력/리포트]
```

---

## 다음에 볼 것

- `src/agents/portfolio_manager.py`, `src/agents/risk_manager.py`
- `src/tools/api.py`(외부 데이터 호출) / `.env` 키 로딩(`src/utils/api_key.py`)

---

*다음 글에서는 실전 사용 패턴을 정리합니다.*

