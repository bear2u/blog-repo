---
layout: post
title: "ai-hedge-fund 완벽 가이드 (04) - 실전 사용 패턴"
date: 2026-03-10
permalink: /ai-hedge-fund-guide-04-usage/
author: virattt
categories: [개발 도구, ai-hedge-fund]
tags: [Trending, GitHub, ai-hedge-fund]
original_url: "https://github.com/virattt/ai-hedge-fund"
excerpt: "티커/기간/로컬LLM 옵션 중심으로 실행 예시를 모읍니다."
---

## 기본 실행

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

---

## 로컬 LLM(Ollama) 사용

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --ollama
```

---

## 기간 지정(백테스트 느낌으로)

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01
```

---

## 팁

- 결과가 이상하면 `.env`에 들어간 키(LLM/금융데이터)부터 점검하세요.
- README에 “실거래는 하지 않는다”고 명시되어 있으니, 연구/학습 목적 범위에서 사용하세요.

---

*다음 글에서는 운영/확장/베스트 프랙티스를 정리합니다.*

