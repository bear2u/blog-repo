---
layout: post
title: "ai-hedge-fund 완벽 가이드 (02) - 설치 및 빠른 시작"
date: 2026-03-10
permalink: /ai-hedge-fund-guide-02-installation/
author: virattt
categories: [개발 도구, ai-hedge-fund]
tags: [Trending, GitHub, ai-hedge-fund]
original_url: "https://github.com/virattt/ai-hedge-fund"
excerpt: "Poetry 기반 설치와 .env 설정 흐름"
---

## 요구사항(README 기준)

- Python + Poetry
- LLM API 키(예: `OPENAI_API_KEY`) 최소 1개
- 티커에 따라 금융 데이터 키(`FINANCIAL_DATASETS_API_KEY`)가 필요할 수 있음

---

## 설치

```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
poetry install
```

---

## API 키 설정

`.env.example` → `.env`로 복사 후 키를 넣습니다.

```bash
cp .env.example .env
```

---

## 첫 실행(CLI)

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

로컬 LLM(Ollama) 경로도 지원합니다(README의 `--ollama` 참고).

---

*다음 글에서는 핵심 개념과 아키텍처를 정리합니다.*

