---
layout: post
title: "Summarize 가이드 (06) - 캐시/비용/출력 제어: SQLite 캐시와 length/token 전략"
date: 2026-02-18
permalink: /summarize-guide-06-cache-cost-output-controls/
author: steipete
categories: ['개발 도구', '운영']
tags: [Cache, SQLite, --length, --max-output-tokens, Metrics]
original_url: "https://github.com/steipete/summarize/blob/main/docs/cache.md"
excerpt: "캐시 설계와 출력 길이 제어 플래그를 묶어, 비용/속도/품질 균형을 잡는 운영 방법을 설명합니다."
---

## 캐시 설계

`docs/cache.md` 기준으로 캐시는 두 층입니다.

1. SQLite (`~/.summarize/cache.sqlite`)
2. 미디어 파일 캐시 (`~/.summarize/cache/media`)

캐시 대상:

- transcript
- extracted content
- summaries
- slides manifest

---

## `--no-cache`와 `--no-media-cache` 차이

여기서 헷갈리기 쉬운 점:

- `--no-cache`: summary cache만 우회
- `--no-media-cache`: media 다운로드 캐시만 비활성

서로 범위가 다르므로 디버깅 시 용도를 구분해야 합니다.

---

## 출력 길이 전략

README/LLM 문서 기준:

- `--length`: 요약 길이 가이드(soft)
- `--max-output-tokens`: 하드 캡(optional)

프로젝트 권장도 `--length` 우선 사용입니다.

---

## extract-only 모드

`--extract`는 LLM 요약 없이 추출 결과를 출력합니다.

- URL 입력 전용
- 필요 시 `--max-extract-characters`로 길이 제한

문서 분석/전처리 단계에서 매우 유용한 모드입니다.

다음 장에서 daemon 아키텍처와 보안 모델을 봅니다.

