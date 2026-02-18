---
layout: post
title: "LLM Reader 가이드 (01) - 소개/포지셔닝: 웹페이지를 LLM 입력으로 바꾸는 전처리 레이어"
date: 2026-02-18
permalink: /llm-reader-guide-01-intro/
author: m92vyas
categories: ['개발 도구', '웹 스크래핑']
tags: [llm-reader, Web Scraping, LLM, HTML Preprocessing, Python]
original_url: "https://github.com/m92vyas/llm-reader"
excerpt: "llm-reader가 어떤 문제를 풀고 어떤 사용 시나리오에서 비용과 정확도를 동시에 개선하는지 정리합니다."
---

## LLM Reader가 푸는 문제

일반 HTML을 그대로 LLM에 넣으면 다음 문제가 자주 발생합니다.

- 광고/스크립트/스타일 노이즈로 필요한 정보가 묻힘
- 상대경로 링크/이미지가 깨져 추출 품질이 떨어짐
- 표 구조가 텍스트로 붕괴되어 컬럼 기반 추출이 어려움

LLM Reader는 이 지점을 "추출 전 전처리"로 해결하려는 도구입니다.

---

## 저장소 기준 핵심 구성

- `get_html_text.py`: Playwright로 페이지 소스 수집
- `get_llm_input_text.py`: HTML 파싱/정리/구조화 텍스트 변환
- `get_llm_ready_text.py`: 수집+변환 래퍼 함수

구조가 단순해서 기존 크롤러에 모듈만 붙여 쓰기 쉽습니다.

---

## 포지셔닝

README는 Firecrawl/Jina Reader API의 대안 관점을 강조합니다.

1. HTML 확보는 원하는 방식(API/프록시/직접 브라우저)으로 수행
2. LLM 친화 텍스트 변환은 로컬 오픈소스 코드로 처리
3. 결과적으로 추출 토큰 비용을 줄이는 운영 모델

즉 "수집"과 "LLM 입력 정규화"를 분리해 비용/유연성을 확보합니다.

---

## 어떤 경우에 잘 맞나

- 카탈로그/리스트 페이지에서 링크+이미지+표를 같이 뽑아야 할 때
- 추출 정확도를 위해 본문 구조를 최대한 보존하고 싶을 때
- 여러 소스에서 HTML만 가져와 공통 후처리 파이프라인을 만들고 싶을 때

다음 장에서는 설치와 첫 실행 경로를 정리합니다.
