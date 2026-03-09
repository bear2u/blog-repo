---
layout: post
title: "Prompt Engineering Guide 가이드 (01) - 소개와 저장소 지도"
date: 2026-02-20
permalink: /prompt-engineering-guide-01-intro-and-repo-map/
author: Elvis Saravia (DAIR.AI)
categories: ['LLM 학습', '프롬프트 엔지니어링']
tags: [Prompt Engineering, LLM, DAIR.AI, Guide]
original_url: "https://github.com/dair-ai/Prompt-Engineering-Guide"
excerpt: "Prompt Engineering Guide 저장소의 목적, 문서 범위, 구조를 빠르게 파악해 학습 경로를 설계합니다."
---

## 이 저장소가 하는 일

`Prompt-Engineering-Guide`는 LLM 활용을 위한 프롬프트 설계 지식을 체계적으로 정리한 오픈 지식베이스입니다.

- 입문: 프롬프트 기본 요소와 설정
- 심화: CoT, ReAct, RAG 같은 고급 기법
- 응용: 코드 생성, 함수 호출, 데이터 생성
- 리스크: 주입 공격, 신뢰성, 바이어스

문서 성격은 "한 번 쓰고 끝"이 아니라 계속 확장되는 레퍼런스에 가깝습니다.

---

## 저장소 구조 핵심

최상위 기준으로 보면 학습에 중요한 축은 아래입니다.

- `guides/`: 전통적인 Prompt Engineering 가이드 문서
- `pages/`: 웹 문서용 MDX 콘텐츠(최신 확장 주제 포함)
- `notebooks/`: 실습용 노트북
- `lecture/`: 강의 슬라이드/자료
- `img/`: 시각 자료

즉, `guides`만 보지 말고 `pages`와 `notebooks`를 함께 보는 것이 좋습니다.

---

## 콘텐츠 범위

`pages/_meta.en.json` 기준으로 소개/기법/에이전트/응용/모델/리스크/논문/툴/데이터셋까지 분리되어 있습니다.

이 구조는 "프롬프트 문구"를 넘어서 실제 제품 개발 흐름(검색, 도구 호출, 평가, 안전성)으로 이어지게 설계된 것입니다.

---

## 이 시리즈 접근 방식

이 가이드는 문서 요약이 아니라 아래 질문에 답하도록 구성합니다.

1. 어떤 기술을 언제 써야 하는가?
2. 어떤 실패 패턴이 반복되는가?
3. 실험 결과를 어떻게 운영 루틴으로 바꿀 것인가?

다음 장에서 로컬 실행과 문서 사이트 구조를 먼저 정리합니다.
