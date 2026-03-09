---
layout: post
title: "Prompt Engineering Guide 가이드 (06) - 채팅 모델과 대화 설계"
date: 2026-02-20
permalink: /prompt-engineering-guide-06-chat-models-and-conversation-design/
author: Elvis Saravia (DAIR.AI)
categories: ['LLM 학습', '프롬프트 엔지니어링']
tags: [Chat Models, Multi-turn, System Prompt, Roles]
original_url: "https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/guides/prompts-chatgpt.md"
excerpt: "Chat 형식 입력에서 system/user/assistant 역할을 어떻게 설계해야 멀티턴과 단일 태스크를 안정적으로 수행할지 정리합니다."
---

## 채팅형 입력의 장점

`prompts-chatgpt.md`는 모델이 대화 히스토리를 입력으로 받는 구조를 기준으로 설명합니다.

핵심은 역할 분리입니다.

- system: 정책/정체성/행동 원칙
- user: 실제 요청
- assistant: 이전 응답(문맥)

---

## 멀티턴 설계 원칙

멀티턴에서 품질이 흔들리는 대표 원인은 "역할 경계 붕괴"입니다.

- system 정책을 짧고 단단하게 유지
- 각 턴 목표를 명시
- 장문 히스토리는 요약 상태로 압축

---

## 싱글턴 태스크에도 유효

채팅 포맷은 대화뿐 아니라 QA/추출/요약 같은 단일 태스크에도 유효합니다.

가이드 예시처럼, 사용자 메시지에 컨텍스트와 출력 조건을 명확히 쓰면 completion 스타일과 유사한 품질을 얻을 수 있습니다.

---

## 실무 체크리스트

1. 역할별 책임을 섞지 않는다.
2. 답변 형식을 메시지에 명시한다.
3. 모델 버전 변경 시 프롬프트 회귀 테스트를 한다.

다음 장에서는 신뢰성 문제(사실성/편향/일반화)를 다룹니다.
