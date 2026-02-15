---
layout: post
title: "Cline 완벽 가이드 (05) - 모델/프로바이더: 비용과 성능을 동시에 잡는 설정"
date: 2026-02-15
permalink: /cline-guide-05-models-and-providers/
author: Cline Bot Inc.
categories: [AI 코딩 에이전트, Cline]
tags: [Cline, Models, Providers, OpenRouter, Anthropic, OpenAI, Gemini, Ollama, LMStudio]
original_url: "https://docs.cline.bot/getting-started/selecting-your-model"
excerpt: "Cline Provider, OpenAI Codex 로그인, API 키 기반 프로바이더, 로컬 모델까지 어떤 기준으로 선택할지 정리합니다."
---

## Cline은 “어떤 모델로도” 굴릴 수 있게 설계됐다

Cline 문서/README는 반복해서 “모델 생태계 락인 없이 선택하라”는 방향을 강조합니다.

실전에서 이게 의미하는 바는:

- 고난도 설계/리팩토링: 더 강한 모델
- 단순 수정/반복 작업: 저렴한 모델
- 큰 컨텍스트(문서/로그): 긴 컨텍스트 모델

처럼 작업 성격에 따라 교체가 쉽다는 점입니다.

---

## 빠른 시작: Cline Provider

문서의 Quick Start는 보통 “Cline을 Provider로 선택”하는 방법을 안내합니다.

장점:

- API 키를 직접 관리하지 않고 시작하기 쉬움
- 여러 모델을 UI에서 빠르게 고를 수 있음

단, 조직 정책이나 비용 정산 방식에 따라 “직접 키를 쓰는 방식”이 더 맞는 경우도 있습니다.

---

## 키 없이 로그인: OpenAI Codex

문서에는 OpenAI 모델을 쓰는 쉬운 방법으로 **OpenAI Codex 로그인** 흐름도 안내합니다.

포인트는:

- API 키를 복붙하는 대신 OAuth 기반으로 연결하고
- 모델 선택/전환을 UI에서 처리한다는 점입니다.

---

## API 키 기반 프로바이더: 선택 기준

Cline은 여러 프로바이더를 지원합니다(예: OpenRouter, Anthropic, OpenAI, Google Gemini, Bedrock 등).

이 때 추천 기준을 간단히 잡으면:

- **모델 다양성/가성비**: OpenRouter 계열을 먼저 고려
- **Claude 직결**: Anthropic
- **GPT 계열 직결**: OpenAI
- **아주 큰 컨텍스트가 필요**: Gemini 계열(작업 특성에 따라)
- **조직 계정/정책 준수**: Bedrock/Vertex/Azure 등

---

## 로컬 모델: Ollama / LM Studio

문서에서 로컬 모델 실행(예: Ollama, LM Studio)도 별도 섹션으로 다룹니다.

실전에서 로컬 모델을 고려할 때의 포인트:

- 네트워크 제약이 심한 환경(내부망)
- 비용 통제가 최우선인 반복 작업
- 민감 데이터가 외부로 나가면 안 되는 작업(정책에 따라)

다만, “에이전트 코딩”은 모델의 추론/도구 사용 안정성이 중요해서,
로컬 모델은 보통 “범위가 잘 정의된 작업”부터 붙이는 게 안전합니다.

---

## 비용 감각: 컨텍스트/토큰을 함께 본다

모델 선택은 가격표만 보면 실패하기 쉽습니다.

- 컨텍스트가 길면 입력 토큰이 커지고
- 에이전트 루프는 여러 번 왕복하므로
- “한 번 비싼 모델”보다 “여러 번의 루프”가 총비용을 좌우할 수 있습니다.

그래서 4장에서 다룬 컨텍스트 관리(Focus Chain/Auto Compact)와 같이 묶어서 보는 게 좋습니다.

---

*다음 글에서는 Cline의 확장 축인 MCP(Model Context Protocol)를 다룹니다. 외부 도구를 붙이는 방식이 Cline을 ‘IDE 챗봇’에서 ‘워크플로우 에이전트’로 바꿉니다.*

