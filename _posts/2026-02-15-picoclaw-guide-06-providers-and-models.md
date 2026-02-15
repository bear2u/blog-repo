---
layout: post
title: "PicoClaw 가이드 (06) - 프로바이더/모델: OpenRouter, Zhipu, Groq와 모델 선택"
date: 2026-02-15
permalink: /picoclaw-guide-06-providers-and-models/
author: Sipeed
categories: [AI 에이전트, 개발 도구]
tags: ["PicoClaw", "Providers", "OpenRouter", "Zhipu", "Groq", "Models"]
original_url: "https://github.com/sipeed/picoclaw#providers"
excerpt: "PicoClaw는 여러 LLM 프로바이더를 지원합니다. README의 가이드를 기준으로 어떤 프로바이더를 언제 쓰면 좋은지, 모델/비용/필터링 이슈까지 정리합니다."
---

## 프로바이더는 “연결 방식”이다

PicoClaw에서 프로바이더는:

- 어떤 회사의 API로 호출할지
- 어떤 키/엔드포인트로 붙을지

를 의미합니다. 모델 선택은 프로바이더별로 다르게 노출될 수 있고, 동일 모델이라도 품질/가격/레이트리밋이 달라질 수 있습니다.

---

## README가 언급하는 프로바이더들

README의 Providers 섹션에는 다음이 정리돼 있습니다(“To be tested” 표기 포함).

- `openrouter`: 여러 모델을 한 곳에서 쓰는 경로(추천 표기)
- `zhipu`: GLM 계열, 콘텐츠 필터링 이슈가 있을 수 있음
- `groq`: LLM + 음성 트랜스크립션(Whisper) 용도도 언급
- `anthropic`, `openai`, `gemini`, `deepseek` 등

---

## 모델 선택의 실전 기준

1. “잘 동작하는 한 모델”을 먼저 정합니다.
2. 그 다음에:
   - 비용이 부담이면 더 저렴한 모델로 내려가고
   - 툴 호출/코딩 품질이 부족하면 더 강한 모델로 올립니다.

PicoClaw의 목표가 “초저사양에서도 도는 에이전트”인 만큼, 모델 자체는 클라우드로 쓰되 로컬 리소스는 아끼는 방향이 기본값입니다.

---

## 자주 겪는 이슈

- **콘텐츠 필터링 오류**: README는 일부 프로바이더(Zhipu 등)가 필터링을 할 수 있으니 프롬프트를 바꾸거나 모델을 바꾸라고 안내합니다.
- **키/베이스 URL 미설정**: 먼저 `providers` 블록에서 키가 제대로 읽히는지부터 확인합니다.

다음 장에서는 실제로 “어디서 메시지를 받을지”를 구성하는 채팅 채널(Telegram/Discord/LINE 등)을 정리합니다.

