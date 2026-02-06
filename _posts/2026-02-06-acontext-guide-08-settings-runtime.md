---
layout: post
title: "Acontext 완벽 가이드 (8) - 설정 & 런타임"
date: 2026-02-06
permalink: /acontext-guide-08-settings-runtime/
author: memodb-io
categories: [AI 에이전트, Acontext]
tags: [Acontext, Settings, Runtime, Self-host, Env]
original_url: "https://docs.acontext.io/settings/local"
excerpt: "로컬 실행, 핵심 환경변수, 런타임 버퍼 설정을 정리합니다."
---

## 로컬 실행 기본

```bash
curl -fsSL https://install.acontext.io | sh
mkdir acontext_server && cd acontext_server
acontext server up
```

필수 키 예시:

```env
LLM_API_KEY="YOUR_OPENAI_API_KEY"
```

## Core 설정 포인트

문서 기준 핵심 변수:

- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_SDK` (`openai`, `anthropic`)
- `LLM_SIMPLE_MODEL`
- `LLM_RESPONSE_TIMEOUT`

## Runtime 튜닝 포인트

Task 추출과 버퍼 처리 품질/비용에 영향을 주는 변수:

- `PROJECT_SESSION_MESSAGE_USE_PREVIOUS_MESSAGES_TURNS`
- `PROJECT_SESSION_MESSAGE_BUFFER_MAX_TURNS`
- `PROJECT_SESSION_MESSAGE_BUFFER_TTL_SECONDS`

운영에서는 대화 길이와 응답 지연 목표에 맞춰 세 값을 함께 조정해야 합니다.
