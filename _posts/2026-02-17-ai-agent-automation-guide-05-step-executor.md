---
layout: post
title: "AI Agent Automation 가이드 (05) - Step Executor: LLM/HTTP/Delay/File/Email/Browser 실행기"
date: 2026-02-17
permalink: /ai-agent-automation-guide-05-step-executor/
author: vmDeshpande
categories: ['AI 에이전트', '실행 엔진']
tags: [Step Executor, LLM, HTTP, File Tool, Email Tool, Browser Tool]
original_url: "https://github.com/vmDeshpande/ai-agent-automation/blob/main/backend/src/agents/executor.js"
excerpt: "executor.js를 기준으로 스텝 타입별 실행 로직과 템플릿 보간(interpolate) 방식, 실패 처리 규칙을 해설합니다."
---

## 지원 스텝 타입

`executeStep(step, context)`는 타입 분기 방식으로 동작합니다.

- `llm`: Groq 기반 텍스트 생성
- `http`: axios 호출
- `delay`: 지정 초만큼 sleep
- `email`: nodemailer 전송
- `file`: read/write/append
- `browser`: puppeteer screenshot/evaluate

---

## LLM 스텝

`llmAdapter.runLLM`가 `@ai-sdk/groq` + `ai.generateText`로 호출됩니다.

- 기본 모델: `llama-3.1-8b-instant`
- 옵션: `temperature`, `maxTokens`

실패 시 `success: false`와 에러를 result에 포함합니다.

---

## HTTP/File/Browser 실무 포인트

- HTTP body는 문자열 보간 후 JSON parse 시도
- File step은 `{{results}}`를 만나면 step 결과 전체를 JSON으로 저장
- Browser step은 `--no-sandbox` 옵션 포함, CI/컨테이너 실행을 고려한 형태

---

## 템플릿 보간

`interpolate()`는 `{{key}}` 및 `{{nested.key}}`를 context 값으로 치환합니다.

예: `{{last}}`, `{{input.text}}`, `{{workflow.name}}`

이 메커니즘으로 step 간 데이터 전달을 간단히 구성할 수 있습니다.

---

## 실패 처리

알 수 없는 스텝 타입이나 런타임 예외는 공통 실패 result로 반환됩니다.

- `success: false`
- `output: err.message`
- `error: stack 일부`

다음 장에서 스케줄러와 웹훅 트리거를 연결해봅니다.

