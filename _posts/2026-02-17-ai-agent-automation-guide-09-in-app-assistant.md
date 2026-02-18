---
layout: post
title: "AI Agent Automation 가이드 (09) - 인앱 어시스턴트: offline/online 모드와 컨텍스트 기반 분석"
date: 2026-02-17
permalink: /ai-agent-automation-guide-09-in-app-assistant/
author: vmDeshpande
categories: ['AI 에이전트', '어시스턴트']
tags: [Assistant, Context Aware, Groq, Offline Mode, Online Mode]
original_url: "https://github.com/vmDeshpande/ai-agent-automation/blob/main/backend/src/controllers/assistant.controller.js"
excerpt: "프런트 컨텍스트 주입과 백엔드 시스템 프롬프트를 연결해 화면 상태 기반 분석 응답을 만드는 방식을 설명합니다."
---

## 모드 전환 구조

어시스턴트는 두 모드가 있습니다.

- offline: 프런트의 규칙 기반 응답(`offline-responder.tsx`)
- online: 백엔드 `/api/assistant/chat` 호출

`settings.assistant.enabled` + API 키 유무에 따라 실질적인 사용 모드가 결정됩니다.

---

## 컨텍스트 주입 방식

각 페이지는 `useAssistantContext().setContext(...)`로 현재 상태를 넣습니다.

예시 컨텍스트:

- 현재 페이지 종류
- workflow/task/step 식별자
- 실패 스텝 정보
- 대시보드 통계/최근 활동

그래서 어시스턴트는 "일반 챗봇"보다 화면 상태 분석에 집중할 수 있습니다.

---

## 서버 프롬프트 설계

`assistant.controller.buildSystemPrompt()`는 규칙이 매우 강합니다.

- 현재 컨텍스트 밖 정보 추측 금지
- 실패 원인 분석 우선
- 장황한 설명 대신 실무형 진단 중심

실패 태스크가 있을 때는 failed step 이름/원인/수정안을 우선하도록 유도합니다.

---

## 호출 경로

```text
AssistantPanel -> onlineRespond()
  -> POST /api/assistant/chat (JWT auth)
    -> Groq chat.completions
```

모델은 현재 `llama-3.1-8b-instant`로 고정 사용됩니다.

다음 장에서 배포와 트러블슈팅 체크리스트로 마무리합니다.

