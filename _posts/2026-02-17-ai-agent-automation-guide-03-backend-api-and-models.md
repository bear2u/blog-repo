---
layout: post
title: "AI Agent Automation 가이드 (03) - 백엔드 API/데이터 모델: Workflow, Task, Agent 스키마 이해"
date: 2026-02-17
permalink: /ai-agent-automation-guide-03-backend-api-and-models/
author: vmDeshpande
categories: ['AI 에이전트', '백엔드']
tags: [Express, MongoDB, Mongoose, REST API, JWT]
original_url: "https://github.com/vmDeshpande/ai-agent-automation/tree/main/backend/src"
excerpt: "Express 라우트와 Mongoose 모델을 기준으로 이 플랫폼의 핵심 데이터 모델과 인증 구조를 해설합니다."
---

## API 라우트 구성

`backend/src/app.js`에서 주요 엔드포인트를 한 번에 등록합니다.

- `/api/auth`, `/api/workflows`, `/api/tasks`, `/api/agents`
- `/api/schedules`, `/api/webhooks`, `/webhook`(public)
- `/api/logs`, `/api/settings`, `/api/system`, `/api/assistant`

기본적으로 대부분 JWT 인증(`auth.middleware`)이 필요합니다.

---

## 핵심 모델 4개

### 1) Workflow
- 메타데이터(`metadata.steps`)에 스텝 정의 저장
- 실행된 Task ID 목록을 `tasks`로 보관

### 2) Task
- 실제 실행 단위
- `steps`는 워크플로우에서 복사된 실행 스냅샷
- `stepResults`에 실행 이력 누적

### 3) Agent
- 모델명/temperature/tool 설정을 `config`로 관리
- `capabilities`로 기능 태깅

### 4) SystemSettings
- worker poll interval, maxAttempts
- UI theme
- assistant on/off

---

## 인증 흐름

- `auth.controller`에서 register/login 시 JWT 발급
- `auth.middleware`가 `Authorization: Bearer <token>` 검증
- 유저 문서 로드 후 `req.user`에 주입

이후 대부분의 CRUD는 `req.user._id` 기준 소유권 검사를 수행합니다.

---

## 설계 포인트

Task에 steps를 복사해두는 구조 덕분에,
워크플로우가 나중에 수정되어도 실행 당시 스냅샷 기준 재현이 쉬워집니다.

다음 장에서는 이 모델들이 실제 실행으로 연결되는 생명주기를 봅니다.

