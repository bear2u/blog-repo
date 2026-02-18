---
layout: post
title: "AI Agent Automation 가이드 (06) - 스케줄러/웹훅: 시간 기반 + 이벤트 기반 자동화"
date: 2026-02-17
permalink: /ai-agent-automation-guide-06-scheduler-and-webhooks/
author: vmDeshpande
categories: ['AI 에이전트', '자동화']
tags: [Scheduler, Cron, Webhook, node-cron, Event Trigger]
original_url: "https://github.com/vmDeshpande/ai-agent-automation/blob/main/backend/src/services/schedulerService.js"
excerpt: "node-cron 기반 스케줄 실행과 public webhook 엔드포인트를 통해 Task를 자동 생성하는 구조를 정리합니다."
---

## Cron 스케줄 경로

`schedulerService.start()`는 enabled schedules를 로드해 메모리 job으로 등록합니다.

- 모델: `Schedule` 컬렉션
- 실행 함수: `createTaskForSchedule(schedule)`
- timezone 지원: `schedule.timezone`

트리거 시 워크플로우를 읽고 Task(status=`pending`)를 생성합니다.

---

## 스케줄 CRUD

`/api/schedules`에서 인증 기반 CRUD를 제공합니다.

생성 시 `workflow.metadata.steps`가 비어 있으면 거부하므로,
잘못된 스케줄 등록을 초기에 막습니다.

---

## 웹훅 구조

### 관리용
- `/api/webhooks` (auth required)
- source/secret/workflowId를 등록

### 공개 수신
- `POST /webhook/:source`
- `?secret=` 또는 `x-webhook-secret`로 검증
- 성공 시 Task를 생성

외부 시스템(예: GitHub/Stripe/custom webhook) 이벤트를 내부 태스크로 흡수하는 형태입니다.

---

## 보안 관점 체크

- secret이 없거나 불일치하면 거부(401/403)
- webhook 설정의 `active` 상태를 검사
- payload와 headers를 Task.input에 저장해 추적 가능

다음 장에서 워커 루프와 큐, 로그 저장을 묶어서 봅니다.

