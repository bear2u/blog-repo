---
layout: post
title: "AI Agent Automation 가이드 (07) - 워커/큐/로그: claim-next-task 루프와 실행 관측"
date: 2026-02-17
permalink: /ai-agent-automation-guide-07-worker-and-observability/
author: vmDeshpande
categories: ['AI 에이전트', '운영']
tags: [Worker, Queue, Retry, Logs, Observability]
original_url: "https://github.com/vmDeshpande/ai-agent-automation/blob/main/backend/src/agents/runner.js"
excerpt: "runner.js와 queueService.js를 기준으로 태스크 클레임-실행-완료 루프와 로그 수집 구조를 정리합니다."
---

## Worker 루프 핵심

`runner.js`는 무한 루프로 동작합니다.

1. `claimNextTask()`로 pending task 1개 선점
2. task 상태를 running으로 갱신
3. steps를 순차 실행
4. 실패 즉시 중단 또는 전부 완료
5. `completeTask()`로 completed/failed 마감

---

## 재시도 제어

`queueService.claimNextTask()`에서 attempts 조건을 사용합니다.

- 조건: `attempts < WORKER_MAX_ATTEMPTS`
- 선점 시 attempts 자동 증가

과도한 재시도를 DB 쿼리 레벨에서 제한하는 방식입니다.

---

## 동적 워커 설정

`SystemSettings.worker`를 주기적으로 읽어 아래 값을 반영합니다.

- `pollIntervalMs`
- `maxAttempts`

DB 조회 실패 시 fallback 설정을 사용해 워커가 완전히 멈추지 않게 구성돼 있습니다.

---

## 로그 저장

`writeLog(message, level, meta)`가 `Log` 모델에 기록합니다.

- 레벨: debug/info/success/warn/error
- 필드: workerId/taskId/workflowId

로그 조회는 `/api/logs`에서 제공합니다.

다음 장에서 Next.js 프런트 구조와 API 연동을 정리합니다.

