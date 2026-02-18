---
layout: post
title: "AI Agent Automation 가이드 (04) - 워크플로우-태스크 생명주기: 정의에서 실행까지"
date: 2026-02-17
permalink: /ai-agent-automation-guide-04-workflow-and-task-lifecycle/
author: vmDeshpande
categories: ['AI 에이전트', '실행 엔진']
tags: [Workflow, Task, Lifecycle, Runner, Metadata]
original_url: "https://github.com/vmDeshpande/ai-agent-automation/blob/main/backend/src/controllers/workflow.controller.js"
excerpt: "워크플로우 작성, 스텝 저장, 즉시 실행(run), 태스크 생성까지의 엔드투엔드 흐름을 코드 기준으로 정리합니다."
---

## 단일 진실원(Single Source of Truth)

이 프로젝트는 워크플로우 정의를 `workflow.metadata.steps`에 저장하고,
실행 시점에 이를 Task의 `steps`로 복제합니다.

- 저장: `PUT /api/workflows/:workflowId/steps`
- 실행: `POST /api/workflows/:workflowId/run`
- 일반 태스크 생성: `POST /api/tasks`

---

## runWorkflowNow 동작

`workflow.controller.runWorkflowNow`는 다음을 수행합니다.

1. 워크플로우 소유권 확인
2. Task 생성(status=`pending`)
3. `metadata.steps`를 Task 쪽에 전달
4. Workflow.tasks에 Task ID 추가

즉 API 호출 시점에는 실제 실행이 아니라 "큐에 올리는 단계"까지 처리됩니다.

---

## createTask 경로

`task.controller.createTask`는 workflowId가 있으면 steps를 검증 후 Task를 생성합니다.

- 스텝이 비어 있으면 `workflow_has_no_steps` 반환
- 생성된 Task의 `steps`가 Runner의 실제 입력

이 제약 덕분에 빈 워크플로우 실행을 초기에 차단합니다.

---

## 실행 메타데이터

Task에는 다음 실행 정보가 남습니다.

- `status`: pending/running/completed/failed
- `attempts`: 재시도 횟수
- `stepResults`: step별 output/history
- `startedAt`, `completedAt`

다음 장에서 step executor 내부를 상세히 분석합니다.

