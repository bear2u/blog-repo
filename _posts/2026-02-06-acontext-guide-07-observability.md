---
layout: post
title: "Acontext 완벽 가이드 (7) - 관측성(Observability)"
date: 2026-02-06
permalink: /acontext-guide-07-observability/
author: memodb-io
categories: [AI 에이전트, Acontext]
tags: [Acontext, Observability, Agent Tasks, Buffer, Tracing]
original_url: "https://docs.acontext.io/observe/agent_tasks"
excerpt: "에이전트 작업 추출과 대시보드/트레이싱 관측 기능을 정리합니다."
---

## Agent Tasks

Acontext는 에이전트 대화에서 계획 단계(예: 1,2,3 단계)를 자동 추출해 추적 가능한 Task로 만듭니다.

- 저장된 메시지 기반 처리
- `flush`로 즉시 처리 트리거 가능
- `get_tasks`로 결과 조회

## Session Buffer

비용 최적화를 위해 메시지를 배치 처리합니다.

- 최대 턴 수(`PROJECT_SESSION_MESSAGE_BUFFER_MAX_TURNS`) 도달 시 처리
- 유휴 시간(`PROJECT_SESSION_MESSAGE_BUFFER_TTL_SECONDS`) 초과 시 처리

## 특정 세션 추적 비활성화

`disable_task_tracking=True`로 세션 단위 비활성화가 가능합니다.

- 메시지 저장은 유지
- Task 자동 추출만 중단

## Dashboard

대시보드에서 다음 뷰를 확인합니다.

- Metrics(BI)
- Traces
- Messages
- Artifacts

## Distributed Tracing

OpenTelemetry/Jaeger 기반으로 API/Core/DB/Cache/Storage/LLM 호출 흐름을 추적합니다.
