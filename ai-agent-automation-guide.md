---
layout: page
title: AI Agent Automation 가이드
permalink: /ai-agent-automation-guide/
icon: fas fa-robot
---

# AI Agent Automation 완벽 가이드

> **로컬 우선(Local-first) AI 워크플로우 엔진을 직접 실행하고, 단계별 실행/스케줄/웹훅/관측까지 이해하는 실전형 해설**

**ai-agent-automation**은 LLM 호출, HTTP 요청, 내부 도구 실행을 "워크플로우 단계"로 정의하고, 이를 작업(Task) 단위로 실행하는 오픈소스 자동화 플랫폼입니다. 백엔드는 Express + MongoDB, 프론트는 Next.js로 구성되어 있습니다.

- 원문 저장소: https://github.com/vmDeshpande/ai-agent-automation
- 문서 사이트: https://vmdeshpande.github.io/ai-automation-platform-website/

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개/포지셔닝](/blog-repo/ai-agent-automation-guide-01-intro/) | 이 프로젝트가 해결하는 문제와 핵심 가치 |
| 02 | [로컬 실행/환경 변수](/blog-repo/ai-agent-automation-guide-02-local-setup/) | 백엔드/워커/프런트 실행과 `.env` 구성 |
| 03 | [백엔드 API/데이터 모델](/blog-repo/ai-agent-automation-guide-03-backend-api-and-models/) | Auth, Workflow, Task, Agent, Settings 스키마 |
| 04 | [워크플로우-태스크 생명주기](/blog-repo/ai-agent-automation-guide-04-workflow-and-task-lifecycle/) | Workflow 정의가 Task 실행으로 이어지는 흐름 |
| 05 | [Step Executor](/blog-repo/ai-agent-automation-guide-05-step-executor/) | LLM/HTTP/Delay/File/Email/Browser 스텝 실행 로직 |
| 06 | [스케줄러/웹훅](/blog-repo/ai-agent-automation-guide-06-scheduler-and-webhooks/) | cron 자동 실행과 외부 이벤트 트리거 |
| 07 | [워커/큐/로그](/blog-repo/ai-agent-automation-guide-07-worker-and-observability/) | claim-complete 루프, 재시도, 실행 로그 |
| 08 | [프런트엔드 구조](/blog-repo/ai-agent-automation-guide-08-frontend-architecture/) | Next.js App Router 기반 화면/상태 관리 |
| 09 | [인앱 어시스턴트](/blog-repo/ai-agent-automation-guide-09-in-app-assistant/) | offline/online 모드와 컨텍스트 기반 응답 |
| 10 | [배포/트러블슈팅](/blog-repo/ai-agent-automation-guide-10-deployment-and-troubleshooting/) | 현재 제약과 운영 전 점검 체크리스트 |
