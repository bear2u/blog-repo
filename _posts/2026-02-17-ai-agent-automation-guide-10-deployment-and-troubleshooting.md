---
layout: post
title: "AI Agent Automation 가이드 (10) - 배포/트러블슈팅: 운영 전 점검 포인트"
date: 2026-02-17
permalink: /ai-agent-automation-guide-10-deployment-and-troubleshooting/
author: vmDeshpande
categories: ['AI 에이전트', '트러블슈팅']
tags: [Deployment, PM2, Troubleshooting, Self Hosting, Production Checklist]
original_url: "https://github.com/vmDeshpande/ai-agent-automation/tree/main/scripts"
excerpt: "현재 저장소 상태를 기준으로 배포 스크립트, 운영 시 주의할 점, 알려진 제약을 체크리스트 형태로 정리합니다."
---

## 배포 스크립트 요약

`scripts/deploy.sh` 기준 배포 흐름:

1. frontend `npm run build`
2. backend 의존성 설치
3. PM2로 backend/worker 재시작

즉 최소 운영 단위는 "API 서버 + 워커 프로세스" 2개입니다.

---

## 운영 전 필수 점검

1. MongoDB 연결 안정성 (`MONGO_URI`)
2. JWT secret 강도 (`JWT_SECRET`)
3. LLM 키 주입 여부(Groq/OpenAI/Gemini/HF)
4. Worker 튜닝(`pollIntervalMs`, `maxAttempts`)
5. webhook secret 관리

---

## 소스 기준 현재 제약

아래 항목은 저장소 상태(2026-02-17 기준)를 보고 확인된 포인트입니다.

- `infra/Dockerfile`, `infra/docker-compose.yml` 파일이 비어 있음
- `workflows/*.json` 샘플 파일이 비어 있음
- `backend/package.json`의 `dev` 엔트리(`src/server.js`)와 실제 엔트리(`server.js`) 경로가 어긋남

즉 컨테이너 배포/샘플 워크플로우는 추가 정비가 필요합니다.

---

## 자주 겪는 문제

### 1) API는 뜨는데 작업이 실행 안 됨
- 워커(`npm run worker`)가 별도 실행 중인지 확인
- Task 상태가 pending에 머무는지 확인

### 2) 어시스턴트가 offline만 동작
- `/api/system/env`에서 키 감지 상태 확인
- Settings의 assistant enabled 상태 확인

### 3) 스케줄은 만들었는데 실행 안 됨
- cron 식/타임존 확인
- workflow에 steps가 실제 저장돼 있는지 확인

---

## 실무 확장 우선순위

1. Docker/compose 파일 채우기
2. 샘플 워크플로우 JSON 제공
3. dev/prod 엔트리 스크립트 정리
4. 테스트/헬스체크 자동화 강화

이 4가지를 먼저 정리하면 로컬 프로젝트에서 팀 운영 가능한 플랫폼으로 빠르게 올릴 수 있습니다.

