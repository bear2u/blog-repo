---
layout: post
title: "AI Agent Automation 가이드 (02) - 로컬 실행/환경 변수: 백엔드, 워커, 프런트 동시 구동"
date: 2026-02-17
permalink: /ai-agent-automation-guide-02-local-setup/
author: vmDeshpande
categories: ['AI 에이전트', '개발 환경']
tags: [Node.js, Express, Next.js, MongoDB, Environment Variables]
original_url: "https://github.com/vmDeshpande/ai-agent-automation/blob/main/README.md"
excerpt: "README와 스크립트 파일 기준으로 백엔드/워커/프런트 실행 순서와 `.env` 필수값을 정리합니다."
---

## 기본 실행 순서

README 기준 로컬 실행은 다음 흐름입니다.

```bash
git clone https://github.com/vmDeshpande/ai-agent-automation.git
cd ai-agent-automation

cd backend
npm install
cp .env.example .env
npm run dev
npm run worker

cd ../frontend
npm install
npm run dev
```

- Backend: `http://localhost:5000`
- Frontend: `http://localhost:3000`

---

## `.env.example` 핵심 변수

`backend/.env.example`에서 중요한 키들:

- `MONGO_URI`, `JWT_SECRET`
- `GROQ_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `HF_API_KEY`
- `WORKER_POLL_INTERVAL_MS`, `WORKER_MAX_ATTEMPTS`
- `EMAIL_*` (email step 테스트용)

LLM 키가 없으면 플랫폼은 돌아가도 AI 어시스턴트는 offline 모드에 머무릅니다.

---

## 개발 스크립트 참고

- 루트 `scripts/start-dev.sh`는 backend + worker + frontend를 순차 실행합니다.
- `scripts/deploy.sh`는 frontend 빌드 후 PM2로 backend/worker 재기동 흐름입니다.

---

## 소스 기준 주의점

코드 기준으로 `backend/package.json`의 `dev` 스크립트는 `nodemon src/server.js`를 가리키지만, 실제 서버 엔트리는 `backend/server.js`입니다.

환경에 따라 `npm run dev`가 바로 동작하지 않으면 아래처럼 `npm start`로 백엔드를 띄우는 우회가 필요할 수 있습니다.

```bash
cd backend
npm start
```

다음 장에서 API와 스키마를 기준으로 데이터 구조를 정리합니다.

