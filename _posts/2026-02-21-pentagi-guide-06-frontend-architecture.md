---
layout: post
title: "PentAGI 가이드 (06) - 프론트엔드 구조: React/Vite + Apollo"
date: 2026-02-21
permalink: /pentagi-guide-06-frontend-architecture/
author: PentAGI Team
categories: ['AI 에이전트', '보안']
tags: [PentAGI, React, Vite, GraphQL, Apollo, Tailwind]
original_url: "https://github.com/vxcontrol/pentagi"
excerpt: "frontend/의 구조와 기술 스택을 빠르게 훑고, 실시간 UI(구독/터미널/로그)가 어떤 방식으로 연결되는지 정리합니다."
---

## 프론트엔드는 “관리 콘솔”에 가깝다

PentAGI 프론트엔드는 단순 채팅 UI가 아니라:

- Flow/Task/Subtask의 상태를 추적하고
- 실행 로그(터미널/도구 호출/검색 결과)를 시각화하며
- 설정(Provider/OAuth/환경)을 관리하는

**운영 콘솔** 성격이 강합니다.

---

## 기술 스택(README 기준)

- React + TypeScript
- Vite 빌드
- TailwindCSS + shadcn/ui + Radix UI
- Apollo Client(GraphQL) + WebSocket subscriptions
- Vitest 테스트

`package.json`을 보면 GraphQL 코드 생성 스크립트도 포함되어 있습니다.

```text
npm run graphql:generate
```

---

## 프로젝트 구조(개념)

README가 제시하는 구조는 다음처럼 feature 중심입니다.

```text
frontend/src/
├─ components/   # 공용 UI
├─ features/     # 도메인 기능(채팅, 인증 등)
├─ graphql/      # GraphQL operation / 타입
├─ hooks/        # 커스텀 훅
├─ lib/          # 유틸/설정
└─ pages/        # 라우트 단위 화면
```

이 구조는 “한 화면이 하는 일”이 많아지기 쉬운 운영 콘솔에서 유지보수에 도움이 됩니다.

---

## GraphQL이 중요한 이유

PentAGI의 UI는 “상태가 계속 변하는 시스템”을 보여줘야 합니다.

- 작업이 생성/대기/실행/종료로 이동하고
- 로그가 계속 쌓이고
- 에이전트 메시지/툴 호출 결과가 스트리밍되고
- 터미널 출력이 실시간으로 갱신됩니다.

이런 UI는 폴링보다 **구독(Subscriptions)** 모델이 자연스럽습니다.

---

## 환경 변수: `VITE_API_URL`

프론트는 백엔드 API URL을 환경 변수로 받습니다.

```bash
VITE_API_URL=your_api_url
```

로컬 개발에서는 백엔드와의 연결(HTTPS 여부, 포트, 프록시)을 함께 조정해야 합니다.

---

## 개발 편의 팁

프론트는 ESLint/Prettier 스크립트를 제공합니다.

```bash
npm run lint
npm run prettier
```

또한 SSL 인증서 생성 스크립트도 포함되어 있어(개발용),
HTTPS 개발 환경을 빠르게 맞추는 데 도움이 됩니다.

---

## 참고 링크

- 인덱스(전체 목차): `/blog-repo/pentagi-guide/`
- 프론트 README: `https://github.com/vxcontrol/pentagi/blob/master/frontend/README.md`

---

다음 글에서는 PentAGI의 핵심인 **멀티 에이전트/프롬프트 템플릿**이 어떤 역할 분리를 갖는지 정리합니다.

