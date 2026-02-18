---
layout: post
title: "AI Agent Automation 가이드 (08) - 프런트엔드 구조: Next.js App Router와 대시보드 화면"
date: 2026-02-17
permalink: /ai-agent-automation-guide-08-frontend-architecture/
author: vmDeshpande
categories: ['AI 에이전트', '프런트엔드']
tags: [Next.js, React, App Router, Dashboard, AuthGuard]
original_url: "https://github.com/vmDeshpande/ai-agent-automation/tree/main/frontend/src"
excerpt: "frontend/src 기준으로 라우트 구조, 사이드바 내비게이션, 인증 가드, API 호출 훅을 정리합니다."
---

## 라우트 구조

`frontend/src/app`는 App Router 기반 페이지를 제공합니다.

- `/dashboard`, `/workflows`, `/tasks`, `/schedules`, `/agents`, `/logs`, `/settings`
- 로그인/회원가입: `/login`, `/register`
- 상세/빌더: `/workflows/[id]/...`, `/tasks/[id]`

`app/page.tsx`는 `/dashboard`로 리다이렉트합니다.

---

## 공통 레이아웃

`client-layout.tsx`에서 전역 Provider를 묶습니다.

- AuthProvider
- SettingsProvider
- AssistantProvider
- ThemeProvider

로그인/회원가입은 public route로 처리하고, 나머지는 `AuthGuard`로 보호합니다.

---

## API 호출 패턴

두 가지 패턴이 공존합니다.

1. `useApi()` 훅 기반 호출
2. 페이지 내부 `fetch("http://localhost:5000/api/..." )` 직접 호출

토큰은 `localStorage`에서 꺼내 `Authorization` 헤더로 전달합니다.

---

## 화면 구성 포인트

- `AppSidebar`가 섹션 네비게이션 담당
- Dashboard는 stats + recent activity 중심
- Workflow Builder는 step 편집 + 저장
- Task Detail은 step timeline/output inspector

다음 장에서 인앱 어시스턴트가 이 화면 컨텍스트를 어떻게 활용하는지 봅니다.

