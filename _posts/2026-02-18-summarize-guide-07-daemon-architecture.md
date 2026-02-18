---
layout: post
title: "Summarize 가이드 (07) - 로컬 데몬 아키텍처: install/run/status와 SSE API"
date: 2026-02-18
permalink: /summarize-guide-07-daemon-architecture/
author: steipete
categories: ['개발 도구', '백엔드']
tags: [Daemon, SSE, Localhost, Token Auth, Chrome Extension]
original_url: "https://github.com/steipete/summarize/blob/main/src/daemon/server.ts"
excerpt: "src/daemon/*와 docs/chrome-extension.md를 기준으로 로컬 daemon의 설치 방식, 인증, 스트리밍 API를 정리합니다."
---

## 왜 daemon이 필요한가

확장은 브라우저 샌드박스 제약이 있어 무거운 추출/전사 작업을 직접 처리하기 어렵습니다.

그래서 Summarize는 로컬 daemon이 다음을 담당합니다.

- URL/미디어 추출
- 모델 호출
- SSE 스트리밍
- 장기 실행 프로세스 관리

---

## 설치 명령 계열

daemon CLI는 `src/daemon/cli.ts`에서 플랫폼별 서비스 설치를 처리합니다.

- `summarize daemon install --token <TOKEN>`
- `summarize daemon status`
- `summarize daemon restart`
- `summarize daemon uninstall`
- `summarize daemon run` (foreground)

플랫폼별 백그라운드 등록:

- macOS: launchd
- Linux: systemd user
- Windows: Scheduled Task

---

## 인증/보안 모델

`docs/chrome-extension.md` 기준 원칙:

- daemon은 `127.0.0.1` 바인딩
- API는 Bearer token 인증
- extension과 daemon이 같은 token을 공유

공개 네트워크 노출이 아니라 로컬 연동에 최적화된 모델입니다.

---

## 주요 엔드포인트

- `GET /health`
- `POST /v1/summarize`
- `GET /v1/summarize/:id/events` (SSE)
- `POST /v1/agent` (SSE/JSON)
- `POST /v1/agent/history`

즉 확장은 요청 생성/표시를 맡고, 실연산은 daemon이 담당합니다.

다음 장에서 확장 UX와 패널 데이터 흐름을 이어서 봅니다.

