---
layout: post
title: "PicoClaw 가이드 (10) - 운영/트러블슈팅: cron/heartbeat, web search 설정, 채널 충돌"
date: 2026-02-15
permalink: /picoclaw-guide-10-ops-and-troubleshooting/
author: Sipeed
categories: [AI 에이전트, 개발 도구]
tags: ["PicoClaw", "Ops", "Cron", "Heartbeat", "Troubleshooting"]
original_url: "https://github.com/sipeed/picoclaw#-troubleshooting"
excerpt: "스케줄 작업(cron/heartbeat)과 gateway 운영에서 자주 만나는 문제를 README 기준으로 정리합니다. web search API 설정, 텔레그램 getUpdates 충돌, 필터링 이슈를 포함합니다."
---

## CLI 레퍼런스(핵심만)

README의 CLI 표에 나온 대표 커맨드:

- `picoclaw onboard`: config/workspace 초기화
- `picoclaw agent`: 대화형 또는 `-m` 단발 실행
- `picoclaw gateway`: 채널 봇 실행
- `picoclaw status`: 상태 확인
- `picoclaw cron ...`: 스케줄 작업 관리

---

## Scheduled Tasks: cron

README는 `cron` 도구로:

- “10분 뒤 알려줘” 같은 단발 리마인더
- “2시간마다 알려줘” 같은 반복 작업
- cron 표현식 기반 스케줄

을 지원한다고 설명합니다.

운영 팁:

- 처음에는 알림/리마인더 같은 “부작용이 적은 작업”부터 사용합니다.

---

## Heartbeat: 주기 작업

README는 워크스페이스에 `HEARTBEAT.md`를 만들면 주기적으로 읽는다고 안내합니다.

- 기본 30분(설정 가능)
- 오래 걸리는 작업은 subagent(spawn)로 비동기 처리 권장

이 기능은 편하지만, 동시에 “자동 실행”이므로 9장의 샌드박스/가드레일과 함께 켜는 편이 안전합니다.

---

## 트러블슈팅(README 기반)

### web search가 “API 설정 문제”라고 뜬다

README의 요지는:

- Brave Search 키를 설정하면 가장 좋은 결과
- 키가 없으면 DuckDuckGo로 자동 폴백하는 흐름이 있다는 점입니다(버전에 따라 설정 구조는 달라질 수 있음).

따라서 먼저 `tools.web` 설정과 키가 정상인지 확인합니다.

### 콘텐츠 필터링 오류

일부 프로바이더는 필터링이 있을 수 있어:

- 프롬프트를 바꾸거나
- 모델/프로바이더를 바꾸는

대응이 필요합니다.

### Telegram bot이 getUpdates 충돌을 낸다

README는 “다른 인스턴스가 동시에 실행 중”일 때 발생한다고 안내합니다.

- 동일 봇 토큰으로 `gateway`를 여러 개 띄우지 않도록 정리합니다.

---

## 마무리

PicoClaw는 “작고 빠른 개인 에이전트”를 목표로 하면서도, 채널/도구/스케줄링까지 범위를 넓히고 있습니다.

운영에서의 기본 전략은 단순합니다.

1. agent 모드로 최소 동작 확인
2. gateway로 채널 1개만 붙여 운영
3. 워크스페이스 제한/allow list 같은 방어선을 먼저 켠 뒤 확장

