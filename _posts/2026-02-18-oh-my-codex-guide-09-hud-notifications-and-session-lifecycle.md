---
layout: post
title: "oh-my-codex 가이드 (09) - HUD/알림/세션 라이프사이클: 운영 가시성 확보"
date: 2026-02-18
permalink: /oh-my-codex-guide-09-hud-notifications-and-session-lifecycle/
author: Yeachan Heo
categories: ['AI 코딩', '운영 관측성']
tags: [HUD, Notifications, Session Lifecycle, tmux, OMX]
original_url: "https://github.com/Yeachan-Heo/oh-my-codex"
excerpt: "hud/index.ts와 notifications 모듈을 기반으로 OMX의 런타임 관측/알림 체계를 정리합니다."
---

## HUD 명령

`omx hud`는 상태를 텍스트 HUD로 보여줍니다.

주요 옵션:

- `--watch`: 1초 주기 갱신
- `--json`: 원시 상태 출력
- `--preset=minimal|focused|full`
- `--tmux`: 분할 pane 자동 실행

---

## tmux 안전성

HUD tmux 실행은 `execFileSync('tmux', args)` 방식으로 shell injection을 줄인 구현입니다.

또한 preset 값을 허용 목록으로 제한해 임의 문자열 유입을 막습니다.

---

## 알림 시스템

`notifications/notifier.ts`는 3채널을 지원합니다.

- Desktop
- Discord webhook
- Telegram bot

설정은 `.omx/notifications.json`을 읽고, 채널별 전송은 best-effort(`Promise.allSettled`)로 처리합니다.

---

## 세션 라이프사이클

CLI 런치 경로는 preLaunch/run/postLaunch를 분리하고, hooks/session 모듈과 연결됩니다.

이 구조 덕분에 시작/종료 지점에서 상태 기록, 훅 dispatch, 정리 작업을 일관되게 수행할 수 있습니다.

---

## 운영 제안

- HUD는 리더 pane 하단 고정(`--tmux`)으로 운영
- 팀/훅 로그를 일자 단위 수집
- Discord/Telegram은 실패 허용이므로 별도 dead-letter 모니터링 고려

다음 장에서 전체 운영 체크리스트와 트러블슈팅을 마무리합니다.
