---
layout: post
title: "oh-my-codex 가이드 (07) - Hooks 확장 런타임: 플러그인 이벤트 파이프라인"
date: 2026-02-18
permalink: /oh-my-codex-guide-07-hooks-extension-runtime/
author: Yeachan Heo
categories: ['AI 코딩', '자동화']
tags: [Hooks, Plugin, Event Dispatch, OMX, Automation]
original_url: "https://github.com/Yeachan-Heo/oh-my-codex"
excerpt: "docs/hooks-extension.md와 extensibility 런타임 코드를 기준으로 OMX 훅 플러그인 활성화 모델과 안전장치를 설명합니다."
---

## 확장 표면

OMX는 `.omx/hooks/*.mjs` 경로의 사용자 플러그인을 지원합니다.

핵심 커맨드:

```bash
omx hooks init
omx hooks status
omx hooks validate
OMX_HOOK_PLUGINS=1 omx hooks test
```

---

## 활성화 모델

플러그인은 기본 비활성입니다.

```bash
export OMX_HOOK_PLUGINS=1
```

`dispatchHookEventRuntime()`는 이 플래그가 꺼져 있으면 즉시 `plugins_disabled` 결과를 반환합니다.

---

## 이벤트 파이프라인

문서 기준 native 이벤트:

- `session-start`
- `session-end`
- `turn-complete`
- `session-idle`

`dispatcher.ts`는 플러그인 탐색/검증 후 별도 runner 프로세스로 실행하고, 타임아웃(`OMX_HOOK_PLUGIN_TIMEOUT_MS`)을 적용합니다.

---

## 팀 안전성 가드

팀 워커(`OMX_TEAM_WORKER`)에서는 side effect를 기본 스킵합니다.

리더 세션만 외부 알림/연동을 대표로 수행하게 만들어, 중복 전송을 막는 설계입니다.

---

## 로그와 관측

플러그인 dispatch 결과는 `.omx/logs/hooks-YYYY-MM-DD.jsonl`에 누적됩니다.

실패 원인(타임아웃, invalid_export, runner_missing)을 구조화 로그로 남겨 운영 추적이 가능합니다.

다음 장에서는 설정 병합과 모델 라우팅을 봅니다.
