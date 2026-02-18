---
layout: post
title: "oh-my-codex 가이드 (08) - Config/모델 라우팅: config.toml 병합과 reasoning 제어"
date: 2026-02-18
permalink: /oh-my-codex-guide-08-config-model-routing-and-reasoning/
author: Yeachan Heo
categories: ['AI 코딩', '설정 관리']
tags: [config.toml, model routing, reasoning, MCP servers, OMX]
original_url: "https://github.com/Yeachan-Heo/oh-my-codex"
excerpt: "config/generator.ts와 models.ts를 기준으로 OMX 설정 병합 규칙과 모드별 모델 선택 전략을 정리합니다."
---

## `mergeConfig()`의 목표

사용자 설정을 지우지 않으면서 OMX가 관리하는 키만 안정적으로 업서트하는 것입니다.

핵심 처리:

- OMX 소유 top-level 키 strip 후 재삽입
- `[features]` 플래그 보정
- OMX 블록(`[mcp_servers.*]`, `[tui]`) 갱신

---

## 삽입되는 MCP 서버

설정 생성 코드 기준 서버 4개:

1. `omx_state`
2. `omx_memory`
3. `omx_code_intel`
4. `omx_trace`

그리고 `[tui] status_line`도 함께 설정됩니다.

---

## 모델 선택 규칙

`getModelForMode(mode)` 해석 순서:

1. `~/.codex/.omx-config.json`의 `models[mode]`
2. `models.default`
3. 하드코딩 기본값 `gpt-5.3-codex`

즉 team, autopilot 같은 모드별 모델 라우팅을 파일로 분리할 수 있습니다.

---

## reasoning 제어

`omx reasoning` 명령은 `model_reasoning_effort`를 직접 설정/조회합니다.

운영 시 자주 쓰는 조합:

- 기본 고정: `high`
- 난도 높은 리팩터/분석: `xhigh`
- 비용 절감 런: `medium`

---

## 실무 팁

config를 여러 도구가 동시에 수정하는 환경에서는 `omx setup --force` 전에 백업 diff를 남기는 것이 안전합니다.

특히 MCP 서버 블록 순서/중복을 CI에서 lint로 검증하면 장애를 줄일 수 있습니다.

다음 장에서는 HUD/알림/세션 수명주기를 다룹니다.
