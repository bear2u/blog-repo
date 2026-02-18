---
layout: post
title: "oh-my-codex 가이드 (05) - Team 모드 런타임: tmux 워커 오케스트레이션"
date: 2026-02-18
permalink: /oh-my-codex-guide-05-team-mode-runtime-and-tmux/
author: Yeachan Heo
categories: ['AI 코딩', '멀티 에이전트']
tags: [Team Mode, tmux, Worker, Orchestrator, OMX]
original_url: "https://github.com/Yeachan-Heo/oh-my-codex"
excerpt: "src/cli/team.ts와 src/team/runtime.ts를 기준으로 OMX team 명령의 파싱, 세션 생성, 워커 부트스트랩을 분석합니다."
---

## team 명령 파싱

`team.ts`는 기본 문법을 `omx team [ralph] [N:agent-type] "task"`로 해석합니다.

- 워커 수 범위: 1~20
- 기본 워커 수: 3
- 기본 agent type: `executor`

task 문자열을 slugify해 팀명을 자동 생성하는 패턴이 포함됩니다.

---

## 상태 모델 연결

팀 시작 시 `ensureTeamModeState()`가 mode state를 갱신합니다.

- `current_phase = team-exec`
- `agent_count`, `agent_types`
- `linked_ralph`

즉 CLI 레벨 입력이 상태 파일로 즉시 반영됩니다.

---

## runtime 시작 조건

`startTeam()` 핵심 가드:

1. nested team 금지(`OMX_TEAM_WORKER` 확인)
2. tmux 필수
3. 리더도 tmux 안에서 실행 중이어야 함
4. 리더 세션 기준 중복 active team 방지

운영 중복과 세션 꼬임을 예방하는 제약이 명확합니다.

---

## 워커 런치

런타임은 워커별 오버레이 지시문을 생성하고, 모델 인자를 주입해 Codex 워커를 띄웁니다.

- 모델은 `.omx-config.json`의 `models.team` 우선
- 없으면 `gpt-5.3-codex` 기본값
- 워커 ready 대기 타임아웃 조절 가능

---

## 팀 라이프사이클

기본 흐름은 아래 커맨드로 관리됩니다.

```bash
omx team <N:agent> "task"
omx team status <team-name>
omx team resume <team-name>
omx team shutdown <team-name>
```

다음 장에서는 팀 상태 저장소와 MCP 툴 계약을 연결해서 봅니다.
