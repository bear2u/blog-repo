---
layout: post
title: "oh-my-codex 가이드 (06) - Team 상태/MCP 연동: mailbox/task/state 계약"
date: 2026-02-18
permalink: /oh-my-codex-guide-06-team-state-and-mcp-ops/
author: Yeachan Heo
categories: ['AI 코딩', 'MCP']
tags: [MCP, State Server, Team Ops, Mailbox, Task Management]
original_url: "https://github.com/Yeachan-Heo/oh-my-codex"
excerpt: "state-server.ts와 team-ops.ts를 기준으로 OMX가 팀 상태를 MCP 도구로 노출하는 방법을 정리합니다."
---

## 상태 서버 역할

`src/mcp/state-server.ts`는 OMX 상태/팀 운영용 MCP 서버입니다.

기본 state 툴:

- `state_read`
- `state_write`
- `state_clear`
- `state_list_active`
- `state_get_status`

---

## 팀 커뮤니케이션 툴

같은 서버에서 팀 전용 툴도 노출합니다.

- 메시지: `team_send_message`, `team_broadcast`, `team_mailbox_list`
- 태스크: `team_create_task`, `team_claim_task`, `team_update_task`
- 워커 상태: heartbeat/status/read/write
- 승인/모니터 스냅샷/종료 제어 툴

즉 "상태 조회"와 "실행 제어"가 동일한 MCP 표면에서 연결됩니다.

---

## team-ops 게이트웨이 패턴

`src/team/team-ops.ts`는 대부분 함수를 `state.ts`에서 재export하는 게이트웨이입니다.

의도는 명확합니다.

- 런타임(runtime.ts)과 MCP 서버(state-server.ts)가 같은 계약을 참조
- 외부 MCP surface와 내부 실행 surface의 의미를 일치

---

## mailbox 전달 흐름

`mcp-comm.ts`는 inbox/직접메시지/브로드캐스트 큐를 다룹니다.

1. 상태 저장소에 메시지 기록
2. notify 콜백으로 워커 pane 트리거
3. 성공 시 notified 마킹

tmux 입력 이벤트와 파일 기반 상태를 함께 엮는 구조입니다.

---

## 운영 포인트

팀 모드 안정성은 "상태 파일 일관성 + tmux 이벤트 전달"의 결합 품질에 좌우됩니다.

그래서 `doctor --team`, monitor snapshot, shutdown ack 체크가 중요한 운영 루틴이 됩니다.

다음 장에서는 hooks 확장 런타임을 해설합니다.
