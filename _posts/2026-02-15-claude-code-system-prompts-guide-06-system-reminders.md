---
layout: post
title: "Claude Code System Prompts 가이드 (06) - System Reminders: Plan Mode와 런타임 이벤트가 만드는 규칙"
date: 2026-02-15
permalink: /claude-code-system-prompts-guide-06-system-reminders/
author: Piebald AI
categories: [AI 코딩 에이전트, Claude Code]
tags: [Claude Code, System Reminders, Plan Mode, Hooks, Notifications, Safety]
original_url: "https://github.com/Piebald-AI/claude-code-system-prompts"
excerpt: "Plan Mode 활성화/재진입, 파일 이벤트, 훅 이벤트 등 런타임 상태 변화가 System Reminder로 어떻게 주입되는지 정리합니다."
---

## System Reminder란 무엇인가

`system-prompts/system-reminder-*.md`는 “Claude Code가 런타임 이벤트에 반응해 모델에게 보내는 큰 알림 텍스트”를 모아둔 것입니다.

중요한 점:

- 이건 “항상 붙는 프롬프트”가 아니라
- 특정 이벤트에서만 주입되는 “상태 머신의 신호”에 가깝습니다.

그래서 System Reminder를 보면, Claude Code가 어떤 상태를 갖고 있고 어떤 순간에 행동을 바꾸는지 역으로 추론할 수 있습니다.

---

## Plan Mode: 가장 영향력이 큰 Reminder

대표 파일:

- `system-reminder-plan-mode-is-active-iterative.md`
- `system-reminder-plan-mode-is-active-5-phase.md`
- `system-reminder-exited-plan-mode.md`
- `system-reminder-plan-mode-re-entry.md`

특히 “Plan mode is active” 류는 강력한 제약을 선언합니다.

- Plan 모드에서는 실행/편집/커밋 같은 비-읽기 전용 도구를 쓰지 말라
- (예외로) 지정된 plan 파일만 편집 가능
- 계획은 “Explore → plan 업데이트 → 질문” 루프를 반복하라
- 승인은 텍스트로 묻지 말고 `ExitPlanMode` 툴을 호출하라

이걸 보면 Plan Mode가 단순 UI 토글이 아니라:

1. 도구 사용 권한을 바꾸고
2. 작업 흐름을 강제하며
3. 사용자 인터뷰를 포함한 “pair planning 프로토콜”을 요구하는

**모드 전환**이라는 걸 알 수 있습니다.

---

## 파일/IDE 이벤트: “컨텍스트 경계”를 관리한다

레포에는 다음처럼 파일 관련 Reminder가 여러 개 있습니다.

- 파일이 비어 있음: `system-reminder-file-exists-but-empty.md`
- 파일이 잘림(truncated): `system-reminder-file-truncated.md`
- IDE에서 파일을 열었음: `system-reminder-file-opened-in-ide.md`
- 요약(compaction) 전에 읽은 파일 참조: `system-reminder-compact-file-reference.md`

이 메시지들은 보통 두 목적을 가집니다.

1. 모델에게 “현재 컨텍스트가 불완전하다/잘렸다/변경됐다”는 사실을 명시
2. 사용자에게 “왜 지금 이 경고가 뜨는지”를 설명

특히 large file truncation은, 모델이 “파일 전체를 읽었다”고 착각하는 사고를 막기 위한 안전장치입니다.

---

## Hooks/자동화 이벤트: ‘설정’이 아니라 ‘프로토콜’이다

Hook 관련 Reminder 예:

- `system-reminder-hook-additional-context.md`
- `system-reminder-hook-blocking-error.md`
- `system-reminder-hook-success.md`
- `system-reminder-hook-stopped-continuation*.md`

이 텍스트를 보면, Claude Code의 Hooks는 단순히 “명령을 실행한다”가 아니라:

- 어떤 이벤트에서 실행되고(PreToolUse/PostToolUse/Stop 등)
- 어떤 경우에 흐름이 막히고(blocking error)
- 어떤 경우에 대화가 이어지지 못하는지(continuation stopped)

까지 모델이 인지하도록 설계돼 있습니다.

Hooks는 “개인 설정”처럼 보이지만, 실제로는 모델과 런타임 사이의 **자동화 프로토콜**입니다.

---

## “왜 이런 리마인더가 중요하나”

System Reminders는 사용자가 체감하는 Claude Code의 “이상한 규칙”을 설명하는 열쇠인 경우가 많습니다.

예:

- Plan Mode에서 편집을 거부하는 이유
- 파일이 일부만 읽혔을 때 답변 품질이 떨어지는 이유
- 훅이 실패했는데도 모델이 계속 진행하려는 것처럼 보이는 이유(혹은 반대로 멈추는 이유)

이런 현상은 “모델 성능” 문제가 아니라 “시스템 메시지로 주입된 제약”일 수 있습니다.

---

## 다음 장 예고: Agent Prompts로 역할 분리 보기

System Reminder가 “상태 이벤트”라면, Agent Prompt는 “역할 분리된 작업자”입니다.

7장에서는:

- Explore 서브에이전트가 왜 read-only로 강제되는지
- Task 도구로 띄운 서브에이전트가 어떤 규칙을 받는지
- Plan 모드와 서브에이전트가 어떻게 연결되는지

를 `agent-prompt-*.md` 파일들을 중심으로 정리합니다.

---

*다음 글에서는 `agent-prompt-explore.md`, `agent-prompt-task-tool.md` 등을 통해 Claude Code의 서브에이전트 구조를 살펴봅니다.*

