---
layout: post
title: "Claude Code System Prompts 가이드 (05) - Builtin Tool Description: 도구 설명이 UX를 만든다"
date: 2026-02-15
permalink: /claude-code-system-prompts-guide-05-builtin-tool-descriptions/
author: Piebald AI
categories: [AI 코딩 에이전트, Claude Code]
tags: [Claude Code, Tools, Bash, ReadFile, Edit, Write, Task, Parallelism]
original_url: "https://github.com/Piebald-AI/claude-code-system-prompts"
excerpt: "Bash/Task 등 내장 툴 설명에 들어 있는 사용 금지/권장/병렬화 규칙을 통해 Claude Code의 도구 UX를 해부합니다."
---

## Tool Description은 “도구 계약(Contract)”이다

Claude Code에서 내장 도구는 단순한 API 호출이 아니라, **시스템 프롬프트 레벨에서 매우 구체적인 사용 규칙**을 가집니다.

그 규칙이 가장 노골적으로 드러나는 곳이 `system-prompts/tool-description-*.md`입니다.

이 장에서는 특히 두 축을 봅니다.

- `tool-description-bash.md`: “셸 실행”의 위험과 범위를 어떻게 통제하는가
- `tool-description-task.md`: “서브에이전트 실행”을 어떤 상황에서 쓰라고(혹은 쓰지 말라고) 하는가

---

## Bash 도구: “터미널 작업만, 파일 작업은 전용 툴로”

`tool-description-bash.md`의 핵심 메시지는 명확합니다.

- Bash는 `git`, `npm`, `docker` 같은 “진짜 터미널 작업”에 쓰라
- `find/grep/cat/head/tail/sed/awk/echo`로 파일 작업을 하지 말라
- 파일 탐색/검색/읽기/편집/작성에는 각각 전용 툴을 쓰라

이 정책은 “편의”가 아니라, 제품 UX 설계에 가깝습니다.

- 전용 툴은 출력이 구조화돼 있어 모델이 추론하기 쉽고
- 보안/권한/로그 정책을 도구 단위로 적용하기 쉽고
- 사용자가 “무엇이 실행되는지”를 더 명확히 추적할 수 있습니다.

---

## 병렬 도구 호출: “독립이면 한 번에 묶어라”

Tool Description/정책 프롬프트에는 병렬화 규칙이 자주 등장합니다.

대표 패턴:

- 서로 의존성이 없는 명령/툴 호출은 병렬로 한 번에 실행한다
- 의존성이 있는 경우에만 순차 실행(예: `mkdir` 후 `cp`)

이 규칙은 Claude Code의 “턴 당 처리량”을 높이고, 사용자 체감 지연을 줄이기 위한 최적화입니다.

---

## Task 도구: “언제 서브에이전트를 띄워야 하는가”

`tool-description-task.md`는 Task 도구(서브에이전트 실행)에 대해 “언제 쓰지 말아야 하는지”를 먼저 나열합니다.

요지는 이렇습니다.

- 특정 파일 경로를 읽고 싶다면: Task 말고 Read/Glob을 써라
- 특정 클래스 정의/문구를 찾고 싶다면: Task 말고 Glob/Grep을 써라
- 2~3개 파일 내에서만 찾고 싶다면: Task 말고 Read를 써라

Task는 “복잡하고 넓은 탐색”이나 “역할 분리가 필요한 작업”에 쓰는 도구입니다.

예를 들어:

- 거대한 레포에서 아키텍처를 빠르게 파악해야 한다
- 보안 리뷰처럼 별도의 관점/체크리스트가 필요하다
- 복수의 탐색을 병렬로 처리해야 한다

같은 상황에서 가치가 커집니다.

---

## 서브에이전트 프롬프트는 또 따로 있다

재미있는 점은, Task 도구 설명(`tool-description-task.md`)과 별개로,

- Task로 실행되는 서브에이전트가 어떤 시스템 프롬프트를 받는지는
- `agent-prompt-task-tool.md` 같은 파일로 따로 존재한다는 점입니다.

즉, “도구 계약”과 “도구가 생성하는 에이전트의 인격”이 분리돼 있습니다.

이 분리는 Claude Code가:

- 메인 모델은 사용자와 대화하며 전체를 조율하고
- 특정 역할의 서브에이전트는 제한된 규칙으로 빠르게 탐색/검증하고
- 결과를 메인 모델이 통합하는

구조를 갖고 있음을 시사합니다.

---

## 실전 감각: Tool Description을 읽으면 무엇이 달라지나

Tool Description을 이해하면, Claude Code 사용자가 실전에서 얻는 이점이 있습니다.

- “왜 이 도구를 쓰라/쓰지 말라 하는지”를 논리로 이해한다
- 에이전트에게 원하는 행동을 더 정확히 유도한다
  - 예: “이건 Read로 충분하니 Task 띄우지 말고 파일 3개만 읽어” 같은 지시
- 권한/안전 정책의 근거를 문서로 확인할 수 있다

그리고 무엇보다, Claude Code의 프롬프트 설계는 “추상적 지침”이 아니라

```text
도구 선택 규칙 + 안전 규칙 + 출력 포맷 규칙
```

의 조합으로 구현돼 있다는 걸 체감하게 됩니다.

---

## 다음 장 예고: System Reminders가 ‘상태 머신’을 만든다

Tool Description이 “도구 계약”이라면, System Reminders는 “상태 변화 이벤트”입니다.

Plan Mode 진입/종료, 파일 열기/잘림, 훅 실행 결과 같은 이벤트가 발생할 때,

- 사용자가 보지 못하는 시스템 메시지
- 혹은 UI에 표시되는 경고/알림 문구

가 “리마인더 프롬프트”로 주입됩니다.

6장에서는 이 System Reminders를 중심으로:

- Plan Mode가 왜 그렇게 동작하는지
- 파일 관련 경고가 어떤 의미인지
- 훅/자동화가 어떤 프로토콜로 연결되는지

를 정리합니다.

---

*다음 글에서는 `system-reminder-plan-mode-is-active-*.md`를 시작으로, ‘런타임 상태 머신’ 관점에서 Claude Code를 봅니다.*

