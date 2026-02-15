---
layout: post
title: "Claude Code System Prompts 가이드 (01) - 소개: 시스템 프롬프트는 하나가 아니다"
date: 2026-02-15
permalink: /claude-code-system-prompts-guide-01-intro/
author: Piebald AI
categories: [AI 코딩 에이전트, Claude Code]
tags: [Claude Code, System Prompts, PiebaldAI, Prompt Engineering, Tool Descriptions, System Reminders]
original_url: "https://github.com/Piebald-AI/claude-code-system-prompts"
excerpt: "Claude Code의 시스템 프롬프트가 단일 문자열이 아니라 110+ 조각으로 구성되는 이유와, 이 레포가 제공하는 가치를 정리합니다."
---

## 이 레포는 무엇을 제공하나

`Piebald-AI/claude-code-system-prompts`는 Anthropic이 공개하지 않는 Claude Code의 소스(거대한 번들 JS)에서 **시스템 프롬프트 관련 문자열을 추출**해, 다음을 “파일 단위 카탈로그”로 제공합니다.

- **System Prompt**: 메인 정체성/정책/작업 가이드라인 조각들
- **Builtin Tool Descriptions**: `Bash`, `ReadFile`, `Edit`, `Write`, `Task` 같은 내장 도구 설명
- **System Reminders**: Plan Mode 진입/종료, 파일 이벤트, 훅 이벤트 등 런타임 이벤트 알림
- **Agent Prompts**: Explore/Plan 같은 서브에이전트, 요약/분석 유틸리티용 프롬프트
- **CHANGELOG**: 2.0.14부터 수십~수백 버전의 변화 내역(토큰 증감 포함)

핵심은 “Claude Code의 프롬프트가 **업데이트마다** 바뀌는데, 우리가 그걸 원문 그대로 추적하기가 너무 어렵다”는 문제를 해결한다는 점입니다.

---

## 왜 ‘시스템 프롬프트’가 여러 개인가

Claude Code는 단순히 “한 줄짜리 System Prompt”만 붙는 제품이 아닙니다. README에서 강조하는 포인트는 다음과 같습니다.

- 환경/설정/기능 플래그에 따라 **조건부로** 시스템 프롬프트 일부가 추가된다
- 내장 도구마다 **Tool Description**이 붙고, 어떤 것들은 길다(토큰을 크게 먹는다)
- Explore/Plan 같은 서브에이전트/유틸리티는 **별도 system prompt**를 가진다
- 대화 압축(compaction), 세션 타이틀 생성, CLAUDE.md 생성 등 “AI 유틸리티 함수”도 프롬프트를 가진다

결과적으로 Claude Code 내부에는 “실제로는 110+ 문자열 조각”이 있고, 이것들이 번들 JS 속에서 계속 움직입니다.

---

## “왜 갑자기 이런 행동을 하지?”를 설명하는 단서

Claude Code를 쓰다 보면 아래 같은 체감이 생깁니다.

- Plan Mode에 들어가면 “왜 갑자기 편집을 안 하지?”
- 파일이 커서 잘렸을 때, “왜 저 경고가 뜨지?”
- 도구를 거절당했는데, “왜 다시 시도하면 안 된다고 하지?”
- 왜 `rg/cat/sed` 대신 전용 툴을 쓰라고 강제하지?

이 레포는 이런 질문에 대해 **정답에 가까운 근거**(실제 문구)를 제공합니다.

- Plan Mode의 제약은 `system-reminder-plan-mode-is-active-*.md` 같은 파일에 드러납니다.
- 도구 호출 규칙은 `system-prompt-tool-usage-policy.md` 같은 정책 프롬프트에 들어 있습니다.
- “행동을 조심하라” 같은 안전 규칙은 `system-prompt-executing-actions-with-care.md`에 따로 분리돼 있습니다.

---

## 이 가이드의 목표

이 시리즈는 “프롬프트 문장을 그대로 나열”하기보다는, 다음을 목표로 합니다.

1. **구조 파악**: 어떤 종류의 프롬프트 조각이 있고 어디에 있는지
2. **역할 이해**: Tool Description / Reminder / Agent Prompt가 각각 어떤 효과를 내는지
3. **변화 추적**: 버전 업 시 어떤 조각이 바뀌는지(CHANGELOG 읽는 법)
4. **현실적인 커스터마이징**: 로컬에서 일부 조각을 바꾸려면 어떻게 해야 하는지(`tweakcc` 소개)

이 레포 자체는 “참조(reference) 자료”에 가깝습니다. 여기 파일을 수정한다고 Claude Code 동작이 바뀌지는 않습니다(로컬 패치는 별도 도구가 필요).

---

## 빠른 길잡이: 다음 장에서 볼 것

- 2장: 파일 네이밍과 메타데이터(`ccVersion`, `variables`)로 “어떤 프롬프트 조각인지” 빠르게 분류하기
- 3장: `tools/updatePrompts.js`가 어떻게 README/토큰 카운트를 자동 갱신하는지

---

*다음 글에서는 `system-prompts/` 디렉토리의 파일 규칙(카테고리, 메타, 변수)을 먼저 잡고 갑니다.*

