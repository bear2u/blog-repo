---
layout: post
title: "Claude Code System Prompts 가이드 (04) - 메인 System Prompt 해부: 정체성, 톤, 안전, 학습모드"
date: 2026-02-15
permalink: /claude-code-system-prompts-guide-04-system-prompt-anatomy/
author: Piebald AI
categories: [AI 코딩 에이전트, Claude Code]
tags: [Claude Code, System Prompt, Tone, Safety, Learning Mode, Policies]
original_url: "https://github.com/Piebald-AI/claude-code-system-prompts"
excerpt: "Main system prompt, tone/style, 안전 가드레일, learning mode 조각을 묶어서 Claude Code의 기본 동작 원리를 설명합니다."
---

## “메인 시스템 프롬프트”는 정말 짧다

`system-prompts/system-prompt-main-system-prompt.md`를 보면, 의외로 핵심 정체성 선언은 짧습니다.

포인트는 두 가지입니다.

1. Claude Code는 “대화형 CLI 도구”이며, **소프트웨어 엔지니어링 작업을 돕는다**
2. 나머지 규칙은 `${...}` 변수로 분리돼 **조건부로 조립**된다

예를 들어 파일에는 이런 구조가 그대로 남습니다.

```text
You are an interactive CLI tool ...
${SECURITY_POLICY}
IMPORTANT: You must NEVER generate or guess URLs ...
```

즉, “핵심 문장 + 정책 조각(SECURITY_POLICY)” 조합입니다.

---

## 톤/스타일은 별도 프롬프트로 분리돼 있다

`system-prompts/system-prompt-tone-and-style.md`는 제품의 UX를 만드는 규칙을 명시합니다.

자주 체감되는 규칙들:

- 사용자가 요구하지 않으면 이모지를 쓰지 않는다
- CLI에서 읽기 쉬운 짧고 단정한 출력(마크다운 허용)
- “도구 호출 앞에 콜론을 붙이지 말라” 같은 UI/렌더링 제약
- 불필요한 과잉 칭찬/감정적 검증을 피하고, 사실/문제해결 중심
- 시간 예측(“몇 분 걸림”)을 하지 않는다

여기서 중요한 점은, 이런 규칙이 “모델의 성격”이 아니라 **시스템 프롬프트의 구성 요소**라는 겁니다.

---

## “조심해서 실행하라”는 안전 레이어다

`system-prompts/system-prompt-executing-actions-with-care.md`는 위험한 행동의 기본 정책을 제공합니다.

핵심은 “되돌리기 어려운 행동은 확인하고 진행”입니다.

- push/force-push, 브랜치 삭제, 파일 삭제 같은 파괴적 작업
- CI/CD 수정, 권한/인프라 변경 같은 공유 상태 작업

등은 기본적으로 사용자 확인을 요구하도록 가이드합니다.

이 프롬프트를 알고 있으면, Claude Code가 왜 자꾸 “확인 질문”을 하는지 이해가 됩니다. 그건 모델의 소심함이 아니라 “안전 설계”입니다.

---

## Learning mode는 ‘협업 프로토콜’을 강제한다

`system-prompts/system-prompt-learning-mode.md`는 이름 그대로 “학습 모드”에서의 행동을 정의합니다.

흥미로운 점:

- 사용자가 직접 2~10줄 정도의 코드를 작성하게 “과제를 던지는” 규칙이 들어 있습니다.
- 그 규칙은 “TODO(human)을 먼저 코드에 넣고, 그 다음 Learn by Doing 요청을 하라”처럼 매우 구체적입니다.
- TodoList(작업 추적)과도 연결돼, 학습 요청을 “작업 항목”으로 남기도록 유도합니다.

이걸 보면 Claude Code는 단순 채팅이 아니라:

1. 에이전트가 구현을 진행하고
2. 인간이 특정 핵심 결정을 기여하고
3. 다시 에이전트가 통합하는

**협업 워크플로우**를 시스템 프롬프트로 설계한 제품이라는 게 드러납니다.

---

## 메인 프롬프트의 “구성 단위”를 이렇게 보면 편하다

이 레포를 읽을 때 추천하는 사고방식은:

- `system-prompt-main-system-prompt.md`: 제품의 “정체성 선언”
- `system-prompt-tone-and-style.md`: 출력/대화 스타일 UX
- `system-prompt-tool-usage-policy.md`: 도구 호출 계약(앞에서 했던/다음에 할 것)
- `system-prompt-executing-actions-with-care.md`: 위험 행동에 대한 안전 가드레일
- `system-prompt-learning-mode*.md`: 학습/교육 협업 프로토콜

즉, “한 덩어리”가 아니라 “레이어”로 보자는 겁니다.

---

## 다음 장 예고: Tool Description으로 넘어가자

System Prompt가 “지침의 레이어”라면, Tool Description은 “실행 계약”입니다.

Claude Code는 `Bash`, `ReadFile`, `Edit`, `Write`, `Task` 같은 도구의 설명 자체가 길고 정교해서:

- 어떤 도구를 언제 쓰고
- 어떤 도구는 절대 쓰지 말고
- 병렬 호출은 어떻게 하며
- 파일 작업은 왜 전용 툴을 쓰라고 강제하는지

가 Tool Description에 박혀 있습니다.

5장에서는 Builtin Tool Description을 중심으로 “도구 사용 정책이 어떻게 프롬프트 조각으로 구현되는지”를 봅니다.

---

*다음 글에서는 `tool-description-bash.md`, `tool-description-task.md`, 그리고 `system-prompt-tool-usage-policy.md`를 엮어 ‘도구 계약’ 관점에서 Claude Code를 해부합니다.*

