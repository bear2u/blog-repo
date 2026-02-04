---
layout: post
title: "Claude Code 2.0 가이드 (6) - 명령어와 커스텀 명령어"
date: 2025-12-28
author: Sankalp
category: AI
tags: [Claude Code, AI, 명령어, 커스텀 명령어, 워크플로우]
series: claude-code-2-guide
part: 6
original_url: "https://sankalp.bearblog.dev/my-experience-with-claude-code-20-and-how-to-get-better-at-using-coding-agents/"
excerpt: "Claude Code의 내장 슬래시 명령어와 커스텀 명령어를 만들고 활용하는 방법을 알아봅니다."
---

## 기능 심층 탐구

다음 몇 서브 섹션은 가장 많이 사용하는 기능들에 대한 것입니다.

---

## 명령어 (Commands)

이전 블로그 포스트에서 명령어를 제대로 다루지 않았습니다. `/`를 사용하여 내장 슬래시 명령어에 접근할 수 있습니다. 이것들은 특정 작업을 수행하는 **미리 정의된 프롬프트**입니다.

명령어를 입력하면, 그 프롬프트가 현재 대화/컨텍스트에 추가되고 메인 에이전트가 작업을 수행하기 시작합니다.

---

## 내장 슬래시 명령어

| 명령어 | 설명 |
|--------|------|
| `/context` | 현재 컨텍스트 사용량 확인 |
| `/compact` | 컨텍스트 압축 |
| `/clear` | 새 대화 시작 |
| `/resume` | 이전 대화 재개 |
| `/rewind` | 특정 체크포인트로 복원 |
| `/ultrathink` | 심층 분석 모드 |
| `/plan` | 계획 모드 |
| `/usage` | 사용량 확인 |
| `/stats` | 통계 확인 |

---

## 커스텀 명령어 만들기

특정 작업이 이것들로 커버되지 않으면, **커스텀 명령어**를 만들 수 있습니다.

명령어는 **프로젝트 레벨** 또는 **글로벌 레벨**로 만들 수 있습니다:

```
프로젝트 레벨: .claude/commands/
글로벌 레벨: ~/.claude/commands/
```

---

## /handoff 명령어 예시

컨텍스트 윈도우가 차기 시작하거나 모델이 복잡한 작업에서 어려워한다고 느낄 때, `/clear`로 새 대화를 시작하고 싶습니다. Claude는 `/compact`를 제공하고 CC 2.0에서 더 빠르게 실행되지만, 때때로 죽이기 전에 현재 세션에서 일어난 일(특정 내용과 함께)을 Claude에게 작성하게 하는 것을 선호합니다. 이를 위해 `/handoff` 명령어를 만들었습니다.

```markdown
# .claude/commands/handoff.md 예시
현재 세션에서 수행한 작업을 요약하고,
다음 세션에서 계속할 수 있도록 중요한 컨텍스트를
스크래치패드 파일에 기록해주세요.

포함할 내용:
- 완료한 작업
- 진행 중인 작업
- 다음 단계
- 중요한 결정 사항
- 관련 파일 목록
```

---

## 커스텀 명령어 작성 팁

반복적으로 무언가에 대한 프롬프트를 작성하고 지시가 정적/정확할 수 있다면, 커스텀 명령어를 만드는 것이 좋은 아이디어입니다.

**Claude에게 커스텀 명령어를 만들라고 말할 수 있습니다.** Claude는 방법을 알고 있습니다(또는 웹을 검색하고 claude-code-guide.md를 통해 알아낼 것입니다). 그러면 만들어줄 것입니다.

```
사용자: "코드 리뷰를 위한 커스텀 명령어를 만들어줘"
Claude: .claude/commands/review.md 파일을 생성합니다...
```

---

## 유용한 리소스

[awesome-claude-code](https://github.com/anthropics/awesome-claude-code)에서 많은 명령어, 훅, 스킬을 찾을 수 있습니다. 하지만 **직접 만들거나 필요할 때만 검색하는 것을 권장합니다.**

---

## bootstrap-repo 명령어 예시

저는 `bootstrap-repo`라는 명령어가 있는데, 10개의 병렬 서브 에이전트로 레포를 검색하여 포괄적인 문서를 만듭니다. 요즘은 거의 사용하지 않고, 너무 많은 병렬 서브 에이전트는 Claude Code 깜빡임 버그로 이어집니다 ㅋㅋ.

```
bootstrap-repo 실행 시:
├── 10개 Explore 서브 에이전트 병렬 실행
├── "running in background" 상태 표시
└── 포괄적인 레포 문서 생성
```

어쨌든, "Explore" 서브 에이전트와 "running in background"를 주목하세요.

---

*다음 글에서는 서브 에이전트(Sub-agents)에 대해 자세히 알아봅니다.*
