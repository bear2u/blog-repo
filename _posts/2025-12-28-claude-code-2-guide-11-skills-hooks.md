---
layout: post
title: "Claude Code 2.0 가이드 (11) - 스킬, 플러그인, 훅"
date: 2025-12-28
permalink: /claude-code-2-guide-11-skills-hooks/
author: Sankalp
category: AI
tags: [Claude Code, AI, 스킬, 플러그인, 훅, 자동화]
series: claude-code-2-guide
part: 11
original_url: "https://sankalp.bearblog.dev/my-experience-with-claude-code-20-and-how-to-get-better-at-using-coding-agents/"
excerpt: "스킬, 플러그인, 훅의 개념과 활용법, 그리고 이것들을 결합하여 워크플로우를 최적화하는 방법을 알아봅니다."
---

## 스킬과 플러그인

Anthropic은 최근 [Agent Skills](https://docs.anthropic.com/en/docs/claude-code/skills)를 도입했고, 이것은 최근 Codex에서도 채택되었습니다.

### 스킬이란?

스킬은 **SKILL.md 파일, 다른 참조 가능한 파일, 사용자 정의 작업을 수행하는 코드 스크립트가 포함된 폴더**입니다.

SKILL.md는 LLM이 어떤 스킬이 사용 가능한지 알 수 있는 일부 **메타데이터**를 포함합니다(메타데이터는 시스템 프롬프트에 추가됩니다). Claude가 스킬이 관련 있다고 느끼면, **도구 호출을 수행하여 스킬의 내용을 읽고** 1999년 매트릭스의 네오처럼 도메인 전문 지식을 다운로드합니다.

```
스킬 로딩 흐름:
1. Claude가 작업 분석
2. 관련 스킬 감지 (메타데이터 기반)
3. 도구 호출로 SKILL.md 읽기
4. 도메인 전문 지식 "다운로드"
5. 작업 수행
```

> "I know Kung Fu" - 스킬은 매트릭스(1999)의 네오처럼 온디맨드로 로드됩니다.

코드 스크립트에는 Claude가 사용할 수 있는 도구가 포함될 수 있습니다.

### 스킬의 장점

보통 도메인 전문 지식을 가르치려면, 그 모든 정보를 시스템 프롬프트에 작성하고 아마도 도구 호출 정의까지 해야 합니다. 스킬을 사용하면, 모델이 **온디맨드로 로드하기 때문에 그렇게 할 필요가 없습니다**. 이는 해당 지시가 항상 필요한지 확실하지 않을 때 특히 유용합니다.

---

## 플러그인 (Plugins)

플러그인은 **스킬, 슬래시 명령어, 서브 에이전트, 훅, MCP 서버를 단일 배포 가능한 단위로 번들링**하는 패키징 메커니즘입니다.

```
플러그인 구조:
my-plugin/
├── skills/
│   └── frontend-design/
│       └── SKILL.md
├── commands/
│   └── review.md
├── agents/
│   └── custom-agent.md
├── hooks/
│   └── on-stop.sh
└── mcp/
    └── config.json
```

`/plugins`를 통해 설치할 수 있고 충돌을 피하기 위해 **네임스페이스**가 지정됩니다(예: `/my-plugin:hello`).

`.claude/`의 독립 실행형 설정은 개인/프로젝트 특정 사용에 좋지만, 플러그인은 **프로젝트와 팀 간에 기능을 쉽게 공유**할 수 있게 합니다.

### frontend-design 플러그인

인기 있는 frontend-design 플러그인은 실제로 **스킬**입니다.

---

## 훅 (Hooks)

훅은 Claude Code와 Cursor에서 사용 가능합니다(Codex는 아직 구현하지 않음). 에이전트 루프 생명주기의 **특정 단계가 시작되거나 끝날 때 관찰**하고 **변경을 위해 전후에 bash 스크립트를 실행**할 수 있게 합니다.

### 훅 유형

| 훅 | 실행 시점 | 예시 사용 |
|----|----------|----------|
| `Stop` | Claude 응답 완료 후 | 알림 재생 |
| `UserPromptSubmit` | 처리 전 프롬프트 제출 시 | 입력 검증 |
| `PreToolUse` | 도구 사용 전 | 권한 확인 |
| `PostToolUse` | 도구 사용 후 | 결과 로깅 |

### 첫 훅 예시

제가 만든 첫 번째 훅은 **Claude가 응답을 끝냈을 때 애니메이션 알림 소리를 재생**하는 것이었습니다. 분명히 Cursor의 알림 소리에서 영감을 받았습니다.

### 재미있는 사용 사례

Claude를 몇 시간 동안 실행시키는 재미있는 사용 사례는 **Stop 훅을 통해 Claude가 현재 작업을 마치면 "Do more" 프롬프트를 실행**하는 것일 수 있습니다.

```bash
# .claude/hooks/on-stop.sh 예시
#!/bin/bash
echo "Do more" | claude-code --continue
```

---

## 훅, 스킬, 리마인더 결합하기

이 블로그 포스트를 위한 리서치 중에 이 포스트를 발견했습니다. 이 분은 지금까지 논의한 개념과 기능을 아름답게 결합했습니다.

**훅을 사용하여 모델에게 스킬에 대해 상기**시킵니다. 유틸리티/요구사항이 발생하면, 커스터마이징을 위한 많은 공간이 있습니다. 이렇게 무거운 커스터마이징이 필요하지 않을 수 있지만, 최소한 영감을 얻을 수 있습니다. (저 자신을 위해 말하는 것입니다 ㅋㅋ)

### 통합 패턴 예시

```
┌─────────────────────────────────────────┐
│           통합 워크플로우                │
│                                         │
│  훅 (UserPromptSubmit)                  │
│    └── 스킬 가용성 확인                  │
│          └── 관련 스킬 로드              │
│                └── 시스템 리마인더 주입   │
│                      └── 작업 수행       │
└─────────────────────────────────────────┘
```

### CLAUDE.md 크기 줄이기

Anthropic은 **skill.md를 500줄 이하로 유지**할 것을 권장하므로, 이 분은 별도의 파일로 나누고 훅과 결합하여 **CLAUDE.md의 크기를 줄였습니다**.

```
지시 분리:
├── CLAUDE.md (간소화)
├── skills/
│   ├── coding-standards/
│   │   └── SKILL.md
│   ├── testing/
│   │   └── SKILL.md
│   └── documentation/
│       └── SKILL.md
└── hooks/
    └── remind-skills.sh
```

---

*다음 글에서는 결론과 참고 자료를 정리합니다.*
