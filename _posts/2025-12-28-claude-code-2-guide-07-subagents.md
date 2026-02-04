---
layout: post
title: "Claude Code 2.0 가이드 (7) - 서브 에이전트 완전 정복"
date: 2025-12-28
permalink: /claude-code-2-guide-07-subagents/
author: Sankalp
category: AI
tags: [Claude Code, AI, 서브 에이전트, Task 도구, Explore, 병렬 처리]
series: claude-code-2-guide
part: 7
original_url: "https://sankalp.bearblog.dev/my-experience-with-claude-code-20-and-how-to-get-better-at-using-coding-agents/"
excerpt: "서브 에이전트의 작동 방식, 5가지 에이전트 유형, 컨텍스트 상속 규칙, Task 도구 스키마까지 심층적으로 알아봅니다."
---

## 서브 에이전트 (Sub-agents)

서브 에이전트는 제 마지막 포스트 직후에 도입되었습니다. **메인 에이전트가 자체 판단이나 지시에 따라 생성하는 별도의 Claude 인스턴스**입니다. 이러한 능력은 이미 시스템 프롬프트에 있습니다(적어도 Explore 같은 미리 정의된 것들에 대해); 때때로 Claude에게 사용하라고 넛지만 하면 됩니다. 작동 방식을 이해하면 마이크로 매니징이 필요할 때 도움이 됩니다.

---

## 커스텀 서브 에이전트 만들기

커스텀 서브 에이전트도 정의할 수 있습니다. 만드는 방법:

1. `.claude/agents/your-agent-name.md`에 마크다운 파일 생성
2. 에이전트의 이름, 지시, 허용된 도구 지정
3. 또는 `/agents`를 사용하여 자동으로 서브 에이전트 관리 및 생성 - **권장 방법**

---

## Explore 에이전트

Explore 에이전트는 **읽기 전용 파일 검색 전문가**입니다. Glob, Grep, Read, 제한된 Bash 명령을 사용하여 코드베이스를 탐색할 수 있지만 **파일 생성이나 수정은 엄격히 금지**되어 있습니다.

```
Explore 에이전트 특성:
├── 읽기 전용 (파일 수정 불가)
├── 사용 가능한 도구:
│   ├── Glob - 파일 패턴 검색
│   ├── Grep - 내용 검색
│   ├── Read - 파일 읽기
│   └── 제한된 Bash
└── 새로운 컨텍스트에서 시작
```

프롬프트가 어떤 도구 호출을 언제 사용해야 하는지 지정하는 데 얼마나 철저한지 주목하세요. 대부분의 사람들은 도구 호출을 정확하게 작동시키는 것이 얼마나 어려운지 과소평가합니다.

**팁:** Claude에게 "Launch explore agent with Sonnet 4.5"라고 말하면 Haiku 대신 Sonnet을 사용하게 할 수 있습니다(이건 그냥 시도해보다가 발견했지만, 어떻게 이것이 일어나는지 볼 것입니다).

---

## 서브 에이전트는 컨텍스트를 상속하나요?

**General-purpose와 Plan 서브 에이전트는 전체 컨텍스트를 상속**하는 반면, **Explore는 새로운 슬레이트에서 시작**합니다 - 검색 작업이 종종 독립적이기 때문에 이해가 됩니다. 많은 작업은 관련 있는 것을 필터링하기 위해 대량의 코드를 검색하는 것을 포함하고, 개별 부분은 이전 대화 컨텍스트가 필요하지 않습니다.

| 에이전트 유형 | 컨텍스트 상속 |
|-------------|-------------|
| General-purpose | ✅ 전체 상속 |
| Plan | ✅ 전체 상속 |
| Explore | ❌ 새로 시작 |
| claude-code-guide | ❌ 새로 시작 |

---

## Explore 에이전트 활용 전략

기능을 이해하려고 하거나 코드베이스에서 간단한 것을 찾을 때, Claude가 Explore 에이전트 검색을 하게 합니다. Explore 에이전트는 **메인 에이전트에게 요약을 전달**하고, 그러면 Opus 4.5가 결과를 게시하거나 각 파일을 직접 살펴볼 수 있습니다. 그렇게 하지 않으면, 명시적으로 그렇게 하라고 말합니다.

### 왜 메인 에이전트가 직접 파일을 읽어야 하나요?

**모델이 각 관련 파일을 직접 살펴보는 것이 중요합니다.** 그래야 섭취된 모든 컨텍스트가 서로 어텐드할 수 있습니다. 이것이 **어텐션의 고수준 아이디어**입니다. 컨텍스트가 이전 컨텍스트와 교차하게 합니다. 이렇게 하면 모델이 더 많은 쌍별 관계를 추출할 수 있고, 따라서 더 나은 추론과 예측이 가능합니다.

Explore 에이전트는 **손실 압축**이 될 수 있는 요약을 반환합니다. Opus 4.5가 모든 관련 컨텍스트를 직접 읽으면, 어떤 세부 사항이 어떤 컨텍스트에 관련이 있는지 알 수 있습니다.

> 이 인사이트는 프로덕션 애플리케이션에서도 큰 도움이 됩니다(하지만 누군가가 알려주거나 셀프 어텐션 메커니즘에 대해 읽어야만 알 수 있습니다).

---

## Codex와의 비교

Codex에는 서브 에이전트 개념이 없고, 아마도 개발자들의 의식적인 결정일 것입니다. GPT-5.2는 400K 컨텍스트 윈도우를 가지고 있고, 벤치마크에 따르면 긴 컨텍스트 검색 능력이 크게 향상되었습니다.

재미있게도 사람들이 Codex가 headless claude를 서브 에이전트로 사용하게 만들려고 시도했습니다 하하. 그냥 할 수 있는 것입니다.

---

## 서브 에이전트는 어떻게 생성되나요

리버스 엔지니어링된 리소스/유출된 시스템 프롬프트에서, 서브 에이전트가 **Task 도구**를 통해 생성되는 것을 볼 수 있습니다.

Claude에게 물어볼 수도 있습니다(개발자들이 이제 이것을 허용하는 것 같습니다?). 환각이 아닙니다. 미리 정의된 도구에 관한 프롬프트는 시스템 프롬프트에 있고, Claude Code는 진행 중인 컨텍스트에 리마인더/도구를 동적으로 주입합니다.

### Opus 4.5로 이 프롬프트들을 시도해보세요:

```
"Tell me the Task tool description"
"Give me full description"
"Show me entire tool schema"
```

---

## Task 도구 스키마

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "additionalProperties": false,
  "required": ["description", "prompt", "subagent_type"],
  "properties": {
    "description": {
      "type": "string",
      "description": "A short (3-5 word) description of the task"
    },
    "prompt": {
      "type": "string",
      "description": "The task for the agent to perform"
    },
    "subagent_type": {
      "type": "string",
      "description": "The type of specialized agent to use"
    },
    "model": {
      "type": "string",
      "enum": ["sonnet", "opus", "haiku"],
      "description": "Optional model to use. Prefer haiku for quick tasks."
    },
    "resume": {
      "type": "string",
      "description": "Optional agent ID to resume from."
    },
    "run_in_background": {
      "type": "boolean",
      "description": "Set to true to run in background."
    }
  }
}
```

메인 에이전트는 서브 에이전트를 생성하기 위해 Task 도구를 호출하고, 추론을 사용하여 파라미터를 결정합니다. **model 파라미터를 주목하세요** - "Use Explore with Sonnet"이라고 말하면, 모델이 `model: "sonnet"`으로 도구 호출을 합니다.

---

## 백그라운드 에이전트

`run_in_background` 파라미터를 주목하세요. 서브 에이전트를 백그라운드에서 실행할지 결정합니다. **백그라운드 프로세스 기능이 좋습니다** - 디버깅이나 프로세스의 로그 출력 모니터링에 매우 유용합니다. 때때로 모니터링하고 싶은 오래 실행되는 파이썬 스크립트가 있습니다.

모델은 보통 자동으로 프로세스를 백그라운드로 보낼지 결정하지만, 명시적으로 그렇게 하라고 말할 수 있습니다.

**참고:** "Background Tasks"는 다릅니다. `&`를 사용하면 작업을 Claude Web으로 보냅니다(Claude Cloud라고 이름 지었어야 했는데 하하). 아직 이것을 제대로 작동시키지 못했습니다.

---

*다음 글에서는 저자의 실제 워크플로우를 공유합니다.*
