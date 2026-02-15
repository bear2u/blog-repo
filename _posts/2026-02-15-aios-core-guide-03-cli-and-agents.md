---
layout: post
title: "Synkra AIOS Core 완벽 가이드 (03) - CLI/에이전트 사용법: @agent와 *command"
date: 2026-02-15
permalink: /aios-core-guide-03-cli-and-agents/
author: Synkra AIOS Team
categories: [AI 코딩 에이전트, AIOS]
tags: [AIOS, Agents, CLI, Commands, Persona, Command Authority]
original_url: "https://github.com/SynkraAI/aios-core"
excerpt: "@dev 같은 에이전트 활성화, *help 같은 커맨드 규약, 커맨드 가시성(키/퀵/풀)과 권한(오너) 개념을 정리합니다."
---

## 에이전트 활성화 규약: `@agent`

AIOS 문서에서 에이전트는 보통 다음 형태로 활성화합니다.

```
@dev
@qa
@architect
@pm
@sm
@aios-master
```

핵심은 “한 번에 한 에이전트 컨텍스트로 대화한다”는 점입니다.

- 같은 요청도 `@pm`은 PRD 관점으로,
- `@dev`는 구현 관점으로,
- `@qa`는 리스크/테스트 관점으로 바라보게 설계됩니다.

---

## 커맨드 규약: `*command`

에이전트가 활성화된 상태에서 실행 커맨드는 `*` 접두사를 사용합니다.

예:

```
*help
*status
*agents
*task <name>
*exit
```

문서에서는 “커맨드의 가시성(visibility)” 개념도 소개합니다.

- `key`: 아주 핵심만(최소 그리팅)
- `quick`: 자주 쓰는 커맨드(퀵 레퍼런스)
- `full`: 전체 목록(`*help`)

개념 예시는 이런 느낌입니다.

```yaml
commands:
  - name: help
    visibility: [full, quick, key]
    description: "Show available commands"

  - name: create-prd
    visibility: [full, quick]
    description: "Create product requirements"

  - name: session-info
    visibility: [full]
    description: "Show session details"
```

---

## 커맨드 권한(Authority): 오너 에이전트가 있다

AIOS는 “커맨드마다 단 한 명의 오너(owner) 에이전트가 있다”는 규칙을 강조합니다.

예:
- `*create-prd`는 `@pm`이 오너
- 구현 관련은 `@dev`가 오너
- 리뷰는 `@qa`가 오너

이 규칙의 실용적인 의미는:

- 여러 에이전트가 비슷한 일을 할 수 있어도, **실행 책임을 분리**하고
- 혼선(중복/상충)을 줄이며
- 워크플로우 단계별 핸드오프(handoff)를 명확히 한다는 점입니다.

---

## 자주 쓰는 “첫 세션” 흐름

설치 직후에 가장 흔한 순서는 아래 정도입니다.

1. 오케스트레이터/마스터부터

```
@aios-master
*help
*status
*agents
```

2. 개발 에이전트로 구현

```
@dev
*help
```

3. QA로 검증

```
@qa
*help
```

---

## CLI 명령(별도 커맨드)도 함께 등장한다

문서에는 `@...`/`*...` 외에도, 별도의 CLI 커맨드가 나옵니다.

예:

```bash
# 에이전트 목록
aios agents list

# 태스크 목록
aios tasks list

# 정보/진단
npx @synkra/aios-core info
npx @synkra/aios-core doctor
```

이 둘을 섞어 이해하면 헷갈릴 수 있는데, 대략:

- `aios ...`는 “도구(프레임워크) 자체를 조작하는 CLI”
- `@agent`/`*command`는 “에이전트가 대화/태스크 실행을 수행”

으로 구분하면 편합니다.

---

## 팁: 권한/안전 모드와 결합해서 쓴다

에이전트 자동화는 강력하지만, 안전장치도 같이 씁니다.

- 탐색만 할 때는 읽기 위주
- 수정이 들어가면 확인 모드
- CI 같은 신뢰된 환경에서만 자동 모드

이 내용은 후반 챕터(권한 모드/보안)에서 더 자세히 다룹니다.

---

*다음 글에서는 “Everything is a task” 관점에서 태스크/워크플로우가 어떻게 구성되고, 어떤 상황에 어떤 워크플로우를 쓰는지 정리합니다.*
