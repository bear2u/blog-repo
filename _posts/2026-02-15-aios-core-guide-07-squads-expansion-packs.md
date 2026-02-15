---
layout: post
title: "Synkra AIOS Core 완벽 가이드 (07) - Squads: 도메인별 에이전트 팀(확장팩) 만들기"
date: 2026-02-15
permalink: /aios-core-guide-07-squads-expansion-packs/
author: Synkra AIOS Team
categories: [AI 코딩 에이전트, AIOS]
tags: [AIOS, Squads, Expansion Pack, Task-First, Validation, Distribution]
original_url: "https://github.com/SynkraAI/aios-core"
excerpt: "Squads(스쿼드)는 에이전트/태스크/워크플로우/템플릿을 한 묶음으로 배포하는 확장팩입니다. 구조, 생성/검증, 배포 레벨, 마이그레이션을 정리합니다."
---

## Squads는 무엇인가?

**Squads**는 AIOS 기능을 특정 도메인(예: ETL, 콘텐츠 제작 등)에 맞게 확장하는 **모듈형 에이전트 팀 패키지**입니다.

전통적인 “에이전트 몇 개 복붙”과 달리,

- 에이전트 정의
- 태스크 정의(Task-First)
- 워크플로우
- 체크리스트/템플릿/도구/스크립트

를 한 패키지로 묶어 **설치/공유/조합**할 수 있게 설계됩니다.

---

## 스쿼드 디렉토리 구조

문서에서 제시하는 표준 구조는 다음과 같습니다.

```
./squads/my-squad/
├── squad.yaml              # 매니페스트(필수)
├── README.md
├── config/
│   ├── coding-standards.md
│   ├── tech-stack.md
│   └── source-tree.md
├── agents/
├── tasks/
├── workflows/
├── checklists/
├── templates/
├── tools/
├── scripts/
└── data/
```

핵심은 `squad.yaml`(manifest)로 “구성요소 목록/호환성/의존성”을 선언한다는 점입니다.

---

## `squad.yaml` 감각(요지)

문서 예시를 요약하면 이런 키들이 중요합니다.

```yaml
name: my-squad
version: 1.0.0
aios:
  minVersion: "2.1.0"
  type: squad
components:
  agents:
    - my-agent.md
  tasks:
    - my-task.md
config:
  extends: extend
  coding-standards: config/coding-standards.md
  tech-stack: config/tech-stack.md
  source-tree: config/source-tree.md
```

---

## 스쿼드 만드는 흐름(가이드)

문서는 `@squad-creator` 에이전트를 중심으로 소개합니다.

1. 문서 기반 설계(권장)

```text
@squad-creator
*design-squad --docs ./docs/prd/my-project.md
*create-squad my-squad --from-design
*validate-squad my-squad
```

2. 바로 생성

```text
@squad-creator
*create-squad my-squad
*validate-squad my-squad
```

---

## 배포 레벨(Distribution Levels)

문서는 스쿼드 배포를 3레벨로 설명합니다.

- Level 1: 로컬 `./squads/` (개인/팀 내부)
- Level 2: 공개 레포(예: `SynkraAI/aios-squads`)
- Level 3: API/마켓플레이스(문서상 `api.synkra.dev`)

실무적으로는:

- 처음엔 로컬에서 만들고
- 팀에 공유되면 레포로 옮기고
- 범용 가치가 있으면 공개 배포

순서로 가는 게 자연스럽습니다.

---

## 레거시 스쿼드 마이그레이션

AIOS 2.1에서 포맷이 바뀌면서, 레거시 스쿼드는 마이그레이션이 필요할 수 있습니다.

문서가 제시하는 신호:

- `config.yaml`만 있고 `squad.yaml`이 없다
- 에이전트 정의가 YAML 형태다

진단/마이그레이션:

```text
@squad-creator
*validate-squad ./squads/legacy-squad
*migrate-squad ./squads/legacy-squad --dry-run
*migrate-squad ./squads/legacy-squad
```

---

## Squads를 잘 쓰는 포인트

- “에이전트”보다 “태스크”를 먼저 설계한다(Task-First)
- 검증(`*validate-squad`)을 CI처럼 습관화한다
- config 상속(extend/override/none)을 남발하지 말고 원칙을 정한다

---

*다음 글에서는 전역 MCP 구성을 통해 여러 프로젝트에서 도구(MCP 서버)를 공유하는 방식과, 추천 아키텍처(직접 MCP + docker-gateway)를 정리합니다.*
