---
layout: post
title: "Synkra AIOS Core 완벽 가이드 (04) - 태스크/워크플로우: Task-First로 개발을 자동화하기"
date: 2026-02-15
permalink: /aios-core-guide-04-tasks-and-workflows/
author: Synkra AIOS Team
categories: [AI 코딩 에이전트, AIOS]
tags: [AIOS, Tasks, Workflows, Orchestration, Greenfield, Brownfield]
original_url: "https://github.com/SynkraAI/aios-core"
excerpt: "AIOS의 Task-First 아키텍처와 워크플로우(YAML 오케스트레이션) 개념을 정리하고, 그린필드/브라운필드에 맞는 선택 기준을 제공합니다."
---

## 태스크(Task)는 “실행 가능한 표준 단위”

AIOS는 태스크를 중심으로 모든 것을 구성합니다.

- 태스크는 “요청을 어떻게 처리할지”를 문서로 표준화한 실행 레시피
- 에이전트는 태스크를 수행하는 실행자
- 워크플로우는 여러 태스크(또는 에이전트 단계)를 순서대로 오케스트레이션

이 구조가 중요한 이유는, “자연어 요청”을 그대로 실행하면 매번 결과가 달라지기 때문입니다.

---

## 태스크 실행 감각

문서에는 보통 이런 형태가 나옵니다.

```text
*task <name> [args]
```

예:

```text
*task develop-story --story=1.1
```

또는 CLI에서 목록을 확인하는 흐름:

```bash
aios tasks list
```

---

## 워크플로우(Workflows)는 “여러 에이전트의 합주”

워크플로우는 YAML로 정의되는 “멀티 스텝 절차”입니다.

AIOS가 제시하는 분류는 크게 두 축입니다.

1. 프로젝트 유형
- Greenfield: 새로 만든다
- Brownfield: 기존 것을 분석/개선한다
- Generic: 유형에 상관 없이 반복되는 프로세스

2. 범위
- Fullstack / UI / Service / Discovery 등

문서 예시로는 이런 워크플로우가 등장합니다.

- Story Development Cycle: `@sm → @po → @dev → @qa`
- Greenfield Fullstack: DevOps → Analyst → PM → UX → Architect → PO → SM → Dev → QA
- Brownfield Discovery: Architect → Data Engineer → UX → QA → Analyst → PM

---

## 워크플로우를 쓰는 이유

태스크만으로도 자동화는 되지만, 워크플로우를 쓰면:

- **단계/순서가 고정**되고
- 각 단계의 **담당 에이전트가 명확**해지며
- 단계별 산출물(문서/코드/테스트)을 강제할 수 있습니다.

특히 “계획(문서) → 실행(스토리/코드) → 검증(QA)”처럼 반복되는 루프는 워크플로우로 묶을 가치가 큽니다.

---

## 선택 가이드: 어떤 워크플로우를 먼저?

1. 이미 요구사항/설계가 꽤 있는 팀
- Story Development Cycle부터: 스토리 하나씩 SM→Dev→QA로 굴리기

2. 새 서비스/앱을 제대로 시작하고 싶다
- Greenfield Fullstack: 플래닝과 문서 정렬부터 시작

3. 레거시가 크고, 상태 파악이 먼저다
- Brownfield Discovery: 현황/부채/리스크를 먼저 수집하고 정리

---

## 워크플로우 정의는 “프로세스의 코드화”다

워크플로우를 정의한다는 건, 팀의 개발 습관을 문서화하고 자동화 가능한 형태로 만들겠다는 뜻입니다.

- 무엇을 언제 만들고
- 누가 승인하고
- 어떤 품질 기준을 통과해야 다음 단계로 가는지

를 고정하는 것이고, 이게 AIOS가 말하는 “에이전트 애자일”의 핵심입니다.

---

*다음 글에서는 플래닝 워크플로우(Brief→PRD→Architecture)에서 어떤 산출물이 나오고, Web UI에서 IDE로 전환하는 지점을 어떻게 잡는지 살펴봅니다.*
