---
layout: post
title: "Synkra AIOS Core 완벽 가이드 (01) - 소개: CLI First 에이전트 프레임워크"
date: 2026-02-15
permalink: /aios-core-guide-01-intro/
author: Synkra AIOS Team
categories: [AI 코딩 에이전트, AIOS]
tags: [AIOS, SynkraAI, aios-core, Agentic Agile, CLI First, Workflows, Tasks]
original_url: "https://github.com/SynkraAI/aios-core"
excerpt: "Synkra AIOS Core가 제공하는 CLI-first 철학, 역할 분리 에이전트, 태스크/워크플로우 기반 개발 방법을 개요로 정리합니다."
---

## AIOS Core는 무엇인가?

**Synkra AIOS Core(aios-core)**는 “AI가 개발을 잘 하게 만드는 프롬프트 모음”이 아니라,

- 에이전트 역할(Dev/QA/PM/SM/Architect 등)을 분리하고
- 태스크와 워크플로우로 실행 단위를 표준화하며
- 문서/스토리/체크리스트를 템플릿화해서

**반복 가능한 개발 프로세스**를 만들기 위한 프레임워크입니다.

이 프로젝트가 강조하는 아키텍처 전제는 단순합니다.

```
CLI First → Observability Second → UI Third
```

즉, “지능은 CLI에 살고”, 대시보드는 CLI에서 일어나는 일을 관측하는 2순위이며, UI는 3순위입니다.

---

## AIOS가 풀려는 문제

전통적인 “AI로 개발하기”의 가장 큰 문제는 다음 2가지로 요약됩니다.

1. 계획이 흔들린다: PRD/아키텍처/스토리 사이가 맞지 않고, 작업이 엉킨다
2. 컨텍스트가 흐른다: 세션이 바뀌면 의도/제약/결정이 사라지고, 다시 설명해야 한다

AIOS는 이를 “역할 분리 + 산출물 표준화 + 품질 게이트”로 완화하려고 합니다.

---

## 핵심 아이디어 1: 역할 분리된 에이전트

AIOS는 한 명의 만능 에이전트보다 “역할이 명확한 여러 에이전트”를 강조합니다.

예:
- `@pm`: PRD/에픽 수준의 요구사항 정리
- `@architect`: 설계/아키텍처/ADR
- `@sm`: 다음 스토리 초안, 프로세스 진행
- `@dev`: 구현과 테스트
- `@qa`: 리뷰/품질 게이트

IDE/Claude Code에서 다음처럼 활성화합니다.

```
@dev
*help
```

---

## 핵심 아이디어 2: Task-First 아키텍처

AIOS 문서에서 반복해서 등장하는 표현이 “Everything is a task”입니다.

- 사용자는 “태스크”로 의도를 표현
- 에이전트는 태스크 정의에 따라 실행
- 여러 태스크를 묶어 “워크플로우”로 자동화

개념적으로는 이렇게 흐릅니다.

```
User Request → Task → Agent Execution → Output
                 │
            Workflow (if multi-step)
```

이 방식의 장점은:
- 작업 단위가 표준화되어 “다음에 뭘 할지”가 명확해지고
- 팀(또는 다른 사용자)에게 재사용 가능한 실행 레시피가 생기며
- QA/보안/검증을 태스크 레벨에서 강제할 수 있다는 점입니다.

---

## 이 시리즈에서 무엇을 다루나

이 시리즈는 “AIOS를 써서 실제로 개발 사이클을 굴리는 것”에 초점을 맞춥니다.

- 설치/진단(`doctor`)과 프로젝트 구조
- 에이전트/커맨드 사용법(`@agent`, `*command`)
- 태스크/워크플로우의 설계와 사용
- 플래닝(Brief→PRD→Architecture)
- 스토리 기반 개발 루프(SM→Dev→QA)
- Squads(확장팩)로 도메인별 팀 구성
- MCP 글로벌 설정으로 도구 공유
- LLM 라우팅으로 비용 최적화
- 보안 하드닝/권한 모드/트러블슈팅

---

## 시작 전에 알아둘 것

1. **AIOS는 “프로젝트 안에 프레임워크를 설치”하는 방식**입니다.
- 보통 `.aios-core/` 같은 디렉토리가 프로젝트에 들어옵니다.

2. **다양한 IDE/클라이언트(Claude Code 등) 통합을 전제**합니다.
- 에이전트 활성화는 클라이언트별로 UX가 다릅니다.

3. **비용과 보안은 별도 레이어로 다룹니다.**
- LLM 라우팅(DeepSeek 등), MCP 격리(Docker) 같은 내용이 별도 챕터로 분리됩니다.

---

*다음 글에서는 설치(install/init)와 `doctor`로 설치 상태를 검증하는 흐름을 정리합니다.*
