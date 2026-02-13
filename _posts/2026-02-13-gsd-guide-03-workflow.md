---
layout: post
title: "GSD (Get Shit Done) 완벽 가이드 (03) - 핵심 워크플로우"
date: 2026-02-13
permalink: /gsd-guide-03-workflow/
author: TÂCHES
categories: [AI 코딩, 개발 도구]
tags: [Claude Code, Workflow, Planning, Execution]
original_url: "https://github.com/gsd-build/get-shit-done"
excerpt: "GSD의 6단계 핵심 워크플로우 이해하기"
---

## 워크플로우 개요

GSD는 **6단계 워크플로우**로 프로젝트를 진행합니다:

```
┌─────────────────────────────────────────────────────────────┐
│                    GSD 워크플로우                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Initialize → /gsd:new-project                           │
│       ↓                                                      │
│  2. Discuss   → /gsd:discuss-phase N                        │
│       ↓                                                      │
│  3. Plan      → /gsd:plan-phase N                           │
│       ↓                                                      │
│  4. Execute   → /gsd:execute-phase N                        │
│       ↓                                                      │
│  5. Verify    → /gsd:verify-work N                          │
│       ↓                                                      │
│  6. Repeat    → 다음 단계 또는 마일스톤 완료                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. 프로젝트 초기화

```
/gsd:new-project
```

하나의 명령어로 전체 흐름이 시작됩니다:

| 단계 | 설명 |
|------|------|
| **Questions** | 아이디어를 완전히 이해할 때까지 질문 |
| **Research** | 병렬 에이전트로 도메인 조사 (선택사항, 권장) |
| **Requirements** | v1, v2, 범위 밖 구분 |
| **Roadmap** | 요구사항에 매핑된 단계 생성 |

**생성 파일:** `PROJECT.md`, `REQUIREMENTS.md`, `ROADMAP.md`, `STATE.md`, `.planning/research/`

### 기존 코드가 있는 경우

먼저 `/gsd:map-codebase`를 실행하세요. 병렬 에이전트가 스택, 아키텍처, 컨벤션, 관심사를 분석합니다. 그러면 `/gsd:new-project`가 코드베이스를 이해한 상태에서 시작합니다.

---

## 2. 단계 논의 (Discuss Phase)

```
/gsd:discuss-phase 1
```

**이 단계에서 구현 방향을 결정합니다.**

로드맵은 단계당 한두 문장만 있습니다. 이것만으로는 당신이 상상하는 대로 구축하기에 충분하지 않습니다.

시스템은 단계를 분석하고 **회색 영역**을 식별합니다:

- **시각적 기능** → 레이아웃, 밀도, 인터랙션, 빈 상태
- **API/CLI** → 응답 형식, 플래그, 에러 처리, 상세도
- **콘텐츠 시스템** → 구조, 톤, 깊이, 흐름
- **조직 작업** → 그룹화 기준, 명명, 중복, 예외

**생성 파일:** `{phase}-CONTEXT.md`

### CONTEXT.md의 역할

CONTEXT.md는 다음 두 단계에 직접 영향을 줍니다:

1. **리서처** → 어떤 패턴을 조사할지 알게 됨
2. **플래너** → 어떤 결정이 확정되었는지 알게 됨

이 단계에서 깊이 들어갈수록 시스템이 당신이 원하는 것을 더 잘 만듭니다.

---

## 3. 단계 계획 (Plan Phase)

```
/gsd:plan-phase 1
```

시스템이 수행하는 작업:

| 단계 | 설명 |
|------|------|
| **Research** | CONTEXT.md 결정에 따라 구현 방법 조사 |
| **Plan** | XML 구조로 2-3개 원자적 태스크 계획 생성 |
| **Verify** | 요구사항 대비 계획 검증, 통과할 때까지 반복 |

각 계획은 **새로운 컨텍스트 윈도우에서 실행 가능한 크기**입니다. 품질 저하 없이 일관된 결과를 보장합니다.

**생성 파일:** `{phase}-RESEARCH.md`, `{phase}-{N}-PLAN.md`

---

## 4. 단계 실행 (Execute Phase)

```
/gsd:execute-phase 1
```

시스템이 수행하는 작업:

| 단계 | 설명 |
|------|------|
| **Waves로 실행** | 가능한 곳은 병렬, 의존성이 있으면 순차 |
| **계획당 새 컨텍스트** | 200k 토큰이 구현에만 사용, 누적 쓰레기 없음 |
| **태스크당 커밋** | 모든 태스크가 자체 원자적 커밋을 받음 |
| **목표 대비 검증** | 코드베이스가 단계에서 약속한 것을 전달하는지 확인 |

자리를 비웠다 돌아오면 **깨끗한 git 히스토리와 함께 완료된 작업**을 발견하게 됩니다.

**생성 파일:** `{phase}-{N}-SUMMARY.md`, `{phase}-VERIFICATION.md`

---

## 5. 작업 검증 (Verify Work)

```
/gsd:verify-work 1
```

**실제로 작동하는지 확인하는 단계입니다.**

자동화된 검증은 코드가 존재하고 테스트가 통과하는지 확인합니다. 하지만 기능이 당신이 기대한 대로 작동하나요?

시스템이 수행하는 작업:

1. **테스트 가능한 결과물 추출** — 이제 할 수 있는 것
2. **하나씩 안내** — "이메일로 로그인할 수 있나요?" 예/아니오 또는 문제 설명
3. **자동 실패 진단** — 디버그 에이전트가 근본 원인 찾기
4. **검증된 수정 계획 생성** — 즉시 재실행 준비

모든 것이 통과하면 다음 단계로. 문제가 있으면 수동 디버깅 없이 `/gsd:execute-phase`를 다시 실행하면 됩니다.

**생성 파일:** `{phase}-UAT.md`, 문제 발견 시 수정 계획

---

## 6. 반복 → 완료 → 다음 마일스톤

```
/gsd:discuss-phase 2
/gsd:plan-phase 2
/gsd:execute-phase 2
/gsd:verify-work 2
...
/gsd:complete-milestone
/gsd:new-milestone
```

**discuss → plan → execute → verify** 루프를 마일스톤 완료까지 반복합니다.

각 단계는:
- 입력 받기 (discuss)
- 적절한 리서치 (plan)
- 깨끗한 실행 (execute)
- 사람 검증 (verify)

컨텍스트는 신선하게 유지되고 품질은 높게 유지됩니다.

---

## Quick Mode

```
/gsd:quick
```

**전체 계획이 필요 없는 애드혹 태스크용.**

Quick 모드는 더 빠른 경로로 GSD 보장(원자적 커밋, 상태 추적)을 제공합니다:

- **동일한 에이전트** — 플래너 + 실행자, 동일한 품질
- **선택 단계 건너뛰기** — 리서치 없음, 계획 체커 없음, 검증자 없음
- **별도 추적** — `.planning/quick/`에 저장, 단계가 아님

**용도:** 버그 수정, 작은 기능, 설정 변경, 일회성 태스크

```
/gsd:quick
> What do you want to do? "Add dark mode toggle to settings"
```

**생성 파일:** `.planning/quick/001-add-dark-mode-toggle/PLAN.md`, `SUMMARY.md`

---

## 요약

| 명령어 | 용도 |
|--------|------|
| `/gsd:new-project` | 프로젝트 초기화 |
| `/gsd:discuss-phase N` | 구현 결정 사항 캡처 |
| `/gsd:plan-phase N` | 리서치 + 계획 + 검증 |
| `/gsd:execute-phase N` | 계획 실행 |
| `/gsd:verify-work N` | 사용자 수용 테스트 |
| `/gsd:complete-milestone` | 마일스톤 완료 |
| `/gsd:new-milestone` | 다음 버전 시작 |
| `/gsd:quick` | 애드혹 태스크 |

---

*다음 글에서는 GSD의 멀티 에이전트 아키텍처를 살펴봅니다.*
