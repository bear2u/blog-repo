---
layout: post
title: "GSD (Get Shit Done) 완벽 가이드 (08) - 템플릿 시스템"
date: 2026-02-13
permalink: /gsd-guide-08-templates/
author: TÂCHES
categories: [AI 코딩, 개발 도구]
tags: [Claude Code, Templates, Project Structure, Documentation]
original_url: "https://github.com/gsd-build/get-shit-done"
excerpt: "GSD 템플릿 시스템과 문서 구조 이해하기"
---

## 템플릿 개요

GSD는 `.planning/` 디렉토리에 체계적인 문서를 생성합니다. 각 템플릿은 특정 목적을 가집니다.

```
.planning/
├── PROJECT.md          # 프로젝트 비전
├── REQUIREMENTS.md     # 요구사항 명세
├── ROADMAP.md          # 로드맵
├── STATE.md            # 현재 상태
├── config.json         # 설정
├── research/           # 리서치 결과
├── todos/              # 캡처된 아이디어
├── quick/              # Quick 모드 작업
└── {phase}/            # 각 단계별 디렉토리
    ├── CONTEXT.md
    ├── RESEARCH.md
    ├── {N}-PLAN.md
    ├── {N}-SUMMARY.md
    └── VERIFICATION.md
```

---

## PROJECT.md

**목적:** 프로젝트의 살아있는 컨텍스트 문서

### 구조

```markdown
# [프로젝트 이름]

## What This Is
[현재 정확한 설명 - 2-3문장]

## Core Value
[가장 중요한 한 가지]

## Requirements

### Validated
[출시되어 가치가 입증된 요구사항]

### Active
[현재 구축 중인 요구사항]

### Out of Scope
[명시적 경계와 이유]

## Context
[구현에 영향을 미치는 배경 정보]

## Constraints
[하드 제한 사항]

## Key Decisions
| Decision | Rationale | Outcome |
|----------|-----------|---------|
| [선택] | [이유] | [결과] |
```

### 진화

PROJECT.md는 프로젝트 수명 주기 동안 진화합니다:

- 각 단계 전환 후 요구사항 업데이트
- 각 마일스톤 후 전체 검토
- Core Value가 여전히 올바른 우선순위인지 확인

---

## REQUIREMENTS.md

**목적:** 범위가 지정된 v1/v2 요구사항

### 구조

```markdown
# Requirements

## v1 (MVP)
- [ ] [필수 요구사항 1]
- [ ] [필수 요구사항 2]

## v2 (Enhanced)
- [ ] [향상된 요구사항 1]

## Out of Scope
- [제외 항목] — [이유]
```

### 단계 추적성

각 요구사항은 특정 단계로 추적 가능해야 합니다.

---

## ROADMAP.md

**목적:** 진행 상황과 완료된 작업 추적

### 구조

```markdown
# Roadmap

## Milestone: v1.0

### Phase 1: [이름]
**Status:** Complete
**Delivered:** [날짜]

### Phase 2: [이름]
**Status:** In Progress
**Started:** [날짜]

### Phase 3: [이름]
**Status:** Pending
```

---

## STATE.md

**목적:** 결정사항, 차단요소, 현재 위치 기억

### 구조

```markdown
# State

## Current Position
- Milestone: v1.0
- Phase: 2
- Status: Executing

## Recent Decisions
- [결정 1] — [이유]

## Blockers
(없음 또는 목록)

## Session Notes
- [날짜]: [노트]
```

### 세션 간 메모리

STATE.md는 Claude가 세션 간에 맥락을 유지하는 데 도움이 됩니다.

---

## CONTEXT.md (단계별)

**목적:** 구현 전 사용자 결정 사항 캡처

### 구조

```markdown
# Phase {N} Context

## Decisions
- [ ] **[주제]**: [사용자 결정]

## Deferred Ideas
- [아이디어] — 나중에 고려

## Claude's Discretion
- [주제]: Claude가 결정
```

### 충실도 규칙

| 섹션 | 규칙 |
|------|------|
| Decisions | 반드시 그대로 구현 |
| Deferred Ideas | 계획에 포함 금지 |
| Claude's Discretion | Claude 판단에 맡김 |

---

## PLAN.md (단계별)

**목적:** XML 구조의 원자적 태스크

### 구조

```markdown
---
phase: 2
plan: 1
type: standard
autonomous: true
wave: 1
depends_on: []
---

# [계획 이름]

## Objective
[목표 설명]

## Context
@PROJECT.md
@REQUIREMENTS.md
@{phase}-CONTEXT.md

## Tasks

### Task 1: [이름]
<task type="auto">
  <name>[태스크 이름]</name>
  <files>[파일 경로]</files>
  <action>[구현 지침]</action>
  <verify>[검증 방법]</verify>
  <done>[완료 기준]</done>
</task>

## Success Criteria
- [성공 기준 1]
- [성공 기준 2]
```

---

## SUMMARY.md (단계별)

**목적:** 수행된 작업 기록

### 구조

```markdown
# Summary: Phase {N} Plan {M}

## Completed Tasks
- [x] Task 1: [이름] (commit: abc123)
- [x] Task 2: [이름] (commit: def456)

## Deviations
- [Rule 2 - Missing Critical] [설명]

## Files Changed
- `path/to/file.ts` — [변경 설명]

## Verification Results
- [x] [검증 항목 1]
- [x] [검증 항목 2]

## Duration
Start: [시간]
End: [시간]
Total: [소요 시간]
```

---

## VERIFICATION.md (단계별)

**목적:** 목표 대비 자동 검증 결과

### 구조

```markdown
# Verification: Phase {N}

## Checks
- [x] Code exists
- [x] Tests pass
- [x] Lint clean

## Goal Verification
- [x] [목표 1 달성]
- [ ] [목표 2 실패] — [이유]

## Issues Found
(없음 또는 목록)

## Fix Plans
(필요한 경우 수정 계획)
```

---

## 템플릿 커스터마이징

GSD의 템플릿은 `.claude/get-shit-done/templates/`에 위치합니다.

### 커스텀 템플릿 생성

1. 기본 템플릿 복사
2. 프로젝트에 맞게 수정
3. 동일한 위치에 저장

---

## 브라운필드 (기존 코드베이스)

기존 코드베이스의 경우:

1. `/gsd:map-codebase` 먼저 실행
2. 기존 코드에서 **Validated 요구사항 추론**:
   - 코드베이스가 실제로 무엇을 하는가?
   - 어떤 패턴이 확립되었는가?
   - 무엇이 작동하고 신뢰되는가?

---

*다음 글에서는 GSD 심화 활용법을 살펴봅니다.*
