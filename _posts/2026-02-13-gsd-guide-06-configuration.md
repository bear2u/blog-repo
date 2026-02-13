---
layout: post
title: "GSD (Get Shit Done) 완벽 가이드 (06) - 설정 및 구성"
date: 2026-02-13
permalink: /gsd-guide-06-configuration/
author: TÂCHES
categories: [AI 코딩, 개발 도구]
tags: [Claude Code, Configuration, Settings, Profiles]
original_url: "https://github.com/gsd-build/get-shit-done"
excerpt: "GSD 설정 및 모델 프로필 구성 가이드"
---

## 설정 파일 위치

GSD는 `.planning/config.json`에 프로젝트 설정을 저장합니다.

`/gsd:new-project` 중에 구성하거나 `/gsd:settings`로 나중에 업데이트할 수 있습니다.

---

## 핵심 설정

| 설정 | 옵션 | 기본값 | 설명 |
|------|------|--------|------|
| `mode` | `yolo`, `interactive` | `interactive` | 각 단계 자동 승인 vs 확인 |
| `depth` | `quick`, `standard`, `comprehensive` | `standard` | 계획 철저성 (단계 × 계획) |

### Mode 설정

- **interactive**: 각 단계마다 사용자 확인
- **yolo**: 모든 단계 자동 승인

### Depth 설정

| Depth | 설명 |
|-------|------|
| `quick` | 빠른 계획, 최소한의 검증 |
| `standard` | 균형 잡힌 계획과 검증 |
| `comprehensive` | 상세한 계획, 철저한 검증 |

---

## 모델 프로필

각 에이전트가 사용할 Claude 모델을 제어합니다. 품질과 토큰 비용의 균형을 조정합니다.

| 프로필 | Planning | Execution | Verification |
|--------|----------|-----------|--------------|
| `quality` | Opus | Opus | Sonnet |
| `balanced` (기본값) | Opus | Sonnet | Sonnet |
| `budget` | Sonnet | Sonnet | Haiku |

### 프로필 전환

```
/gsd:set-profile budget
```

또는 `/gsd:settings`를 통해 구성합니다.

### 프로필 선택 가이드

| 프로필 | 추천 상황 |
|--------|----------|
| `quality` | 복잡한 프로젝트, 최고 품질 필요 |
| `balanced` | 일반적인 사용, 균형 잡힌 접근 |
| `budget` | 토큰 비용 절약, 단순한 작업 |

---

## 워크플로우 에이전트

계획/실행 중 추가 에이전트를 스폰합니다. 품질을 향상시키지만 토큰과 시간을 추가합니다.

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `workflow.research` | `true` | 각 단계 계획 전 도메인 리서치 |
| `workflow.plan_check` | `true` | 실행 전 계획이 단계 목표 달성 검증 |
| `workflow.verifier` | `true` | 실행 후 필수 기능 전달 확인 |

### 토글 방법

`/gsd:settings`를 사용하거나 호출 시 재정의:

```bash
/gsd:plan-phase --skip-research   # 리서치 건너뛰기
/gsd:plan-phase --skip-verify     # 계획 검증 건너뛰기
```

### 언제 끄나요?

- **research 끄기**: 잘 알려진 도메인, 빠른 반복 필요
- **plan_check 끄기**: 단순한 작업, 시간 절약
- **verifier 끄기**: 신뢰할 수 있는 실행, 빠른 진행

---

## 실행 설정

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `parallelization.enabled` | `true` | 독립적인 계획 동시 실행 |
| `planning.commit_docs` | `true` | `.planning/`을 git에 추적 |

### 병렬화

```json
{
  "parallelization": {
    "enabled": true
  }
}
```

독립적인 계획은 동시에 실행됩니다. 의존성이 있으면 순차적으로 실행됩니다.

### 문서 커밋

```json
{
  "planning": {
    "commit_docs": true
  }
}
```

`.planning/` 디렉토리의 모든 문서를 git에 추적합니다.

---

## Git 브랜칭 설정

실행 중 GSD가 브랜치를 처리하는 방법을 제어합니다.

| 설정 | 옵션 | 기본값 | 설명 |
|------|------|--------|------|
| `git.branching_strategy` | `none`, `phase`, `milestone` | `none` | 브랜치 생성 전략 |
| `git.phase_branch_template` | string | `gsd/phase-{phase}-{slug}` | 단계 브랜치 템플릿 |
| `git.milestone_branch_template` | string | `gsd/{milestone}-{slug}` | 마일스톤 브랜치 템플릿 |

### 전략 설명

| 전략 | 동작 |
|------|------|
| `none` | 현재 브랜치에 커밋 (기본 GSD 동작) |
| `phase` | 단계당 브랜치 생성, 단계 완료 시 병합 |
| `milestone` | 전체 마일스톤에 하나의 브랜치, 완료 시 병합 |

### 마일스톤 완료 시

GSD는 스쿼시 병합(권장) 또는 히스토리 포함 병합을 제안합니다.

---

## 설정 파일 예시

```json
{
  "mode": "interactive",
  "depth": "standard",
  "model_profile": "balanced",
  "workflow": {
    "research": true,
    "plan_check": true,
    "verifier": true
  },
  "parallelization": {
    "enabled": true
  },
  "planning": {
    "commit_docs": true
  },
  "git": {
    "branching_strategy": "none",
    "phase_branch_template": "gsd/phase-{phase}-{slug}",
    "milestone_branch_template": "gsd/{milestone}-{slug}"
  }
}
```

---

## 설정 변경 방법

### 대화형

```
/gsd:settings
```

### 직접 파일 편집

`.planning/config.json`을 직접 편집할 수 있습니다.

### 프로필만 변경

```
/gsd:set-profile quality
```

---

*다음 글에서는 보안 및 문제 해결을 살펴봅니다.*
