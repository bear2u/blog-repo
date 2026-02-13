---
layout: post
title: "GSD (Get Shit Done) 완벽 가이드 (01) - 소개 및 개요"
date: 2026-02-13
permalink: /gsd-guide-01-intro/
author: TÂCHES
categories: [AI 코딩, 개발 도구]
tags: [Claude Code, OpenCode, Gemini CLI, Meta-Prompting, Context Engineering, Spec-Driven Development]
original_url: "https://github.com/gsd-build/get-shit-done"
excerpt: "Claude Code의 컨텍스트 품질 저하 문제를 해결하는 메타 프롬프팅 시스템"
---

## GSD란?

**GSD(Get Shit Done)**는 Claude Code, OpenCode, Gemini CLI를 위한 **경량화되고 강력한 메타 프롬프팅, 컨텍스트 엔지니어링, 스펙 주도 개발 시스템**입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                      GSD 시스템                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Meta-       │  │ Context     │  │ Spec-Driven │          │
│  │ Prompting   │  │ Engineering │  │ Development │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                              │
│  → Claude Code, OpenCode, Gemini CLI 지원                   │
│  → Context Rot 문제 해결                                     │
│  → 일관된 고품질 코드 생성                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 왜 GSD가 필요한가?

### Context Rot (컨텍스트 부패) 문제

Claude Code는 강력하지만, **컨텍스트 윈도우가 채워질수록 품질이 저하**되는 문제가 있습니다.

| 컨텍스트 사용량 | 품질 | Claude 상태 |
|----------------|------|------------|
| 0-30% | **최고** | 철저하고 종합적 |
| 30-50% | **양호** | 자신감 있고 견고 |
| 50-70% | 저하됨 | 효율 모드 시작 |
| 70%+ | **나쁨** | 성급하고 최소화 |

GSD는 **컨텍스트 엔지니어링**을 통해 이 문제를 해결합니다. 작업을 작은 단위로 분할하고, 각 단위에 **새로운 컨텍스트 윈도우**를 할당하여 일관된 품질을 유지합니다.

---

## Vibecoding의 문제점

"Vibecoding"은 AI가 코드를 생성하게 하지만, 종종 **일관성 없는 결과물**을 만들어냅니다:

- 요구사항이 명확하지 않음
- 컨텍스트가 누적되며 품질 저하
- 대규모에서 붕괴하는 구조

GSD는 이 문제를 **컨텍스트 엔지니어링 레이어**로 해결합니다:

```
아이디어 설명 → 시스템이 필요한 것 추출 → Claude Code 작업 수행
```

---

## 주요 특징

### 1. Context Engineering (컨텍스트 엔지니어링)

Claude에게 필요한 컨텍스트를 자동으로 관리합니다.

| 파일 | 역할 |
|------|------|
| `PROJECT.md` | 프로젝트 비전, 항상 로드됨 |
| `research/` | 생태계 지식 (스택, 기능, 아키텍처, 주의사항) |
| `REQUIREMENTS.md` | v1/v2 범위 요구사항 |
| `ROADMAP.md` | 진행 상황, 완료된 작업 |
| `STATE.md` | 결정사항, 차단요소, 현재 위치 |
| `PLAN.md` | XML 구조의 원자적 태스크 |
| `SUMMARY.md` | 수행 내용, 변경사항 |

### 2. XML 프롬프트 포맷팅

모든 계획은 Claude에 최적화된 **XML 구조**로 작성됩니다:

```xml
<task type="auto">
  <name>로그인 엔드포인트 생성</name>
  <files>src/app/api/auth/login/route.ts</files>
  <action>
    JWT를 위해 jose 사용 (jsonwebtoken 아님 - CommonJS 문제).
    users 테이블로 자격증명 검증.
    성공 시 httpOnly 쿠키 반환.
  </action>
  <verify>curl -X POST localhost:3000/api/auth/login returns 200 + Set-Cookie</verify>
  <done>유효한 자격증명은 쿠키 반환, 무효하면 401</done>
</task>
```

### 3. 멀티 에이전트 오케스트레이션

각 단계에서 **전문 에이전트**를 병렬로 실행합니다:

| 단계 | 오케스트레이터 | 에이전트 |
|------|---------------|----------|
| 리서치 | 조정, 결과 제시 | 4개 병렬 리서처 |
| 계획 | 검증, 반복 관리 | 플래너, 체커 |
| 실행 | 웨이브 그룹화, 진행 추적 | 병렬 실행자 |
| 검증 | 결과 제시, 라우팅 | 검증자, 디버거 |

### 4. 원자적 Git 커밋

각 태스크는 완료 직후 **자체 커밋**을 받습니다:

```bash
abc123f docs(08-02): 사용자 등록 계획 완료
def456g feat(08-02): 이메일 확인 플로우 추가
hij789k feat(08-02): 비밀번호 해싱 구현
lmn012o feat(08-02): 등록 엔드포인트 생성
```

---

## 누구를 위한 것인가?

GSD는 다음을 원하는 사람들을 위한 것입니다:

- 50인 규모의 엔지니어링 조직인 척하지 않고
- 원하는 것을 설명하고 **올바르게 구축**하고 싶은 사람

**불필요한 것들:**
- 스프린트 의식, 스토리 포인트
- 이해관계자 동기화, 회고
- Jira 워크플로우

GSD는 **복잡성은 시스템 안에, 워크플로우는 단순하게** 유지합니다.

---

## 지원 플랫폼

| 플랫폼 | 설치 경로 |
|--------|----------|
| Claude Code | `~/.claude/` (global) 또는 `./.claude/` (local) |
| OpenCode | `~/.config/opencode/` |
| Gemini CLI | `~/.gemini/` |

---

*다음 글에서는 GSD 설치 및 기본 설정을 살펴봅니다.*
