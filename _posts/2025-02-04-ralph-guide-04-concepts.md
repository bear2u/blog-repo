---
layout: post
title: "Ralph 가이드 04 - 핵심 개념"
date: 2025-02-04
categories: [AI, Claude Code, Ralph]
tags: [ralph, autonomous-loop, exit-signal, concepts]
permalink: /ralph-guide-04-concepts/
---

# 핵심 개념

## 자율 개발 루프

Ralph의 핵심은 자율 개발 루프입니다. 인간의 개입 없이 프로젝트 완료까지 반복 실행됩니다.

### 루프 구조

```
┌───────────────────────────────────────────────────────────────┐
│                      RALPH AUTONOMOUS LOOP                     │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐│
│   │  INIT   │ ──▶ │  READ   │ ──▶ │ EXECUTE │ ──▶ │ UPDATE  ││
│   │ Session │     │ Context │     │  Task   │     │  State  ││
│   └─────────┘     └─────────┘     └─────────┘     └─────────┘│
│        │               │               │               │      │
│        │               │               │               │      │
│        ▼               ▼               ▼               ▼      │
│   Load .ralphrc   PROMPT.md      Implement       fix_plan.md │
│   Load state      fix_plan.md    Test & Fix      status.json │
│   Check limits    AGENT.md       Commit          logs/       │
│                   specs/                                      │
│                                                                │
│        └───────────────────────────────────────────┘          │
│                          │                                     │
│                          ▼                                     │
│                   ┌─────────────┐                             │
│                   │   CHECK     │                             │
│                   │  EXIT COND  │                             │
│                   └─────────────┘                             │
│                     /         \                               │
│                    /           \                              │
│               [NO]              [YES]                         │
│                /                   \                          │
│               ▼                     ▼                         │
│        ┌───────────┐         ┌───────────┐                   │
│        │  CONTINUE │         │    EXIT   │                   │
│        │   LOOP    │         │   LOOP    │                   │
│        └───────────┘         └───────────┘                   │
│              │                                                │
│              └────────────────────────────┐                  │
│                                           │                  │
│                                           ▼                  │
│                                     Back to READ             │
└───────────────────────────────────────────────────────────────┘
```

### 각 단계 설명

| 단계 | 동작 | 파일 |
|------|------|------|
| **INIT** | 세션 초기화, 상태 로드 | `.ralphrc`, `status.json` |
| **READ** | 컨텍스트 읽기 | `PROMPT.md`, `fix_plan.md`, `specs/` |
| **EXECUTE** | 작업 구현 및 테스트 | 프로젝트 파일 |
| **UPDATE** | 상태 갱신 | `fix_plan.md`, `status.json`, `logs/` |
| **CHECK** | 종료 조건 확인 | `status.json` |

## 종료 감지 시스템

### Dual-Condition 체크

Ralph는 두 가지 조건을 모두 확인해야 종료합니다:

```javascript
// Ralph 종료 조건
const shouldExit = (
  allTasksComplete()     // 조건 1: 모든 작업 완료
  && EXIT_SIGNAL === true  // 조건 2: 명시적 종료 신호
);
```

### 왜 두 조건인가?

**단일 조건의 문제:**

| 단일 조건 | 문제점 |
|-----------|--------|
| 작업 완료만 체크 | 추가 작업이 필요할 수 있음 |
| EXIT_SIGNAL만 체크 | 실수로 설정될 수 있음 |

**Dual-Condition 장점:**
- 조기 종료 방지
- 명시적 확인 필요
- 안전한 완료 보장

### 완료 감지 흐름

```
fix_plan.md 체크
        │
        ▼
┌───────────────────┐
│ 미완료 작업 있음?  │
└───────────────────┘
        │
    YES │ NO
        │  │
        ▼  ▼
     작업   ┌───────────────────┐
     계속   │ completion_indicators │
            │ 확인                │
            └───────────────────┘
                    │
                    ▼
            ┌───────────────────┐
            │ EXIT_SIGNAL 설정? │
            └───────────────────┘
                 │         │
              NO │         │ YES
                 │         │
                 ▼         ▼
            추가 검토    종료 확인
            또는 대기    루프 종료
```

## EXIT_SIGNAL

### 개념

`EXIT_SIGNAL`은 Ralph가 프로젝트 완료를 명시적으로 선언하는 메커니즘입니다.

```javascript
// status.json 구조
{
  "session_id": "abc123",
  "loop_count": 15,
  "last_activity": "2024-01-15T10:30:00Z",
  "EXIT_SIGNAL": false,  // 진행 중
  "completion_indicators": {
    "all_tasks_checked": false,
    "tests_passing": true,
    "no_errors": true
  }
}
```

### EXIT_SIGNAL 설정 조건

Ralph가 EXIT_SIGNAL을 true로 설정하는 경우:

```
1. fix_plan.md의 모든 [ ] 가 [x]로 변경됨
2. 모든 테스트 통과
3. 빌드 성공
4. 추가 작업 없음 확인
```

### completion_indicators

```javascript
const completion_indicators = {
  all_tasks_checked: boolean,     // 모든 체크박스 완료
  tests_passing: boolean,         // 테스트 통과
  no_errors: boolean,             // 에러 없음
  build_successful: boolean,      // 빌드 성공
  no_pending_issues: boolean      // 미해결 이슈 없음
};
```

## 세션과 루프

### 용어 정의

| 용어 | 정의 | 지속 시간 |
|------|------|-----------|
| **세션(Session)** | Ralph 실행 전체 | `ralph` 시작부터 종료까지 |
| **루프(Loop)** | 단일 작업 사이클 | 몇 분 |
| **이터레이션** | Claude API 호출 1회 | 몇 초 |

### 세션 상태

```
┌─────────────────────────────────────────────────────────────┐
│                         SESSION                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │ Loop 1  │ → │ Loop 2  │ → │ Loop 3  │ → │ Loop N  │     │
│  │ Task A  │   │ Task B  │   │ Task C  │   │ Exit    │     │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
│       │             │             │             │           │
│       ▼             ▼             ▼             ▼           │
│  status.json 업데이트 (매 루프)                              │
└─────────────────────────────────────────────────────────────┘
```

## 컨텍스트 관리

### 컨텍스트 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    RALPH CONTEXT                            │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Project Vision (PROMPT.md)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - Project goals                                      │   │
│  │ - Key principles                                     │   │
│  │ - Technology stack                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                              │
│                              ▼                              │
│  Layer 2: Detailed Specs (specs/)                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - API contracts                                      │   │
│  │ - Data models                                        │   │
│  │ - Integration requirements                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                              │
│                              ▼                              │
│  Layer 3: Current Tasks (fix_plan.md)                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - [ ] Pending tasks                                  │   │
│  │ - [x] Completed tasks                               │   │
│  │ - Priority order                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                              │
│                              ▼                              │
│  Layer 4: Build Instructions (AGENT.md)                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ - Build commands                                     │   │
│  │ - Test commands                                      │   │
│  │ - Environment setup                                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 컨텍스트 우선순위

1. **PROMPT.md**: 항상 로드 (높은 우선순위)
2. **fix_plan.md**: 항상 로드 (현재 작업 결정)
3. **관련 specs/**: 필요시 로드
4. **AGENT.md**: 빌드/테스트 시 로드

## 작업 상태 전이

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ PENDING  │ -> │ ACTIVE   │ -> │ COMPLETE │
│   [ ]    │    │ (작업 중) │    │   [x]    │
└──────────┘    └──────────┘    └──────────┘
     │                               │
     │                               │
     └─────────────────────────────┘
              (발견된 새 작업)
```

### 상태 전이 규칙

| 현재 상태 | 이벤트 | 다음 상태 |
|-----------|--------|-----------|
| PENDING | 작업 시작 | ACTIVE |
| ACTIVE | 구현 완료 | COMPLETE |
| ACTIVE | 블로커 발견 | PENDING + 새 작업 |
| COMPLETE | - | (변경 없음) |

---

**이전 장:** [파일 구조](/ralph-guide-03-files/) | **다음 장:** [CLI 명령어](/ralph-guide-05-commands/)
