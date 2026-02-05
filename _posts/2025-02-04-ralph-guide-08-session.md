---
layout: post
title: "Ralph 가이드 08 - 세션 관리"
date: 2025-02-04
categories: [AI, Claude Code, Ralph]
tags: [ralph, session, continuity, state-management]
permalink: /ralph-guide-08-session/
---

# 세션 관리

## 세션 개념

### 세션이란?

Ralph 세션은 `ralph` 명령 실행부터 종료까지의 전체 작업 단위입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                      RALPH SESSION                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Session ID: my-project-abc123                               │
│  Started: 2024-01-15 10:00:00                               │
│  Status: RUNNING                                             │
│                                                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │ Loop 1  │ → │ Loop 2  │ → │ Loop 3  │ → │ Loop N  │     │
│  │ 10:00   │   │ 10:05   │   │ 10:12   │   │ ...     │     │
│  │ Task A  │   │ Task B  │   │ Task B  │   │         │     │
│  │ ✓       │   │ ✓       │   │ (fix)   │   │         │     │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
│                                                              │
│  Ended: 2024-01-15 11:30:00                                 │
│  Duration: 1h 30m                                            │
│  Tasks Completed: 12                                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 세션 vs 루프

| 구분 | 세션 (Session) | 루프 (Loop) |
|------|---------------|-------------|
| 범위 | 전체 실행 | 단일 작업 사이클 |
| 지속 시간 | 수 시간 | 수 분 |
| 상태 저장 | 종료 시 | 매 루프 후 |
| 재개 가능 | 예 | 아니오 |

## 세션 연속성

### 상태 저장

Ralph는 매 루프 후 상태를 저장합니다:

```json
// .ralph/status.json
{
  "session_id": "my-project-abc123",
  "started_at": "2024-01-15T10:00:00Z",
  "last_activity": "2024-01-15T10:45:30Z",
  "loop_count": 15,
  "status": "RUNNING",

  "progress": {
    "tasks_total": 12,
    "tasks_completed": 7,
    "current_task": "Implement user authentication"
  },

  "api_usage": {
    "calls_this_hour": 45,
    "calls_total": 67,
    "hour_started": "2024-01-15T10:00:00Z"
  },

  "circuit_breaker": {
    "state": "CLOSED",
    "error_count": 0
  },

  "EXIT_SIGNAL": false,
  "completion_indicators": {
    "all_tasks_checked": false,
    "tests_passing": true,
    "no_errors": true
  }
}
```

### 세션 재개

```bash
# 이전 세션 재개
ralph --resume

# 특정 세션 재개
ralph --resume --session my-project-abc123
```

재개 시 복원되는 것:

| 항목 | 복원됨 | 설명 |
|------|--------|------|
| 작업 진행 상황 | ✓ | fix_plan.md의 체크 상태 |
| 루프 카운트 | ✓ | 연속 번호 유지 |
| API 사용량 | ✓ | 시간당 호출 카운트 |
| 서킷 브레이커 | ✓ | 에러 카운트 유지 |
| 대화 컨텍스트 | ✗ | 새로 시작 |

### 연속성 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                    SESSION CONTINUITY                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Session Start                                               │
│       │                                                      │
│       ▼                                                      │
│  ┌────────────────────────┐                                 │
│  │ status.json 확인        │                                 │
│  └────────────────────────┘                                 │
│       │                                                      │
│       ├── 없음 ──────────────────┐                          │
│       │                          │                           │
│       ▼                          ▼                           │
│  ┌─────────────────┐     ┌─────────────────┐               │
│  │ 기존 세션 로드   │     │ 새 세션 생성     │               │
│  │ 상태 복원        │     │ ID 생성         │               │
│  └─────────────────┘     └─────────────────┘               │
│       │                          │                           │
│       └──────────┬───────────────┘                          │
│                  │                                           │
│                  ▼                                           │
│         ┌─────────────────┐                                 │
│         │ 루프 실행        │                                 │
│         └─────────────────┘                                 │
│                  │                                           │
│                  ▼                                           │
│         ┌─────────────────┐                                 │
│         │ 상태 저장        │                                 │
│         │ status.json     │                                 │
│         └─────────────────┘                                 │
│                  │                                           │
│                  ▼                                           │
│             다음 루프                                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 세션 만료

### 만료 조건

| 조건 | 설정 | 결과 |
|------|------|------|
| 타임아웃 | `SESSION_TIMEOUT` | 세션 일시 중지 |
| 유휴 타임아웃 | `IDLE_TIMEOUT` | 세션 일시 중지 |
| 최대 루프 | `MAX_LOOPS_PER_SESSION` | 세션 종료 |
| API 한도 | `MAX_CALLS_PER_HOUR` | 쿨다운 후 계속 |

### 만료 처리

```
세션 실행 중
     │
     ▼
┌────────────────────────┐
│ 만료 조건 체크          │
└────────────────────────┘
     │
     ├── SESSION_TIMEOUT ────────┐
     │                           │
     ├── IDLE_TIMEOUT ──────────┤
     │                           │
     ├── MAX_LOOPS ─────────────┤
     │                           │
     └── MAX_CALLS ────────┐    │
                           │    │
                           ▼    ▼
                    ┌────────────────┐
                    │ 쿨다운         │
                    │ (재개 가능)    │
                    └────────────────┘
                           │
                           ▼
                    ┌────────────────┐
                    │ 상태 저장       │
                    │ 재개 대기       │
                    └────────────────┘
```

### 만료 후 재개

```bash
# 만료된 세션 확인
ralph --status

# 출력:
# Session: my-project-abc123
# Status: PAUSED (timeout)
# Can resume: yes
# Time paused: 2h 15m

# 재개
ralph --resume
```

## 리셋 트리거

### 자동 리셋

특정 상황에서 세션이 자동으로 리셋됩니다:

| 트리거 | 동작 | 복구 방법 |
|--------|------|-----------|
| `fix_plan.md` 변경 감지 | 작업 목록 다시 읽기 | 자동 |
| `PROMPT.md` 변경 감지 | 컨텍스트 재로드 | 자동 |
| 치명적 에러 | 세션 중지 | 수동 재개 |
| 버전 업그레이드 | 호환성 체크 | 새 세션 시작 |

### 수동 리셋

```bash
# 세션 완전 리셋
ralph --reset

# 상태 파일 삭제
rm .ralph/status.json

# 로그 정리
rm -rf .ralph/logs/*

# 새 세션으로 시작
ralph --monitor
```

### 부분 리셋

```bash
# 서킷 브레이커만 리셋
ralph --reset-circuit

# API 카운터만 리셋
ralph --reset-api-counter

# 루프 카운터만 리셋
ralph --reset-loop-counter
```

## 세션 ID 관리

### ID 형식

```
{project_name}-{random_hash}

예: my-project-abc123
예: todo-cli-xyz789
```

### 세션 목록 확인

```bash
# 로그 디렉토리에서 세션 확인
ls .ralph/logs/

# 출력:
# session_my-project-abc123.log
# session_my-project-def456.log
# session_latest.log -> session_my-project-abc123.log
```

### 세션 정리

```bash
# 오래된 세션 로그 정리
find .ralph/logs -name "session_*.log" -mtime +7 -delete

# 마지막 3개만 유지
ls -t .ralph/logs/session_*.log | tail -n +4 | xargs rm -f
```

## 다중 세션

### 동시 실행 방지

```
┌─────────────────────────────────────────────────────────────┐
│                  CONCURRENT SESSION CHECK                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ralph 실행 시도                                             │
│       │                                                      │
│       ▼                                                      │
│  ┌────────────────────────┐                                 │
│  │ .ralph/lock 파일 확인   │                                 │
│  └────────────────────────┘                                 │
│       │                                                      │
│       ├── 없음 ──────────────────┐                          │
│       │                          │                           │
│       ▼                          ▼                           │
│  ┌─────────────────┐     ┌─────────────────┐               │
│  │ 에러 메시지      │     │ lock 파일 생성   │               │
│  │ "세션 실행 중"   │     │ 세션 시작        │               │
│  └─────────────────┘     └─────────────────┘               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 잠금 파일

```bash
# 잠금 파일 내용
cat .ralph/lock
# PID: 12345
# Started: 2024-01-15T10:00:00Z
# Session: my-project-abc123

# 강제 잠금 해제 (주의!)
rm .ralph/lock
```

## 세션 모니터링

### 실시간 상태

```bash
# 상태 지속 확인
watch -n 5 ralph --status

# 로그 스트리밍
tail -f .ralph/logs/session_latest.log
```

### 상태 요약

```bash
ralph --status --json | jq '.'
```

출력:
```json
{
  "session_id": "my-project-abc123",
  "status": "RUNNING",
  "duration": "1h 30m",
  "loops": 45,
  "tasks": {
    "completed": 8,
    "remaining": 4,
    "total": 12
  },
  "api": {
    "calls_this_hour": 67,
    "limit": 100
  },
  "circuit": "CLOSED"
}
```

---

**이전 장:** [서킷 브레이커](/ralph-guide-07-circuit-breaker/) | **다음 장:** [모니터링](/ralph-guide-09-monitoring/)
