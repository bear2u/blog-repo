---
layout: post
title: "Ralph 가이드 07 - 서킷 브레이커"
date: 2025-02-04
categories: [AI 코딩 에이전트, Ralph]
tags: [ralph, circuit-breaker, error-handling, safety]
permalink: /ralph-guide-07-circuit-breaker/
---

# 서킷 브레이커

## 개념

서킷 브레이커는 전기 회로의 차단기에서 영감을 받은 패턴입니다. 연속적인 실패를 감지하면 자동으로 실행을 중단하여 시스템을 보호합니다.

### 왜 필요한가?

| 문제 상황 | 서킷 브레이커 없이 | 서킷 브레이커 있을 때 |
|----------|-----------------|-------------------|
| 무한 루프 | API 비용 폭증 | 자동 중단 |
| 반복 에러 | 같은 실패 반복 | 일시 중지 후 재시도 |
| 잘못된 프롬프트 | 의미없는 작업 반복 | 인간 개입 요청 |
| API 장애 | 연속 실패 | 백오프 후 복구 |

## 상태 전이

```
┌───────────────────────────────────────────────────────────────┐
│                    CIRCUIT BREAKER STATES                      │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│                    정상 작동                                    │
│                        │                                       │
│                        ▼                                       │
│            ┌──────────────────────┐                           │
│            │       CLOSED         │◄───────────────┐          │
│            │    (정상 운영)        │               │          │
│            └──────────────────────┘               │          │
│                        │                          │          │
│              에러 누적 │                          │          │
│         (threshold 도달)│                          │          │
│                        ▼                          │          │
│            ┌──────────────────────┐               │          │
│            │        OPEN          │               │          │
│            │   (모든 요청 차단)     │               │          │
│            └──────────────────────┘               │          │
│                        │                          │          │
│           타임아웃 후  │                          │          │
│                        ▼                          │          │
│            ┌──────────────────────┐      성공     │          │
│            │     HALF-OPEN        │───────────────┘          │
│            │  (테스트 요청 허용)   │                          │
│            └──────────────────────┘                          │
│                        │                                      │
│                   실패 │                                      │
│                        ▼                                      │
│                   다시 OPEN                                   │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 상태 설명

| 상태 | 동작 | 전환 조건 |
|------|------|-----------|
| **CLOSED** | 정상 운영, 모든 요청 처리 | 에러가 threshold 도달 → OPEN |
| **OPEN** | 모든 요청 즉시 거부 | 타임아웃 경과 → HALF-OPEN |
| **HALF-OPEN** | 테스트 요청 1개 허용 | 성공 → CLOSED, 실패 → OPEN |

## 에러 감지

### 감지되는 에러 유형

```javascript
// Ralph가 감지하는 에러 패턴
const errorPatterns = {
  // API 에러
  api_errors: [
    "rate_limit_exceeded",
    "context_length_exceeded",
    "invalid_request",
    "authentication_error"
  ],

  // 실행 에러
  execution_errors: [
    "command_failed",
    "test_failed_repeatedly",
    "build_failed",
    "timeout_exceeded"
  ],

  // 논리 에러
  logic_errors: [
    "same_task_repeated",      // 같은 작업 3회 이상 반복
    "no_progress_detected",    // 진행 없음
    "circular_dependency",     // 순환 의존성
    "conflicting_changes"      // 충돌하는 변경
  ]
};
```

### 에러 카운팅

```
Loop 실행
    │
    ▼
┌────────────────────┐
│ 작업 실행          │
└────────────────────┘
    │
    ├── 성공 ──────────────────┐
    │                          │
    ▼                          ▼
┌────────────────────┐   ┌────────────────────┐
│ 에러 발생          │   │ 에러 카운터 리셋    │
│ error_count++      │   │ error_count = 0    │
└────────────────────┘   └────────────────────┘
    │
    ▼
┌────────────────────────────┐
│ error_count >= threshold?  │
└────────────────────────────┘
    │
   YES ────────────┐
    │              │
    ▼              ▼
계속 실행    서킷 OPEN
```

## 설정

### 기본 설정

```bash
# .ralphrc
CIRCUIT_BREAKER_ENABLED=true
ERROR_THRESHOLD=5              # OPEN 전환까지 허용 에러 수
HALF_OPEN_TIMEOUT=300          # OPEN → HALF-OPEN 대기 시간 (초)
CONSECUTIVE_FAILURES=3         # 연속 실패 횟수
```

### 설정 옵션

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `CIRCUIT_BREAKER_ENABLED` | true | 서킷 브레이커 활성화 |
| `ERROR_THRESHOLD` | 5 | OPEN 전환까지 누적 에러 수 |
| `HALF_OPEN_TIMEOUT` | 300 | HALF-OPEN 전환 대기 시간 |
| `CONSECUTIVE_FAILURES` | 3 | 연속 실패 시 즉시 OPEN |
| `SUCCESS_THRESHOLD` | 2 | CLOSED 복귀까지 연속 성공 수 |

### 시나리오별 설정

**엄격한 설정 (프로덕션):**
```bash
ERROR_THRESHOLD=3
CONSECUTIVE_FAILURES=2
HALF_OPEN_TIMEOUT=600
```

**느슨한 설정 (개발):**
```bash
ERROR_THRESHOLD=10
CONSECUTIVE_FAILURES=5
HALF_OPEN_TIMEOUT=120
```

## 자동 복구

### 복구 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                      AUTO RECOVERY                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   OPEN 상태                                                  │
│       │                                                      │
│       ▼                                                      │
│   ┌────────────────────────┐                                │
│   │ 타이머 시작             │                                │
│   │ wait(HALF_OPEN_TIMEOUT)│                                │
│   └────────────────────────┘                                │
│       │                                                      │
│       ▼                                                      │
│   HALF-OPEN 전환                                            │
│       │                                                      │
│       ▼                                                      │
│   ┌────────────────────────┐                                │
│   │ 테스트 루프 실행        │                                │
│   │ (가벼운 작업)          │                                │
│   └────────────────────────┘                                │
│       │                                                      │
│       ├── 성공 ─────────┐                                   │
│       │                 │                                    │
│       ▼                 ▼                                    │
│   ┌─────────┐    ┌─────────────────┐                       │
│   │  OPEN   │    │ success_count++ │                       │
│   │ (재시도)│    └─────────────────┘                       │
│   └─────────┘           │                                   │
│                         ▼                                    │
│                 ┌──────────────────────┐                    │
│                 │ success >= threshold?│                    │
│                 └──────────────────────┘                    │
│                         │                                    │
│                    YES  │                                    │
│                         ▼                                    │
│                 ┌─────────────┐                             │
│                 │   CLOSED    │                             │
│                 │ (정상 복귀)  │                             │
│                 └─────────────┘                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 복구 전략

| 전략 | 설명 | 사용 시점 |
|------|------|-----------|
| **즉시 재시도** | HALF-OPEN에서 바로 다음 작업 | 일시적 네트워크 오류 |
| **백오프** | 점진적으로 대기 시간 증가 | API 레이트 리밋 |
| **수동 복구** | 인간 개입 필요 | 논리적 오류 |

### 지수 백오프

```javascript
// 백오프 계산
function calculateBackoff(attempt) {
  const base = 60;  // 기본 60초
  const max = 3600; // 최대 1시간
  const backoff = Math.min(base * Math.pow(2, attempt), max);
  return backoff + Math.random() * 30; // 지터 추가
}

// 시도별 대기 시간
// 1차: 60초 + 랜덤
// 2차: 120초 + 랜덤
// 3차: 240초 + 랜덤
// 4차: 480초 + 랜덤
// 5차+: 3600초 + 랜덤
```

## 모니터링

### 상태 확인

```bash
# 현재 서킷 상태 확인
ralph --status
```

출력:
```
Circuit Breaker Status
======================
State: HALF-OPEN
Error Count: 4/5
Last Error: test_failed_repeatedly (2 min ago)
Recovery Attempts: 2
Next Test: in 3 minutes
```

### status.json 구조

```json
{
  "circuit_breaker": {
    "state": "HALF-OPEN",
    "error_count": 4,
    "consecutive_failures": 2,
    "last_error": {
      "type": "test_failed_repeatedly",
      "message": "Test suite failed 3 times",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    "recovery": {
      "attempts": 2,
      "last_attempt": "2024-01-15T10:35:00Z",
      "next_attempt": "2024-01-15T10:40:00Z"
    }
  }
}
```

### 로그 메시지

```
[WARN] Circuit breaker triggered: error_count=5
[INFO] Circuit state: CLOSED -> OPEN
[INFO] Waiting 300s before recovery attempt
[INFO] Circuit state: OPEN -> HALF-OPEN
[INFO] Testing recovery with lightweight task
[INFO] Recovery successful, circuit state: HALF-OPEN -> CLOSED
```

## 수동 개입

### 강제 리셋

```bash
# 서킷 브레이커 강제 리셋
ralph --reset-circuit

# 상태 파일 직접 수정
cat > .ralph/status.json << EOF
{
  "circuit_breaker": {
    "state": "CLOSED",
    "error_count": 0
  }
}
EOF
```

### 일시 비활성화

```bash
# 세션 동안 비활성화
CIRCUIT_BREAKER_ENABLED=false ralph --monitor

# .ralphrc에서 비활성화
echo "CIRCUIT_BREAKER_ENABLED=false" >> .ralphrc
```

---

**이전 장:** [구성 및 설정](/ralph-guide-06-configuration/) | **다음 장:** [세션 관리](/ralph-guide-08-session/)
