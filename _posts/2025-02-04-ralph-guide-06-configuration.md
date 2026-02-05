---
layout: post
title: "Ralph 가이드 06 - 구성 및 설정"
date: 2025-02-04
categories: [AI 코딩 에이전트, Ralph]
tags: [ralph, configuration, settings, rate-limit]
permalink: /ralph-guide-06-configuration/
---

# 구성 및 설정

## 설정 파일 계층

Ralph는 여러 수준의 설정을 지원합니다:

```
┌─────────────────────────────────────────────────────────────┐
│  Priority (낮음 → 높음)                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 기본값 (내장)                                            │
│     │                                                        │
│     ▼                                                        │
│  2. 글로벌 설정 (~/.ralphrc)                                 │
│     │                                                        │
│     ▼                                                        │
│  3. 프로젝트 설정 (.ralphrc)                                 │
│     │                                                        │
│     ▼                                                        │
│  4. 환경 변수 (RALPH_*)                                      │
│     │                                                        │
│     ▼                                                        │
│  5. CLI 옵션 (--option)                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## .ralphrc 파일

### 위치

| 파일 | 적용 범위 |
|------|-----------|
| `~/.ralphrc` | 모든 프로젝트 (글로벌) |
| `<project>/.ralphrc` | 해당 프로젝트만 |

### 기본 구조

```bash
# .ralphrc - Ralph Configuration

#======================================
# Project Information
#======================================
PROJECT_NAME="my-project"
PROJECT_TYPE="typescript"

#======================================
# Rate Limiting
#======================================
MAX_CALLS_PER_HOUR=100
MAX_LOOPS_PER_SESSION=200
COOLDOWN_MINUTES=5

#======================================
# Timeouts
#======================================
SESSION_TIMEOUT=3600        # 1 hour
LOOP_TIMEOUT=300            # 5 minutes
IDLE_TIMEOUT=600            # 10 minutes

#======================================
# Circuit Breaker
#======================================
CIRCUIT_BREAKER_ENABLED=true
ERROR_THRESHOLD=5
HALF_OPEN_TIMEOUT=300

#======================================
# Tool Permissions
#======================================
ALLOWED_TOOLS="Write,Read,Edit,Bash(git *),Bash(npm *),Bash(pytest)"

#======================================
# Monitoring
#======================================
ENABLE_DASHBOARD=true
LOG_LEVEL="info"
```

## 속도 제한 설정

### 기본 설정

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `MAX_CALLS_PER_HOUR` | 100 | 시간당 최대 API 호출 수 |
| `MAX_LOOPS_PER_SESSION` | 200 | 세션당 최대 루프 수 |
| `COOLDOWN_MINUTES` | 5 | 한도 도달 시 대기 시간 |

### 속도 제한 동작

```
API 호출
    │
    ▼
┌────────────────────────┐
│ 호출 카운터 증가        │
│ current_calls++        │
└────────────────────────┘
    │
    ▼
┌────────────────────────┐
│ 한도 확인               │
│ current >= MAX_CALLS?  │
└────────────────────────┘
    │
   YES ────────────────┐
    │                  │
    ▼                  ▼
계속 실행         ┌────────────────────────┐
                  │ 쿨다운 대기             │
                  │ sleep(COOLDOWN_MINUTES)│
                  └────────────────────────┘
                           │
                           ▼
                  ┌────────────────────────┐
                  │ 카운터 리셋             │
                  │ current_calls = 0      │
                  └────────────────────────┘
```

### 사용 시나리오별 설정

**개발/테스트 (보수적):**
```bash
MAX_CALLS_PER_HOUR=50
MAX_LOOPS_PER_SESSION=100
```

**프로덕션 작업 (적극적):**
```bash
MAX_CALLS_PER_HOUR=200
MAX_LOOPS_PER_SESSION=500
```

**빠른 프로토타이핑:**
```bash
MAX_CALLS_PER_HOUR=300
COOLDOWN_MINUTES=2
```

## 타임아웃 설정

### 타임아웃 유형

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `SESSION_TIMEOUT` | 3600초 (1시간) | 전체 세션 제한 |
| `LOOP_TIMEOUT` | 300초 (5분) | 단일 루프 제한 |
| `IDLE_TIMEOUT` | 600초 (10분) | 비활성 상태 제한 |
| `API_TIMEOUT` | 120초 (2분) | API 응답 대기 |

### 타임아웃 처리

```
┌─────────────────────────────────────────────────────────────┐
│                    타임아웃 체크                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Session Start                                               │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Loop 실행                                            │   │
│  │                                                      │   │
│  │   if (now - loop_start > LOOP_TIMEOUT) {            │   │
│  │     saveState();                                     │   │
│  │     restartLoop();                                   │   │
│  │   }                                                  │   │
│  │                                                      │   │
│  │   if (now - last_activity > IDLE_TIMEOUT) {         │   │
│  │     saveState();                                     │   │
│  │     pauseSession();                                  │   │
│  │   }                                                  │   │
│  │                                                      │   │
│  │   if (now - session_start > SESSION_TIMEOUT) {      │   │
│  │     saveState();                                     │   │
│  │     endSession();                                    │   │
│  │   }                                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 도구 권한 설정

### ALLOWED_TOOLS 형식

```bash
# 기본 도구만 허용
ALLOWED_TOOLS="Write,Read,Edit"

# Bash 명령어 패턴 허용
ALLOWED_TOOLS="Write,Read,Edit,Bash(git *),Bash(npm *)"

# 특정 Bash 명령어만 허용
ALLOWED_TOOLS="Write,Read,Edit,Bash(git status),Bash(npm test)"

# 모든 Bash 허용 (주의!)
ALLOWED_TOOLS="Write,Read,Edit,Bash(*)"
```

### 권한 패턴 예시

| 패턴 | 허용되는 명령 |
|------|--------------|
| `Bash(git *)` | `git status`, `git commit`, `git push` 등 |
| `Bash(npm *)` | `npm install`, `npm test`, `npm run build` 등 |
| `Bash(pytest)` | `pytest`만 |
| `Bash(docker *)` | 모든 docker 명령 |

### 보안 권장 설정

```bash
# 최소 권한 (권장)
ALLOWED_TOOLS="Write,Read,Edit,Bash(git status),Bash(git add),Bash(git commit),Bash(npm test)"

# 개발 환경
ALLOWED_TOOLS="Write,Read,Edit,Bash(git *),Bash(npm *),Bash(npx *)"

# 위험 - 피해야 함
ALLOWED_TOOLS="Bash(*)"  # 모든 명령 허용
```

## 환경별 설정

### 개발 환경

```bash
# .ralphrc (development)
PROJECT_NAME="my-app-dev"
PROJECT_TYPE="typescript"

# 느슨한 제한
MAX_CALLS_PER_HOUR=200
SESSION_TIMEOUT=7200

# 상세 로깅
LOG_LEVEL="debug"
ENABLE_DASHBOARD=true

# 넓은 권한
ALLOWED_TOOLS="Write,Read,Edit,Bash(git *),Bash(npm *),Bash(npx *)"
```

### 프로덕션 환경

```bash
# .ralphrc (production)
PROJECT_NAME="my-app-prod"
PROJECT_TYPE="typescript"

# 엄격한 제한
MAX_CALLS_PER_HOUR=100
MAX_LOOPS_PER_SESSION=100
SESSION_TIMEOUT=3600

# 최소 로깅
LOG_LEVEL="warn"

# 제한된 권한
ALLOWED_TOOLS="Write,Read,Edit,Bash(git status),Bash(npm test)"

# 서킷 브레이커 활성화
CIRCUIT_BREAKER_ENABLED=true
ERROR_THRESHOLD=3
```

### CI/CD 환경

```bash
# .ralphrc (ci)
PROJECT_NAME="my-app-ci"

# 짧은 타임아웃
SESSION_TIMEOUT=1800
LOOP_TIMEOUT=180

# 실패 즉시 중지
ERROR_THRESHOLD=1
CIRCUIT_BREAKER_ENABLED=true

# 읽기 위주 작업
ALLOWED_TOOLS="Read,Bash(npm test),Bash(npm run lint)"
```

## 설정 검증

### 설정 확인 명령

```bash
# 현재 적용된 설정 확인
ralph --config

# 특정 설정 값 확인
ralph --config MAX_CALLS_PER_HOUR
```

### 일반적인 설정 오류

| 오류 | 원인 | 해결 |
|------|------|------|
| 세션 즉시 종료 | `SESSION_TIMEOUT=0` | 적절한 값 설정 |
| API 호출 안됨 | `MAX_CALLS_PER_HOUR=0` | 양수 값 설정 |
| 무한 루프 | `ERROR_THRESHOLD` 너무 높음 | 5 이하로 설정 |
| 권한 오류 | `ALLOWED_TOOLS` 누락 | 필요한 도구 추가 |

---

**이전 장:** [CLI 명령어](/ralph-guide-05-commands/) | **다음 장:** [서킷 브레이커](/ralph-guide-07-circuit-breaker/)
