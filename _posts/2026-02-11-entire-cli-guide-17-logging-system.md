---
layout: post
title: "Entire CLI 완벽 가이드 (17) - Logging 시스템"
date: 2026-02-11
permalink: /entire-cli-guide-17-logging-system/
author: Entire Team
categories: [AI 코딩, 개발 도구]
tags: [Entire, Logging, Privacy, Structured Logging, slog]
original_url: "https://github.com/entireio/cli"
excerpt: "Entire CLI의 구조화된 로깅 시스템과 프라이버시 보호 메커니즘 완벽 이해"
---

## 개요

Entire CLI는 **구조화된 로깅(structured logging)**을 통해 모든 작업을 기록합니다. 하지만 사용자 컨텐츠는 절대 로깅하지 않아 **프라이버시를 보호**합니다. 이 챕터에서는 Entire의 로깅 시스템 아키텍처와 사용법을 살펴봅니다.

```
┌─────────────────────────────────────────────────────┐
│                 Entire CLI Logging                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Internal Operations          User-Facing Output    │
│         │                            │               │
│         ▼                            ▼               │
│  .entire/logs/entire.log      stdout/stderr         │
│  (slog JSON format)           (human-readable)       │
│                                                      │
│  - Session IDs                - Success messages     │
│  - Tool Call IDs              - Error messages       │
│  - File paths                 - Prompts             │
│  - Durations                                         │
│  - Counts                                            │
│                                                      │
│  ✗ NO USER CONTENT            ✓ User content OK     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 로깅 패키지 (cmd/entire/cli/logging)

### 아키텍처

Entire는 Go의 표준 `log/slog` 패키지를 사용합니다.

```go
package logging

import (
    "log/slog"
)

// 패키지 레벨 logger
var logger *slog.Logger

// 로그 파일
var logFile *os.File
```

**주요 특징:**

- **JSON 형식** - 구조화된 로그, 파싱 가능
- **버퍼링** - 8KB 버퍼로 성능 최적화
- **Context 기반** - 세션 ID, 도구 ID 자동 추가
- **레벨 제어** - DEBUG, INFO, WARN, ERROR

---

## 로그 파일 위치

### 저장 경로

```
.entire/logs/entire.log
```

**특징:**

- Repository root 기준 상대 경로
- 모든 세션의 로그가 하나의 파일에 추가됨
- Session ID로 필터링 가능
- **`.gitignore`에 자동 추가됨** - 커밋되지 않음

### 파일 권한

```bash
# 디렉토리 권한
.entire/logs/          0750 (rwxr-x---)

# 파일 권한
.entire/logs/entire.log  0600 (rw-------)
```

사용자만 읽고 쓸 수 있습니다.

---

## 초기화 및 종료

### Init() - 로거 초기화

```go
func Init(sessionID string) error
```

**역할:**

1. 로그 파일 열기 (또는 생성)
2. 버퍼링된 writer 설정
3. slog.Logger 인스턴스 생성
4. Session ID 저장 (모든 로그에 자동 추가)

**사용 예시:**

```go
import "github.com/entireio/cli/cmd/entire/cli/logging"

// 세션 시작 시
if err := logging.Init(sessionID); err != nil {
    return fmt.Errorf("failed to initialize logging: %w", err)
}
defer logging.Close()
```

**Fallback:**

로그 파일을 생성할 수 없으면 `stderr`로 fallback합니다.

```go
if err := os.MkdirAll(logsPath, 0o750); err != nil {
    // stderr로 fallback
    logger = createLogger(os.Stderr, level)
    return nil
}
```

### Close() - 로거 종료

```go
func Close()
```

**역할:**

1. 버퍼 flush (쓰지 않은 데이터 기록)
2. 로그 파일 닫기

**특징:**

- 여러 번 호출해도 안전 (idempotent)
- `defer`와 함께 사용

---

## 로그 레벨

### 지원되는 레벨

| 레벨 | slog 상수 | 용도 |
|-----|----------|-----|
| **DEBUG** | `slog.LevelDebug` | 상세한 디버깅 정보 |
| **INFO** | `slog.LevelInfo` | 일반 작업 정보 (기본) |
| **WARN** | `slog.LevelWarn` | 경고 메시지 |
| **ERROR** | `slog.LevelError` | 에러 메시지 |

### 레벨 설정

#### 1. 환경 변수 (최우선)

```bash
export ENTIRE_LOG_LEVEL=debug
entire status

# 또는 일회성
ENTIRE_LOG_LEVEL=debug entire status
```

#### 2. Settings 파일 (fallback)

```json
{
  "log_level": "debug"
}
```

**유효한 값:**

- `"debug"`, `"DEBUG"`
- `"info"`, `"INFO"` (기본)
- `"warn"`, `"WARN"`
- `"error"`, `"ERROR"`

**잘못된 값 처리:**

```go
func parseLogLevel(levelStr string) slog.Level {
    switch strings.ToUpper(levelStr) {
    case "DEBUG":
        return slog.LevelDebug
    case "WARN":
        return slog.LevelWarn
    case "ERROR":
        return slog.LevelError
    default:
        return slog.LevelInfo // 기본값
    }
}
```

잘못된 레벨은 경고와 함께 INFO로 fallback:

```
[entire] Warning: invalid log level "verbose", defaulting to INFO
```

---

## Context 기반 로깅

### Context에 데이터 추가

```go
// Session ID 추가
ctx = logging.WithSession(ctx, sessionID)

// Tool Call ID 추가
ctx = logging.WithToolCall(ctx, toolCallID)

// Agent 추가
ctx = logging.WithAgent(ctx, "claude-code")

// Component 추가
ctx = logging.WithComponent(ctx, "rewind")
```

### Context에서 데이터 추출

```go
func Debug(ctx context.Context, msg string, attrs ...slog.Attr)
func Info(ctx context.Context, msg string, attrs ...slog.Attr)
func Warn(ctx context.Context, msg string, attrs ...slog.Attr)
func Error(ctx context.Context, msg string, attrs ...slog.Attr)
```

**자동으로 추가되는 필드:**

- `session_id` - Context에서 추출
- `tool_call_id` - Context에서 추출 (있으면)
- `agent` - Context에서 추출 (있으면)
- `component` - Context에서 추출 (있으면)

**사용 예시:**

```go
ctx = logging.WithSession(ctx, "2026-02-11-abc123...")
ctx = logging.WithComponent(ctx, "rewind")

logging.Info(ctx, "rewind started",
    slog.String("checkpoint_id", "a3b2c4d5e6f7"),
    slog.Int("file_count", 10),
)
```

**로그 출력:**

```json
{
  "time": "2026-02-11T10:30:00Z",
  "level": "INFO",
  "msg": "rewind started",
  "session_id": "2026-02-11-abc123...",
  "component": "rewind",
  "checkpoint_id": "a3b2c4d5e6f7",
  "file_count": 10
}
```

---

## 로그 형식

### JSON Lines (JSONL)

각 로그 라인은 독립적인 JSON 객체입니다.

```jsonl
{"time":"2026-02-11T10:30:00Z","level":"INFO","msg":"session started","session_id":"2026-02-11-abc123..."}
{"time":"2026-02-11T10:30:05Z","level":"DEBUG","msg":"hook invoked","session_id":"2026-02-11-abc123...","hook":"UserPromptSubmit"}
{"time":"2026-02-11T10:30:10Z","level":"INFO","msg":"checkpoint saved","session_id":"2026-02-11-abc123...","checkpoint_id":"a3b2c4d5e6f7"}
```

**장점:**

- 라인별로 파싱 가능
- `jq`, `grep` 등 표준 도구로 필터링 가능
- 스트리밍 처리 가능

### 필드 설명

| 필드 | 타입 | 설명 |
|-----|------|-----|
| `time` | ISO 8601 | UTC 타임스탬프 |
| `level` | string | 로그 레벨 (DEBUG, INFO, WARN, ERROR) |
| `msg` | string | 로그 메시지 |
| `session_id` | string | 세션 ID (자동 추가) |
| `checkpoint_id` | string | 체크포인트 ID (해당 시) |
| `tool_call_id` | string | 도구 호출 ID (해당 시) |
| `agent` | string | 에이전트 이름 |
| `component` | string | 컴포넌트 이름 |
| `error` | string | 에러 메시지 (ERROR 레벨) |

**커스텀 필드:**

추가 속성은 `slog.Attr`로 전달:

```go
logging.Info(ctx, "condensation completed",
    slog.Duration("duration", elapsed),
    slog.Int("file_count", len(files)),
    slog.String("branch", branchName),
)
```

---

## 프라이버시 보호

### 절대 로깅하지 않는 것

Entire는 다음 데이터를 **절대 로깅하지 않습니다**:

1. **사용자 프롬프트** - "Add user authentication" 등
2. **AI 응답 내용** - 생성된 코드, 설명 등
3. **파일 내용** - 소스 코드, 설정 파일 등
4. **커밋 메시지** - "Fix login bug" 등
5. **환경 변수 값** - API 키, 비밀번호 등

### 로깅하는 것 (메타데이터만)

**OK:**

- Session ID
- Checkpoint ID
- Tool Call ID
- 파일 **경로** (내용 제외)
- 파일 **개수**
- 작업 **시간**
- 에러 **타입** (메시지 제외, 사용자 데이터 포함 가능)

**예시:**

```go
// ✓ OK - 메타데이터만
logging.Info(ctx, "checkpoint saved",
    slog.String("checkpoint_id", "a3b2c4d5e6f7"),
    slog.Int("file_count", 5),
    slog.Duration("duration", elapsed),
)

// ✗ WRONG - 사용자 콘텐츠 포함
logging.Info(ctx, "user prompt",
    slog.String("prompt", "Add user authentication"), // 절대 안됨!
)

// ✗ WRONG - 파일 내용 포함
logging.Debug(ctx, "file modified",
    slog.String("content", fileContent), // 절대 안됨!
)
```

### 에러 메시지 처리

에러 메시지는 사용자 데이터를 포함할 수 있으므로 주의해야 합니다.

```go
// ✓ OK - 에러 타입만
logging.Error(ctx, "failed to save checkpoint",
    slog.String("error", "permission denied"),
)

// ✗ WRONG - 경로나 사용자 데이터 포함 가능
logging.Error(ctx, "operation failed",
    slog.String("error", err.Error()), // err.Error()에 경로 포함 가능
)
```

**안전한 방법:**

```go
// 에러 타입만 추출
logging.Error(ctx, "operation failed",
    slog.String("error_type", fmt.Sprintf("%T", err)),
)
```

---

## 로깅 vs 사용자 출력

### 명확한 구분

| 용도 | 대상 | 도구 | 예시 |
|-----|------|-----|------|
| **Internal Logging** | `.entire/logs/entire.log` | `logging.Info()` | "checkpoint saved" |
| **User Output** | stdout/stderr | `fmt.Fprintf()` | "Checkpoint created: a3b2c4d5e6f7" |

**규칙:**

- **작업 로깅** - `logging` 패키지 사용
- **사용자 메시지** - `fmt.Fprint*` 사용

### 예시

```go
// Internal logging (작업 기록)
logging.Info(ctx, "condensation started",
    slog.String("session_id", sessionID),
    slog.Int("checkpoint_count", len(checkpoints)),
)

// User-facing output (사용자에게 보여줌)
fmt.Fprintf(cmd.OutOrStdout(), "Condensing %d checkpoints...\n", len(checkpoints))

// Internal logging (완료)
logging.Info(ctx, "condensation completed",
    slog.Duration("duration", elapsed),
)

// User-facing output (결과)
fmt.Fprintf(cmd.OutOrStdout(), "Checkpoint created: %s\n", checkpointID)
```

---

## 실전 예제

### Hook Handler 로깅

```go
func handleUserPromptSubmit(ctx context.Context, hookData map[string]interface{}) error {
    sessionID := hookData["session_id"].(string)

    // Context에 session ID 추가
    ctx = logging.WithSession(ctx, sessionID)
    ctx = logging.WithComponent(ctx, "hooks")

    logging.Debug(ctx, "hook invoked",
        slog.String("hook", "UserPromptSubmit"),
    )

    // 프롬프트는 절대 로깅하지 않음!
    // prompt := hookData["prompt"].(string) // 사용만 하고 로깅 안함

    // 작업 수행...

    logging.Info(ctx, "hook completed",
        slog.String("hook", "UserPromptSubmit"),
        slog.Duration("duration", elapsed),
    )

    return nil
}
```

### Condensation 로깅

```go
func (s *Strategy) Condense(sessionID string) error {
    ctx := logging.WithSession(context.Background(), sessionID)
    ctx = logging.WithComponent(ctx, "condensation")

    logging.Info(ctx, "condensation started")

    startTime := time.Now()

    // 메타데이터 복사
    err := copyMetadata(sessionID, checkpointPath)
    if err != nil {
        logging.Error(ctx, "failed to copy metadata",
            slog.String("error", "io error"),
        )
        return err
    }

    elapsed := time.Since(startTime)

    logging.Info(ctx, "condensation completed",
        slog.Duration("duration", elapsed),
        slog.String("checkpoint_id", checkpointID.String()),
    )

    return nil
}
```

### Rewind 로깅

```go
func (s *Strategy) Rewind(point RewindPoint) error {
    ctx := logging.WithSession(context.Background(), point.SessionID)
    ctx = logging.WithComponent(ctx, "rewind")

    logging.Info(ctx, "rewind started",
        slog.String("checkpoint_id", point.CheckpointID.String()),
        slog.Int("file_count", len(point.FilesToRestore)),
    )

    // 파일 복원
    for _, file := range point.FilesToRestore {
        logging.Debug(ctx, "restoring file",
            slog.String("path", file.Path), // 경로는 OK (내용 제외)
        )
        // 복원 작업...
    }

    logging.Info(ctx, "rewind completed",
        slog.Duration("duration", elapsed),
    )

    return nil
}
```

---

## 로그 분석

### jq로 필터링

```bash
# 특정 세션 로그만 조회
cat .entire/logs/entire.log | jq 'select(.session_id == "2026-02-11-abc123...")'

# ERROR 레벨만
cat .entire/logs/entire.log | jq 'select(.level == "ERROR")'

# Hook 호출만
cat .entire/logs/entire.log | jq 'select(.component == "hooks")'

# 최근 10개 로그
tail -n 10 .entire/logs/entire.log | jq .

# 특정 checkpoint 관련 로그
cat .entire/logs/entire.log | jq 'select(.checkpoint_id == "a3b2c4d5e6f7")'
```

### grep으로 검색

```bash
# 특정 세션 검색
grep "2026-02-11-abc123" .entire/logs/entire.log

# 에러 검색
grep '"level":"ERROR"' .entire/logs/entire.log

# condensation 작업 검색
grep '"component":"condensation"' .entire/logs/entire.log
```

### 성능 분석

```bash
# duration이 있는 로그만 (작업 시간 측정)
cat .entire/logs/entire.log | jq 'select(.duration != null) | {msg, duration}'

# 느린 작업 찾기 (1초 이상)
cat .entire/logs/entire.log | jq 'select(.duration > 1000000000) | {msg, duration}'
```

---

## 설정

### Log Level Getter

Settings 패키지와의 순환 의존성을 피하기 위해 콜백 사용:

```go
import "github.com/entireio/cli/cmd/entire/cli/logging"
import "github.com/entireio/cli/cmd/entire/cli/settings"

// 초기화 시 log level getter 설정
logging.SetLogLevelGetter(func() string {
    s, err := settings.Load()
    if err != nil {
        return ""
    }
    return s.LogLevel
})

// 이후 Init() 호출 시 자동으로 settings에서 레벨 읽음
logging.Init(sessionID)
```

---

## 디버깅

### 로그 확인

```bash
# 로그 파일 위치 확인
ls -la .entire/logs/

# 실시간 로그 보기
tail -f .entire/logs/entire.log

# 포맷팅해서 보기
tail -f .entire/logs/entire.log | jq .
```

### 디버그 모드 활성화

```bash
# 일회성
ENTIRE_LOG_LEVEL=debug entire rewind

# 영구 설정
export ENTIRE_LOG_LEVEL=debug
entire rewind
```

### 로그 크기 관리

로그 파일이 너무 커지면:

```bash
# 로그 크기 확인
du -h .entire/logs/entire.log

# 오래된 로그 삭제 (백업 후)
cp .entire/logs/entire.log .entire/logs/entire.log.bak
> .entire/logs/entire.log  # 파일 비우기
```

---

## 다음 단계

Logging 시스템을 이해했습니다! 다음 챕터에서는:

- **Rewind 메커니즘** - 체크포인트로 되돌리기 상세 구현
- **Resume 기능** - 이전 세션 복원 프로세스
- **Auto-Summarization** - AI 기반 자동 요약

---

*다음 글에서는 Entire CLI의 Rewind 메커니즘을 깊이 있게 살펴봅니다.*
