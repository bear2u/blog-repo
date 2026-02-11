---
layout: post
title: "Entire CLI 완벽 가이드 (23) - 코드 구조"
date: 2026-02-11
permalink: /entire-cli-guide-23-code-structure/
author: Entire Team
categories: [AI 코딩, 개발 도구, 아키텍처]
tags: [Entire, Architecture, Go, Code Structure]
original_url: "https://github.com/entireio/cli"
excerpt: "Entire CLI의 코드 구조 - 패키지 구성과 주요 파일 완벽 가이드"
---

## 개요

Entire CLI는 **모듈화된 Go 아키텍처**를 사용하여 확장 가능하고 유지보수하기 쉬운 구조를 가지고 있습니다. 이 챕터에서는 전체 코드 구조와 주요 패키지를 살펴봅니다.

---

## 디렉토리 구조

```
cli/
├── cmd/
│   └── entire/
│       ├── main.go              # Entry point
│       └── cli/                 # CLI 구현
│           ├── root.go          # Root command
│           ├── setup.go         # enable 명령
│           ├── status.go        # status 명령
│           ├── rewind.go        # rewind 명령
│           ├── resume.go        # resume 명령
│           ├── explain.go       # explain 명령
│           ├── hooks.go         # Git/Agent 훅 설치
│           ├── agent/           # Agent 추상화
│           ├── checkpoint/      # Checkpoint 저장소
│           ├── strategy/        # Strategy 구현
│           ├── session/         # Session 상태 관리
│           ├── logging/         # 구조화된 로깅
│           ├── settings/        # 설정 관리
│           ├── summarize/       # AI 요약
│           └── integration_test/ # 통합 테스트
├── redact/                      # 민감 정보 제거
├── scripts/                     # 빌드 스크립트
├── go.mod                       # Go 모듈
├── mise.toml                    # mise 설정
└── .golangci.yaml               # 린터 설정
```

---

## 주요 패키지

### 1. cmd/entire/cli (Core CLI)

**책임:**

- Cobra 명령어 정의
- 사용자 입력 처리
- Strategy 호출
- 출력 포맷팅

**주요 파일:**

| 파일 | 역할 |
|-----|------|
| `root.go` | Root command, global flags |
| `setup.go` | `entire enable` 구현 |
| `status.go` | `entire status` 구현 |
| `rewind.go` | `entire rewind` 구현 |
| `resume.go` | `entire resume` 구현 |
| `explain.go` | `entire explain` 구현 |
| `hooks.go` | Git/Agent 훅 설치 |
| `errors.go` | 에러 타입 (SilentError) |
| `config.go` | 설정 로딩 (settings 사용) |

### 2. cmd/entire/cli/strategy

**책임:**

- Checkpoint 생성 및 관리
- Rewind 로직
- Session condensation
- Git 작업 (커밋, 브랜치)

**구조:**

```
strategy/
├── strategy.go              # Strategy 인터페이스
├── registry.go              # Strategy 등록/탐색
├── common.go                # 공통 헬퍼
├── session.go               # 세션 데이터 구조
├── manual_commit.go         # Manual-commit 메인
├── manual_commit_types.go   # 타입 정의
├── manual_commit_session.go # 세션 상태 관리
├── manual_commit_condensation.go # Condensation
├── manual_commit_rewind.go  # Rewind 구현
├── manual_commit_git.go     # Git 작업
├── manual_commit_logs.go    # 로그 조회
├── manual_commit_hooks.go   # 훅 핸들러
├── auto_commit.go           # Auto-commit 전략
└── hooks.go                 # Git 훅 설치
```

**Strategy 인터페이스:**

```go
type Strategy interface {
    Name() string
    AllowsMainBranch() bool

    // Checkpoint 관리
    SaveChanges(ctx SaveContext) error
    SaveTaskCheckpoint(ctx TaskCheckpointContext) error

    // Rewind
    GetRewindPoints(repo *git.Repository) ([]RewindPoint, error)
    Rewind(repo *git.Repository, point *RewindPoint) error

    // Session 조회
    GetSessionLog(sessionID string) ([]byte, error)
    GetSessionInfo(sessionID string) (*SessionInfo, error)
    ListSessions() ([]SessionInfo, error)

    // Git 훅
    PrepareCommitMsg(repo *git.Repository, ...) error
    PostCommit(repo *git.Repository, ...) error
    PrePush(repo *git.Repository, ...) error
}
```

### 3. cmd/entire/cli/checkpoint

**책임:**

- Checkpoint 데이터 타입
- Shadow 브랜치 작업
- Metadata 브랜치 작업

**파일:**

| 파일 | 역할 |
|-----|------|
| `checkpoint.go` | 데이터 타입 정의 |
| `store.go` | GitStore (git.Repository 래퍼) |
| `temporary.go` | Shadow 브랜치 ops (WriteTemporary, ReadTemporary) |
| `committed.go` | Metadata 브랜치 ops (WriteCommitted, ReadCommitted) |

**주요 타입:**

```go
type Checkpoint struct {
    CheckpointID     id.CheckpointID
    Message          string
    Timestamp        time.Time
    IsTaskCheckpoint bool
    ToolUseID        string
}

type TemporaryCheckpoint struct {
    Checkpoint
    TreeHash plumbing.Hash  // Shadow 브랜치 tree
}

type CommittedCheckpoint struct {
    Checkpoint
    Metadata *CommittedMetadata
}

type CommittedMetadata struct {
    SessionID    string
    CheckpointID id.CheckpointID
    Strategy     string
    CreatedAt    time.Time
    FilesTouched []string
    TokenUsage   *TokenUsage
    Summary      *Summary  // AI 생성 요약
}
```

### 4. cmd/entire/cli/session

**책임:**

- Session 상태 관리
- Phase state machine
- Session 저장소

**파일:**

| 파일 | 역할 |
|-----|------|
| `session.go` | Session 데이터 타입 |
| `state.go` | StateStore (`.git/entire-sessions/` 관리) |
| `phase.go` | Phase state machine |

**Phase State Machine:**

```go
type Phase string

const (
    PhaseIdle            Phase = "idle"
    PhaseActive          Phase = "active"
    PhaseActiveCommitted Phase = "active_committed"
    PhaseEnded           Phase = "ended"
)

type Event int

const (
    EventTurnStart    Event = iota
    EventTurnEnd
    EventGitCommit
    EventSessionStart
    EventSessionStop
)

func Transition(current Phase, event Event, ctx TransitionContext) TransitionResult {
    // 상태 전환 로직
}
```

### 5. cmd/entire/cli/agent

**책임:**

- Agent 추상화
- Transcript 파싱
- Session 복원

**구조:**

```
agent/
├── agent.go             # Agent 인터페이스
├── types.go             # 공통 타입
├── claudecode/          # Claude Code 구현
│   ├── agent.go
│   ├── transcript.go
│   └── types.go
└── geminicli/           # Gemini CLI 구현
    ├── agent.go
    ├── transcript.go
    └── types.go
```

**Agent 인터페이스:**

```go
type Agent interface {
    Name() string
    SetupHooks(repoPath string) error

    // Session 관리
    GetSessionDir(repoPath string) (string, error)
    GetTranscriptPath(sessionID string) (string, error)
    ReadSession(sessionRef string) (*AgentSession, error)
    WriteSession(session *AgentSession) error

    // ID 변환
    GenerateSessionID(agentSessionID string) string
    ExtractAgentSessionID(sessionID string) string

    // 명령어
    FormatResumeCommand(agentSessionID string) string
}
```

### 6. cmd/entire/cli/logging

**책임:**

- 구조화된 로깅 (slog)
- Context 기반 로깅
- 프라이버시 보호

**파일:**

| 파일 | 역할 |
|-----|------|
| `logger.go` | Logger 초기화, 로그 함수 |
| `context.go` | Context에 데이터 추가/추출 |

**사용 예시:**

```go
import "github.com/entireio/cli/cmd/entire/cli/logging"

// 초기화
logging.Init(sessionID)
defer logging.Close()

// Context에 데이터 추가
ctx = logging.WithSession(ctx, sessionID)
ctx = logging.WithComponent(ctx, "rewind")

// 로깅
logging.Info(ctx, "rewind started",
    slog.String("checkpoint_id", "a3b2c4d5e6f7"),
)
```

### 7. cmd/entire/cli/settings

**책임:**

- 설정 파일 로딩
- 순환 의존성 방지

**파일:**

```
settings/
└── settings.go
```

**EntireSettings:**

```go
type EntireSettings struct {
    Enabled        bool   `json:"enabled"`
    Strategy       string `json:"strategy"`
    Agent          string `json:"agent"`
    PushSessions   bool   `json:"push_sessions"`
    LogLevel       string `json:"log_level"`
    Summarize      *SummarizeSettings `json:"summarize,omitempty"`
}

// Load settings
settings, err := settings.Load()

// Helper functions
if settings.IsSummarizeEnabled() {
    // ...
}
```

### 8. cmd/entire/cli/summarize

**책임:**

- AI 기반 요약 생성
- Transcript condensation
- Claude API 통합

**파일:**

| 파일 | 역할 |
|-----|------|
| `summarize.go` | Core 로직, Generator 인터페이스 |
| `claude.go` | ClaudeGenerator 구현 |

---

## 데이터 흐름

### Enable 워크플로우

```
1. User: entire enable
   ↓
2. cli/setup.go: newEnableCmd().RunE()
   ↓
3. strategy/registry.go: Get("manual-commit")
   ↓
4. cli/hooks.go: InstallGitHooks()
   ↓
5. agent/claudecode/agent.go: SetupHooks()
   ↓
6. settings/settings.go: Save(EntireSettings)
```

### Checkpoint 생성 (Manual-Commit)

```
1. Git Hook: post-commit
   ↓
2. cli/hooks.go: handlePostCommit()
   ↓
3. strategy/manual_commit_hooks.go: PostCommit()
   ↓
4. session/phase.go: Transition(EventGitCommit)
   ↓
5. strategy/manual_commit_condensation.go: Condense()
   ↓
6. checkpoint/committed.go: WriteCommitted()
```

### Rewind 워크플로우

```
1. User: entire rewind
   ↓
2. cli/rewind.go: newRewindCmd().RunE()
   ↓
3. strategy/manual_commit_rewind.go: GetRewindPoints()
   ↓
4. User: Select checkpoint
   ↓
5. strategy/manual_commit_rewind.go: Rewind()
   ↓
6. checkpoint/temporary.go: ReadTemporary()
   ↓
7. Restore files from tree
```

---

## 핵심 패턴

### 1. Strategy Pattern

모든 checkpoint 로직은 Strategy 인터페이스로 추상화됩니다.

```go
// Registry에 등록
func init() {
    Register(&ManualCommitStrategy{})
    Register(&AutoCommitStrategy{})
}

// CLI에서 사용
strategy := GetStrategy()
strategy.SaveChanges(ctx)
```

### 2. Error Handling

**SilentError:**

```go
// 이미 에러 메시지를 출력한 경우
fmt.Fprintln(cmd.ErrOrStderr(), "Not a git repository.")
return NewSilentError(errors.New("not a git repository"))

// main.go에서 처리
if !errors.As(err, &SilentError{}) {
    fmt.Fprintf(os.Stderr, "Error: %v\n", err)
}
```

### 3. Settings 순환 의존성 방지

```go
// cli 패키지에서 settings 사용
import "github.com/entireio/cli/cmd/entire/cli/settings"

s, err := settings.Load()

// strategy 패키지에서도 settings 사용 가능
// (settings는 cli도 strategy도 import 안함)
```

### 4. Logging vs User Output

```go
// Internal logging
logging.Info(ctx, "condensation started")

// User-facing output
fmt.Fprintf(cmd.OutOrStdout(), "Condensing checkpoints...\n")
```

### 5. Context 전파

```go
// Component 추가
ctx = logging.WithComponent(context.Background(), "rewind")

// Session ID 추가
ctx = logging.WithSession(ctx, sessionID)

// 전파
someFunction(ctx, ...)
```

---

## 테스트 구조

### Unit Tests

```
cmd/entire/cli/strategy/
├── manual_commit_test.go
├── auto_commit_test.go
├── common_test.go
└── rewind_test.go
```

**패턴:**

```go
func TestManualCommit_SaveChanges(t *testing.T) {
    t.Parallel()

    // Setup
    repo := createTestRepo(t)
    strategy := &ManualCommitStrategy{}

    // Execute
    err := strategy.SaveChanges(ctx)

    // Assert
    assert.NoError(t, err)
}
```

### Integration Tests

```
cmd/entire/cli/integration_test/
├── testenv.go                      # TestEnv helper
├── manual_commit_workflow_test.go
├── auto_commit_workflow_test.go
├── rewind_test.go
└── phase_transitions_test.go
```

**패턴:**

```go
func TestWorkflow_BasicFlow(t *testing.T) {
    t.Parallel()

    RunForAllStrategies(t, func(t *testing.T, env *TestEnv, strategy string) {
        // Test workflow...
    })
}
```

**TestEnv:**

```go
type TestEnv struct {
    RepoPath  string
    Repo      *git.Repository
    Strategy  string
    Cleanup   func()
}

func NewTestEnv(t *testing.T, strategy string) *TestEnv {
    // Create temp repo
    // Initialize entire
    // Return env with cleanup
}
```

---

## 확장 포인트

### 1. 새 Strategy 추가

```go
// 1. strategy 패키지에 파일 생성
// custom_strategy.go

type CustomStrategy struct{}

func (s *CustomStrategy) Name() string {
    return "custom"
}

func (s *CustomStrategy) SaveChanges(ctx SaveContext) error {
    // 구현...
}

// 2. Registry에 등록
func init() {
    Register(&CustomStrategy{})
}
```

### 2. 새 Agent 추가

```go
// 1. agent/<name>/ 디렉토리 생성
// agent/myagent/agent.go

type MyAgent struct{}

func (a *MyAgent) Name() string {
    return "myagent"
}

// Agent 인터페이스 구현...

// 2. agent/registry.go에 등록
func init() {
    RegisterAgent(&MyAgent{})
}
```

### 3. 새 명령어 추가

```go
// 1. cli/ 디렉토리에 파일 생성
// cli/mycommand.go

func newMyCommand() *cobra.Command {
    return &cobra.Command{
        Use:   "mycommand",
        Short: "Description",
        RunE:  runMyCommand,
    }
}

// 2. root.go에 추가
func init() {
    rootCmd.AddCommand(newMyCommand())
}
```

---

## 다음 단계

코드 구조를 이해했습니다! 다음 챕터에서는:

- **Agent 통합** - Gemini CLI, 새 Agent 추가 상세
- **Contributing** - 기여 가이드, 테스트, PR 프로세스

---

*다음 글에서는 Entire CLI의 Agent 통합 방법을 살펴봅니다.*
