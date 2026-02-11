---
layout: post
title: "Entire CLI 완벽 가이드 (16) - Subagent Tracking"
date: 2026-02-11
permalink: /entire-cli-guide-16-subagent-tracking/
author: Entire Team
categories: [AI 코딩, 개발 도구]
tags: [Entire, Subagent, Task, TodoWrite, Checkpoint]
original_url: "https://github.com/entireio/cli"
excerpt: "Entire CLI의 Subagent Tracking 시스템 - Task와 TodoWrite 체크포인트 완벽 이해"
---

## 개요

**Subagent Tracking**은 AI 에이전트가 하위 작업(subtask)을 실행할 때 각 작업의 체크포인트를 개별적으로 추적하는 기능입니다. Claude Code의 `Task` 도구나 `TodoWrite` 도구를 사용할 때, 각 작업마다 별도의 체크포인트가 생성되어 세밀한 되돌리기가 가능합니다.

```
Session (전체 세션)
├── Checkpoint 1 (일반)
├── Checkpoint 2 (일반)
├── Task Checkpoint 3.1 (하위 작업 1)
├── Task Checkpoint 3.2 (하위 작업 2)
└── Checkpoint 4 (일반)
```

---

## Task Checkpoint란?

### 정의

**Task Checkpoint**는 AI 에이전트가 `Task` 또는 `TodoWrite` 도구를 사용하여 하위 에이전트를 실행할 때 생성되는 특수한 체크포인트입니다.

**일반 체크포인트와의 차이:**

| 항목 | 일반 체크포인트 | Task 체크포인트 |
|-----|---------------|----------------|
| **트리거** | AI 응답 완료 | Task/TodoWrite 도구 실행 |
| **스코프** | 전체 세션 | 개별 하위 작업 |
| **메타데이터** | `.entire/metadata/<session-id>/` | `.entire/metadata/<session-id>/tasks/<tool-use-id>/` |
| **Rewind** | 전체 세션 상태 복원 | 특정 Task만 복원 가능 |

### Task Checkpoint ID

Task 체크포인트는 다음 정보로 식별됩니다:

```go
type TaskCheckpointInfo struct {
    ToolUseID    string    // Task/TodoWrite 도구 실행 ID
    TaskID       string    // Task 고유 ID (UUID)
    Description  string    // Task 설명
    Timestamp    time.Time // 생성 시간
}
```

**예시:**

```
Tool Use ID: task_abc123def456
Task ID: 550e8400-e29b-41d4-a716-446655440000
Description: "Run integration tests"
```

---

## 지원되는 도구

### 1. Task 도구

Claude Code의 `Task` 도구는 장기 실행 작업을 백그라운드에서 실행합니다.

**사용 예시:**

```
사용자: "Run integration tests in the background"

Claude: [Task 도구 사용]
  - Command: mise run test:integration
  - Background: true
  - Tool Use ID: task_abc123

→ Task Checkpoint 생성
```

**저장되는 데이터:**

```
.entire/metadata/<session-id>/tasks/task_abc123/
├── checkpoint.json      # Task ID 매핑
├── agent-<uuid>.jsonl   # Subagent transcript
└── result.json          # Task 실행 결과
```

### 2. TodoWrite 도구

Claude Code의 `TodoWrite` 도구는 작업 목록을 관리합니다.

**사용 예시:**

```
사용자: "Add these tasks to the todo list"

Claude: [TodoWrite 도구 사용]
  - Todo 1: Implement login feature
  - Todo 2: Add unit tests
  - Tool Use ID: todo_def456

→ Task Checkpoint 생성
```

**저장되는 데이터:**

```
.entire/metadata/<session-id>/tasks/todo_def456/
├── checkpoint.json      # Task ID 매핑
└── agent-<uuid>.jsonl   # TodoWrite 상태 변경
```

---

## Task Checkpoint 생성 과정

### 1. 도구 실행 감지

Strategy의 `SaveTaskCheckpoint()` 메서드가 호출됩니다.

```go
type TaskCheckpointContext struct {
    SessionID    string    // 세션 ID
    ToolUseID    string    // 도구 사용 ID
    Transcript   []byte    // Subagent transcript
    FilesTouched []string  // 변경된 파일 목록
}

func (s *Strategy) SaveTaskCheckpoint(ctx TaskCheckpointContext) error {
    // Task 체크포인트 생성
}
```

### 2. 메타데이터 저장

Task별 디렉토리에 메타데이터를 저장합니다.

```
.entire/metadata/<session-id>/tasks/<tool-use-id>/
```

**checkpoint.json:**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "tool_use_id": "task_abc123",
  "description": "Run integration tests",
  "created_at": "2026-02-11T10:30:00Z"
}
```

**agent-<uuid>.jsonl:**

```jsonl
{"type":"user","content":"Run integration tests"}
{"type":"assistant","content":"Running tests..."}
{"type":"tool_result","tool":"Bash","output":"..."}
```

### 3. Shadow 브랜치에 커밋

Manual-commit 전략에서는 shadow 브랜치에 Task 체크포인트를 커밋합니다.

```bash
# Shadow 브랜치 확인
git log entire/<commit-hash>-<worktree>

# Commit 메시지 예시
Task: Run integration tests

Entire-Session: 2026-02-11-abc123...
Entire-Task-Metadata: .entire/metadata/<session-id>/tasks/task_abc123
Entire-Strategy: manual-commit
```

---

## Task Checkpoint 구조

### 디렉토리 레이아웃

```
Shadow Branch (manual-commit)
└── .entire/metadata/<session-id>/
    ├── full.jsonl                    # 전체 세션 transcript
    ├── prompt.txt                    # 사용자 프롬프트
    ├── context.md                    # 컨텍스트
    └── tasks/
        ├── task_abc123/              # Task 1
        │   ├── checkpoint.json
        │   ├── agent-<uuid>.jsonl
        │   └── result.json
        └── todo_def456/              # Task 2
            ├── checkpoint.json
            └── agent-<uuid>.jsonl

Metadata Branch (entire/checkpoints/v1)
└── <checkpoint-id[:2]>/<checkpoint-id[2:]>/
    └── 0/                            # 세션 0
        ├── metadata.json
        ├── full.jsonl
        └── tasks/
            ├── task_abc123/
            │   ├── checkpoint.json
            │   └── agent-<uuid>.jsonl
            └── todo_def456/
                ├── checkpoint.json
                └── agent-<uuid>.jsonl
```

### Condensation 시 처리

커밋 시 Task 체크포인트도 함께 condensation됩니다.

```go
// Manual-commit condensation
func (s *ManualCommitStrategy) Condense(sessionID string) error {
    // 1. 메인 세션 메타데이터 복사
    CopySessionMetadata(sessionID, checkpointPath)

    // 2. Task 체크포인트도 함께 복사
    for _, taskDir := range FindTaskDirectories(sessionID) {
        CopyTaskMetadata(taskDir, checkpointPath + "/tasks")
    }

    // 3. Shadow 브랜치 정리
    CleanupShadowBranch()
}
```

---

## Rewind와 Task Checkpoint

### Task별 Rewind

Task 체크포인트는 개별적으로 rewind할 수 있습니다.

```bash
entire rewind

# 출력:
┌─────────────────────────────────────────────────────┐
│ Select a checkpoint to rewind to:                   │
├─────────────────────────────────────────────────────┤
│ > Checkpoint 4 - Implement login feature (latest)   │
│   Task: Run integration tests (task_abc123)         │
│   Task: Add unit tests (todo_def456)                │
│   Checkpoint 3 - Add authentication                 │
│   Checkpoint 2 - Initial setup                      │
└─────────────────────────────────────────────────────┘
```

**선택 시:**

- **일반 체크포인트 선택** - 전체 세션 상태 복원
- **Task 체크포인트 선택** - 해당 Task만 복원

### 구현 예시

```go
type RewindPoint struct {
    CheckpointID     id.CheckpointID
    CommitHash       string
    Message          string
    Timestamp        time.Time
    IsTaskCheckpoint bool           // Task 체크포인트 여부
    ToolUseID        string          // Task 도구 사용 ID
    TaskDescription  string          // Task 설명
    SessionID        string          // 세션 ID
}

func (s *Strategy) GetRewindPoints() ([]RewindPoint, error) {
    points := []RewindPoint{}

    // 일반 체크포인트 추가
    points = append(points, GetRegularCheckpoints()...)

    // Task 체크포인트 추가
    for _, taskDir := range FindTaskDirectories() {
        taskPoint := RewindPoint{
            IsTaskCheckpoint: true,
            ToolUseID:        ExtractToolUseID(taskDir),
            TaskDescription:  ReadTaskDescription(taskDir),
        }
        points = append(points, taskPoint)
    }

    return points, nil
}
```

---

## Task Metadata 형식

### checkpoint.json

Task ID와 도구 정보를 저장합니다.

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "tool_use_id": "task_abc123",
  "tool_name": "Task",
  "description": "Run integration tests",
  "created_at": "2026-02-11T10:30:00Z",
  "completed_at": "2026-02-11T10:35:00Z",
  "status": "completed",
  "exit_code": 0
}
```

### agent-<uuid>.jsonl

Subagent의 전체 transcript를 저장합니다.

```jsonl
{"type":"user","content":"Run integration tests","timestamp":"2026-02-11T10:30:00Z"}
{"type":"assistant","content":"I'll run the integration tests using mise.","timestamp":"2026-02-11T10:30:05Z"}
{"type":"tool_use","tool":"Bash","command":"mise run test:integration","timestamp":"2026-02-11T10:30:06Z"}
{"type":"tool_result","tool":"Bash","output":"Running tests...\nAll tests passed","timestamp":"2026-02-11T10:35:00Z"}
{"type":"assistant","content":"All integration tests passed successfully.","timestamp":"2026-02-11T10:35:01Z"}
```

### result.json (Task 도구만)

Task 실행 결과를 저장합니다.

```json
{
  "exit_code": 0,
  "stdout": "Running tests...\nAll tests passed\n",
  "stderr": "",
  "duration_ms": 300000,
  "completed_at": "2026-02-11T10:35:00Z"
}
```

---

## Strategy별 구현

### Manual-Commit Strategy

**Task 체크포인트 생성:**

1. Shadow 브랜치에 task 메타데이터 커밋
2. `.git/entire-sessions/<session-id>.json`에 task 정보 추가
3. Rewind 시 task 디렉토리에서 복원

**Condensation:**

```
Shadow Branch                 → Metadata Branch
─────────────────────────────────────────────────
.entire/metadata/<sid>/       → <cpid[:2]>/<cpid[2:]>/0/
├── full.jsonl                  ├── full.jsonl
├── tasks/                      └── tasks/
│   └── task_abc123/                └── task_abc123/
│       └── agent-<uuid>.jsonl          └── agent-<uuid>.jsonl
```

### Auto-Commit Strategy

**Task 체크포인트 생성:**

1. 자동 커밋 생성 (코드 변경사항 포함)
2. Metadata 브랜치에 task 메타데이터 저장
3. 커밋 메시지에 `Entire-Task-Checkpoint` trailer 추가

```
Commit Subject: Task: Run integration tests

Entire-Checkpoint: a3b2c4d5e6f7
Entire-Task-Checkpoint: task_abc123
```

---

## 실습: Task Checkpoint 사용

### 1. Task 도구로 체크포인트 생성

```bash
# Claude Code 세션 시작
claude "Run the integration tests in the background"

# Claude가 Task 도구 사용:
# [Task tool: mise run test:integration, background: true]

# 상태 확인
entire status

# 출력:
# Session: 2026-02-11-abc123...
# Checkpoints: 2 (1 regular, 1 task)
#   - Checkpoint 1: Initial setup
#   - Task: Run integration tests (task_abc123)
```

### 2. TodoWrite 도구로 체크포인트 생성

```bash
claude "Add these tasks to my todo list: implement login, add tests"

# Claude가 TodoWrite 도구 사용
# Task checkpoint 생성됨

entire status

# 출력:
# Checkpoints: 3 (1 regular, 2 tasks)
#   - Checkpoint 1: Initial setup
#   - Task: Run integration tests (task_abc123)
#   - Task: Update todo list (todo_def456)
```

### 3. Task Checkpoint로 Rewind

```bash
entire rewind

# Task 체크포인트 선택
# → 해당 Task 시점으로 복원
```

---

## 고급 기능

### 1. Task별 Transcript 조회

```bash
# Shadow 브랜치에서 task transcript 확인
git show entire/<commit>-<worktree>:.entire/metadata/<session-id>/tasks/task_abc123/agent-<uuid>.jsonl

# Metadata 브랜치에서 확인
git show entire/checkpoints/v1:<cpid[:2]>/<cpid[2:]>/0/tasks/task_abc123/agent-<uuid>.jsonl
```

### 2. Task ID로 검색

```bash
# Task ID로 체크포인트 찾기
grep -r "550e8400-e29b-41d4-a716-446655440000" .entire/metadata/

# 출력:
# .entire/metadata/2026-02-11-abc123.../tasks/task_abc123/checkpoint.json
```

### 3. 다중 Task 추적

여러 Task가 동시에 실행될 수 있습니다:

```
Session
├── Task 1: Running tests (background)
├── Task 2: Building project (background)
└── Task 3: Formatting code (background)
```

각 Task는 독립적인 체크포인트를 가집니다.

---

## 제한사항 및 주의사항

### 1. Task 체크포인트는 메타데이터만

Task 체크포인트는 **transcript와 실행 결과만** 저장합니다. 코드 변경사항은 부모 세션의 일반 체크포인트에 포함됩니다.

### 2. Rewind 시 코드 복원

Task 체크포인트로 rewind하면:
- Task의 transcript는 복원됨
- 코드는 **해당 Task가 속한 일반 체크포인트 시점**으로 복원됨

### 3. Auto-commit에서의 Task

Auto-commit 전략에서는 Task마다 별도의 커밋이 생성되지 않습니다. Task 메타데이터만 별도 저장됩니다.

---

## 디버깅

### Task Checkpoint 확인

```bash
# 현재 세션의 Task 목록
ls .entire/metadata/$(entire status --json | jq -r .session_id)/tasks/

# Task 메타데이터 확인
cat .entire/metadata/<session-id>/tasks/task_abc123/checkpoint.json

# Task transcript 확인
cat .entire/metadata/<session-id>/tasks/task_abc123/agent-*.jsonl
```

### Shadow 브랜치에서 Task 조회

```bash
# Shadow 브랜치 커밋 확인
git log entire/<commit>-<worktree> --oneline

# Task 커밋 찾기
git log entire/<commit>-<worktree> --grep="Task:"

# Task 메타데이터 조회
git show entire/<commit>-<worktree>:.entire/metadata/<session-id>/tasks/
```

---

## 다음 단계

Subagent Tracking을 이해했습니다! 다음 챕터에서는:

- **Logging 시스템** - 구조화된 로깅과 프라이버시 보호
- **Rewind 메커니즘** - 체크포인트로 되돌리기 상세
- **Resume 기능** - 이전 세션 복원 깊이 파기

---

*다음 글에서는 Entire CLI의 Logging 시스템을 살펴봅니다.*
