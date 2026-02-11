---
layout: post
title: "Entire CLI 완벽 가이드 (18) - Rewind 메커니즘"
date: 2026-02-11
permalink: /entire-cli-guide-18-rewind-mechanism/
author: Entire Team
categories: [AI 코딩, 개발 도구]
tags: [Entire, Rewind, Checkpoint, Git, Restore]
original_url: "https://github.com/entireio/cli"
excerpt: "Entire CLI의 Rewind 메커니즘 - 체크포인트로 안전하게 되돌리는 방법 완벽 이해"
---

## 개요

**Rewind**는 Entire의 핵심 기능으로, 이전 체크포인트로 코드와 세션 상태를 되돌리는 기능입니다. Git의 `reset --hard`처럼 작동하지만, AI 세션의 메타데이터도 함께 복원합니다.

```
┌─────────────────────────────────────────────────────┐
│                   Rewind Process                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Current State              Target Checkpoint       │
│       │                            ▲                │
│       │                            │                │
│       ▼                            │                │
│  [Checkpoint 5]                    │                │
│  [Checkpoint 4]                    │                │
│  [Checkpoint 3] ───────────────────┘                │
│  [Checkpoint 2]                                     │
│  [Checkpoint 1]                                     │
│                                                      │
│  Restore:                                            │
│  ✓ Code files                                       │
│  ✓ Session transcript                               │
│  ✓ Metadata                                         │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Rewind 개념

### 정의

**Rewind**는 워크트리를 이전 체크포인트 시점으로 복원하는 작업입니다.

**복원되는 것:**

1. **코드 파일** - 체크포인트 시점의 내용으로 복원
2. **Session Transcript** - 해당 시점까지의 AI 대화 기록
3. **메타데이터** - 프롬프트, 컨텍스트, Task 정보

**복원되지 않는 것:**

- Git 커밋 히스토리 (manual-commit)
- 추적되지 않는 파일 (untracked files)
- `.entire/` 디렉토리 자체

### Rewind 타입

| 타입 | 설명 | Strategy |
|-----|------|----------|
| **Full Rewind** | 코드 + 메타데이터 모두 복원 | Manual-commit (항상)<br>Auto-commit (feature 브랜치) |
| **Logs-Only Rewind** | 메타데이터만 복원 (코드 유지) | Auto-commit (main 브랜치) |

---

## Rewind Point

### RewindPoint 구조

```go
type RewindPoint struct {
    CheckpointID     id.CheckpointID // 체크포인트 ID
    CommitHash       string          // Git 커밋 해시
    Message          string          // 커밋 메시지
    Timestamp        time.Time       // 생성 시간
    FilesTouched     []string        // 변경된 파일 목록

    // Full vs Logs-Only
    IsLogsOnly       bool            // Logs-only rewind 여부

    // Task checkpoint 관련
    IsTaskCheckpoint bool            // Task 체크포인트 여부
    ToolUseID        string          // Task 도구 사용 ID

    // Multi-session 관련
    SessionID        string          // 세션 ID
    SessionPrompt    string          // 세션 프롬프트 (식별용)

    // Agent 정보
    Agent            string          // 에이전트 이름
}
```

### Rewind Point 조회

```go
func (s *Strategy) GetRewindPoints(repo *git.Repository) ([]RewindPoint, error)
```

**Manual-commit:**

1. Shadow 브랜치에서 커밋 목록 조회
2. 각 커밋에서 메타데이터 추출
3. RewindPoint 리스트 생성

**Auto-commit:**

1. 현재 브랜치의 커밋 목록 조회
2. `Entire-Checkpoint` trailer 있는 커밋만 선택
3. Main 브랜치와 비교하여 logs-only 여부 결정

---

## Manual-Commit Rewind

### 원리

Manual-commit 전략에서는 **shadow 브랜치**에서 파일을 복원합니다.

```
Shadow Branch: entire/<commit[:7]>-<worktree[:6]>
├── Checkpoint 1 commit
├── Checkpoint 2 commit
├── Checkpoint 3 commit
└── Checkpoint 4 commit (latest)

Rewind to Checkpoint 2:
1. Shadow 브랜치의 Checkpoint 2 커밋에서 파일 tree 읽기
2. 워크트리에 파일 복원
3. Session transcript 복원
```

**특징:**

- **Non-destructive** - Git 히스토리 변경 안함
- **항상 Full Rewind** - 코드 + 메타데이터 모두 복원
- **main 브랜치에서도 안전** - 커밋 생성 안함

### 구현 흐름

```go
func (s *ManualCommitStrategy) Rewind(repo *git.Repository, point *RewindPoint) error {
    // 1. Shadow 브랜치 열기
    shadowBranch := GetShadowBranchName(point.CommitHash)

    // 2. Checkpoint 커밋 찾기
    commit := FindCheckpointCommit(shadowBranch, point.CheckpointID)

    // 3. 파일 tree 읽기
    tree, _ := commit.Tree()

    // 4. 워크트리에 파일 복원
    RestoreFilesFromTree(tree, point.FilesTouched)

    // 5. Session transcript 복원
    RestoreSessionLog(point.SessionID, commit)

    return nil
}
```

### 파일 복원 로직

```go
func RestoreFilesFromTree(tree *object.Tree, files []string) error {
    repoRoot, _ := paths.RepoRoot()

    for _, relPath := range files {
        // Tree에서 파일 읽기
        entry, err := tree.File(relPath)
        if err != nil {
            // 파일이 tree에 없으면 삭제
            absPath := filepath.Join(repoRoot, relPath)
            os.Remove(absPath)
            continue
        }

        // 파일 내용 읽기
        content, _ := entry.Contents()

        // 워크트리에 쓰기
        absPath := filepath.Join(repoRoot, relPath)
        os.MkdirAll(filepath.Dir(absPath), 0o750)
        os.WriteFile(absPath, []byte(content), 0o644)
    }

    return nil
}
```

**주의사항:**

- `.entire/` 디렉토리는 복원하지 않음 (필터링)
- 디렉토리 권한 유지
- Symlink는 지원 안함

---

## Auto-Commit Rewind

### Full Rewind (Feature 브랜치)

Feature 브랜치에서는 `git reset --hard`로 복원합니다.

```bash
# 현재 상태
feature-branch: [Commit 5] [Commit 4] [Commit 3] [Commit 2] [Commit 1]
                                         ▲
                                    Target checkpoint

# Rewind 실행
git reset --hard <commit-3-hash>

# 결과
feature-branch: [Commit 3] [Commit 2] [Commit 1]
```

**구현:**

```go
func (s *AutoCommitStrategy) Rewind(repo *git.Repository, point *RewindPoint) error {
    if point.IsLogsOnly {
        return s.rewindLogsOnly(point)
    }

    // Full rewind - git reset --hard
    cmd := exec.Command("git", "reset", "--hard", point.CommitHash)
    return cmd.Run()
}
```

### Logs-Only Rewind (Main 브랜치)

Main 브랜치에서는 코드를 유지하고 **메타데이터만** 복원합니다.

```
main: [Commit 5] [Commit 4] [Commit 3] [Commit 2] [Commit 1]
                              ▲
                         Target checkpoint

Logs-Only Rewind:
- 코드는 Commit 5 유지
- Session transcript만 Commit 3 시점으로 복원
```

**이유:**

- Main 브랜치의 커밋 히스토리를 보존
- 협업 시 문제 방지
- Push된 커밋은 되돌리지 않음

**구현:**

```go
func (s *AutoCommitStrategy) rewindLogsOnly(point *RewindPoint) error {
    // 1. Checkpoint 메타데이터 읽기
    metadata := ReadCheckpointMetadata(point.CheckpointID)

    // 2. Session transcript 복원
    for _, sessionID := range metadata.SessionIDs {
        logContent := ReadSessionLog(point.CheckpointID, sessionID)
        WriteSessionLog(sessionID, logContent)
    }

    // 3. 사용자에게 안내
    fmt.Println("Logs-only rewind: session transcripts restored")
    fmt.Println("Code remains at current state")

    return nil
}
```

---

## Multi-Session Rewind

### 동시 세션 처리

하나의 체크포인트에 여러 세션이 포함될 수 있습니다.

```
Checkpoint ID: a3b2c4d5e6f7
├── Session 1: 2026-02-11-abc123... ("Add login")
├── Session 2: 2026-02-11-def456... ("Fix tests")
└── Session 3: 2026-02-11-ghi789... ("Update docs")
```

**Rewind 시:**

1. 모든 세션의 transcript 복원
2. 사용자에게 복원된 세션 목록 표시
3. 각 세션의 resume 명령어 출력

### 예시

```bash
entire rewind

# 체크포인트 선택: a3b2c4d5e6f7

# 출력:
Restored 3 sessions:

Session 1: 2026-02-11-abc123...
  Prompt: "Add login"
  Resume: claude resume abc123...

Session 2: 2026-02-11-def456...
  Prompt: "Fix tests"
  Resume: claude resume def456...

Session 3: 2026-02-11-ghi789... (most recent)
  Prompt: "Update docs"
  Resume: claude resume ghi789...
```

---

## Rewind 명령어

### 기본 사용법

```bash
entire rewind

# Interactive 선택:
┌─────────────────────────────────────────────────────┐
│ Select a checkpoint to rewind to:                   │
├─────────────────────────────────────────────────────┤
│ > Checkpoint 4 - Add login feature (10:30 AM)       │
│   Checkpoint 3 - Fix authentication (10:15 AM)      │
│   Checkpoint 2 - Initial setup (10:00 AM)           │
│   Checkpoint 1 - Project init (9:45 AM)             │
└─────────────────────────────────────────────────────┘
```

### Flags

```bash
# 확인 없이 강제 실행
entire rewind --force

# Logs-only rewind (강제)
entire rewind --logs-only

# 특정 체크포인트로 바로 이동 (non-interactive)
entire rewind --to a3b2c4d5e6f7
```

### 예시

#### 1. 일반 Rewind

```bash
entire rewind

# 체크포인트 선택
# → 코드와 메타데이터 복원
# → Session transcript 업데이트
```

#### 2. Logs-Only Rewind

```bash
# Main 브랜치에서 (자동으로 logs-only)
git checkout main
entire rewind

# 또는 명시적으로
entire rewind --logs-only
```

#### 3. Task Checkpoint Rewind

```bash
entire rewind

# Task 체크포인트 선택:
┌─────────────────────────────────────────────────────┐
│ > Task: Run integration tests (task_abc123)         │
│   Checkpoint 3 - Add authentication                 │
└─────────────────────────────────────────────────────┘

# → Task transcript만 복원
```

---

## 안전장치

### 1. Uncommitted Changes 확인

Rewind 전에 uncommitted changes를 확인합니다.

```go
func checkUncommittedChanges(repo *git.Repository) error {
    worktree, _ := repo.Worktree()
    status, _ := worktree.Status()

    if !status.IsClean() {
        return errors.New("you have uncommitted changes")
    }
    return nil
}
```

**에러 메시지:**

```
You have uncommitted changes. Please commit or stash them first.
```

### 2. 확인 프롬프트

Full rewind 시 확인을 요청합니다.

```go
func confirmRewind(point *RewindPoint) (bool, error) {
    var confirmed bool

    form := huh.NewForm(
        huh.NewGroup(
            huh.NewConfirm().
                Title(fmt.Sprintf("Rewind to: %s?", point.Message)).
                Description("This will restore code and session to checkpoint state.").
                Value(&confirmed),
        ),
    )

    return confirmed, form.Run()
}
```

### 3. Logs-Only 경고

Main 브랜치에서 rewind 시 경고를 표시합니다.

```
Warning: Rewinding on main branch - logs-only mode
Code will remain at current state. Only session transcripts will be restored.

Continue? [y/N]
```

---

## Rewind Point 표시

### 체크포인트 목록 포맷

```
Checkpoint 4 - Add login feature (10:30 AM) [latest]
  Files: auth.go, login.go (2 files)
  Session: 2026-02-11-abc123...

Task: Run integration tests (task_abc123) (10:25 AM)
  Session: 2026-02-11-abc123...

Checkpoint 3 - Fix authentication (10:15 AM)
  Files: auth.go (1 file)
  Session: 2026-02-11-def456...
```

**정보:**

- 커밋 메시지
- 생성 시간
- 변경된 파일 수
- 세션 ID
- Task 여부

### 상대 시간 표시

```go
func formatTimestamp(t time.Time) string {
    now := time.Now()
    diff := now.Sub(t)

    switch {
    case diff < time.Minute:
        return "just now"
    case diff < time.Hour:
        mins := int(diff.Minutes())
        return fmt.Sprintf("%d minute%s ago", mins, plural(mins))
    case diff < 24*time.Hour:
        hours := int(diff.Hours())
        return fmt.Sprintf("%d hour%s ago", hours, plural(hours))
    default:
        return t.Format("Jan 2, 3:04 PM")
    }
}
```

---

## 성능 최적화

### 1. 파일 필터링

변경된 파일만 복원합니다.

```go
// FilesTouched만 복원 (전체 tree 순회 안함)
for _, file := range point.FilesTouched {
    RestoreFile(tree, file)
}
```

### 2. 병렬 복원

큰 파일은 병렬로 복원할 수 있습니다.

```go
var wg sync.WaitGroup
for _, file := range largeFiles {
    wg.Add(1)
    go func(f string) {
        defer wg.Done()
        RestoreFile(tree, f)
    }(file)
}
wg.Wait()
```

### 3. Metadata 캐싱

Rewind point 조회 시 metadata를 캐싱합니다.

```go
var metadataCache map[string]*Metadata

func getMetadata(checkpointID id.CheckpointID) (*Metadata, error) {
    if cached, ok := metadataCache[checkpointID.String()]; ok {
        return cached, nil
    }

    metadata := ReadMetadata(checkpointID)
    metadataCache[checkpointID.String()] = metadata
    return metadata, nil
}
```

---

## 트러블슈팅

### 1. "No checkpoints found"

**원인:** Shadow 브랜치나 metadata가 없음

**해결:**

```bash
# 세션 상태 확인
entire status

# Shadow 브랜치 확인
git branch -a | grep entire/

# Metadata 브랜치 확인
git show-ref entire/checkpoints/v1
```

### 2. "Checkpoint not found"

**원인:** Checkpoint ID가 metadata에 없음

**해결:**

```bash
# Checkpoint ID 확인
git log --grep="Entire-Checkpoint:"

# Metadata 브랜치에서 확인
git show entire/checkpoints/v1:<cpid[:2]>/<cpid[2:]>/metadata.json
```

### 3. Rewind 실패 후 복구

```bash
# Uncommitted changes 있으면 stash
git stash

# Rewind 다시 시도
entire rewind

# Stash 복원
git stash pop
```

---

## 다음 단계

Rewind 메커니즘을 이해했습니다! 다음 챕터에서는:

- **Resume 기능** - 이전 세션 복원 프로세스 상세
- **Auto-Summarization** - AI 기반 자동 요약
- **Token Usage Tracking** - 사용량 추적 및 분석

---

*다음 글에서는 Entire CLI의 Resume 기능을 살펴봅니다.*
