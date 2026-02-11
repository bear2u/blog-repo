---
layout: post
title: "Entire CLI 완벽 가이드 (19) - Resume 기능"
date: 2026-02-11
permalink: /entire-cli-guide-19-resume-feature/
author: Entire Team
categories: [AI 코딩, 개발 도구]
tags: [Entire, Resume, Session, Branch, Restore]
original_url: "https://github.com/entireio/cli"
excerpt: "Entire CLI의 Resume 기능 - 브랜치 전환과 세션 복원 완벽 가이드"
---

## 개요

**Resume**은 다른 브랜치로 전환하면서 해당 브랜치의 마지막 세션을 복원하는 기능입니다. 팀 협업이나 여러 feature 브랜치를 오가며 작업할 때 유용합니다.

```
┌─────────────────────────────────────────────────────┐
│                  Resume Workflow                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  feature-A                   feature-B               │
│      │                           │                   │
│      ▼                           ▼                   │
│  [Commit with                [Commit with            │
│   Checkpoint]                Checkpoint]             │
│      │                           │                   │
│      │  entire resume feature-B  │                   │
│      └───────────────────────────┘                   │
│                                  │                   │
│  1. git checkout feature-B       │                   │
│  2. Find checkpoint              │                   │
│  3. Restore session logs         │                   │
│  4. Show resume command ─────────┘                   │
│                                                      │
│  $ claude resume <session-id>                        │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Resume vs Rewind

### 차이점

| 기능 | Rewind | Resume |
|-----|--------|--------|
| **브랜치 전환** | 없음 | 브랜치 전환 포함 |
| **대상** | 현재 브랜치의 체크포인트 | 다른 브랜치의 마지막 체크포인트 |
| **코드 복원** | 항상 (Full) 또는 안함 (Logs-only) | 브랜치 전환으로 자동 |
| **메타데이터 복원** | 항상 | 항상 |
| **사용 시나리오** | 실수 되돌리기 | 브랜치 간 작업 전환 |

### 공통점

- Session transcript 복원
- Metadata 복원
- Multi-session 지원
- Agent 정보 추출

---

## Resume 명령어

### 기본 사용법

```bash
entire resume <branch-name>
```

**동작 순서:**

1. 브랜치 체크아웃
2. 브랜치 고유 커밋에서 checkpoint 찾기
3. Metadata 브랜치에서 세션 로그 조회
4. Session transcript 복원
5. Resume 명령어 출력

### 예시

```bash
# feature-login 브랜치로 전환 및 세션 복원
entire resume feature-login

# 출력:
Switched to branch 'feature-login'
Session: 2026-02-11-abc123de-f456-7890-abcd-ef1234567890

To continue this session, run:
  claude resume abc123de-f456-7890-abcd-ef1234567890
```

---

## Checkpoint 탐색 로직

### Branch-Only Commits

Resume은 **브랜치 고유 커밋**만 검색합니다.

```
main:     [C1] ─── [C2] ─── [C3]
                     │
feature:             └─── [C4] ─── [C5] ─── [C6]
                           ↑               ↑
                      Checkpoint      Checkpoint
                      (찾기 O)        (찾기 O)

Merge base: C2
Branch-only commits: C4, C5, C6
```

**이유:**

- main의 커밋은 무시 (다른 브랜치의 작업)
- feature 브랜치 고유 작업만 관련

### Merge Commit 제외

Merge 커밋은 "branch work"로 간주하지 않습니다.

```
feature: [C4] ─── [Merge C3] ─── [C5]
          ↑           ↑            ↑
      Checkpoint   (무시)      Checkpoint

C5가 checkpoint 없으면:
→ C4로 resume (C5는 merge이므로 "newer commit" 경고 안함)
```

### 탐색 알고리즘

```go
func findBranchCheckpoint(repo *git.Repository, branchName string) (*branchCheckpointResult, error) {
    // 1. HEAD 커밋 가져오기
    head := repo.Head()

    // 2. HEAD에 checkpoint 있으면 바로 반환
    if cpID, found := trailers.ParseCheckpoint(head.Message); found {
        return &branchCheckpointResult{
            checkpointID: cpID,
            newerCommitsExist: false,
        }, nil
    }

    // 3. Default 브랜치 찾기 (main or master)
    defaultBranch := getDefaultBranchFromRemote(repo)

    // 4. Merge base 찾기
    mergeBase := headCommit.MergeBase(defaultCommit)

    // 5. HEAD → merge base까지 순회
    return findCheckpointInHistory(headCommit, mergeBase)
}
```

### Newer Commits 경고

Checkpoint보다 최신 커밋이 있으면 경고합니다.

```bash
entire resume feature-login

# 출력:
Found checkpoint in an older commit.
There are 2 newer commit(s) on this branch without checkpoints.
Checkpoint from: a3f2b1c Add login feature

Resume from this older checkpoint? [y/N]
```

**Merge 커밋은 제외:**

```bash
# Merge 커밋은 카운트 안됨
feature: [Checkpoint] ─── [Merge main] ─── [Merge hotfix]

→ "There are 0 newer commits" (merge 커밋 무시)
```

---

## Remote Branch Fetching

### 로컬에 없는 브랜치

브랜치가 로컬에 없으면 origin에서 fetch를 제안합니다.

```bash
entire resume feature-new

# 출력:
Branch 'feature-new' not found locally. Fetch from origin? [y/N]
```

**사용자가 Yes 선택:**

```bash
Fetching branch 'feature-new' from origin...
Switched to branch 'feature-new'
Session: 2026-02-11-xyz789...

To continue this session, run:
  claude resume xyz789...
```

### Fetch 로직

```go
func FetchAndCheckoutRemoteBranch(branchName string) error {
    // 1. Remote branch fetch
    cmd := exec.Command("git", "fetch", "origin",
        fmt.Sprintf("%s:%s", branchName, branchName))
    if err := cmd.Run(); err != nil {
        return fmt.Errorf("failed to fetch: %w", err)
    }

    // 2. Checkout
    cmd = exec.Command("git", "checkout", branchName)
    return cmd.Run()
}
```

---

## Metadata Branch 처리

### Local Metadata 확인

```go
metadataTree, err := strategy.GetMetadataBranchTree(repo)
if err != nil {
    // Metadata 브랜치 없음 → Remote 확인
    return checkRemoteMetadata(repo, checkpointID)
}
```

### Remote Metadata 자동 Fetch

로컬에 metadata가 없으면 자동으로 fetch합니다.

```bash
entire resume feature-login

# 출력:
Fetching session metadata from origin...
Session restored to: .claude/sessions/abc123.../full.jsonl
Session: 2026-02-11-abc123...

To continue this session, run:
  claude resume abc123...
```

**Fetch 로직:**

```go
func checkRemoteMetadata(repo *git.Repository, checkpointID id.CheckpointID) error {
    // 1. Remote metadata tree 가져오기
    remoteTree := strategy.GetRemoteMetadataBranchTree(repo)

    // 2. Checkpoint 존재 확인
    metadata := strategy.ReadCheckpointMetadata(remoteTree, checkpointID.Path())

    // 3. Fetch
    FetchMetadataBranch()

    // 4. Session 복원
    resumeSession(metadata.SessionID, checkpointID, false)
}
```

---

## Multi-Session Resume

### 하나의 Checkpoint에 여러 Session

동시 세션이 condensation된 경우 모두 복원합니다.

```
Checkpoint ID: a3b2c4d5e6f7
├── Session 0: 2026-02-11-abc123... ("Add login")
├── Session 1: 2026-02-11-def456... ("Fix tests")
└── Session 2: 2026-02-11-ghi789... ("Update docs") [latest]
```

**Resume 시:**

```bash
entire resume feature-login

# 출력:
Restored 3 sessions. To continue, run:
  claude resume abc123...  # Add login
  claude resume def456...  # Fix tests
  claude resume ghi789...  # Update docs (most recent)
```

### Session Restore Info

```go
type SessionRestoreInfo struct {
    SessionID      string
    Prompt         string
    Agent          string
    Status         TimestampStatus  // LocalNewer, CheckpointNewer, Equal
    LocalTime      time.Time
    CheckpointTime time.Time
}
```

---

## Timestamp 비교

### Local vs Checkpoint

로컬 세션이 checkpoint보다 최신이면 경고합니다.

```go
type TimestampStatus int

const (
    StatusEqual TimestampStatus = iota
    StatusLocalNewer
    StatusCheckpointNewer
)

func ClassifyTimestamps(local, checkpoint time.Time) TimestampStatus {
    if local.IsZero() {
        return StatusCheckpointNewer
    }
    if checkpoint.IsZero() {
        return StatusLocalNewer
    }

    diff := local.Sub(checkpoint)
    if diff.Abs() < time.Second {
        return StatusEqual
    }
    if diff > 0 {
        return StatusLocalNewer
    }
    return StatusCheckpointNewer
}
```

### Overwrite 확인

```bash
entire resume feature-login

# 로컬 세션이 더 최신인 경우:
Warning: Local session log has newer timestamps than checkpoint
  Local:      2026-02-11 10:35:00
  Checkpoint: 2026-02-11 10:30:00

This will overwrite your local session log. Continue? [y/N]
```

**Force flag로 건너뛰기:**

```bash
entire resume feature-login --force
```

---

## Agent 통합

### Agent 정보 추출

Checkpoint metadata에서 agent 정보를 가져옵니다.

```go
metadata, _ := strategy.ReadCheckpointMetadata(tree, checkpointID.Path())

// Agent 이름 추출 (예: "claude-code", "gemini")
agent := metadata.Agent

// Agent 인스턴스 생성
ag, _ := strategy.ResolveAgentForRewind(agent)
```

### Resume Command 생성

각 Agent는 자신의 resume 명령어 형식을 가집니다.

```go
type Agent interface {
    FormatResumeCommand(agentSessionID string) string
}

// Claude Code
func (a *ClaudeCodeAgent) FormatResumeCommand(sessionID string) string {
    return fmt.Sprintf("claude resume %s", sessionID)
}

// Gemini CLI
func (a *GeminiAgent) FormatResumeCommand(sessionID string) string {
    return fmt.Sprintf("gemini chat resume %s", sessionID)
}
```

---

## Session Log 복원

### Transcript Path 결정

```go
func resolveTranscriptPath(sessionID string, ag agent.Agent) (string, error) {
    repoRoot, _ := paths.RepoRoot()
    sessionDir := ag.GetSessionDir(repoRoot)

    // Agent별 경로 형식
    return ag.GetTranscriptPath(sessionID)
}
```

**예시:**

- Claude Code: `.claude/sessions/<uuid>/full.jsonl`
- Gemini CLI: `.gemini/sessions/<date>-<uuid>.jsonl`

### 로그 쓰기

```go
func resumeSingleSession(ag agent.Agent, sessionID string, logContent []byte) error {
    // 1. Session directory 생성
    sessionDir := ag.GetSessionDir(repoRoot)
    os.MkdirAll(sessionDir, 0o700)

    // 2. AgentSession 생성
    agentSession := &agent.AgentSession{
        SessionID:  ag.ExtractAgentSessionID(sessionID),
        AgentName:  ag.Name(),
        RepoPath:   repoRoot,
        NativeData: logContent,
    }

    // 3. Agent의 WriteSession 호출
    return ag.WriteSession(agentSession)
}
```

---

## 실습 예제

### 1. 브랜치 전환 및 세션 복원

```bash
# 현재 브랜치 확인
git branch
# * main

# feature-login으로 전환 및 복원
entire resume feature-login

# 출력:
Switched to branch 'feature-login'
Session: 2026-02-11-abc123de-f456-7890-abcd-ef1234567890

To continue this session, run:
  claude resume abc123de-f456-7890-abcd-ef1234567890

# 세션 계속
claude resume abc123de-f456-7890-abcd-ef1234567890
```

### 2. Remote 브랜치 Fetch

```bash
# 팀원이 만든 브랜치
entire resume feature-auth

# 출력:
Branch 'feature-auth' not found locally. Fetch from origin? [y/N] y
Fetching branch 'feature-auth' from origin...
Switched to branch 'feature-auth'

Session: 2026-02-11-xyz789...
To continue this session, run:
  claude resume xyz789...
```

### 3. 여러 브랜치 오가기

```bash
# feature-A 작업
git checkout feature-A
claude "Implement feature A"
git commit -m "Add feature A"

# feature-B로 전환
entire resume feature-B
claude resume <session-id>
# feature-B 작업...

# 다시 feature-A로 돌아오기
entire resume feature-A
claude resume <session-id>
```

---

## Flags

### --force, -f

확인 프롬프트 없이 강제 실행합니다.

```bash
# 로컬 세션이 더 최신이어도 덮어쓰기
entire resume feature-login --force

# Newer commits 경고 무시
entire resume feature-login -f
```

---

## 에러 처리

### 1. Branch Not Found

```bash
entire resume non-existent-branch

# 출력:
Branch 'non-existent-branch' not found locally or on origin
```

### 2. No Checkpoint Found

```bash
entire resume feature-no-checkpoint

# 출력:
Switched to branch 'feature-no-checkpoint'
No Entire checkpoint found on branch 'feature-no-checkpoint'
```

### 3. Uncommitted Changes

```bash
entire resume feature-login

# 출력:
You have uncommitted changes. Please commit or stash them first.
```

**해결:**

```bash
git stash
entire resume feature-login
git stash pop
```

### 4. Metadata Not Available

```bash
entire resume feature-old

# 출력:
Switched to branch 'feature-old'
Checkpoint 'a3b2c4d5e6f7' found in commit but session metadata not available
The entire/checkpoints/v1 branch may not exist locally or on the remote.
```

**해결:**

```bash
# Metadata 브랜치 fetch
git fetch origin entire/checkpoints/v1:entire/checkpoints/v1

# 다시 시도
entire resume feature-old
```

---

## 고급 사용법

### 1. 이미 브랜치에 있는 경우

```bash
# feature-login 브랜치에 있는 상태
git branch
# * feature-login

# Resume 실행 (브랜치 전환 건너뜀)
entire resume feature-login

# 출력:
Session: 2026-02-11-abc123...
To continue this session, run:
  claude resume abc123...
```

### 2. Worktree에서 사용

각 worktree는 독립적인 세션을 가집니다.

```bash
# Worktree 생성
git worktree add ../feature-login feature-login

# Worktree로 이동
cd ../feature-login

# Resume
entire resume feature-login
```

---

## 성능 최적화

### 1. Checkpoint 검색 제한

최근 100개 커밋만 검색합니다.

```go
const maxCommits = 100

func findCheckpointInHistory(start *object.Commit) *branchCheckpointResult {
    totalChecked := 0
    for current != nil && totalChecked < maxCommits {
        // ...
        totalChecked++
    }
}
```

### 2. Metadata 캐싱

동일한 checkpoint ID는 캐시됩니다.

### 3. Parallel Session Restore

여러 세션을 병렬로 복원합니다.

```go
var wg sync.WaitGroup
for _, session := range sessions {
    wg.Add(1)
    go func(s SessionRestoreInfo) {
        defer wg.Done()
        restoreSessionLog(s)
    }(session)
}
wg.Wait()
```

---

## 다음 단계

Resume 기능을 이해했습니다! 다음 챕터에서는:

- **Auto-Summarization** - AI 기반 자동 요약
- **Token Usage Tracking** - 사용량 추적 및 분석
- **개발 환경 설정** - mise, Go, 테스트 실행

---

*다음 글에서는 Entire CLI의 Auto-Summarization 기능을 살펴봅니다.*
