---
layout: post
title: "Beads 완벽 가이드 (6) - 동기화 메커니즘"
date: 2025-02-04
permalink: /beads-guide-06-sync/
author: Steve Yegge
categories: [AI]
tags: [Beads, Sync, Import, Export, JSONL, Git]
original_url: "https://github.com/steveyegge/beads"
excerpt: "Beads의 SQLite-JSONL 양방향 동기화 메커니즘을 분석합니다."
---

## 동기화 개요

Beads의 동기화는 **SQLite(로컬 캐시)**와 **JSONL(Git 진실의 원천)** 사이의 양방향 변환입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Sync Architecture                            │
│                                                                  │
│   SQLite Database          ←→          JSONL File               │
│   (.beads/beads.db)                    (.beads/issues.jsonl)    │
│                                                                  │
│   ┌─────────────┐     Export      ┌─────────────┐              │
│   │  Fast Query │  ────────────▶  │  Git Track  │              │
│   │  Indexing   │                 │  Merge-safe │              │
│   │  Relations  │  ◀────────────  │  Portable   │              │
│   └─────────────┘     Import      └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Export 로직

### FlushManager

```go
// internal/flush/manager.go

type FlushManager struct {
    store       storage.Store
    workspace   string
    dirty       atomic.Bool
    debounceMs  int
    ticker      *time.Ticker
    stopChan    chan struct{}
}

func NewFlushManager(store storage.Store, workspace string) *FlushManager {
    return &FlushManager{
        store:      store,
        workspace:  workspace,
        debounceMs: 5000, // 5초 기본값
        stopChan:   make(chan struct{}),
    }
}

func (f *FlushManager) Start() {
    f.ticker = time.NewTicker(time.Duration(f.debounceMs) * time.Millisecond)

    for {
        select {
        case <-f.ticker.C:
            if f.dirty.Load() {
                f.flush()
                f.dirty.Store(false)
            }
        case <-f.stopChan:
            return
        }
    }
}

func (f *FlushManager) MarkDirty() {
    f.dirty.Store(true)
}
```

### 증분 익스포트

```go
// cmd/bd/export.go

func exportIssues(store storage.Store, outputPath string) error {
    // 모든 이슈 가져오기
    issues, err := store.ListIssues(nil)
    if err != nil {
        return err
    }

    // JSONL 파일 생성
    file, err := os.Create(outputPath)
    if err != nil {
        return err
    }
    defer file.Close()

    encoder := json.NewEncoder(file)

    for _, issue := range issues {
        // Tombstone 포함 (삭제된 이슈도 기록)
        if err := encoder.Encode(issue); err != nil {
            return err
        }
    }

    return nil
}
```

### 콘텐츠 해시 기반 변경 감지

```go
// internal/export/hash.go

type ExportHashDB struct {
    db *sql.DB
}

func (h *ExportHashDB) NeedsExport(issue *types.Issue) bool {
    currentHash := issue.ComputeContentHash()

    var storedHash string
    err := h.db.QueryRow(
        "SELECT hash FROM export_hashes WHERE id = ?",
        issue.ID,
    ).Scan(&storedHash)

    if err == sql.ErrNoRows {
        return true // 새 이슈
    }

    return currentHash != storedHash
}

func (h *ExportHashDB) UpdateHash(issue *types.Issue) error {
    _, err := h.db.Exec(`
        INSERT OR REPLACE INTO export_hashes (id, hash, exported_at)
        VALUES (?, ?, ?)
    `, issue.ID, issue.ComputeContentHash(), time.Now())
    return err
}
```

---

## Import 로직

### 자동 임포트 감지

```go
// internal/autoimport/detector.go

type ImportDetector struct {
    workspace  string
    store      storage.Store
    lastMtime  time.Time
}

func (d *ImportDetector) CheckForUpdates() bool {
    jsonlPath := filepath.Join(d.workspace, ".beads", "issues.jsonl")

    info, err := os.Stat(jsonlPath)
    if err != nil {
        return false
    }

    if info.ModTime().After(d.lastMtime) {
        d.lastMtime = info.ModTime()
        return true
    }

    return false
}
```

### JSONL 파싱

```go
// internal/importer/jsonl.go

func ImportJSONL(path string, store storage.Store) (*ImportResult, error) {
    file, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    result := &ImportResult{}
    scanner := bufio.NewScanner(file)

    for scanner.Scan() {
        line := scanner.Bytes()
        if len(line) == 0 {
            continue
        }

        var issue types.Issue
        if err := json.Unmarshal(line, &issue); err != nil {
            result.Errors = append(result.Errors, err)
            continue
        }

        action, err := mergeIssue(store, &issue)
        if err != nil {
            result.Errors = append(result.Errors, err)
            continue
        }

        switch action {
        case ActionCreated:
            result.Created++
        case ActionUpdated:
            result.Updated++
        case ActionSkipped:
            result.Skipped++
        }
    }

    return result, scanner.Err()
}
```

### 머지 로직

```go
// internal/importer/merge.go

type MergeAction int

const (
    ActionSkipped MergeAction = iota
    ActionCreated
    ActionUpdated
)

func mergeIssue(store storage.Store, incoming *types.Issue) (MergeAction, error) {
    existing, err := store.GetIssue(incoming.ID)

    if err == storage.ErrNotFound {
        // 새 이슈 - 생성
        if err := store.CreateIssue(incoming); err != nil {
            return ActionSkipped, err
        }
        return ActionCreated, nil
    }

    if err != nil {
        return ActionSkipped, err
    }

    // 콘텐츠 해시 비교
    incomingHash := incoming.ComputeContentHash()
    existingHash := existing.ComputeContentHash()

    if incomingHash == existingHash {
        // 동일 - 스킵
        return ActionSkipped, nil
    }

    // 타임스탬프 기반 충돌 해결
    if incoming.UpdatedAt.After(existing.UpdatedAt) {
        if err := store.UpdateIssue(incoming); err != nil {
            return ActionSkipped, err
        }
        return ActionUpdated, nil
    }

    // 로컬이 더 새로움 - 스킵
    return ActionSkipped, nil
}
```

---

## 충돌 해결 전략

### Last-Write-Wins

```
┌─────────────────────────────────────────────────────────────────┐
│                   Conflict Resolution                            │
│                                                                  │
│   Local (UpdatedAt: 14:00)    vs    Remote (UpdatedAt: 15:00)   │
│                                                                  │
│   Result: Remote wins (더 최근 타임스탬프)                       │
└─────────────────────────────────────────────────────────────────┘
```

### 해시 ID로 충돌 방지

```go
// internal/types/id.go

func GenerateID(prefix string) string {
    // 랜덤 바이트 생성
    bytes := make([]byte, 4)
    rand.Read(bytes)

    // Base36 인코딩 (0-9, a-z)
    hash := base36Encode(bytes)

    // 접두사 + 해시
    return fmt.Sprintf("%s-%s", prefix, hash[:4])
}
```

다른 머신에서 동시에 이슈를 생성해도 ID 충돌 확률이 매우 낮습니다.

---

## Git 통합

### 자동 커밋

```go
// internal/sync/git.go

func (s *SyncManager) CommitChanges() error {
    // Git 상태 확인
    status, err := s.git.Status()
    if err != nil {
        return err
    }

    // 변경된 JSONL 파일 확인
    jsonlPath := ".beads/issues.jsonl"
    if !status.IsModified(jsonlPath) {
        return nil // 변경 없음
    }

    // 스테이징
    if err := s.git.Add(jsonlPath); err != nil {
        return err
    }

    // 커밋
    message := fmt.Sprintf("beads: sync %d issues", s.changeCount)
    return s.git.Commit(message)
}
```

### Sync 브랜치 지원

보호된 main 브랜치가 있는 경우, 별도 sync 브랜치 사용:

```go
// internal/sync/branch.go

func (s *SyncManager) GetSyncBranch() string {
    if s.config.SyncBranch != "" {
        return s.config.SyncBranch
    }

    // 기본값: beads-metadata
    return "beads-metadata"
}

func (s *SyncManager) SyncToRemote() error {
    branch := s.GetSyncBranch()

    // sync 브랜치로 체크아웃
    if err := s.git.Checkout(branch); err != nil {
        // 브랜치가 없으면 생성
        if err := s.git.CreateBranch(branch); err != nil {
            return err
        }
    }

    // 변경 커밋
    if err := s.CommitChanges(); err != nil {
        return err
    }

    // 푸시
    return s.git.Push(branch)
}
```

---

## 동기화 명령어

### bd sync

```bash
# 기본 동기화 (export + git commit)
bd sync

# 강제 전체 익스포트
bd sync --force

# 푸시 포함
bd sync --push

# 드라이런
bd sync --dry-run
```

### bd import / bd export

```bash
# 수동 임포트
bd import -i .beads/issues.jsonl

# 특정 파일에서 임포트
bd import -i backup/issues.jsonl

# 수동 익스포트
bd export -o .beads/issues.jsonl

# 다른 위치로 익스포트
bd export -o /tmp/issues-backup.jsonl
```

---

## 디바운스 설정

```yaml
# .beads/config.yaml

sync:
  debounce_ms: 5000    # 5초 (기본값)
  auto_commit: true
  auto_push: false
  sync_branch: ""      # 비어있으면 현재 브랜치
```

```go
// 환경 변수로 오버라이드
// BEADS_SYNC_DEBOUNCE=10000  (10초)
// BEADS_AUTO_PUSH=true
```

---

## 동기화 상태 확인

```bash
# 동기화 상태
bd info

# 출력 예시:
# Workspace: /Users/alice/projects/webapp
# Database: .beads/beads.db
# JSONL: .beads/issues.jsonl
#
# Sync Status:
#   Last export: 2 minutes ago
#   Last import: 5 minutes ago
#   Pending changes: 3
#   Dirty: yes
```

---

## 오류 처리

### 임포트 실패 시

```go
type ImportResult struct {
    Created int
    Updated int
    Skipped int
    Errors  []error
}

func (r *ImportResult) Summary() string {
    return fmt.Sprintf(
        "Import complete: %d created, %d updated, %d skipped, %d errors",
        r.Created, r.Updated, r.Skipped, len(r.Errors),
    )
}
```

### 복구 명령어

```bash
# 데이터베이스 상태 진단
bd doctor

# 문제 발견 시 자동 수정
bd doctor --fix

# JSONL에서 강제 재임포트
bd import --force -i .beads/issues.jsonl
```

---

*다음 글에서는 의존성 관리와 Ready 작업 감지를 살펴봅니다.*
