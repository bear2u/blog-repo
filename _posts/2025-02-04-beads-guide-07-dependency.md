---
layout: post
title: "Beads 완벽 가이드 (7) - 의존성 관리"
date: 2025-02-04
permalink: /beads-guide-07-dependency/
author: Steve Yegge
categories: [AI]
tags: [Beads, Dependency, Graph, Ready, Topological Sort]
original_url: "https://github.com/steveyegge/beads"
excerpt: "Beads의 의존성 그래프와 Ready 작업 감지 메커니즘을 분석합니다."
---

## 의존성 개요

Beads의 핵심 기능 중 하나는 **의존성 인식 작업 관리**입니다. `bd ready` 명령어는 차단된 작업을 제외하고 즉시 시작 가능한 작업만 표시합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                   Dependency Graph Example                       │
│                                                                  │
│   bd-epic (Epic)                                                │
│       │                                                          │
│       ├── blocks ──▶ bd-auth (Task) ──── READY ✓                │
│       │                                                          │
│       └── blocks ──▶ bd-api (Task)                              │
│                          │                                       │
│                          └── blocks ──▶ bd-ui (Task) ── BLOCKED │
└─────────────────────────────────────────────────────────────────┘
```

---

## 의존성 타입

### 4가지 의존성 유형

```go
// internal/types/dependency.go

type DependencyType string

const (
    // 차단 의존성 (Ready 영향)
    DepBlocks      DependencyType = "blocks"
    DepParentChild DependencyType = "parent-child"

    // 참조 의존성 (Ready 영향 없음)
    DepRelated        DependencyType = "related"
    DepDiscoveredFrom DependencyType = "discovered-from"
)
```

### 의존성 의미론

| 타입 | 방향 | Ready 영향 | 사용 사례 |
|------|------|------------|----------|
| `blocks` | A blocks B | Yes | A 완료 전 B 시작 불가 |
| `parent-child` | Parent → Child | Yes | Epic/Sub-task 계층 |
| `related` | A ↔ B | No | 참조 링크 |
| `discovered-from` | A → B | No | 작업 중 발견된 이슈 |

---

## 의존성 저장

### SQLite 스키마

```sql
CREATE TABLE dependencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    type TEXT NOT NULL,
    FOREIGN KEY (from_id) REFERENCES issues(id) ON DELETE CASCADE,
    FOREIGN KEY (to_id) REFERENCES issues(id) ON DELETE CASCADE,
    UNIQUE(from_id, to_id, type)
);

CREATE INDEX idx_dependencies_from ON dependencies(from_id);
CREATE INDEX idx_dependencies_to ON dependencies(to_id);
```

### 의존성 추가/제거

```go
// internal/storage/sqlite/dependencies.go

func (s *SQLiteStore) AddDependency(dep *types.Dependency) error {
    _, err := s.db.Exec(`
        INSERT OR IGNORE INTO dependencies (from_id, to_id, type)
        VALUES (?, ?, ?)
    `, dep.FromID, dep.ToID, dep.Type)
    return err
}

func (s *SQLiteStore) RemoveDependency(fromID, toID string) error {
    _, err := s.db.Exec(`
        DELETE FROM dependencies
        WHERE from_id = ? AND to_id = ?
    `, fromID, toID)
    return err
}
```

---

## Ready 작업 감지

### 핵심 쿼리

```go
// internal/storage/sqlite/ready.go

func (s *SQLiteStore) GetReadyIssues() ([]*types.Issue, error) {
    query := `
        SELECT i.* FROM issues i
        WHERE i.status = 'open'
          AND i.deleted_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM dependencies d
              JOIN issues blocker ON d.from_id = blocker.id
              WHERE d.to_id = i.id
                AND d.type IN ('blocks', 'parent-child')
                AND blocker.status != 'closed'
                AND blocker.deleted_at IS NULL
          )
        ORDER BY i.priority ASC, i.created_at ASC
    `
    return s.queryIssues(query)
}
```

### 동작 원리

```
┌─────────────────────────────────────────────────────────────────┐
│                     Ready Detection Logic                        │
│                                                                  │
│  1. status = 'open' 인 이슈 선택                                │
│  2. 삭제되지 않은 이슈만 (deleted_at IS NULL)                   │
│  3. 차단 의존성 확인:                                            │
│     - blocks 또는 parent-child 타입                             │
│     - 차단자(blocker)가 아직 열려있는지 확인                    │
│  4. 차단자가 없거나 모두 closed → READY                         │
│  5. 우선순위순 정렬 (P0 먼저)                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## CLI 명령어

### bd dep add

```bash
# 차단 의존성 추가 (기본값)
bd dep add bd-task bd-blocker

# 명시적 타입 지정
bd dep add bd-task bd-blocker --blocks
bd dep add bd-child bd-parent --parent-child
bd dep add bd-a bd-b --related
bd dep add bd-new bd-origin --discovered-from
```

```go
// cmd/bd/dep.go

var depAddCmd = &cobra.Command{
    Use:   "add <from-id> <to-id>",
    Short: "Add a dependency between issues",
    RunE: func(cmd *cobra.Command, args []string) error {
        fromID := args[0]
        toID := args[1]

        depType := types.DepBlocks // 기본값
        if parentChild {
            depType = types.DepParentChild
        } else if related {
            depType = types.DepRelated
        } else if discoveredFrom {
            depType = types.DepDiscoveredFrom
        }

        dep := &types.Dependency{
            FromID: fromID,
            ToID:   toID,
            Type:   depType,
        }

        return store.AddDependency(dep)
    },
}
```

### bd dep rm

```bash
# 의존성 제거
bd dep rm bd-task bd-blocker
```

### bd dep list

```bash
# 특정 이슈의 의존성 목록
bd dep list bd-a1b2

# 출력:
# FROM      TO        TYPE
# bd-a1b2   bd-f14c   blocks
# bd-a1b2   bd-x9z3   related
```

### bd dep graph

```bash
# 전체 의존성 그래프
bd dep graph

# DOT 형식 출력 (Graphviz)
bd dep graph --format dot > deps.dot
dot -Tpng deps.dot -o deps.png
```

---

## 그래프 알고리즘

### 순환 감지

```go
// internal/graph/cycle.go

func DetectCycle(store storage.Store) ([]string, error) {
    deps, err := store.ListDependencies()
    if err != nil {
        return nil, err
    }

    // 인접 리스트 구성
    graph := make(map[string][]string)
    for _, dep := range deps {
        if dep.Type == types.DepBlocks || dep.Type == types.DepParentChild {
            graph[dep.FromID] = append(graph[dep.FromID], dep.ToID)
        }
    }

    // DFS로 순환 감지
    visited := make(map[string]bool)
    recStack := make(map[string]bool)
    var cyclePath []string

    var dfs func(node string) bool
    dfs = func(node string) bool {
        visited[node] = true
        recStack[node] = true

        for _, neighbor := range graph[node] {
            if !visited[neighbor] {
                if dfs(neighbor) {
                    cyclePath = append([]string{node}, cyclePath...)
                    return true
                }
            } else if recStack[neighbor] {
                cyclePath = []string{node, neighbor}
                return true
            }
        }

        recStack[node] = false
        return false
    }

    for node := range graph {
        if !visited[node] {
            if dfs(node) {
                return cyclePath, nil
            }
        }
    }

    return nil, nil // 순환 없음
}
```

### 위상 정렬

```go
// internal/graph/topo.go

func TopologicalSort(store storage.Store) ([]string, error) {
    deps, err := store.ListDependencies()
    if err != nil {
        return nil, err
    }

    // 진입 차수 계산
    inDegree := make(map[string]int)
    graph := make(map[string][]string)

    for _, dep := range deps {
        if dep.Type == types.DepBlocks || dep.Type == types.DepParentChild {
            graph[dep.FromID] = append(graph[dep.FromID], dep.ToID)
            inDegree[dep.ToID]++
            if inDegree[dep.FromID] == 0 {
                inDegree[dep.FromID] = 0
            }
        }
    }

    // 진입 차수 0인 노드부터 시작
    var queue []string
    for node, degree := range inDegree {
        if degree == 0 {
            queue = append(queue, node)
        }
    }

    var result []string
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        result = append(result, node)

        for _, neighbor := range graph[node] {
            inDegree[neighbor]--
            if inDegree[neighbor] == 0 {
                queue = append(queue, neighbor)
            }
        }
    }

    return result, nil
}
```

---

## 계층적 ID

Epic과 Sub-task를 위한 계층적 ID 지원:

```
bd-a3f8           (Epic)
├── bd-a3f8.1     (Task 1)
│   └── bd-a3f8.1.1   (Sub-task)
└── bd-a3f8.2     (Task 2)
```

```go
// internal/types/id.go

func CreateChildID(parentID string) string {
    // 기존 자식 수 확인
    children := store.GetChildCount(parentID)
    return fmt.Sprintf("%s.%d", parentID, children+1)
}

func IsChildOf(childID, parentID string) bool {
    return strings.HasPrefix(childID, parentID+".")
}
```

---

## Ready 명령어 옵션

```bash
# 기본 Ready 작업
bd ready

# 특정 우선순위만
bd ready -p 0       # P0만
bd ready -p 0,1     # P0, P1

# 특정 라벨
bd ready --label urgent

# JSON 출력 (AI 에이전트용)
bd ready --json

# 간략 출력
bd ready --brief

# 첫 번째 작업만
bd ready --first
```

**출력 예시:**

```
READY TASKS (3 items)

ID        PRI  TYPE     TITLE
bd-a1b2   P0   feature  Add OAuth login
bd-f14c   P1   bug      Fix session timeout
bd-x9z3   P2   task     Update documentation
```

---

## 이벤트 추적

의존성 변경은 감사 로그에 기록됩니다:

```go
// internal/events/dependency.go

func RecordDependencyAdded(store storage.Store, dep *types.Dependency) error {
    event := &types.Event{
        ID:        generateEventID(),
        IssueID:   dep.ToID,
        Type:      types.EventDependencyAdded,
        Data:      mustMarshal(dep),
        Actor:     getCurrentActor(),
        Timestamp: time.Now(),
    }
    return store.CreateEvent(event)
}
```

```bash
# 이슈의 감사 로그 확인
bd show bd-a1b2 --audit

# 출력:
# Audit Trail:
#   2025-02-04 10:30  Created by alice
#   2025-02-04 11:00  Dependency added: blocks bd-f14c
#   2025-02-04 14:00  Status changed: open → in_progress
```

---

*다음 글에서는 Molecules와 Wisps 시스템을 살펴봅니다.*
