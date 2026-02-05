---
layout: post
title: "Beads 완벽 가이드 (4) - 데이터 모델"
date: 2025-02-04
permalink: /beads-guide-04-data-model/
author: Steve Yegge
categories: [AI]
tags: [Beads, Data Model, JSONL, Schema, Types]
original_url: "https://github.com/steveyegge/beads"
excerpt: "Beads의 핵심 데이터 타입과 JSONL 스키마를 분석합니다."
---

## 핵심 데이터 타입

Beads의 핵심 타입은 `internal/types/types.go`에 정의되어 있습니다.

```go
// internal/types/types.go

type Issue struct {
    ID                 string       `json:"id"`
    Title              string       `json:"title"`
    Description        string       `json:"description,omitempty"`
    Design             string       `json:"design,omitempty"`
    AcceptanceCriteria string       `json:"acceptance_criteria,omitempty"`
    Notes              string       `json:"notes,omitempty"`
    Status             IssueStatus  `json:"status,omitempty"`
    Priority           int          `json:"priority"`
    IssueType          IssueType    `json:"issue_type,omitempty"`
    Assignee           string       `json:"assignee,omitempty"`
    EstimatedMinutes   int          `json:"estimated_minutes,omitempty"`
    CreatedAt          time.Time    `json:"created_at"`
    CreatedBy          string       `json:"created_by,omitempty"`
    UpdatedAt          time.Time    `json:"updated_at"`
    ClosedAt           *time.Time   `json:"closed_at,omitempty"`
    CloseReason        string       `json:"close_reason,omitempty"`
    ExternalRef        string       `json:"external_ref,omitempty"`
    Labels             []string     `json:"labels,omitempty"`
    Dependencies       []Dependency `json:"dependencies,omitempty"`
    Comments           []Comment    `json:"comments,omitempty"`

    // Tombstone 필드 (소프트 삭제)
    DeletedAt      *time.Time `json:"deleted_at,omitempty"`
    DeletedBy      string     `json:"deleted_by,omitempty"`
    DeleteReason   string     `json:"delete_reason,omitempty"`
    OriginalType   string     `json:"original_type,omitempty"`

    // 내부 필드 (JSONL에 포함 안됨)
    ContentHash string `json:"-"`
    SourceRepo  string `json:"-"`
    IDPrefix    string `json:"-"`
}
```

---

## Issue Status

```go
type IssueStatus string

const (
    StatusOpen       IssueStatus = "open"
    StatusInProgress IssueStatus = "in_progress"
    StatusBlocked    IssueStatus = "blocked"
    StatusDeferred   IssueStatus = "deferred"
    StatusClosed     IssueStatus = "closed"
    StatusTombstone  IssueStatus = "tombstone"
    StatusPinned     IssueStatus = "pinned"
    StatusHooked     IssueStatus = "hooked"
)
```

### 상태 흐름

```
open ──▶ in_progress ──▶ closed
  │           │            │
  │           ▼            │
  │       blocked          │
  │           │            │
  └───────────┴────────────┘
           (reopen)
```

---

## Issue Type

```go
type IssueType string

const (
    TypeTask         IssueType = "task"
    TypeBug          IssueType = "bug"
    TypeFeature      IssueType = "feature"
    TypeEpic         IssueType = "epic"
    TypeChore        IssueType = "chore"
    TypeMessage      IssueType = "message"
    TypeMergeRequest IssueType = "merge-request"
    TypeMolecule     IssueType = "molecule"
    TypeGate         IssueType = "gate"
    TypeAgent        IssueType = "agent"
    TypeRole         IssueType = "role"
    TypeConvoy       IssueType = "convoy"
)
```

---

## Priority

| 우선순위 | 값 | 의미 |
|----------|---|------|
| P0 | 0 | Critical - 즉시 처리 필요 |
| P1 | 1 | High - 빠른 처리 필요 |
| P2 | 2 | Medium - 일반 작업 (기본값) |
| P3 | 3 | Low - 나중에 처리 |
| P4 | 4 | Backlog - 언젠가 |

---

## Dependency

```go
type Dependency struct {
    FromID string         `json:"from_id"`
    ToID   string         `json:"to_id"`
    Type   DependencyType `json:"type"`
}

type DependencyType string

const (
    DepBlocks         DependencyType = "blocks"
    DepRelated        DependencyType = "related"
    DepParentChild    DependencyType = "parent-child"
    DepDiscoveredFrom DependencyType = "discovered-from"
)
```

### 의존성 의미론

| 타입 | 의미 | Ready 영향 |
|------|------|------------|
| `blocks` | FromID가 완료되어야 ToID 시작 가능 | Yes |
| `parent-child` | 계층 구조 (FromID가 ToID의 자식) | Yes |
| `related` | 참조 링크 | No |
| `discovered-from` | 작업 중 발견됨 | No |

---

## Comment

```go
type Comment struct {
    ID        string    `json:"id"`
    IssueID   string    `json:"issue_id"`
    Author    string    `json:"author,omitempty"`
    Content   string    `json:"content"`
    CreatedAt time.Time `json:"created_at"`
}
```

---

## Event (감사 추적)

```go
type Event struct {
    ID        string          `json:"id"`
    IssueID   string          `json:"issue_id"`
    Type      EventType       `json:"type"`
    Data      json.RawMessage `json:"data,omitempty"`
    Actor     string          `json:"actor,omitempty"`
    Timestamp time.Time       `json:"timestamp"`
}

type EventType string

const (
    EventCreated          EventType = "created"
    EventUpdated          EventType = "updated"
    EventClosed           EventType = "closed"
    EventReopened         EventType = "reopened"
    EventPriorityChanged  EventType = "priority_changed"
    EventStatusChanged    EventType = "status_changed"
    EventAssigneeChanged  EventType = "assignee_changed"
    EventDependencyAdded  EventType = "dependency_added"
    EventDependencyRemoved EventType = "dependency_removed"
    EventCommentAdded     EventType = "comment_added"
    EventLabelAdded       EventType = "label_added"
    EventLabelRemoved     EventType = "label_removed"
)
```

---

## Label

```go
type Label struct {
    Name        string `json:"name"`
    Color       string `json:"color,omitempty"`
    Description string `json:"description,omitempty"`
}
```

---

## JSONL 스키마

`.beads/issues.jsonl`의 각 라인은 JSON 객체입니다.

### 이슈 예시

```json
{
  "id": "bd-a1b2",
  "title": "Add user authentication",
  "description": "Implement OAuth 2.0 with Google and GitHub",
  "status": "open",
  "priority": 0,
  "issue_type": "feature",
  "assignee": "agent-1",
  "created_at": "2025-02-04T10:30:00Z",
  "created_by": "alice",
  "updated_at": "2025-02-04T14:22:00Z",
  "labels": ["auth", "security", "p0"],
  "dependencies": [
    {"from_id": "bd-a1b2", "to_id": "bd-f14c", "type": "blocks"}
  ]
}
```

### 최소 이슈

```json
{"id":"bd-x7y9","title":"Quick fix","priority":2,"created_at":"2025-02-04T10:30:00Z","updated_at":"2025-02-04T10:30:00Z"}
```

### Tombstone (삭제된 이슈)

```json
{
  "id": "bd-dead",
  "title": "Deleted issue",
  "status": "tombstone",
  "deleted_at": "2025-02-04T15:00:00Z",
  "deleted_by": "admin",
  "delete_reason": "Duplicate of bd-a1b2",
  "original_type": "task"
}
```

---

## SQLite 스키마

```sql
-- internal/storage/sqlite/schema.sql

CREATE TABLE issues (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    design TEXT,
    acceptance_criteria TEXT,
    notes TEXT,
    status TEXT DEFAULT 'open',
    priority INTEGER DEFAULT 2,
    issue_type TEXT DEFAULT 'task',
    assignee TEXT,
    estimated_minutes INTEGER,
    created_at DATETIME NOT NULL,
    created_by TEXT,
    updated_at DATETIME NOT NULL,
    closed_at DATETIME,
    close_reason TEXT,
    external_ref TEXT,
    content_hash TEXT,
    deleted_at DATETIME,
    deleted_by TEXT,
    delete_reason TEXT,
    original_type TEXT
);

CREATE TABLE dependencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    type TEXT NOT NULL,
    FOREIGN KEY (from_id) REFERENCES issues(id) ON DELETE CASCADE,
    FOREIGN KEY (to_id) REFERENCES issues(id) ON DELETE CASCADE,
    UNIQUE(from_id, to_id, type)
);

CREATE TABLE labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id TEXT NOT NULL,
    name TEXT NOT NULL,
    FOREIGN KEY (issue_id) REFERENCES issues(id) ON DELETE CASCADE,
    UNIQUE(issue_id, name)
);

CREATE TABLE comments (
    id TEXT PRIMARY KEY,
    issue_id TEXT NOT NULL,
    author TEXT,
    content TEXT NOT NULL,
    created_at DATETIME NOT NULL,
    FOREIGN KEY (issue_id) REFERENCES issues(id) ON DELETE CASCADE
);

CREATE TABLE events (
    id TEXT PRIMARY KEY,
    issue_id TEXT NOT NULL,
    type TEXT NOT NULL,
    data TEXT,
    actor TEXT,
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (issue_id) REFERENCES issues(id) ON DELETE CASCADE
);

-- 인덱스
CREATE INDEX idx_issues_status ON issues(status);
CREATE INDEX idx_issues_priority ON issues(priority);
CREATE INDEX idx_dependencies_from ON dependencies(from_id);
CREATE INDEX idx_dependencies_to ON dependencies(to_id);
CREATE INDEX idx_labels_name ON labels(name);
CREATE INDEX idx_events_issue ON events(issue_id);
```

---

## 콘텐츠 해시

변경 감지를 위해 각 이슈의 콘텐츠 해시를 계산합니다.

```go
func (i *Issue) ComputeContentHash() string {
    h := sha256.New()

    // 핵심 필드만 해시에 포함
    h.Write([]byte(i.ID))
    h.Write([]byte(i.Title))
    h.Write([]byte(i.Description))
    h.Write([]byte(i.Status))
    h.Write([]byte(fmt.Sprintf("%d", i.Priority)))
    h.Write([]byte(i.Assignee))
    h.Write([]byte(i.UpdatedAt.Format(time.RFC3339)))

    return hex.EncodeToString(h.Sum(nil))[:16]
}
```

---

*다음 글에서는 데몬 시스템을 살펴봅니다.*
