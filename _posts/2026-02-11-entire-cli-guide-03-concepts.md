---
layout: post
title: "Entire CLI 완벽 가이드 (03) - 핵심 개념"
date: 2026-02-11
permalink: /entire-cli-guide-03-concepts/
author: Entire Team
categories: [AI 코딩, 개발 도구]
tags: [Entire, Session, Checkpoint, Strategy, Git]
original_url: "https://github.com/entireio/cli"
excerpt: "Entire CLI의 핵심 개념인 Session, Checkpoint, Strategy 완벽 이해"
---

## 개요

Entire CLI는 세 가지 핵심 개념을 기반으로 작동합니다:

1. **Session** - AI 에이전트와의 완전한 상호작용
2. **Checkpoint** - 세션 내에서 되돌릴 수 있는 저장 지점
3. **Strategy** - 체크포인트를 언제, 어떻게 저장할지 결정

이 챕터에서는 각 개념을 깊이 있게 이해합니다.

---

## Session (세션)

### 정의

**Session**은 AI 에이전트와의 완전한 상호작용 단위입니다. 시작부터 끝까지의 모든 프롬프트, 응답, 파일 변경사항을 포함합니다.

```
Session = 시작 → 여러 프롬프트/응답 → 종료
```

### Session ID 형식

```
YYYY-MM-DD-<UUID>

예시:
2026-02-11-abc123de-f456-7890-abcd-ef1234567890
```

- **날짜 접두사** - 세션 생성 날짜 (KST)
- **UUID** - AI 에이전트의 고유 세션 ID

### Session 구조

```go
type Session struct {
    ID          string       // 세션 ID
    Description string       // 첫 프롬프트 또는 요약
    Strategy    string       // 사용된 전략
    StartTime   time.Time    // 시작 시간
    Checkpoints []Checkpoint // 체크포인트 목록
}
```

### Session 생명주기

```
┌─────────────────────────────────────────┐
│                                          │
│  SessionStart Hook                       │
│  ↓                                       │
│  Session ID 생성 및 저장                 │
│  ↓                                       │
│  [IDLE Phase]                            │
│                                          │
│  UserPromptSubmit Hook                   │
│  ↓                                       │
│  [ACTIVE Phase]                          │
│  - 프롬프트 캡처                          │
│  - AI 작업 중                            │
│  ↓                                       │
│  Stop Hook                               │
│  ↓                                       │
│  Checkpoint 생성                         │
│  ↓                                       │
│  [IDLE Phase]                            │
│                                          │
│  사용자 커밋 (manual-commit)              │
│  ↓                                       │
│  PostCommit Hook                         │
│  ↓                                       │
│  Condensation (메타데이터 저장)           │
│  ↓                                       │
│  [ENDED Phase]                           │
│                                          │
└─────────────────────────────────────────┘
```

### Session 저장 위치

| 저장소 | 위치 | 내용 |
|-------|------|------|
| **Session State** | `.git/entire-sessions/<id>.json` | 활성 세션 상태 |
| **Temporary** | `entire/<commit[:7]>-<worktree[:6]>` 브랜치 | 미커밋 체크포인트 |
| **Committed** | `entire/checkpoints/v1` 브랜치 | 영구 메타데이터 |

---

## Checkpoint (체크포인트)

### 정의

**Checkpoint**는 세션 내에서 **되돌릴 수 있는 저장 지점**입니다. 코드 상태와 메타데이터의 스냅샷을 포함합니다.

```
Checkpoint = 코드 스냅샷 + 메타데이터
```

### Checkpoint ID 형식

```
12-hex-character 랜덤 ID

예시:
a3b2c4d5e6f7
```

이 ID는:
- **커밋 메시지 Trailer**에 추가됨
- **entire/checkpoints/v1 디렉토리 샤딩**에 사용됨
- **커밋과 메타데이터 연결**에 사용됨

### Checkpoint 구조

```go
type Checkpoint struct {
    CheckpointID     id.CheckpointID // 12-hex ID
    Message          string          // 커밋 메시지
    Timestamp        time.Time       // 생성 시간
    IsTaskCheckpoint bool            // 태스크 체크포인트 여부
    ToolUseID        string          // 도구 사용 ID (태스크용)
}
```

### Checkpoint 타입

#### 1. Temporary Checkpoint

**위치:** Shadow 브랜치 (`entire/<commit[:7]>-<worktree[:6]>`)

**내용:**
- 전체 워크트리 스냅샷
- 메타데이터 오버레이

**용도:**
- 세션 중 rewind
- 커밋 전 임시 저장

**생명주기:** 커밋 시 삭제됨

```
.entire/metadata/<session-id>/
├── full.jsonl           # 전체 transcript
├── prompt.txt           # 사용자 프롬프트
├── context.md           # 생성된 컨텍스트
└── tasks/<tool-use-id>/ # 태스크 체크포인트
```

#### 2. Committed Checkpoint

**위치:** `entire/checkpoints/v1` 브랜치

**내용:**
- 메타데이터만 (코드는 커밋 참조)
- 샤딩된 디렉토리 구조

**용도:**
- 영구 기록
- 커밋 후 rewind
- 세션 조회

**생명주기:** 영구 보관

```
<id[:2]>/<id[2:]>/
├── metadata.json        # CheckpointSummary
├── 0/                   # 첫 번째 세션
│   ├── metadata.json    # CommittedMetadata
│   ├── full.jsonl
│   ├── prompt.txt
│   ├── context.md
│   └── content_hash.txt
└── 1/                   # 두 번째 세션 (동시 세션)
```

### Checkpoint 생성 시점

| Strategy | Temporary | Committed |
|---------|-----------|-----------|
| **Manual-commit** | AI 응답마다 | 사용자 커밋 시 |
| **Auto-commit** | 없음 | AI 응답마다 |

---

## Strategy (전략)

### 정의

**Strategy**는 체크포인트를 **언제, 어떻게 저장할지** 결정하는 정책입니다.

```go
type Strategy interface {
    SaveChanges(ctx SaveContext) error
    SaveTaskCheckpoint(ctx TaskCheckpointContext) error
    GetRewindPoints(repo *git.Repository) ([]RewindPoint, error)
    Rewind(repo *git.Repository, rewindPoint *RewindPoint) error
    // ...
}
```

### Manual-Commit Strategy (기본)

**동작 방식:**

1. **AI 응답 시** - Temporary 체크포인트 생성 (shadow 브랜치)
2. **사용자 커밋 시** - Committed 체크포인트로 condensation
3. **Rewind** - Shadow 브랜치에서 파일 복원

**특징:**

```
장점:
✓ 깔끔한 Git 히스토리 (사용자가 직접 커밋)
✓ main 브랜치에서 안전
✓ Flexible rewind (항상 가능)

단점:
✗ 수동 커밋 필요
✗ Shadow 브랜치 관리 필요
```

**사용 예시:**

```bash
# 활성화
entire enable --strategy manual-commit

# 작업 흐름
claude "Add feature"
# → Temporary 체크포인트 생성

git commit -m "Add feature"
# → Committed 체크포인트로 condensation
# → Shadow 브랜치 삭제
```

### Auto-Commit Strategy

**동작 방식:**

1. **AI 응답 시** - 자동으로 커밋 생성 + Committed 체크포인트
2. **Rewind** - Feature 브랜치에서 `git reset --hard`

**특징:**

```
장점:
✓ 완전 자동화
✓ 세밀한 체크포인트
✓ Shadow 브랜치 불필요

단점:
✗ 많은 커밋 생성
✗ main 브랜치에서 주의 필요
✗ Rewind 제한적 (main에서는 로그만)
```

**사용 예시:**

```bash
# 활성화
entire enable --strategy auto-commit

# 작업 흐름
claude "Add feature"
# → 자동으로 커밋 + 체크포인트 생성

git log
# → AI 응답마다 커밋이 생성됨
```

### Strategy 비교

| 항목 | Manual-Commit | Auto-Commit |
|-----|---------------|-------------|
| **코드 커밋** | 사용자가 직접 | 자동 생성 |
| **체크포인트 빈도** | 커밋 시 | AI 응답마다 |
| **Git 히스토리** | 깔끔 | 많은 커밋 |
| **main 브랜치** | 안전 | 주의 필요 |
| **Rewind** | 항상 가능 | 제한적 |
| **적합한 용도** | 대부분의 워크플로우 | 자동 커밋 원하는 팀 |

---

## 워크플로우 예시

### Manual-Commit 워크플로우

```
1. entire enable
   ↓
2. claude "Add feature"
   ↓
   [Temporary checkpoint created]
   ↓
3. entire status
   → "1 uncommitted checkpoint"
   ↓
4. git commit -m "Add feature"
   ↓
   [Condensation to entire/checkpoints/v1]
   [Shadow branch deleted]
   ↓
5. git push
   [entire/checkpoints/v1 also pushed]
```

### Auto-Commit 워크플로우

```
1. entire enable --strategy auto-commit
   ↓
2. claude "Add feature"
   ↓
   [Auto commit created]
   [Committed checkpoint created]
   ↓
3. git log
   → "Commit with Entire-Checkpoint trailer"
   ↓
4. git push
   [Code + entire/checkpoints/v1 pushed]
```

---

## 메타데이터 구조

### CheckpointSummary (Root)

```json
{
  "checkpoint_id": "a3b2c4d5e6f7",
  "strategy": "manual-commit",
  "branch": "main",
  "checkpoints_count": 3,
  "files_touched": ["file1.go", "file2.go"],
  "sessions": [
    {
      "metadata": "/a3/b2c4d5e6f7/0/metadata.json",
      "transcript": "/a3/b2c4d5e6f7/0/full.jsonl",
      "context": "/a3/b2c4d5e6f7/0/context.md",
      "prompt": "/a3/b2c4d5e6f7/0/prompt.txt"
    }
  ],
  "token_usage": {
    "input_tokens": 1500,
    "cache_creation_tokens": 200,
    "cache_read_tokens": 800,
    "output_tokens": 500,
    "api_call_count": 3
  }
}
```

### CommittedMetadata (Session)

```json
{
  "session_id": "2026-02-11-abc123...",
  "checkpoint_id": "a3b2c4d5e6f7",
  "strategy": "manual-commit",
  "created_at": "2026-02-11T10:30:00Z",
  "files_touched": ["file1.go"],
  "token_usage": {
    "input_tokens": 1500,
    "output_tokens": 500
  }
}
```

---

## Checkpoint ID 연결

### Bidirectional Linking

```
User Commit (main 브랜치)
↓
"Entire-Checkpoint: a3b2c4d5e6f7" (trailer)
↓ ↑
Checkpoint ID로 연결
↓ ↑
entire/checkpoints/v1 Commit
↓
Tree: a3/b2c4d5e6f7/
└── metadata.json
    └── full.jsonl
```

### 조회 방법

```bash
# 커밋에서 메타데이터 찾기
git log --format="%H %s" | grep "Entire-Checkpoint: a3b2c4d5e6f7"
→ 커밋 해시 찾기
→ entire/checkpoints/v1에서 a3/b2c4d5e6f7/ 디렉토리 조회

# 메타데이터에서 커밋 찾기
git log --all --grep="Entire-Checkpoint: a3b2c4d5e6f7"
```

---

## Phase State Machine

세션은 상태 머신을 통해 관리됩니다.

### Phases

```
IDLE → ACTIVE → IDLE → ACTIVE_COMMITTED → IDLE → ENDED
```

| Phase | 설명 |
|-------|------|
| **IDLE** | 대기 중 (프롬프트 대기) |
| **ACTIVE** | AI 작업 중 |
| **ACTIVE_COMMITTED** | 작업 중 커밋 발생 |
| **ENDED** | 세션 종료 |

### Events

| Event | Trigger | Phase 전환 |
|-------|---------|-----------|
| **TurnStart** | UserPromptSubmit | IDLE → ACTIVE |
| **TurnEnd** | Stop | ACTIVE → IDLE |
| **GitCommit** | PostCommit | ACTIVE → ACTIVE_COMMITTED |
| **SessionStop** | 명시적 종료 | ANY → ENDED |

---

## 다음 단계

핵심 개념을 이해했습니다! 다음 챕터에서는:

- **일반적인 워크플로우** - Enable, Work, Rewind, Resume 실습
- **명령어 레퍼런스** - 모든 명령어 상세 설명
- **Strategy 상세** - Manual-commit과 Auto-commit 깊이 파기

---

*다음 글에서는 Entire CLI의 일반적인 워크플로우를 실제 예제와 함께 살펴봅니다.*
