---
layout: post
title: "Beads 완벽 가이드 (2) - 아키텍처"
date: 2025-02-04
permalink: /beads-guide-02-architecture/
author: Steve Yegge
categories: [AI]
tags: [Beads, Architecture, SQLite, JSONL, Git]
original_url: "https://github.com/steveyegge/beads"
excerpt: "Beads의 3계층 데이터 모델과 분산 아키텍처를 분석합니다."
---

## 3계층 데이터 모델

Beads의 핵심 설계는 **분산 Git 기반 이슈 트래커가 중앙 집중식 데이터베이스처럼 느껴지게** 하는 것입니다. 이 "마법"은 세 개의 동기화된 계층에서 나옵니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Layer                                 │
│                                                                  │
│  bd create, list, update, close, ready, show, dep, sync, ...    │
│  - Cobra 명령어 (cmd/bd/)                                       │
│  - 모든 명령어 --json 지원                                       │
│  - 데몬 RPC 우선, 직접 DB 접근 폴백                              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SQLite Database                              │
│                     (.beads/beads.db)                            │
│                                                                  │
│  - 로컬 작업 복사본 (gitignored)                                 │
│  - 빠른 쿼리, 인덱스, 외래 키                                    │
│  - Issues, dependencies, labels, comments, events                │
│  - 각 머신이 자체 복사본 보유                                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                         auto-sync
                        (5초 디바운스)
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       JSONL File                                 │
│                   (.beads/issues.jsonl)                          │
│                                                                  │
│  - Git 추적되는 진실의 원천                                      │
│  - 엔티티당 한 JSON 라인 (issue, dep, label, comment)            │
│  - 머지 친화적: 추가는 거의 충돌하지 않음                        │
│  - git push/pull로 머신 간 공유                                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                          git push/pull
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Remote Repository                            │
│                    (GitHub, GitLab, etc.)                        │
│                                                                  │
│  - JSONL을 일반 레포 히스토리의 일부로 저장                      │
│  - 모든 협업자가 동일한 이슈 데이터베이스 공유                   │
│  - 보호된 브랜치 지원 (별도 sync 브랜치)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 왜 이 설계인가?

### SQLite for Speed

```go
// 로컬 쿼리는 밀리초 단위로 완료
// 복잡한 의존성 그래프, 전문 검색, 조인이 빠름

func (s *SQLiteStore) GetReadyIssues() ([]*Issue, error) {
    query := `
        SELECT i.* FROM issues i
        WHERE i.status = 'open'
        AND NOT EXISTS (
            SELECT 1 FROM dependencies d
            JOIN issues blocker ON d.from_id = blocker.id
            WHERE d.to_id = i.id
            AND d.type = 'blocks'
            AND blocker.status != 'closed'
        )
    `
    return s.queryIssues(query)
}
```

### JSONL for Git

```json
{"id":"bd-a1b2","title":"Add OAuth","status":"open","priority":0}
{"id":"bd-f14c","title":"Add Stripe","status":"in_progress","priority":1}
{"id":"bd-x9z3","title":"Fix bug","status":"closed","priority":2}
```

- 한 줄에 하나의 엔티티
- Git diff가 읽기 쉬움
- 머지가 보통 자동 성공

### Git for Distribution

- 특별한 동기화 서버 불필요
- 이슈가 코드와 함께 이동
- 오프라인 작업이 그냥 동작함

---

## Write Path

이슈를 생성하거나 수정할 때의 흐름:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Command   │───▶│  SQLite Write   │───▶│  Mark Dirty     │
│   (bd create)   │    │  (immediate)    │    │  (trigger sync) │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                                              5초 디바운스
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Git Commit    │◀───│  JSONL Export   │◀───│  FlushManager   │
│   (git hooks)   │    │  (incremental)  │    │  (background)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

1. **명령 실행**: `bd create "New feature"`가 SQLite에 즉시 기록
2. **더티 마킹**: 작업이 데이터베이스를 익스포트 필요로 표시
3. **디바운스 윈도우**: 배치 작업을 위해 5초 대기 (설정 가능)
4. **JSONL 익스포트**: 변경된 엔티티만 추가/업데이트
5. **Git 커밋**: git hooks가 설치되어 있으면 자동 커밋

---

## Read Path

`git pull` 후 이슈 쿼리 시:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   git pull      │───▶│  Auto-Import    │───▶│  SQLite Update  │
│   (new JSONL)   │    │  (on next cmd)  │    │  (merge logic)  │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                                                       ▼
                                               ┌─────────────────┐
                                               │  CLI Query      │
                                               │  (bd ready)     │
                                               └─────────────────┘
```

1. **Git pull**: 원격에서 업데이트된 JSONL 가져옴
2. **자동 임포트 감지**: 첫 bd 명령이 JSONL이 DB보다 새로운지 확인
3. **SQLite로 임포트**: JSONL 파싱, 콘텐츠 해시로 로컬 상태와 머지
4. **쿼리**: 명령어가 빠른 로컬 SQLite에서 읽음

---

## 해시 기반 충돌 방지

### 문제: 순차 ID

```bash
Branch A: bd create "Add OAuth"   → bd-10
Branch B: bd create "Add Stripe"  → bd-10 (충돌!)
```

### 해결: 해시 기반 ID

```bash
Branch A: bd create "Add OAuth"   → bd-a1b2
Branch B: bd create "Add Stripe"  → bd-f14c (충돌 없음)
```

### 동작 원리

```
┌─────────────────────────────────────────────────────────────────┐
│                        Import Logic                              │
│                                                                  │
│  For each issue in JSONL:                                       │
│    1. 콘텐츠 해시 계산                                          │
│    2. ID로 기존 이슈 조회                                       │
│    3. 해시 비교:                                                 │
│       - 같은 해시 → 스킵 (이미 임포트됨)                        │
│       - 다른 해시 → 업데이트 (새 버전)                          │
│       - 매치 없음 → 생성 (새 이슈)                              │
└─────────────────────────────────────────────────────────────────┘
```

- **점진적 스케일링**: ID가 4자로 시작, 데이터베이스 성장에 따라 5-6자로 확장
- **콘텐츠 해싱**: 각 이슈가 변경 감지를 위한 콘텐츠 해시 보유

---

## 데몬 아키텍처

각 워크스페이스가 자동 동기화를 위한 자체 백그라운드 데몬 실행:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Per-Workspace Daemon                         │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ RPC Server  │    │  Auto-Sync  │    │  Background │         │
│  │ (bd.sock)   │    │  Manager    │    │  Tasks      │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                  │                  │
│         └──────────────────┴──────────────────┘                  │
│                            │                                     │
│                            ▼                                     │
│                   ┌─────────────┐                                │
│                   │   SQLite    │                                │
│                   │   Database  │                                │
│                   └─────────────┘                                │
└─────────────────────────────────────────────────────────────────┘

     CLI commands ───RPC───▶ Daemon ───SQL───▶ Database
                              or
     CLI commands ───SQL───▶ Database (데몬 불가 시)
```

### 왜 데몬인가?

- 익스포트 전에 여러 작업을 배치 처리
- 데이터베이스 연결을 열어둠 (빠른 쿼리)
- 자동 동기화 타이밍 조정
- 워크스페이스당 하나의 데몬 (LSP 유사 모델)

### 통신

- Unix 도메인 소켓: `.beads/bd.sock` (Windows: named pipes)
- 프로토콜: `internal/rpc/protocol.go`
- CLI가 데몬 우선 시도, 직접 DB 접근으로 폴백

---

## 디렉토리 구조

```
.beads/
├── beads.db          # SQLite 데이터베이스 (gitignored)
├── issues.jsonl      # JSONL 진실의 원천 (git-tracked)
├── bd.sock           # 데몬 소켓 (gitignored)
├── daemon.log        # 데몬 로그 (gitignored)
├── config.yaml       # 프로젝트 설정 (optional)
└── export_hashes.db  # 익스포트 추적 (gitignored)
```

---

## 핵심 코드 경로

| 영역 | 파일 |
|------|------|
| CLI 진입점 | `cmd/bd/main.go` |
| 스토리지 인터페이스 | `internal/storage/storage.go` |
| SQLite 구현 | `internal/storage/sqlite/` |
| RPC 프로토콜 | `internal/rpc/protocol.go`, `server_*.go` |
| 익스포트 로직 | `cmd/bd/export.go`, `autoflush.go` |
| 임포트 로직 | `cmd/bd/import.go`, `internal/importer/` |
| 자동 동기화 | `internal/autoimport/`, `internal/flush/` |

---

*다음 글에서는 CLI 명령어를 상세히 살펴봅니다.*
