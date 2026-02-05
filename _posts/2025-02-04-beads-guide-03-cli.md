---
layout: post
title: "Beads 완벽 가이드 (3) - CLI 명령어"
date: 2025-02-04
permalink: /beads-guide-03-cli/
author: Steve Yegge
categories: [AI]
tags: [Beads, CLI, Commands, Cobra, Go]
original_url: "https://github.com/steveyegge/beads"
excerpt: "Beads CLI의 주요 명령어와 사용법을 상세히 알아봅니다."
---

## CLI 구조

Beads CLI는 **Cobra** 프레임워크로 구축되어 있으며, 모든 명령어가 `--json` 출력을 지원합니다.

```
cmd/bd/
├── main.go           # 진입점
├── create.go         # 이슈 생성
├── list.go           # 이슈 목록
├── show.go           # 이슈 상세
├── update.go         # 이슈 수정
├── close.go          # 이슈 종료
├── ready.go          # Ready 작업 목록
├── dep.go            # 의존성 관리
├── sync.go           # Git 동기화
├── init.go           # 초기화
└── daemon.go         # 데몬 관리
```

---

## 초기화

### bd init

```bash
# 기본 초기화
bd init

# Stealth 모드 (로컬만, Git 커밋 안함)
bd init --stealth

# Contributor 모드 (별도 레포로 라우팅)
bd init --contributor

# 커스텀 접두사
bd init --prefix myproj

# Sync 브랜치 지정
bd init --branch beads-metadata
```

```go
// cmd/bd/init.go
func runInit(cmd *cobra.Command, args []string) error {
    opts := &InitOptions{
        Stealth:     stealth,
        Contributor: contributor,
        Prefix:      prefix,
        Branch:      branch,
    }

    // .beads 디렉토리 생성
    if err := os.MkdirAll(".beads", 0755); err != nil {
        return err
    }

    // SQLite 데이터베이스 초기화
    store, err := sqlite.NewStore(".beads/beads.db")
    if err != nil {
        return err
    }

    // 설정 파일 생성
    return writeConfig(opts)
}
```

---

## 이슈 생성

### bd create

```bash
# 기본 생성
bd create "Add user authentication"

# 우선순위 지정 (P0=critical, P4=backlog)
bd create "Fix login bug" -p 0

# 타입 지정
bd create "Refactor database" --type task

# 설명 포함
bd create "Add OAuth" -d "Implement OAuth 2.0 with Google and GitHub"

# 부모 이슈 지정 (하위 작업)
bd create "Add Google OAuth" --parent bd-a1b2

# JSON 출력
bd create "New feature" --json
```

**출력 예시:**
```json
{
  "id": "bd-x7y9",
  "title": "Add user authentication",
  "status": "open",
  "priority": 2,
  "created_at": "2025-02-04T10:30:00Z"
}
```

---

## 이슈 목록

### bd list

```bash
# 모든 열린 이슈
bd list

# 상태별 필터
bd list --status open
bd list --status in_progress
bd list --status closed

# 우선순위별 필터
bd list -p 0          # P0만
bd list -p 0,1        # P0, P1

# 라벨별 필터
bd list --label urgent

# 정렬
bd list --sort priority
bd list --sort created_at

# JSON 출력
bd list --json

# 간략 출력
bd list --brief
```

**출력 예시:**
```
ID        PRIORITY  STATUS       TITLE
bd-a1b2   P0        open         Fix critical bug
bd-f14c   P1        in_progress  Add OAuth
bd-x9z3   P2        open         Update docs
```

---

## Ready 작업

### bd ready

`bd ready`는 **열린 차단이 없는 작업**만 표시합니다. AI 에이전트의 핵심 명령어입니다.

```bash
# Ready 작업 목록
bd ready

# 특정 우선순위만
bd ready -p 0,1

# JSON 출력
bd ready --json

# 간략 출력
bd ready --brief
```

```go
// cmd/bd/ready.go
func runReady(cmd *cobra.Command, args []string) error {
    store := getStore()

    // 차단되지 않은 이슈만 조회
    issues, err := store.GetReadyIssues()
    if err != nil {
        return err
    }

    // 우선순위순 정렬
    sort.Slice(issues, func(i, j int) bool {
        return issues[i].Priority < issues[j].Priority
    })

    return outputIssues(issues, jsonOutput)
}
```

---

## 이슈 상세

### bd show

```bash
# 이슈 상세 보기
bd show bd-a1b2

# 감사 로그 포함
bd show bd-a1b2 --audit

# 의존성 그래프 포함
bd show bd-a1b2 --deps

# JSON 출력
bd show bd-a1b2 --json
```

**출력 예시:**
```
Issue: bd-a1b2
Title: Add user authentication
Status: open
Priority: P0 (critical)
Type: feature
Created: 2025-02-04 10:30:00
Updated: 2025-02-04 14:22:00

Description:
  Implement user authentication with OAuth 2.0

Dependencies:
  Blocks: bd-f14c (Add OAuth)
  Related: bd-x9z3 (Update docs)

Labels: auth, security, p0

Audit Trail:
  2025-02-04 10:30:00  Created by agent-1
  2025-02-04 12:00:00  Priority changed: P2 → P0
  2025-02-04 14:22:00  Added dependency: blocks bd-f14c
```

---

## 이슈 수정

### bd update

```bash
# 제목 변경
bd update bd-a1b2 --title "New title"

# 상태 변경
bd update bd-a1b2 --status in_progress

# 우선순위 변경
bd update bd-a1b2 -p 0

# 설명 추가
bd update bd-a1b2 -d "Updated description"

# 담당자 지정
bd update bd-a1b2 --assignee agent-1

# 라벨 추가
bd update bd-a1b2 --add-label urgent
```

---

## 이슈 종료

### bd close

```bash
# 이슈 종료
bd close bd-a1b2

# 이유 포함
bd close bd-a1b2 --reason "Completed implementation"

# 여러 이슈 종료
bd close bd-a1b2 bd-f14c bd-x9z3
```

---

## 의존성 관리

### bd dep

```bash
# 의존성 추가
bd dep add bd-child bd-parent              # parent-child
bd dep add bd-task bd-blocker --blocks     # blocks
bd dep add bd-a bd-b --related             # related

# 의존성 제거
bd dep rm bd-child bd-parent

# 의존성 목록
bd dep list bd-a1b2

# 의존성 그래프
bd dep graph
bd dep graph --format dot > deps.dot
```

### 의존성 타입

| 타입 | 설명 | `bd ready` 영향 |
|------|------|-----------------|
| `blocks` | X가 완료되어야 Y 시작 가능 | Yes |
| `parent-child` | 계층 (에픽/서브태스크) | Yes |
| `related` | 소프트 링크 (참조용) | No |
| `discovered-from` | 작업 중 발견됨 | No |

---

## 동기화

### bd sync

```bash
# Git과 동기화 (export + commit)
bd sync

# 강제 익스포트
bd sync --force

# 자동 푸시 포함
bd sync --push
```

### bd import / bd export

```bash
# 수동 임포트
bd import -i .beads/issues.jsonl

# 수동 익스포트
bd export -o .beads/issues.jsonl

# 강제 덮어쓰기
bd export --force
```

---

## 데몬 관리

### bd daemons

```bash
# 실행 중인 데몬 목록
bd daemons list

# 데몬 시작
bd daemons start

# 데몬 중지
bd daemons stop

# 데몬 재시작
bd daemons restart

# 데몬 상태
bd daemons status

# JSON 출력
bd daemons list --json
```

---

## 유틸리티 명령어

### bd info

```bash
# 현재 상태 정보
bd info

# 출력 예시:
# Workspace: /Users/alice/projects/webapp
# Database: .beads/beads.db (2.3 MB)
# Issues: 47 open, 123 closed
# Daemon: running (pid 12345)
# Last sync: 2 minutes ago
```

### bd doctor

```bash
# 상태 진단
bd doctor

# 수정 시도
bd doctor --fix
```

### bd cleanup

```bash
# 오래된 데이터 정리
bd cleanup

# 드라이런
bd cleanup --dry-run
```

---

## 전역 플래그

| 플래그 | 설명 |
|--------|------|
| `--json` | JSON 출력 |
| `--no-daemon` | 데몬 우회, 직접 DB 접근 |
| `--quiet` | 출력 최소화 |
| `--verbose` | 상세 출력 |
| `--help` | 도움말 |

---

*다음 글에서는 데이터 모델을 상세히 살펴봅니다.*
