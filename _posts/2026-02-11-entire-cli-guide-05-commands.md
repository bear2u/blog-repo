---
layout: post
title: "Entire CLI 완벽 가이드 (05) - 명령어 레퍼런스"
date: 2026-02-11
permalink: /entire-cli-guide-05-commands/
author: Entire Team
categories: [AI 코딩, 개발 도구]
tags: [Entire, CLI, Commands, Reference]
original_url: "https://github.com/entireio/cli"
excerpt: "Entire CLI의 모든 명령어 완벽 레퍼런스"
---

## 명령어 개요

Entire CLI는 다음 명령어들을 제공합니다:

| 명령어 | 설명 | 사용 빈도 |
|-------|------|----------|
| `enable` | Entire 활성화 | 프로젝트당 1회 |
| `disable` | Entire 비활성화 | 필요시 |
| `status` | 현재 상태 확인 | 자주 |
| `rewind` | 체크포인트로 되돌리기 | 필요시 |
| `resume` | 세션 복원 | 브랜치 전환 시 |
| `explain` | 커밋/세션 설명 | 조회 시 |
| `clean` | 고아 데이터 정리 | 가끔 |
| `doctor` | 문제 진단 및 수정 | 문제 발생 시 |
| `reset` | 상태 초기화 | 문제 발생 시 |
| `version` | 버전 확인 | 필요시 |

---

## entire enable

Entire를 프로젝트에서 활성화합니다.

### 기본 사용법

```bash
entire enable
```

### 플래그

| 플래그 | 설명 | 기본값 |
|-------|------|-------|
| `--agent <name>` | AI 에이전트 선택 | `claude-code` |
| `--strategy <name>` | 전략 선택 | `manual-commit` |
| `--force`, `-f` | 강제 재설치 | false |
| `--local` | 로컬 설정에만 저장 | false |
| `--project` | 프로젝트 설정에 강제 저장 | false |
| `--skip-push-sessions` | 자동 푸시 비활성화 | false |
| `--telemetry=false` | 텔레메트리 비활성화 | true |

### 예제

```bash
# 기본 활성화 (manual-commit, claude-code)
entire enable

# Auto-commit 전략 사용
entire enable --strategy auto-commit

# Gemini CLI 사용
entire enable --agent gemini

# 로컬 설정으로만 저장
entire enable --local

# 텔레메트리 비활성화
entire enable --telemetry=false

# 강제 재설치
entire enable --force

# 자동 푸시 비활성화
entire enable --skip-push-sessions
```

### 생성되는 파일

```
.entire/
├── settings.json          # 프로젝트 설정 (--local 없을 때)
├── settings.local.json    # 로컬 설정 (--local 사용 시)
└── logs/                  # 로그 디렉토리

.git/hooks/
├── prepare-commit-msg
├── post-commit
└── pre-push

.claude/settings.json      # Claude Code 훅 (--agent claude-code)
.gemini/settings.json      # Gemini CLI 훅 (--agent gemini)
```

---

## entire disable

Entire를 비활성화하고 훅을 제거합니다.

### 기본 사용법

```bash
entire disable
```

### 동작

- Git 훅 제거 (prepare-commit-msg, post-commit, pre-push)
- AI 에이전트 훅 제거
- **데이터는 삭제하지 않음** (`.entire/`, `entire/checkpoints/v1`)

### 예제

```bash
# Entire 비활성화
entire disable

# 데이터 삭제 (수동)
rm -rf .entire/
git branch -D entire/checkpoints/v1
```

---

## entire status

현재 세션과 프로젝트 상태를 표시합니다.

### 기본 사용법

```bash
entire status
```

### 출력 예시

**활성 세션 없을 때:**
```
Strategy: manual-commit
Status: No active session
Branch: main
Checkpoints: 0
```

**활성 세션 있을 때:**
```
Strategy: manual-commit
Status: Active session
Session ID: 2026-02-11-abc123de-f456-7890-abcd-ef1234567890
Branch: main
Base commit: a1b2c3d4
Started: 2026-02-11 10:30:00

Checkpoints:
- 3 uncommitted checkpoints

Modified files:
- models/user.go (new)
- handlers/auth.go (modified)
- utils/jwt.go (new)

Shadow branch: entire/a1b2c3d-abc123
```

**동시 세션:**
```
Strategy: manual-commit
Status: 2 active sessions

Session 1: 2026-02-11-abc123...
- Checkpoints: 2
- Modified: file1.go, file2.go

Session 2: 2026-02-11-def456...
- Checkpoints: 1
- Modified: file3.go
```

---

## entire rewind

이전 체크포인트로 코드를 되돌립니다.

### 기본 사용법

```bash
entire rewind
```

### 인터랙티브 모드

```
Select a checkpoint to rewind to:

Uncommitted checkpoints:
[1] 2026-02-11 11:00:00 - Add password reset
    Session: 2026-02-11-def456...
    Modified: 3 files

[2] 2026-02-11 10:45:30 - Add input validation
    Session: 2026-02-11-abc123...
    Modified: 4 files

Committed checkpoints:
[3] 2026-02-11 10:30:00 - Add user authentication
    Commit: a3f2b1c4
    Checkpoint ID: a3b2c4d5e6f7
    Modified: 3 files

> Enter checkpoint number: 2
```

### 동작

1. **Uncommitted checkpoint** 선택 시:
   - Shadow 브랜치에서 파일 복원
   - 이후 체크포인트는 유지

2. **Committed checkpoint** 선택 시:
   - `entire/checkpoints/v1`에서 파일 복원
   - 코드 커밋은 유지 (메타데이터만 사용)

### 주의사항

- **Manual-commit**: 항상 가능, 비파괴적
- **Auto-commit**:
  - Feature 브랜치: `git reset --hard` 사용
  - Main 브랜치: 로그만 표시 (rewind 불가)

---

## entire resume

브랜치로 전환하고 최신 세션을 복원합니다.

### 기본 사용법

```bash
entire resume <branch>
```

### 동작

1. 브랜치로 checkout
2. `entire/checkpoints/v1`에서 최신 체크포인트 조회
3. 세션 메타데이터 복원
4. 계속할 명령어 출력

### 예제

```bash
# Feature 브랜치로 전환 및 세션 복원
entire resume feature/payment
```

**출력:**
```
✓ Checked out branch: feature/payment
✓ Restored session: 2026-02-11-abc123...
✓ Latest checkpoint: a3b2c4d5e6f7

Session details:
- Start time: 2026-02-11 10:00:00
- Checkpoints: 3
- Last prompt: "Add payment integration"
- Modified files: 5

To continue this session, run:
  claude --session 2026-02-11-abc123...
```

---

## entire explain

커밋이나 세션의 상세 정보를 표시합니다.

### 기본 사용법

```bash
entire explain <commit-or-session-id>
```

### 예제

```bash
# 최근 커밋 설명
entire explain HEAD

# 특정 커밋 설명
entire explain a3f2b1c4

# 세션 ID로 설명
entire explain 2026-02-11-abc123
```

### 출력 예시

**커밋 설명:**
```
Commit: a3f2b1c4
Message: Add user authentication system
Author: John Doe <john@example.com>
Date: 2026-02-11 10:30:00
Checkpoint ID: a3b2c4d5e6f7

Session Details:
- Session ID: 2026-02-11-abc123...
- Strategy: manual-commit
- Start time: 2026-02-11 10:00:00
- Duration: 30 minutes
- Checkpoints: 3

User Prompts:
1. "Create a user authentication system"
2. "Add input validation"
3. "Add JWT token generation"

Modified Files:
- models/user.go (new, 150 lines)
- handlers/auth.go (new, 200 lines)
- utils/jwt.go (new, 80 lines)
- handlers/validation.go (new, 50 lines)

Token Usage:
- Input tokens: 1500
- Cache creation: 200
- Cache read: 800
- Output tokens: 500
- API calls: 3

Transcript: .entire/metadata/.../full.jsonl
```

---

## entire clean

고아(orphaned) Entire 데이터를 정리합니다.

### 기본 사용법

```bash
entire clean
```

### 동작

- 대응하는 세션 상태 파일이 없는 shadow 브랜치 삭제
- 더 이상 참조되지 않는 세션 상태 파일 삭제
- 오래된 로그 파일 정리

### 예제

```bash
entire clean
```

**출력:**
```
✓ Cleaned 2 orphaned shadow branches
✓ Cleaned 1 orphaned session state
✓ Cleaned 5 old log files
```

---

## entire doctor

Entire 상태를 진단하고 문제를 수정합니다.

### 기본 사용법

```bash
entire doctor
```

### 동작

1. **진단**:
   - Git 훅 설치 상태
   - AI 에이전트 훅 상태
   - 세션 상태 일관성
   - Shadow 브랜치 상태

2. **수정**:
   - 누락된 훅 재설치
   - 불일치 해결
   - 충돌 정리

### 예제

```bash
entire doctor
```

**출력:**
```
Diagnosing Entire installation...

✓ Git repository: OK
✓ Git hooks: OK
✗ Claude Code hooks: Missing
✓ Session state: OK
✗ Shadow branch: Orphaned

Fixing issues...
✓ Reinstalled Claude Code hooks
✓ Cleaned orphaned shadow branch

All issues resolved.
```

---

## entire reset

현재 HEAD 커밋의 shadow 브랜치와 세션 상태를 삭제합니다.

### 기본 사용법

```bash
entire reset
```

### 플래그

| 플래그 | 설명 |
|-------|------|
| `--force` | 확인 없이 삭제 |

### 동작

- 현재 커밋의 shadow 브랜치 삭제
- 대응하는 세션 상태 파일 삭제
- **entire/checkpoints/v1은 삭제하지 않음**

### 예제

```bash
# 확인 후 삭제
entire reset

# 강제 삭제
entire reset --force
```

**출력:**
```
This will delete:
- Shadow branch: entire/a1b2c3d-abc123
- Session state: .git/entire-sessions/2026-02-11-abc123...

Continue? (y/n): y

✓ Shadow branch deleted
✓ Session state deleted
```

---

## entire version

Entire CLI 버전을 표시합니다.

### 기본 사용법

```bash
entire version
```

### 출력 예시

```
entire version v0.1.0
commit: a3f2b1c4
```

---

## 훅 명령어 (내부 사용)

이 명령어들은 Git 훅과 AI 에이전트 훅에서 내부적으로 사용됩니다.

### entire hooks claude-code session-start

Claude Code SessionStart 훅 핸들러.

```bash
entire hooks claude-code session-start
```

### entire hooks claude-code user-prompt-submit

UserPromptSubmit 훅 핸들러.

```bash
entire hooks claude-code user-prompt-submit
```

### entire hooks claude-code stop

Stop 훅 핸들러.

```bash
entire hooks claude-code stop
```

### entire hooks claude-code pre-task

PreToolUse[Task] 훅 핸들러.

```bash
entire hooks claude-code pre-task
```

### entire hooks claude-code post-task

PostToolUse[Task] 훅 핸들러.

```bash
entire hooks claude-code post-task
```

### entire hooks claude-code post-todo

PostToolUse[TodoWrite] 훅 핸들러.

```bash
entire hooks claude-code post-todo
```

---

## 환경 변수

| 변수 | 설명 | 기본값 |
|-----|------|-------|
| `ENTIRE_LOG_LEVEL` | 로그 레벨 | `info` |
| `ACCESSIBLE` | 접근성 모드 | `false` |

### 예제

```bash
# Debug 로그 활성화
ENTIRE_LOG_LEVEL=debug entire status

# 접근성 모드
ACCESSIBLE=1 entire enable
```

---

## 도움말

모든 명령어는 `--help` 플래그를 지원합니다.

```bash
# 전체 도움말
entire --help

# 특정 명령어 도움말
entire enable --help
entire rewind --help
entire resume --help
```

---

## 다음 단계

명령어를 이해했습니다! 다음 챕터에서는:

- **Strategy 개요** - Manual-commit과 Auto-commit 비교
- **Manual-Commit Strategy** - Shadow 브랜치 메커니즘
- **Auto-Commit Strategy** - 자동 커밋 동작

---

*다음 글에서는 Entire의 Strategy 시스템을 자세히 살펴봅니다.*
