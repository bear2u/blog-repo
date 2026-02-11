---
layout: post
title: "Entire CLI 완벽 가이드 (04) - 일반적인 워크플로우"
date: 2026-02-11
permalink: /entire-cli-guide-04-workflow/
author: Entire Team
categories: [AI 코딩, 개발 도구]
tags: [Entire, Workflow, Enable, Rewind, Resume]
original_url: "https://github.com/entireio/cli"
excerpt: "Entire CLI의 일반적인 워크플로우: Enable, Work, Rewind, Resume 완벽 가이드"
---

## 워크플로우 개요

Entire CLI의 일반적인 워크플로우는 다음 4단계로 구성됩니다:

```
1. Enable  → Entire 활성화
2. Work    → AI와 작업
3. Rewind  → 필요시 되돌리기
4. Resume  → 이전 세션 복원
```

이 챕터에서는 각 단계를 실제 예제와 함께 살펴봅니다.

---

## 1. Enable - Entire 활성화

### 기본 활성화

```bash
cd your-project
entire enable
```

**출력:**
```
✓ Git hooks installed
✓ Claude Code hooks configured
✓ Entire enabled with strategy: manual-commit
✓ Settings saved to .entire/settings.json
```

### 생성된 파일들

```bash
tree -a .entire/
```

```
.entire/
├── settings.json          # 프로젝트 설정
├── settings.local.json    # 개인 설정 (gitignored)
└── logs/                  # 로그 디렉토리

.git/hooks/
├── prepare-commit-msg     # Checkpoint ID 추가
├── post-commit            # Condensation 실행
└── pre-push               # 메타데이터 푸시

.claude/settings.json      # Claude Code 훅 설정 업데이트
```

### 다른 옵션으로 활성화

```bash
# Auto-commit 전략 사용
entire enable --strategy auto-commit

# Gemini CLI 사용
entire enable --agent gemini

# 로컬 설정만 저장
entire enable --local

# 자동 푸시 비활성화
entire enable --skip-push-sessions

# 강제 재설치
entire enable --force
```

---

## 2. Work - AI와 작업

### 시나리오: 새 기능 추가

#### Step 1: Claude Code 시작

```bash
claude
```

#### Step 2: 첫 번째 프롬프트

```
Create a user authentication system with the following:
1. User model with email and password
2. Login endpoint
3. JWT token generation
```

Claude가 파일들을 생성합니다:

```
Created: models/user.go
Created: handlers/auth.go
Created: utils/jwt.go
```

#### Step 3: 상태 확인

```bash
entire status
```

**출력:**
```
Strategy: manual-commit
Status: Active session
Session ID: 2026-02-11-abc123de-f456-7890-abcd-ef1234567890
Branch: main
Base commit: a1b2c3d4

Checkpoints:
- 1 uncommitted checkpoint

Modified files:
- models/user.go (new)
- handlers/auth.go (new)
- utils/jwt.go (new)

Shadow branch: entire/a1b2c3d-abc123
```

#### Step 4: 추가 작업

```
Add input validation for the login endpoint
```

Claude가 코드를 수정합니다.

```bash
entire status
```

**출력:**
```
Checkpoints:
- 2 uncommitted checkpoints

Modified files:
- models/user.go (new)
- handlers/auth.go (new, modified)
- utils/jwt.go (new)
- handlers/validation.go (new)
```

#### Step 5: 커밋

```bash
git add .
git commit -m "Add user authentication system"
```

**Entire가 자동으로:**

1. **Checkpoint ID 추가** (prepare-commit-msg hook)
   ```
   Add user authentication system

   Entire-Checkpoint: a3b2c4d5e6f7
   ```

2. **Condensation 실행** (post-commit hook)
   - Shadow 브랜치 읽기
   - `entire/checkpoints/v1`에 메타데이터 저장
   - Shadow 브랜치 삭제

3. **상태 업데이트**
   ```bash
   entire status
   ```

   **출력:**
   ```
   Status: No active session
   Checkpoints: 0
   Last committed session: 2026-02-11-abc123...
   ```

---

## 3. Rewind - 되돌리기

### 시나리오: 이전 체크포인트로 되돌아가기

#### Step 1: 계속 작업

```
Add password reset functionality
```

Claude가 작업을 완료했지만, 접근 방식이 마음에 들지 않습니다.

#### Step 2: Rewind 실행

```bash
entire rewind
```

**인터랙티브 프롬프트:**
```
Select a checkpoint to rewind to:

[1] 2026-02-11 10:45:30 - Add password reset functionality
    Session: 2026-02-11-def456...
    Modified: 3 files

[2] 2026-02-11 10:30:15 - Add input validation
    Session: 2026-02-11-abc123...
    Modified: 4 files

[3] 2026-02-11 10:15:00 - Create user authentication system
    Session: 2026-02-11-abc123...
    Modified: 3 files

> Enter checkpoint number: 2
```

#### Step 3: 확인

```bash
# 파일 상태 확인
git status

# 변경사항 확인
git diff
```

**결과:**
- 선택한 체크포인트의 파일 상태로 복원
- 이후 체크포인트는 그대로 유지 (삭제 안 됨)

#### Step 4: 다시 시도

```
Add password reset with email verification
```

더 나은 구현으로 계속 작업할 수 있습니다.

### Committed Checkpoint Rewind

커밋된 체크포인트로도 되돌릴 수 있습니다:

```bash
entire rewind
```

```
Select a checkpoint to rewind to:

Uncommitted checkpoints:
[1] 2026-02-11 11:00:00 - Current work

Committed checkpoints:
[2] 2026-02-11 10:50:00 - Add user authentication system
    Commit: a3f2b1c4
    Checkpoint ID: a3b2c4d5e6f7

[3] 2026-02-11 09:30:00 - Initial setup
    Commit: 9e8d7c6b
    Checkpoint ID: 9e8d7c6b5a4f

> Enter checkpoint number: 2
```

---

## 4. Resume - 이전 세션 복원

### 시나리오: 다른 브랜치로 전환 후 복원

#### Step 1: 다른 브랜치로 전환

```bash
# Feature 브랜치에서 작업 중
git checkout feature/payment

# 긴급 버그 수정 필요
git checkout main
git checkout -b hotfix/auth-bug
```

#### Step 2: 버그 수정 후 돌아가기

```bash
# 버그 수정 완료
git commit -m "Fix auth bug"
git checkout main
git merge hotfix/auth-bug

# 원래 작업으로 돌아가기
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

To continue this session, run:
  claude --session 2026-02-11-abc123...
```

#### Step 3: 세션 계속하기

```bash
claude --session 2026-02-11-abc123...
```

이제 이전 컨텍스트와 함께 작업을 계속할 수 있습니다.

---

## 고급 워크플로우

### 동시 세션 처리

#### 시나리오: 같은 커밋에서 두 번째 세션 시작

```bash
# 첫 번째 세션 진행 중
claude "Add feature A"
# 작업 중... (커밋 안 함)

# 두 번째 터미널에서
claude "Add feature B"
```

**경고 메시지:**
```
⚠ Warning: Concurrent session detected

Another session is active on commit a1b2c3d4:
- Session ID: 2026-02-11-abc123...
- Prompt: "Add feature A"
- Started: 2026-02-11 10:00:00

Both sessions can proceed, but consider:
1. Committing the first session before continuing
2. Using different worktrees for parallel work

Continue this session? (y/n)
```

#### 두 세션 모두 진행 시

```bash
# 두 세션 모두 체크포인트 생성
entire status
```

**출력:**
```
Active sessions: 2

Session 1: 2026-02-11-abc123...
- Checkpoints: 2
- Modified: file1.go, file2.go

Session 2: 2026-02-11-def456...
- Checkpoints: 1
- Modified: file3.go
```

#### 커밋 시

```bash
git commit -m "Add features A and B"
```

**Entire가 자동으로:**
- 두 세션을 하나의 커밋으로 condensation
- `entire/checkpoints/v1`에 numbered subdirectories로 저장
  ```
  a3/b2c4d5e6f7/
  ├── 0/  # Session 1
  └── 1/  # Session 2
  ```

### Worktree 사용

#### 병렬 작업을 위한 Worktree

```bash
# 메인 워크트리
cd ~/project
entire enable

# 두 번째 워크트리 생성
git worktree add ../project-feature feature/payment

# 각 워크트리에서 독립적인 세션
cd ~/project
claude "Work on main branch"

cd ~/project-feature
claude "Work on feature branch"
```

**각 워크트리는:**
- 독립적인 세션 추적
- 별도의 shadow 브랜치 사용
  ```
  entire/a1b2c3d-abc123  # main worktree
  entire/d4e5f6g-def456  # feature worktree
  ```

---

## 일일 워크플로우 예시

### 아침

```bash
# 1. 프로젝트로 이동
cd ~/projects/myapp

# 2. 최신 변경사항 가져오기
git pull

# 3. Feature 브랜치 생성
git checkout -b feature/user-profile

# 4. Entire 확인
entire status

# 5. 작업 시작
claude "Implement user profile page"
```

### 점심 전

```bash
# 진행 상황 확인
entire status

# 커밋
git add .
git commit -m "WIP: User profile page"

# 푸시
git push -u origin feature/user-profile
```

### 오후

```bash
# 계속 작업
claude "Add avatar upload functionality"

# 테스트
npm test

# 문제 발생 시 rewind
entire rewind
# → 이전 체크포인트로 되돌리기

# 다시 시도
claude "Add avatar upload with error handling"
```

### 퇴근 전

```bash
# 최종 커밋
git add .
git commit -m "Complete user profile with avatar"

# PR 생성
gh pr create --title "Add user profile" --body "..."

# 상태 확인
entire status
```

---

## 모범 사례

### 1. 자주 커밋하기

```bash
# Good: 논리적 단위로 커밋
claude "Add user model"
git commit -m "Add user model"

claude "Add user endpoints"
git commit -m "Add user endpoints"

# Bad: 너무 큰 작업을 한 커밋에
claude "Build entire user system"
# ... 많은 작업 ...
git commit -m "Add user system"
```

### 2. 의미있는 프롬프트 작성

```bash
# Good: 구체적인 프롬프트
claude "Add input validation for email field in user registration"

# Bad: 모호한 프롬프트
claude "Fix it"
```

프롬프트는 체크포인트 설명으로 사용되므로 명확하게 작성하세요.

### 3. Rewind 전 상태 저장

```bash
# Rewind 전에 현재 상태 확인
git status
git diff

# 필요시 stash
git stash

# Rewind
entire rewind

# 다시 적용 (필요시)
git stash pop
```

### 4. Resume으로 컨텍스트 유지

```bash
# 브랜치 전환 전에 커밋
git commit -m "WIP: Current progress"

# 다른 브랜치에서 작업
git checkout other-branch

# 돌아올 때 resume 사용
entire resume previous-branch
claude --session <session-id>
```

---

## 문제 해결

### "No checkpoints found"

```bash
# AI와 작업했는지 확인
entire status

# 세션이 있는지 확인
ls .git/entire-sessions/

# 없다면 새 세션 시작
claude "Start working"
```

### "Concurrent session warning"

```bash
# 옵션 1: 이전 세션 커밋
git commit -m "Complete previous work"

# 옵션 2: 계속 진행
# 경고를 읽고 'y' 입력

# 옵션 3: Worktree 사용
git worktree add ../project-parallel parallel-branch
```

### "Shadow branch conflict"

```bash
# 상태 확인
entire doctor

# 수정
entire reset --force

# 재시작
entire enable --force
```

---

## 다음 단계

워크플로우를 이해했습니다! 다음 챕터에서는:

- **명령어 레퍼런스** - 모든 명령어 상세 설명
- **Strategy 상세** - Manual-commit 깊이 파기
- **Session 관리** - 고급 세션 관리 기법

---

*다음 글에서는 Entire CLI의 모든 명령어를 상세히 살펴봅니다.*
