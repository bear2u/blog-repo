---
layout: post
title: "Entire CLI 완벽 가이드 - 08. Auto-Commit Strategy"
date: 2026-02-11
categories: [AI, Development Tools]
tags: [Entire, CLI, AI Coding, Auto Commit, Git]
description: "Entire CLI Auto-Commit 전략의 자동 커밋 메커니즘과 빠른 반복 개발 워크플로우"
---

## 08. Auto-Commit Strategy

Auto-Commit 전략은 AI의 모든 변경사항을 현재 브랜치에 자동으로 커밋하는 Entire CLI의 빠른 개발 모드입니다. 이 장에서는 자동 커밋 메커니즘과 효율적인 사용 방법을 다룹니다.

### Auto-Commit 개념

```
작동 원리:
┌────────────────────────────────────────────┐
│ 사용자 프롬프트                            │
│      ↓                                     │
│ AI 응답 및 코드 변경                       │
│      ↓                                     │
│ 자동 커밋 (즉시)                           │
│      ↓                                     │
│ Main 브랜치 업데이트                       │
└────────────────────────────────────────────┘

특징:
1. Shadow branch 없음
2. 즉각적인 변경 반영
3. 자동 checkpoint 생성
4. 빠른 되돌리기 가능
```

### 브랜치 구조 비교

```
Manual-Commit:
main:     ───o───o───o───o (사용자 작업)
              │
shadow:       └──x──x──x (AI 작업, 격리)

Auto-Commit:
main:     ───o───o───x───x───x (모든 작업)
         (사용자) (AI) (AI) (AI)
```

### 세션 시작

#### 기본 시작

```bash
# Auto-commit 세션 시작
entire session start --strategy auto-commit \
  --message "빠른 기능 개발"

# 출력:
# ✓ Session created: abc123def456
# ✓ Strategy: auto-commit
# ✓ Working on branch: main
# ✓ Base commit: 9a7f3c2
#
# All AI changes will be automatically committed to main.
# You can undo changes with:
#   git reset HEAD~N
#   entire checkpoint restore <id>
```

#### 설정 옵션

```bash
# 1. 커밋 메시지 자동 생성 방식 설정
entire session start --strategy auto-commit \
  --commit-style descriptive \
  --message "문서 업데이트"

# 스타일 옵션:
# - concise: 간결한 메시지 (기본)
# - descriptive: 상세한 메시지
# - conventional: Conventional Commits 형식

# 2. 자동 checkpoint 주기 설정
entire session start --strategy auto-commit \
  --auto-checkpoint every-5-commits \
  --message "장기 작업"

# Checkpoint 주기:
# - never: 자동 checkpoint 생성 안 함
# - every-commit: 매 커밋마다
# - every-N-commits: N개 커밋마다
# - every-N-minutes: N분마다
```

### 자동 커밋 동작

#### 기본 동작

```bash
# AI와 상호작용
$ AI: 로그인 페이지를 만들어줘

# AI 응답 및 자동 처리:
# Creating login page...
#
# Created:
#   - src/pages/Login.tsx
#   - src/components/LoginForm.tsx
#   - src/styles/login.css
#
# ✓ Auto-committed: 3d8k5m2
#   Message: "Create login page with form and styling"
#   Files: 3 added
#   Checkpoint: 4a7b3c (temporary)
```

#### 내부 동작 상세

```bash
# Entire가 내부적으로 수행하는 작업:

# 1. AI 응답 완료 감지
on_ai_response_complete() {

  # 2. 작업 디렉토리 변경 감지
  CHANGED_FILES=$(git status --porcelain)

  if [ -n "$CHANGED_FILES" ]; then

    # 3. 모든 변경사항 스테이징
    git add -A

    # 4. 커밋 메시지 생성
    COMMIT_MSG=$(generate_commit_message \
      --prompt "$USER_PROMPT" \
      --response "$AI_RESPONSE" \
      --files "$CHANGED_FILES" \
      --style "$COMMIT_STYLE")

    # 5. 커밋 생성
    git commit -m "$COMMIT_MSG

Session: ${SESSION_ID}
Checkpoint: temporary
Timestamp: $(date -Iseconds)"

    COMMIT_SHA=$(git rev-parse HEAD)

    # 6. Temporary checkpoint 생성
    CHECKPOINT_ID=$(generate_checkpoint_id)

    # 7. 메타데이터 업데이트
    update_metadata \
      --session "$SESSION_ID" \
      --checkpoint "$CHECKPOINT_ID" \
      --commit "$COMMIT_SHA" \
      --type temporary

    # 8. 사용자에게 알림
    echo "✓ Auto-committed: ${COMMIT_SHA:0:7}"
    echo "  Message: $COMMIT_MSG"
    echo "  Checkpoint: $CHECKPOINT_ID (temporary)"
  fi
}
```

### 커밋 메시지 스타일

#### Concise (기본)

```bash
entire session start --strategy auto-commit \
  --commit-style concise

# 예시 메시지:
# "Add login functionality"
# "Fix authentication bug"
# "Update user interface"
# "Refactor database queries"
```

#### Descriptive

```bash
entire session start --strategy auto-commit \
  --commit-style descriptive

# 예시 메시지:
# "Add login functionality
#
# - Implement LoginForm component with email/password fields
# - Add form validation using Yup schema
# - Integrate with authentication API endpoint
# - Handle success/error states with appropriate UI feedback"

# "Fix authentication bug
#
# The token expiration check was comparing timestamps incorrectly,
# causing valid tokens to be rejected. Updated the comparison logic
# to use UTC timestamps consistently."
```

#### Conventional Commits

```bash
entire session start --strategy auto-commit \
  --commit-style conventional

# 예시 메시지:
# "feat(auth): add login functionality"
# "fix(auth): correct token expiration check"
# "docs(api): update authentication endpoints"
# "refactor(db): optimize user queries"
# "test(auth): add login form tests"

# 형식:
# <type>(<scope>): <subject>
#
# Types: feat, fix, docs, style, refactor, test, chore
```

### Checkpoint 관리

#### Temporary Checkpoint

자동으로 생성되는 임시 checkpoint입니다.

```bash
# 매 커밋마다 temporary checkpoint 자동 생성
$ AI: 기능 추가해줘
# ✓ Checkpoint: 4a7b3c (temporary)

$ AI: 테스트 추가해줘
# ✓ Checkpoint: 9e2f1d (temporary)

# Temporary checkpoint 조회
entire checkpoint list --type temporary

# 출력:
# Session: abc123
#
# Temporary Checkpoints (auto-created):
#   4a7b3c - Add feature (3d8k5m2)
#   9e2f1d - Add tests (7f4a2b1)
```

#### Committed Checkpoint

명시적으로 생성하는 영구 checkpoint입니다.

```bash
# 중요한 시점에 committed checkpoint 생성
$ AI: 핵심 기능 완료

entire checkpoint create --message "핵심 기능 완성"

# 출력:
# ✓ Checkpoint created: 6h4j2p
# ✓ Type: committed
# ✓ Commit: 8r6t3v9
# ✓ Message: 핵심 기능 완성
#
# This is a permanent checkpoint.
# Previous temporary checkpoints are preserved.

# Committed checkpoint는 특별 표시
entire checkpoint list

# 출력:
# Session: abc123
#
# Committed Checkpoints:
#   ★ 6h4j2p - 핵심 기능 완성 (8r6t3v9)
#
# Temporary Checkpoints:
#   4a7b3c - Add feature (3d8k5m2)
#   9e2f1d - Add tests (7f4a2b1)
```

#### Checkpoint 자동 정리

```bash
# 오래된 temporary checkpoint 자동 정리
entire config set checkpoint.cleanup.enabled true
entire config set checkpoint.cleanup.keep-temporary 20
entire config set checkpoint.cleanup.keep-committed all

# 설정 내용:
# - temporary는 최근 20개만 유지
# - committed는 모두 유지
# - 정리는 세션 종료 시 자동 실행

# 수동 정리
entire checkpoint cleanup --session abc123

# 출력:
# Analyzing checkpoints for session abc123...
#
# Temporary: 45 total, 20 will be kept, 25 will be deleted
# Committed: 3 total, all will be kept
#
# Proceed with cleanup? (y/N): y
# ✓ Deleted 25 temporary checkpoints
# ✓ Kept 3 committed checkpoints
# ✓ Metadata updated
```

### 변경 되돌리기

#### Git Reset 사용

```bash
# 1. 마지막 커밋 취소 (변경사항 유지)
git reset --soft HEAD~1

# 2. 마지막 커밋 취소 (변경사항도 취소)
git reset --hard HEAD~1

# 3. 여러 커밋 취소
git reset --hard HEAD~5

# 4. 특정 커밋으로 되돌리기
git reset --hard 9a7f3c2
```

#### Checkpoint Restore 사용

```bash
# 1. Checkpoint 목록 확인
entire checkpoint list

# 출력:
# Session: abc123
#
# Committed Checkpoints:
#   ★ 6h4j2p - 핵심 기능 완성 (8r6t3v9)
#   ★ 3k8m2n - 기본 구조 완료 (5d6e7f8)
#
# Temporary Checkpoints:
#   4a7b3c - Add feature (3d8k5m2)
#   9e2f1d - Add tests (7f4a2b1)

# 2. 특정 checkpoint로 복원
entire checkpoint restore 6h4j2p

# 출력:
# Restoring checkpoint 6h4j2p...
#
# This will:
#   - Reset HEAD to commit 8r6t3v9
#   - Discard all changes after this checkpoint
#   - Mark subsequent checkpoints as archived
#
# Continue? (y/N): y
# ✓ Restored to checkpoint 6h4j2p
# ✓ 2 checkpoints archived

# 3. 아카이브된 checkpoint 확인
entire checkpoint list --include-archived

# 4. 아카이브 복원 (실수한 경우)
entire checkpoint unarchive 9e2f1d
```

#### 선택적 되돌리기

```bash
# 특정 파일만 되돌리기
git checkout HEAD~1 -- src/components/Login.tsx

# 특정 커밋의 변경만 되돌리기 (revert)
git revert 3d8k5m2

# AI에게 되돌리기 요청
$ AI: 방금 한 변경을 되돌려줘
# AI가 자동으로 이전 버전으로 복원
# 새로운 커밋으로 기록됨
```

### 충돌 처리

#### 자동 충돌 감지

```bash
# AI 변경 중 충돌 발생
$ AI: 로그인 폼 업데이트해줘

# 출력:
# ⚠ Conflict detected
#
# File: src/components/LoginForm.tsx
# Reason: File was modified outside of this session
#
# Options:
#   1. Use AI's version (overwrite)
#   2. Keep current version (skip AI changes)
#   3. Merge manually
#   4. Ask AI to resolve
#
# Choose [1-4]:
```

#### AI 기반 충돌 해결

```bash
# 옵션 4 선택: AI에게 해결 요청
Choose [1-4]: 4

# AI가 자동으로 분석 및 해결
# Analyzing conflict...
#
# Current version:
#   - Uses Formik for form handling
#   - Has custom validation
#
# AI's version:
#   - Uses React Hook Form
#   - Has Yup schema validation
#
# Suggested resolution:
#   - Keep Formik (already integrated)
#   - Add Yup validation (improves UX)
#   - Merge both approaches
#
# Apply this resolution? (y/N): y
# ✓ Conflict resolved
# ✓ Auto-committed: 7f4a2b1
```

### 브랜치 보호

Auto-commit에서 실수를 방지하는 안전 장치:

```bash
# 1. Protected branch 설정
entire config set protected-branches "main,production,release/*"

# 2. Protected branch에서 auto-commit 시도
git checkout main
entire session start --strategy auto-commit

# 출력:
# ⚠ Warning: 'main' is a protected branch
#
# Auto-commit on protected branches can be risky.
#
# Recommendations:
#   1. Use manual-commit strategy instead
#   2. Create a feature branch
#   3. Disable branch protection (not recommended)
#
# Continue anyway? (y/N):

# 3. 안전한 대안
git checkout -b feature/new-feature
entire session start --strategy auto-commit
# 이제 안전하게 작업 가능
```

### 성능 최적화

#### Commit 배치 처리

```bash
# 여러 작은 변경을 하나의 커밋으로
entire session start --strategy auto-commit \
  --batch-commits 5m \
  --message "UI 개선 작업"

# 5분 동안의 모든 변경을 하나로 커밋
$ AI: 버튼 스타일 바꿔줘
$ AI: 색상도 변경해줘
$ AI: 폰트 크기 조정해줘
# ... 5분 후 ...
# ✓ Batch committed: 3 changes in 1 commit
```

#### 선택적 파일 커밋

```bash
# 특정 파일만 자동 커밋
entire session start --strategy auto-commit \
  --include "src/**/*.ts" \
  --exclude "**/*.test.ts" \
  --message "소스 코드만 커밋"

# 테스트 파일은 자동 커밋 안 됨
$ AI: 기능 추가하고 테스트도 만들어줘
# ✓ Auto-committed: src/feature.ts
# ⊘ Skipped: src/feature.test.ts (excluded)
```

### 실전 워크플로우

#### 빠른 프로토타이핑

```bash
# 1. 프로토타입 브랜치 생성
git checkout -b prototype/new-idea

# 2. Auto-commit 세션 시작
entire session start --strategy auto-commit \
  --commit-style concise \
  --message "아이디어 프로토타이핑"

# 3. 빠른 반복
$ AI: 기본 UI 만들어줘
$ AI: 데이터 바인딩 추가해줘
$ AI: 애니메이션 넣어줘
$ AI: 반응형으로 만들어줘

# 각 단계가 즉시 커밋되어 히스토리 유지

# 4. 테스트
npm start
# 브라우저에서 확인

# 5. 만족하면 PR 생성
gh pr create --title "새 아이디어 프로토타입"

# 6. 불만족하면 브랜치 삭제
git checkout main
git branch -D prototype/new-idea
```

#### 버그 수정

```bash
# 1. 버그 수정 세션
git checkout -b fix/login-issue
entire session start --strategy auto-commit \
  --commit-style conventional \
  --message "로그인 버그 수정"

# 2. AI에게 버그 분석 및 수정 요청
$ AI: 로그인 시 토큰이 저장 안 되는 버그 찾아서 고쳐줘

# AI가 분석 및 수정
# ✓ Auto-committed: fix(auth): save token to localStorage

# 3. 즉시 테스트
npm test

# 4. 추가 수정 필요
$ AI: 토큰 만료 처리도 추가해줘
# ✓ Auto-committed: feat(auth): add token expiration handling

# 5. 검증
npm test
npm run e2e

# 6. 병합
git checkout main
git merge fix/login-issue
git push
```

#### 문서 작업

```bash
# 문서는 auto-commit이 최적
entire session start --strategy auto-commit \
  --commit-style descriptive \
  --include "**/*.md" \
  --message "문서 업데이트"

$ AI: README에 설치 방법 추가해줘
$ AI: API 문서 업데이트해줘
$ AI: 예제 코드 개선해줘

# 각 업데이트가 즉시 커밋
# 문서 변경 이력이 명확하게 기록됨
```

### 고급 기능

#### Interactive Commit Mode

```bash
# 커밋 전 확인 모드
entire session start --strategy auto-commit \
  --interactive \
  --message "신중한 작업"

$ AI: 코드 변경해줘

# AI 응답 후:
# Changes to be committed:
#   modified: src/feature.ts
#   new file: src/helper.ts
#
# Commit message:
#   "Add helper utility for feature processing"
#
# Actions:
#   [y] Commit now
#   [e] Edit commit message
#   [s] Skip this commit
#   [d] Show diff
#   [q] Quit session
#
# Choose:
```

#### Conditional Auto-Commit

```bash
# 조건부 자동 커밋
entire session start --strategy auto-commit \
  --commit-if "tests-pass" \
  --message "테스트 기반 개발"

$ AI: 새 기능 추가해줘

# AI 변경 후 자동으로 테스트 실행
# Running tests...
# ✓ All tests passed (23/23)
# ✓ Auto-committed: 3d8k5m2
#
# 만약 테스트 실패:
# ✗ Tests failed (2/23)
# ⊘ Commit skipped
#
# Fix the failing tests before committing:
#   - test/feature.test.ts (2 failing)
```

#### Smart Commit Grouping

```bash
# AI가 관련 변경을 자동으로 그룹화
entire session start --strategy auto-commit \
  --smart-grouping \
  --message "기능 개발"

$ AI: 로그인 기능의 프론트엔드와 백엔드를 모두 구현해줘

# AI가 여러 파일 변경
# 자동으로 논리적 그룹으로 커밋:
#
# ✓ Commit 1: "Add login API endpoint"
#   - src/api/auth.ts
#   - src/routes/auth.ts
#
# ✓ Commit 2: "Add login UI components"
#   - src/components/LoginForm.tsx
#   - src/styles/login.css
#
# ✓ Commit 3: "Add login tests"
#   - test/auth.test.ts
#   - test/login.test.tsx
```

### 모니터링 및 디버깅

#### 실시간 로그

```bash
# 상세 로깅 활성화
entire session start --strategy auto-commit \
  --verbose \
  --message "디버깅 세션"

# 출력:
# [10:30:15] Session started: abc123
# [10:30:20] User prompt received (45 chars)
# [10:30:25] AI response started
# [10:30:35] AI response completed (1250 chars)
# [10:30:36] Detecting file changes...
# [10:30:36] Found changes: 3 files
# [10:30:37] Staging files...
# [10:30:37] Generating commit message...
# [10:30:38] Creating commit...
# [10:30:38] Commit created: 3d8k5m2
# [10:30:39] Creating temporary checkpoint...
# [10:30:39] Checkpoint created: 4a7b3c
# [10:30:40] Updating metadata...
# [10:30:40] ✓ Complete
```

#### 통계

```bash
# 세션 통계 확인
entire session stats

# 출력:
# Session: abc123
# Strategy: auto-commit
# Duration: 2h 15m
#
# Commits:
#   Total: 47
#   Average interval: 2m 52s
#   Files changed: 125
#   Lines added: 3,450
#   Lines deleted: 892
#
# Checkpoints:
#   Temporary: 47
#   Committed: 5
#
# AI Interactions:
#   Prompts: 52
#   Average response time: 8.3s
#   Tokens used: 245,320
```

### 베스트 프랙티스

```bash
# 1. 실험용 브랜치 사용
git checkout -b experiment/new-idea
entire session start --strategy auto-commit

# 2. Committed checkpoint로 중요 시점 표시
$ AI: 핵심 기능 완료
entire checkpoint create --message "Milestone: 핵심 기능"

# 3. 정기적인 테스트
$ AI: 기능 추가해줘
npm test  # 즉시 검증

# 4. 의미있는 세션 메시지
entire session start --message "구체적인 작업 내용"
# 나쁜 예: --message "작업"

# 5. 세션 종료 전 검토
entire session info
git log --oneline -10
entire session stop
```

### 다음 장 예고

다음 장에서는 **Session 관리**를 다룹니다.

- Session 생성 및 추적
- 여러 세션 동시 관리
- Session 메타데이터
- Session 이력 조회

---

**관련 문서**:
- [06. Strategy 개요](/2026/02/11/entire-cli-guide-06-strategy-overview/)
- [07. Manual-Commit Strategy](/2026/02/11/entire-cli-guide-07-manual-commit/)
- [09. Session 관리](/2026/02/11/entire-cli-guide-09-session-management/)
