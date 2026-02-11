---
layout: post
title: "Entire CLI 완벽 가이드 - 13. Git 통합"
date: 2026-02-11
categories: [AI, Development Tools]
tags: [Entire, CLI, AI Coding, Git, Hooks, Worktree]
description: "Entire CLI의 Git Hooks 연동, Worktree 지원 및 Git 워크플로우 통합"
---

## 13. Git 통합

Entire CLI는 Git와 깊이 통합되어 있습니다. 이 장에서는 Git Hooks, Worktree, 브랜치 전략 등 Git와의 통합 방법을 다룹니다.

### Git 저장소 요구사항

#### 필수 조건

```bash
# 1. Git 저장소 확인
git rev-parse --git-dir
# .git

# 2. Git 버전 확인
git --version
# git version 2.40.0 이상 권장

# 3. Entire CLI 초기화
entire init

# 출력:
# Checking Git repository...
# ✓ Git repository found
# ✓ Git version: 2.40.0
#
# Initializing Entire CLI...
# ✓ Created .entire/ directory
# ✓ Created Git hooks
# ✓ Updated .gitignore
# ✓ Configured Git attributes
#
# Entire CLI is ready!
```

#### .gitignore 설정

```bash
# entire init가 자동으로 추가

cat .gitignore
# 추가된 내용:
# .entire/cache/
# .entire/temp/
# .entire/logs/
# .entire/worktrees/  # Worktree 디렉토리
```

#### .gitattributes 설정

```bash
cat .gitattributes
# Entire가 추가한 내용:

# Entire metadata는 항상 텍스트로 처리
refs/entire/metadata/** text

# Checkpoint 데이터는 diff 비활성화
refs/entire/metadata/**/checkpoints.json -diff

# 대화 파일은 병합 전략 지정
refs/entire/metadata/**/prompts/* merge=union
refs/entire/metadata/**/responses/* merge=union
```

### Git Hooks

Entire CLI는 여러 Git Hooks를 사용합니다.

#### Hook 설치

```bash
# 자동 설치
entire init

# 수동 설치
entire hooks install

# 출력:
# Installing Git hooks...
# ✓ pre-commit
# ✓ post-commit
# ✓ pre-merge-commit
# ✓ post-merge
# ✓ pre-push
#
# All hooks installed.

# 설치된 hooks 확인
ls -la .git/hooks/
# -rwxr-xr-x pre-commit
# -rwxr-xr-x post-commit
# -rwxr-xr-x pre-merge-commit
# -rwxr-xr-x post-merge
# -rwxr-xr-x pre-push
```

#### pre-commit Hook

세션 상태를 자동으로 저장합니다.

```bash
# .git/hooks/pre-commit
#!/bin/bash

# Entire CLI pre-commit hook

# 활성 세션이 있는지 확인
if entire session current > /dev/null 2>&1; then
  SESSION=$(entire session current --id)

  echo "Active Entire session: $SESSION"

  # Temporary checkpoint 생성
  entire checkpoint create-temp \
    --message "Auto-save before commit" \
    --type pre-commit

  # 메타데이터 업데이트
  entire session update-metadata
fi

exit 0
```

**동작 예시**:
```bash
git add .
git commit -m "Add feature"

# 출력:
# Active Entire session: abc123
# ✓ Temporary checkpoint created: 4a7b3c
# ✓ Metadata updated
# [main 3d8k5m2] Add feature
```

#### post-commit Hook

커밋 후 세션 메타데이터를 업데이트합니다.

```bash
# .git/hooks/post-commit
#!/bin/bash

# Entire CLI post-commit hook

if entire session current > /dev/null 2>&1; then
  SESSION=$(entire session current --id)
  COMMIT=$(git rev-parse HEAD)

  # 커밋을 세션에 연결
  entire session link-commit $SESSION $COMMIT

  # 통계 업데이트
  entire session update-stats $SESSION
fi

exit 0
```

#### pre-merge-commit Hook

Shadow branch 병합 시 검증합니다.

```bash
# .git/hooks/pre-merge-commit
#!/bin/bash

# Shadow branch 병합 감지
MERGE_HEAD=$(cat .git/MERGE_HEAD 2>/dev/null)
MERGE_BRANCH=$(git rev-parse --abbrev-ref $MERGE_HEAD 2>/dev/null)

if [[ $MERGE_BRANCH == entire/shadow/* ]]; then
  echo "Merging Entire shadow branch: $MERGE_BRANCH"

  # 세션 ID 추출
  SESSION_ID=${MERGE_BRANCH#entire/shadow/session-}

  # 세션 검증
  if ! entire session validate $SESSION_ID; then
    echo "Error: Invalid session state"
    exit 1
  fi

  # Checkpoint 확인
  CHECKPOINTS=$(entire checkpoint list $SESSION_ID --count)
  echo "Merging $CHECKPOINTS checkpoints"

  # 사용자 확인
  read -p "Continue with merge? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

exit 0
```

#### post-merge Hook

병합 후 정리 작업을 수행합니다.

```bash
# .git/hooks/post-merge
#!/bin/bash

# Shadow branch 병합 완료 처리
MERGE_HEAD=$(cat .git/ORIG_HEAD 2>/dev/null)

if entire session find-by-commit $MERGE_HEAD > /dev/null 2>&1; then
  SESSION=$(entire session find-by-commit $MERGE_HEAD)

  echo "Shadow branch merged for session: $SESSION"

  # 세션 상태 업데이트
  entire session update $SESSION --merged true

  # Shadow branch 정리 제안
  echo "Shadow branch can now be deleted:"
  echo "  git branch -d entire/shadow/session-$SESSION"
fi

exit 0
```

#### pre-push Hook

푸시 전 세션 상태를 확인합니다.

```bash
# .git/hooks/pre-push
#!/bin/bash

# 활성 세션 확인
if entire session current > /dev/null 2>&1; then
  SESSION=$(entire session current --id)

  echo "Active Entire session: $SESSION"

  # 커밋되지 않은 변경사항 확인
  if entire session has-uncommitted $SESSION; then
    echo "Warning: Session has uncommitted changes"
    read -p "Continue with push? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      exit 1
    fi
  fi

  # 메타데이터 푸시 여부 확인
  read -p "Push session metadata? (Y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    # 메타데이터 branch push
    git push origin refs/entire/metadata/session-$SESSION
  fi
fi

exit 0
```

### Git Worktree 지원

#### Worktree 개념

```
Worktree를 사용한 세션 격리:

프로젝트 루트/
├── .git/                    (메인 Git 디렉토리)
├── src/                     (main 브랜치)
├── package.json
└── .entire/
    └── worktrees/
        ├── session-abc123/  (Shadow branch 전용)
        │   ├── src/
        │   └── package.json
        └── session-def456/  (다른 Shadow branch)
            ├── src/
            └── package.json

각 worktree는 독립적인 작업 디렉토리
```

#### Worktree 생성

```bash
# Manual-commit 세션에 worktree 사용
entire session start \
  --strategy manual-commit \
  --use-worktree \
  --message "독립 작업 환경"

# 출력:
# ✓ Session created: abc123
# ✓ Shadow branch: entire/shadow/session-abc123
# ✓ Worktree created: .entire/worktrees/session-abc123
#
# Working directory:
#   Main: /home/user/project (main branch)
#   Session: /home/user/project/.entire/worktrees/session-abc123
#
# To work in the session worktree:
#   cd .entire/worktrees/session-abc123
```

#### Worktree 작업

```bash
# 1. Worktree로 이동
cd .entire/worktrees/session-abc123

# 2. 독립적인 작업
npm install  # 별도 node_modules
npm test     # main에 영향 없음
npm run build

# 3. AI와 작업
$ AI: 코드 변경해줘
# Shadow branch에만 영향

# 4. 메인 디렉토리는 영향 없음
cd /home/user/project
git status
# On branch main
# nothing to commit, working tree clean
```

#### Worktree 관리

```bash
# 모든 worktree 목록
git worktree list

# 출력:
# /home/user/project              (main)
# /home/user/project/.entire/worktrees/session-abc123  (entire/shadow/session-abc123)
# /home/user/project/.entire/worktrees/session-def456  (entire/shadow/session-def456)

# Entire 명령으로 조회
entire session list --worktrees

# 출력:
# Sessions with worktrees:
#   abc123 - .entire/worktrees/session-abc123
#            Branch: entire/shadow/session-abc123
#            Size: 145 MB
#
#   def456 - .entire/worktrees/session-def456
#            Branch: entire/shadow/session-def456
#            Size: 132 MB

# Worktree 정리
entire session stop abc123

# 출력:
# ✓ Session abc123 stopped
# ✓ Worktree removed: .entire/worktrees/session-abc123
#
# Shadow branch preserved:
#   entire/shadow/session-abc123
```

#### Worktree 동기화

```bash
# Worktree와 main 동기화
entire worktree sync session-abc123

# 출력:
# Syncing worktree with main branch...
#
# Main branch updates:
#   package.json (dependency updates)
#   src/config.ts (configuration changes)
#
# Apply to worktree? (y/N): y
# ✓ Worktree updated
# ✓ Rebased on latest main
```

### Git 브랜치 전략

#### Feature Branch 통합

```bash
# 1. Feature branch 생성
git checkout -b feature/user-auth

# 2. Entire 세션 시작
entire session start \
  --strategy auto-commit \
  --message "사용자 인증 구현"

# 3. 개발 진행
$ AI: 인증 시스템 구현해줘
# 자동 커밋됨

# 4. 완료 후 PR
entire session stop
git push origin feature/user-auth
gh pr create --title "사용자 인증 추가"
```

#### Git Flow 통합

```bash
# Develop branch에서 작업
git checkout develop

# Feature 시작
git flow feature start user-auth

# Entire 세션
entire session start \
  --strategy manual-commit \
  --message "사용자 인증"

# 개발...
$ AI: 작업해줘

# Feature 완료
entire checkpoint create --message "완성"
git merge entire/shadow/session-abc123
git flow feature finish user-auth
```

#### Trunk-Based Development

```bash
# Main에서 직접 작업 (짧은 주기)
git checkout main

# 짧은 세션
entire session start \
  --strategy auto-commit \
  --message "빠른 기능 추가"

# 빠르게 개발
$ AI: 간단한 기능 추가해줘

# 즉시 푸시
git push origin main
entire session stop
```

### Git 명령어 통합

#### Git Status 확장

```bash
# Entire 정보 포함 git status
entire git status

# 출력:
# On branch main
# Your branch is up to date with 'origin/main'.
#
# Entire Sessions:
#   abc123 (active) - 사용자 인증 구현
#     Shadow: entire/shadow/session-abc123 (3 commits ahead)
#     Uncommitted changes: 2 files
#
# Changes not staged for commit:
#   modified:   src/auth/index.ts
#   new file:   src/auth/oauth.ts
```

#### Git Log 확장

```bash
# Checkpoint 정보 포함 git log
entire git log

# 출력:
# commit 3d8k5m2a1b4c (entire/shadow/session-abc123)
# Checkpoint: 4a7b3c9e (OAuth 통합 완료)
# Author: AI Assistant <ai@entire.dev>
# Date:   Tue Feb 11 12:30:00 2026 +0900
#
#     Add OAuth integration
#
#     Session: abc123
#     Prompt: "OAuth 프로바이더 추가해줘"
#     Tokens: 3,456
#
# commit 9a7f3c2d5e6f (main)
# Author: Developer <dev@example.com>
# Date:   Tue Feb 11 10:00:00 2026 +0900
#
#     Initial auth structure
```

#### Git Diff 확장

```bash
# Checkpoint 간 diff
entire git diff 4a7b3c 6h4j2p

# AI 설명 포함 diff
entire git diff --ai-explain 4a7b3c 6h4j2p

# 출력:
# Comparing checkpoints 4a7b3c and 6h4j2p...
#
# AI Summary:
# Added OAuth integration with Google and GitHub providers.
# Updated authentication service to support multiple auth methods.
# Added new configuration for OAuth client IDs and secrets.
#
# Files changed:
# diff --git a/src/auth/oauth.ts b/src/auth/oauth.ts
# new file mode 100644
# index 0000000..3d8k5m2
# [... diff content ...]
```

### Git 태그 통합

#### Checkpoint 태깅

```bash
# 중요한 checkpoint에 Git 태그 추가
entire checkpoint tag 4a7b3c --git-tag v1.0.0-auth

# 출력:
# ✓ Entire tag added: milestone
# ✓ Git tag created: v1.0.0-auth
#   Points to commit: 3d8k5m2
#
# You can now:
#   git push origin v1.0.0-auth

# Git 태그 조회
git tag -l "v1.0.*"
# v1.0.0-auth

# Git 태그에서 checkpoint 찾기
entire checkpoint find-by-git-tag v1.0.0-auth
# Checkpoint: 4a7b3c
# Message: OAuth 통합 완료
```

### Git Submodule 지원

```bash
# Submodule이 있는 프로젝트
git submodule status
# a1b2c3d libs/auth (v1.0.0)
# d4e5f6g libs/payment (v2.1.0)

# Entire 세션 시작
entire session start --strategy manual-commit \
  --include-submodules

# AI가 submodule 인식
$ AI: auth 라이브러리 사용해서 로그인 구현해줘

# Submodule 업데이트도 checkpoint에 포함
entire checkpoint create --message "Auth 통합"

# Checkpoint 정보
entire checkpoint info 4a7b3c

# 출력:
# Submodules:
#   libs/auth: a1b2c3d → e7f8g9h (updated)
#   libs/payment: d4e5f6g (unchanged)
```

### Git LFS 통합

```bash
# Git LFS 파일 처리
git lfs track "*.psd"
git lfs track "*.ai"

# Entire가 LFS 파일 인식
entire session start --strategy auto-commit

$ AI: 로고 파일 추가해줘
# logo.psd (LFS 파일)

# LFS 파일도 checkpoint에 포함
entire checkpoint create --message "로고 추가"

# Checkpoint 크기 확인
entire checkpoint info 4a7b3c

# 출력:
# Files:
#   logo.psd (LFS: 45.2 MB)
#   src/assets/logo.svg (2.1 KB)
#
# Total size: 45.2 MB (LFS: 45.2 MB)
```

### Git Reflog 통합

```bash
# Entire 작업도 reflog에 기록
git reflog

# 출력:
# 3d8k5m2 HEAD@{0}: entire checkpoint (abc123): OAuth 통합 완료
# 9a7f3c2 HEAD@{1}: entire checkpoint (abc123): 사용자 인증 완료
# 7f4a2b1 HEAD@{2}: commit: Initial auth structure

# Reflog에서 checkpoint 복원
entire checkpoint restore-from-reflog HEAD@{1}
```

### 충돌 해결 통합

```bash
# Git 충돌 발생
git merge feature/other-auth
# CONFLICT in src/auth/index.ts

# Entire가 AI로 충돌 해결
entire git resolve-conflicts

# 출력:
# Analyzing conflicts...
#
# Conflict in src/auth/index.ts:
# <<<<<<< HEAD
# export function authenticate(user: User) {
#   return jwt.sign(user);
# }
# =======
# export async function authenticate(user: User) {
#   return await oauth.authenticate(user);
# }
# >>>>>>> feature/other-auth
#
# AI suggestion:
# Merge both approaches - support JWT and OAuth
#
# export async function authenticate(
#   user: User,
#   method: 'jwt' | 'oauth' = 'jwt'
# ) {
#   if (method === 'oauth') {
#     return await oauth.authenticate(user);
#   }
#   return jwt.sign(user);
# }
#
# Apply? (y/N): y
# ✓ Conflict resolved
# ✓ Changes staged
```

### Git 설정

#### Entire 전용 설정

```bash
# Entire 관련 Git 설정
git config --local entire.enabled true
git config --local entire.autoCommit true
git config --local entire.shadowBranchPrefix "entire/shadow"
git config --local entire.metadataBranchPrefix "entire/metadata"

# 설정 확인
git config --local --get-regexp entire
# entire.enabled true
# entire.autocommit true
# entire.shadowbranchprefix entire/shadow
# entire.metadatabranchprefix entire/metadata
```

#### Git Aliases

```bash
# Entire 명령을 Git alias로 추가
git config --local alias.entire-start '!entire session start'
git config --local alias.entire-stop '!entire session stop'
git config --local alias.entire-checkpoint '!entire checkpoint create'

# 사용
git entire-start --message "새 작업"
git entire-checkpoint --message "완료"
git entire-stop
```

### 원격 저장소 연동

#### 메타데이터 푸시

```bash
# 세션 메타데이터를 원격에 푸시
entire push abc123

# 출력:
# Pushing session abc123 metadata...
#
# Branch: refs/entire/metadata/session-abc123
# Target: origin
#
# Objects:
#   Checkpoints: 5
#   Conversations: 18
#   Metadata: 2.3 MB
#
# Push? (y/N): y
# ✓ Metadata pushed to origin

# 자동 푸시 설정
entire config set push.metadata auto

# 이제 git push 시 자동으로 메타데이터도 푸시
git push
# ✓ Code pushed
# ✓ Metadata pushed (session abc123)
```

#### 메타데이터 풀

```bash
# 다른 개발자의 세션 가져오기
entire pull def456

# 출력:
# Pulling session def456 from origin...
#
# Session info:
#   Author: john@example.com
#   Message: 결제 시스템 구현
#   Checkpoints: 8
#
# Download? (y/N): y
# ✓ Session imported
# ✓ Checkpoints available
# ✓ Conversations downloaded
#
# You can now:
#   entire session switch def456
#   entire checkpoint list def456
```

### Git 퍼포먼스 최적화

#### Shallow Clone 지원

```bash
# Shallow clone에서 Entire 사용
git clone --depth 1 https://github.com/user/repo.git

# Entire 초기화
cd repo
entire init

# 경고:
# ⚠ Shallow clone detected
#
# Entire features with limitations:
#   - Checkpoint history may be incomplete
#   - Some git operations may fail
#
# Recommended:
#   git fetch --unshallow
#
# Continue anyway? (y/N)
```

#### Sparse Checkout 지원

```bash
# Sparse checkout 설정
git sparse-checkout init --cone
git sparse-checkout set src

# Entire가 sparse checkout 인식
entire session start --strategy auto-commit

$ AI: src/ 디렉토리만 수정해줘
# ✓ Working within sparse checkout boundaries
```

### 실전 예시

#### CI/CD 파이프라인

```yaml
# .github/workflows/entire.yml
name: Entire Session CI

on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # 전체 히스토리

      - name: Setup Entire CLI
        run: npm install -g @entire/cli

      - name: Extract sessions from commits
        run: |
          # 최근 커밋에서 세션 추출
          SESSIONS=$(git log --grep="Session:" -1 --format="%b" | grep "Session:" | cut -d: -f2)
          echo "SESSIONS=$SESSIONS" >> $GITHUB_ENV

      - name: Validate sessions
        run: |
          for session in $SESSIONS; do
            entire session validate $session
          done

      - name: Generate report
        run: |
          entire session export --all --format markdown > report.md

      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('report.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: report
            });
```

### 베스트 프랙티스

```bash
# 1. Hooks 활용
entire hooks install

# 2. Worktree로 격리
entire session start --use-worktree

# 3. 메타데이터 백업
git push origin refs/entire/metadata/*

# 4. 정기적 검증
entire git verify

# 5. 충돌 조기 감지
entire git status  # 자주 확인

# 6. 의미있는 커밋 메시지
entire checkpoint create --message "명확한 설명"
```

### 다음 장 예고

다음 장에서는 **Storage 구조**를 다룹니다.

- Shadow branch 저장 구조
- Metadata branch 구조
- 파일 시스템 레이아웃
- 데이터 최적화

---

**관련 문서**:
- [07. Manual-Commit Strategy](/2026/02/11/entire-cli-guide-07-manual-commit/)
- [12. Multi-Session 처리](/2026/02/11/entire-cli-guide-12-multi-session/)
- [14. Storage 구조](/2026/02/11/entire-cli-guide-14-storage-structure/)
