---
layout: post
title: "Entire CLI 완벽 가이드 - 12. Multi-Session 처리"
date: 2026-02-11
categories: [AI, Development Tools]
tags: [Entire, CLI, AI Coding, Multi-Session, Concurrency]
description: "Entire CLI 동시 세션 관리 및 충돌 해결 메커니즘"
---

## 12. Multi-Session 처리

Entire CLI는 여러 AI 세션을 동시에 관리할 수 있습니다. 이 장에서는 동시 세션 실행, 충돌 처리, 세션 간 협업을 다룹니다.

### Multi-Session 개념

```
단일 저장소에서 여러 세션 동시 진행:

Session A (feature/auth)        Session B (feature/payment)
├─ checkpoint 1                 ├─ checkpoint 1
├─ checkpoint 2                 ├─ checkpoint 2
└─ checkpoint 3                 └─ checkpoint 3

Session C (bugfix/login)        Session D (docs/api)
├─ checkpoint 1                 ├─ checkpoint 1
└─ checkpoint 2                 └─ checkpoint 2

각 세션은 독립적이지만 공통 코드베이스 공유
```

### 세션 격리

#### Manual-Commit 격리

```bash
# Session A: Shadow branch로 격리
entire session start --strategy manual-commit \
  --message "인증 시스템"

# 출력:
# Session: abc123
# Shadow branch: entire/shadow/session-abc123
# Working on: main (unchanged)

# Session B: 별도 shadow branch
entire session start --strategy manual-commit \
  --message "결제 시스템"

# 출력:
# Session: def456
# Shadow branch: entire/shadow/session-def456
# Working on: main (unchanged)

# 두 세션 모두 main에 영향 없음
# 완전히 독립적으로 작업 가능
```

#### Auto-Commit 격리

```bash
# Auto-commit은 브랜치로 격리 필요

# Session A: feature/auth 브랜치
git checkout -b feature/auth
entire session start --strategy auto-commit \
  --message "인증 시스템"

# Session B: feature/payment 브랜치
git checkout -b feature/payment
entire session start --strategy auto-commit \
  --message "결제 시스템"

# 각 브랜치에서 독립적으로 작업
```

### 세션 목록 관리

#### 활성 세션 확인

```bash
# 모든 세션 나열
entire session list

# 출력:
# Active Sessions (3):
#   abc123 - 인증 시스템
#            Strategy: manual-commit
#            Branch: main → entire/shadow/session-abc123
#            Updated: 10m ago
#
#   def456 - 결제 시스템
#            Strategy: auto-commit
#            Branch: feature/payment
#            Updated: 5m ago
#
#   ghi789 - 문서 업데이트
#            Strategy: auto-commit
#            Branch: docs/api
#            Updated: 2h ago
#
# Paused Sessions (1):
#   jkl012 - UI 리팩토링
#            Strategy: manual-commit
#            Branch: main → entire/shadow/session-jkl012
#            Paused: 1 day ago
```

#### 세션 전환

```bash
# 현재 세션 확인
entire session current

# 출력:
# Current session: abc123
# Message: 인증 시스템
# Status: active

# 다른 세션으로 전환
entire session switch def456

# 출력:
# Switching sessions...
#
# Pausing session abc123...
# ✓ Session abc123 paused
#
# Resuming session def456...
# ✓ Session def456 resumed
#
# Current session: def456
# Branch: feature/payment
# Last checkpoint: 4a7b3c (5m ago)
```

### 세션 간 충돌

#### 파일 충돌 감지

```bash
# Session A가 src/config.ts 수정
# Session B도 src/config.ts 수정

# Session A
$ AI: config에 auth 설정 추가해줘
entire checkpoint create --message "A"

# Session B로 전환
entire session switch def456

$ AI: config에 payment 설정 추가해줘
entire checkpoint create --message "B"

# 충돌 감지
# ⚠ Potential conflict detected
#
# File: src/config.ts
# Also modified by:
#   Session abc123 (10m ago)
#   Checkpoint: 4a7b3c
#
# Options:
#   [v] View other session's changes
#   [c] Continue (will need merge later)
#   [s] Switch to other session
#   [a] Ask AI to merge
#
# Choose:
```

#### 충돌 조회

```bash
# 세션 간 충돌 확인
entire session conflicts

# 출력:
# Conflicts between sessions:
#
# 1. src/config.ts
#    Sessions: abc123, def456
#    Type: Content conflict
#
#    abc123 (10m ago):
#      Added auth configuration
#
#    def456 (5m ago):
#      Added payment configuration
#
#    Suggestion: Can be auto-merged
#
# 2. package.json
#    Sessions: abc123, ghi789
#    Type: Dependency conflict
#
#    abc123:
#      Added: jsonwebtoken@9.0.0
#
#    ghi789:
#      Added: express@5.0.0
#
#    Suggestion: Manual review needed
#
# Total conflicts: 2
# Auto-mergeable: 1
# Need review: 1
```

### 충돌 해결

#### 자동 병합

```bash
# AI 기반 자동 병합
entire session merge-conflicts --auto

# 출력:
# Analyzing conflicts...
#
# Conflict 1/2: src/config.ts
# ├─ Session abc123: auth configuration
# ├─ Session def456: payment configuration
# └─ Strategy: Merge both sections
#
# Merged content:
# ```typescript
# export const config = {
#   // From session abc123
#   auth: {
#     jwtSecret: process.env.JWT_SECRET,
#     tokenExpiry: '1h'
#   },
#   // From session def456
#   payment: {
#     apiKey: process.env.PAYMENT_API_KEY,
#     currency: 'USD'
#   }
# };
# ```
#
# Apply? (y/N): y
# ✓ Merged src/config.ts
#
# Conflict 2/2: package.json
# ├─ Session abc123: jsonwebtoken@9.0.0
# ├─ Session ghi789: express@5.0.0
# └─ Strategy: Merge dependencies
#
# ✓ Merged package.json
#
# All conflicts resolved!
```

#### 수동 병합

```bash
# 특정 충돌 수동 해결
entire session resolve-conflict src/config.ts

# 출력:
# Conflict in src/config.ts
#
# Session abc123 (10m ago):
# ```diff
# + auth: {
# +   jwtSecret: process.env.JWT_SECRET
# + }
# ```
#
# Session def456 (5m ago):
# ```diff
# + payment: {
# +   apiKey: process.env.PAYMENT_API_KEY
# + }
# ```
#
# Resolution options:
#   [1] Use abc123's version
#   [2] Use def456's version
#   [3] Merge both
#   [4] Edit manually
#   [5] Ask AI to resolve
#
# Choose: 5

# AI 분석 및 해결
$ AI: 이 두 변경사항을 병합해줘

# AI가 자동으로 병합
# ✓ Conflict resolved
# ✓ Changes applied to both sessions
```

### 세션 병합

#### 전체 병합

```bash
# 두 세션을 하나로 병합
entire session merge abc123 def456 \
  --into merged-session \
  --message "인증과 결제 통합"

# 출력:
# Merging sessions...
#
# Source sessions:
#   abc123 - 인증 시스템
#     Strategy: manual-commit
#     Checkpoints: 5
#     Files changed: 12
#
#   def456 - 결제 시스템
#     Strategy: auto-commit
#     Checkpoints: 8
#     Files changed: 15
#
# Target session:
#   ID: xyz789 (new)
#   Message: 인증과 결제 통합
#   Strategy: manual-commit (from abc123)
#
# Merge plan:
#   1. Merge code changes
#   2. Combine checkpoints
#   3. Merge conversation histories
#   4. Resolve conflicts (if any)
#
# Conflicts detected: 2
# Auto-mergeable: 2
#
# Proceed? (y/N): y
#
# ✓ Code merged
# ✓ Conflicts resolved
# ✓ Checkpoints combined (13 total)
# ✓ Conversations merged
# ✓ New session created: xyz789
#
# Original sessions preserved.
```

#### 선택적 병합

```bash
# 특정 checkpoint만 병합
entire session cherry-pick \
  --from def456 \
  --checkpoints 4a7b3c,9e2f1d \
  --into abc123

# 출력:
# Cherry-picking checkpoints...
#
# From session def456 (결제 시스템):
#   4a7b3c - 결제 API 통합 (3 files)
#   9e2f1d - 카드 검증 추가 (2 files)
#
# Into session abc123 (인증 시스템):
#   Strategy: manual-commit
#   Current checkpoints: 5
#
# Checking for conflicts...
# ✓ No conflicts
#
# Apply? (y/N): y
# ✓ Checkpoint 4a7b3c applied
# ✓ Checkpoint 9e2f1d applied
# ✓ Session abc123 updated (7 checkpoints)
```

### 세션 동기화

#### 변경사항 공유

```bash
# Session A의 변경을 Session B로 가져오기
entire session sync abc123 def456 \
  --files "src/utils/*"

# 출력:
# Syncing files from abc123 to def456...
#
# Files to sync:
#   src/utils/crypto.ts (new in abc123)
#   src/utils/validator.ts (modified in abc123)
#
# Target session def456:
#   src/utils/api.ts (unique to def456)
#
# No conflicts.
#
# Apply sync? (y/N): y
# ✓ Synced 2 files
# ✓ Session def456 updated
```

#### 양방향 동기화

```bash
# 두 세션 간 변경사항 교환
entire session sync-bidirectional abc123 def456

# 출력:
# Bidirectional sync between abc123 and def456...
#
# abc123 → def456:
#   src/auth/* (3 files)
#
# def456 → abc123:
#   src/payment/* (4 files)
#
# Shared files:
#   src/config.ts (conflict!)
#
# Resolving conflicts...
# ✓ Conflict resolved (merged both changes)
#
# Proceed? (y/N): y
# ✓ Synced abc123 → def456 (3 files)
# ✓ Synced def456 → abc123 (4 files)
# ✓ Both sessions updated
```

### 세션 의존성

#### 의존성 선언

```bash
# Session B가 Session A에 의존
entire session depend def456 --on abc123 \
  --reason "결제는 인증이 필요"

# 출력:
# ✓ Dependency created
#
# Session def456 (결제 시스템) depends on:
#   abc123 (인증 시스템)
#
# This means:
#   - def456 should be merged after abc123
#   - Changes in abc123 may affect def456
#   - Checkpoint sync is recommended
```

#### 의존성 확인

```bash
# 세션 의존성 그래프
entire session dependencies

# 출력:
# Session Dependency Graph:
#
# abc123 (인증 시스템)
#   ↓ required-by
# def456 (결제 시스템)
#   ↓ required-by
# ghi789 (주문 시스템)
#
# jkl012 (문서 업데이트)
#   (no dependencies)
#
# Merge order:
#   1. abc123 (인증 시스템)
#   2. def456 (결제 시스템)
#   3. ghi789 (주문 시스템)
#   4. jkl012 (문서 업데이트) - can be parallel
```

### 동시성 제어

#### 락 메커니즘

```bash
# 파일 락으로 충돌 방지
entire session lock abc123 --files "src/config.ts"

# 출력:
# ✓ File locked by session abc123
#
# File: src/config.ts
# Locked by: abc123 (인증 시스템)
# Lock type: exclusive
#
# Other sessions cannot modify this file.

# 다른 세션에서 수정 시도
entire session switch def456
$ AI: config.ts 수정해줘

# 출력:
# ⚠ File locked
#
# File: src/config.ts
# Locked by: Session abc123
# Lock acquired: 5m ago
#
# Options:
#   [w] Wait for lock release
#   [r] Request lock transfer
#   [c] Cancel
#   [f] Force (override lock - dangerous!)
#
# Choose:
```

#### 낙관적 동시성

```bash
# 낙관적 동시성 제어 (기본)
entire config set concurrency.strategy optimistic

# 세션들이 자유롭게 수정
# 병합 시점에 충돌 검사

# Session A
$ AI: config.ts 수정
entire checkpoint create

# Session B (동시)
$ AI: config.ts 수정
entire checkpoint create
# ✓ Created (충돌은 나중에 검사)

# 병합 시
git merge entire/shadow/session-abc123
git merge entire/shadow/session-def456
# ⚠ Conflict in config.ts (이제 해결)
```

#### 비관적 동시성

```bash
# 비관적 동시성 제어
entire config set concurrency.strategy pessimistic

# 파일 수정 전 자동 락

# Session A
$ AI: config.ts 수정
# ✓ Auto-locked config.ts for session abc123

# Session B (동시)
$ AI: config.ts 수정
# ⚠ File is locked by session abc123
# Waiting for release...
```

### 세션 조정

#### 세션 마스터

```bash
# 주 세션 지정
entire session set-master abc123

# 출력:
# ✓ Session abc123 set as master
#
# Master session benefits:
#   - Priority in conflict resolution
#   - Other sessions can sync from master
#   - Master merges first

# 다른 세션들이 마스터와 동기화
entire session sync-from-master def456

# 출력:
# Syncing def456 from master abc123...
# ✓ Pulled 3 checkpoints
# ✓ Updated 5 files
```

#### 세션 팀

```bash
# 관련 세션을 그룹으로 관리
entire session team create backend-team \
  --sessions abc123,def456,ghi789

# 출력:
# ✓ Team created: backend-team
#
# Sessions:
#   abc123 - 인증 시스템
#   def456 - 결제 시스템
#   ghi789 - 주문 시스템
#
# Team operations available:
#   - Sync all sessions
#   - Merge team sessions
#   - View team conflicts

# 팀 단위 작업
entire session team sync backend-team
entire session team conflicts backend-team
entire session team merge backend-team
```

### 세션 통계

```bash
# 전체 세션 통계
entire session stats --all

# 출력:
# Multi-Session Statistics
#
# Total sessions: 4
#   Active: 3
#   Paused: 1
#
# Sessions by strategy:
#   manual-commit: 2
#   auto-commit: 2
#
# Total checkpoints: 23
# Total commits: 45
# Total files changed: 67
#
# Conflicts:
#   Detected: 5
#   Resolved: 3
#   Pending: 2
#
# Session interaction:
#   Independent: 1 (jkl012)
#   Shared files: 3 sessions
#   Dependencies: 3 chains
#
# Merge readiness:
#   Ready: 1 (abc123)
#   Needs sync: 2 (def456, ghi789)
#   Blocked: 1 (jkl012 - paused)
```

### 실전 시나리오

#### 시나리오 1: 기능 팀 협업

```bash
# 3명이 하나의 큰 기능 개발

# 개발자 A: 백엔드
git checkout -b feature/backend
entire session start --strategy auto-commit \
  --message "백엔드 API"
# Session: aaa111

# 개발자 B: 프론트엔드
git checkout -b feature/frontend
entire session start --strategy auto-commit \
  --message "프론트엔드 UI"
# Session: bbb222

# 개발자 C: 데이터베이스
git checkout -b feature/database
entire session start --strategy auto-commit \
  --message "데이터베이스 스키마"
# Session: ccc333

# 팀 생성
entire session team create feature-team \
  --sessions aaa111,bbb222,ccc333

# 정기적 동기화
entire session team sync feature-team

# 최종 병합
entire session team merge feature-team \
  --into feature-complete
```

#### 시나리오 2: 실험적 접근

```bash
# 여러 구현 방식 동시 시도

# 접근 A: REST API
entire session start --strategy manual-commit \
  --message "REST API 구현"
# Session: rest111

# 접근 B: GraphQL
entire session start --strategy manual-commit \
  --message "GraphQL 구현"
# Session: grql222

# 접근 C: gRPC
entire session start --strategy manual-commit \
  --message "gRPC 구현"
# Session: grpc333

# 각 접근 개발
entire session switch rest111
$ AI: REST API 만들어줘

entire session switch grql222
$ AI: GraphQL API 만들어줘

entire session switch grpc333
$ AI: gRPC API 만들어줘

# 비교
entire session compare rest111 grql222 grpc333

# 최적 선택
entire session merge grql222 --into main
entire session archive rest111
entire session archive grpc333
```

#### 시나리오 3: 버그 수정 + 기능 개발

```bash
# 긴급 버그 수정 중 새 기능 요청

# 현재: 기능 개발 중
entire session current
# Session: feat123 (active)

# 긴급 버그 발생
git stash  # 현재 작업 저장
git checkout -b hotfix/critical-bug

entire session start --strategy auto-commit \
  --message "긴급 버그 수정"
# Session: hotfix456

$ AI: 버그 고쳐줘
# ... 수정 완료 ...

# 병합 및 배포
git checkout main
git merge hotfix/critical-bug
git push

# 원래 작업 복귀
git checkout feature/new-feature
git stash pop
entire session switch feat123
# 기능 개발 계속
```

### 베스트 프랙티스

```bash
# 1. 명확한 세션 범위
entire session start --message "구체적이고 좁은 범위"

# 2. 의존성 명시
entire session depend <child> --on <parent>

# 3. 정기적 동기화
entire session sync-from-master <session>

# 4. 충돌 조기 감지
entire session conflicts  # 자주 확인

# 5. 팀 단위 관리
entire session team create <name> --sessions <list>

# 6. 완료된 세션 정리
entire session archive <completed-session>

# 7. 병합 순서 준수
entire session dependencies  # 순서 확인
```

### 다음 장 예고

다음 장에서는 **Git 통합**을 다룹니다.

- Git Hooks 연동
- Worktree 지원
- Git 명령어 통합
- 브랜치 전략

---

**관련 문서**:
- [09. Session 관리](/2026/02/11/entire-cli-guide-09-session-management/)
- [11. Checkpoint ID 연결](/2026/02/11/entire-cli-guide-11-checkpoint-linking/)
- [13. Git 통합](/2026/02/11/entire-cli-guide-13-git-integration/)
