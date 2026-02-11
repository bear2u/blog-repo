---
layout: post
title: "Entire CLI 완벽 가이드 - 09. Session 관리"
date: 2026-02-11
categories: [AI, Development Tools]
tags: [Entire, CLI, AI Coding, Session Management, Tracking]
description: "Entire CLI Session 생성, 추적, 조회 및 다중 세션 관리"
---

## 09. Session 관리

Session은 Entire CLI에서 AI와의 작업을 추적하고 관리하는 기본 단위입니다. 이 장에서는 세션의 생성, 추적, 조회 및 관리 방법을 상세히 다룹니다.

### Session 개념

```
Session의 역할:
┌───────────────────────────────────────────┐
│ Session = AI와의 하나의 작업 단위        │
│                                           │
│ 포함 내용:                                │
│ • 모든 대화 (프롬프트 + 응답)            │
│ • 모든 코드 변경사항                      │
│ • 모든 Checkpoint                         │
│ • 메타데이터 (시간, 전략, 브랜치 등)     │
└───────────────────────────────────────────┘

생명주기:
시작 → 활성 → 일시정지 → 재개 → 종료 → 보관
```

### Session ID 체계

#### ID 생성

```bash
# Session ID는 12자리 16진수
# 예시: abc123def456

생성 방식:
1. 타임스탬프 기반 (첫 8자리)
2. 랜덤 데이터 (마지막 4자리)
3. 충돌 방지 검증

형식:
[timestamp:8][random:4]
 ↓         ↓
 65e9a1b3  c2f4
```

#### ID 사용

```bash
# 1. 전체 ID 사용
entire session info abc123def456

# 2. 짧은 형태 (처음 6자리)
entire session info abc123

# 3. 충돌 방지 (같은 앞 6자리가 있으면 경고)
entire session info abc123
# Multiple sessions match 'abc123':
#   abc123def456
#   abc123789abc
# Please use more characters.

# 4. 패턴 매칭
entire session list --filter "abc*"
```

### Session 생성

#### 기본 생성

```bash
# 기본 설정으로 생성
entire session start

# 출력:
# ✓ Session created: abc123def456
# ✓ Strategy: manual-commit (default)
# ✓ Base branch: main
# ✓ Base commit: 9a7f3c2
#
# Start chatting with AI!
```

#### 고급 생성

```bash
# 모든 옵션 지정
entire session start \
  --message "사용자 인증 시스템 구현" \
  --strategy auto-commit \
  --branch feature/auth \
  --tags "auth,security,backend" \
  --metadata '{"priority":"high","sprint":"Q1-2026"}' \
  --description "OAuth 2.0 기반 인증 시스템 개발"

# 출력:
# ✓ Session created: def456789abc
# ✓ Strategy: auto-commit
# ✓ Branch: feature/auth
# ✓ Tags: auth, security, backend
# ✓ Metadata: priority=high, sprint=Q1-2026
#
# Description:
#   OAuth 2.0 기반 인증 시스템 개발
```

#### 템플릿 기반 생성

```bash
# 자주 사용하는 설정을 템플릿으로 저장
entire template create feature-dev \
  --strategy manual-commit \
  --use-worktree \
  --commit-style conventional \
  --tags "feature,development"

# 템플릿 사용
entire session start --template feature-dev \
  --message "결제 시스템 개발"

# 또 다른 템플릿
entire template create quick-fix \
  --strategy auto-commit \
  --commit-style concise \
  --tags "bugfix,hotfix"

entire session start --template quick-fix \
  --message "긴급 버그 수정"
```

### Session 상태

#### 상태 종류

```
Available States:
┌─────────────────────────────────────────┐
│ active     현재 진행 중                 │
│ paused     일시 정지                    │
│ stopped    종료됨 (재개 가능)          │
│ archived   보관됨 (읽기 전용)          │
│ deleted    삭제됨 (복구 불가)          │
└─────────────────────────────────────────┘
```

#### 상태 전환

```bash
# 1. Active → Paused
entire session pause
# ✓ Session abc123 paused
# You can switch to other sessions.

# 2. Paused → Active
entire session resume abc123
# ✓ Session abc123 resumed
# Continue working where you left off.

# 3. Active → Stopped
entire session stop
# ✓ Session abc123 stopped
# All changes are saved.

# 4. Stopped → Archived
entire session archive abc123
# ✓ Session abc123 archived
# Metadata preserved, session is now read-only.

# 5. Any → Deleted
entire session delete abc123
# ⚠ This will permanently delete:
#   - All checkpoints
#   - All metadata
#   - Conversation history
#
# Shadow branch will be preserved.
#
# Continue? (y/N):
```

### Session 정보 조회

#### 기본 정보

```bash
# 현재 세션 정보
entire session info

# 출력:
# Session: abc123def456
# Status: active
# Strategy: manual-commit
#
# Created: 2026-02-11 10:30:00 +0900 (2 hours ago)
# Updated: 2026-02-11 12:15:00 +0900 (15 minutes ago)
#
# Message: 사용자 인증 시스템 구현
# Description: OAuth 2.0 기반 인증 시스템 개발
#
# Branch: feature/auth
# Base Commit: 9a7f3c2 (main)
# Shadow Branch: entire/shadow/session-abc123def456
#
# Tags: auth, security, backend
# Metadata:
#   priority: high
#   sprint: Q1-2026
#
# Statistics:
#   Checkpoints: 5 (3 committed, 2 temporary)
#   Commits: 12
#   Files changed: 23
#   Lines added: 1,245
#   Lines deleted: 387
#   AI interactions: 18
#   Duration: 2h 15m
```

#### 상세 정보

```bash
# 모든 세부 정보 포함
entire session info --verbose

# 추가 출력:
#
# Checkpoints:
#   ★ 4a7b3c - 기본 인증 완료 (2h ago)
#   ★ 9e2f1d - OAuth 통합 (1h 30m ago)
#   ★ 6h4j2p - 권한 관리 추가 (45m ago)
#     3k8m2n - UI 개선 (30m ago)
#     7f4a2b - 테스트 추가 (15m ago)
#
# Recent AI Interactions:
#   [12:00] User: OAuth 프로바이더 추가해줘
#   [12:01] AI: Google, GitHub OAuth 추가했습니다
#   [12:10] User: 에러 처리 개선해줘
#   [12:11] AI: 상세 에러 메시지와 로깅 추가했습니다
#   [12:15] User: 테스트 작성해줘
#   [12:16] AI: 단위 테스트 15개 추가했습니다
#
# Files Modified:
#   src/auth/oauth.ts (created)
#   src/auth/providers/google.ts (created)
#   src/auth/providers/github.ts (created)
#   src/auth/errors.ts (modified)
#   test/auth/oauth.test.ts (created)
#   ... (18 more files)
```

### Session 목록

#### 기본 목록

```bash
# 모든 세션 조회
entire session list

# 출력:
# Active Sessions (2):
#   abc123 - 사용자 인증 시스템 구현
#            feature/auth | manual-commit | 2h 15m ago
#
#   def456 - 결제 시스템 개발
#            feature/payment | auto-commit | 30m ago
#
# Paused Sessions (1):
#   ghi789 - 문서 업데이트
#            main | auto-commit | 1 day ago
#
# Recent Stopped Sessions (5):
#   jkl012 - 버그 수정 (2 days ago)
#   mno345 - UI 리팩토링 (3 days ago)
#   pqr678 - 성능 최적화 (5 days ago)
#   stu901 - API 추가 (1 week ago)
#   vwx234 - 데이터베이스 마이그레이션 (1 week ago)
```

#### 필터링

```bash
# 1. 상태별 필터
entire session list --status active
entire session list --status paused
entire session list --status stopped

# 2. 전략별 필터
entire session list --strategy manual-commit
entire session list --strategy auto-commit

# 3. 브랜치별 필터
entire session list --branch main
entire session list --branch "feature/*"

# 4. 태그별 필터
entire session list --tag auth
entire session list --tag bugfix

# 5. 날짜 범위 필터
entire session list --since "2026-02-01"
entire session list --since "7 days ago"
entire session list --until "yesterday"

# 6. 복합 필터
entire session list \
  --status active \
  --strategy manual-commit \
  --tag feature \
  --since "1 week ago"
```

#### 정렬

```bash
# 1. 생성 시간순 (기본)
entire session list --sort created

# 2. 업데이트 시간순
entire session list --sort updated

# 3. 이름순
entire session list --sort name

# 4. 역순 정렬
entire session list --sort updated --reverse
```

### Session 검색

#### 텍스트 검색

```bash
# 메시지나 설명에서 검색
entire session search "인증"

# 출력:
# Found 3 sessions:
#
# abc123 - 사용자 인증 시스템 구현
#   Match: message
#   Branch: feature/auth
#   Created: 2 hours ago
#
# def456 - 백엔드 API 개발
#   Match: description ("...인증 미들웨어...")
#   Branch: feature/api
#   Created: 2 days ago
#
# ghi789 - 보안 강화
#   Match: checkpoint message ("인증 토큰 검증")
#   Branch: main
#   Created: 1 week ago
```

#### 고급 검색

```bash
# 1. 대화 내용 검색
entire session search "OAuth" --in conversations

# 2. 코드 변경 검색
entire session search "class User" --in code

# 3. 파일 이름 검색
entire session search "auth.ts" --in files

# 4. 복합 검색
entire session search \
  --message "인증" \
  --tag security \
  --since "1 month ago" \
  --has-checkpoints
```

### Session 전환

#### 빠른 전환

```bash
# 1. 현재 세션 일시정지
entire session pause

# 2. 다른 세션으로 전환
entire session switch def456

# 출력:
# ✓ Session abc123 paused
# ✓ Session def456 resumed
#
# Now working on: 결제 시스템 개발
# Strategy: auto-commit
# Branch: feature/payment

# 3. 다시 원래 세션으로
entire session switch abc123
```

#### 안전한 전환

```bash
# 변경사항 있으면 경고
entire session switch def456

# 출력:
# ⚠ Current session has uncommitted changes:
#   modified: src/auth/index.ts
#   new file: src/auth/oauth.ts
#
# Options:
#   [c] Commit changes before switching
#   [s] Stash changes
#   [d] Discard changes
#   [a] Abort switching
#
# Choose:
```

### Session 병합

여러 세션의 작업을 통합:

```bash
# 두 세션 병합
entire session merge abc123 def456 \
  --into ghi789 \
  --message "인증과 결제 시스템 통합"

# 출력:
# Merging sessions...
#
# Source sessions:
#   abc123 - 사용자 인증 시스템 (12 commits)
#   def456 - 결제 시스템 (8 commits)
#
# Target session:
#   ghi789 - 통합 시스템 (new)
#
# This will:
#   - Combine all checkpoints
#   - Merge conversation histories
#   - Preserve all metadata
#   - Create new merged commits
#
# Continue? (y/N): y
#
# ✓ Sessions merged successfully
# ✓ New session: ghi789
# ✓ Total checkpoints: 20
# ✓ Original sessions preserved
```

### Session 복제

기존 세션을 기반으로 새 세션:

```bash
# 세션 복제
entire session clone abc123 \
  --message "인증 시스템 v2" \
  --branch feature/auth-v2

# 출력:
# Cloning session abc123...
#
# ✓ New session created: jkl012
# ✓ All checkpoints copied
# ✓ Conversation history copied
# ✓ Metadata copied
#
# New session details:
#   ID: jkl012
#   Message: 인증 시스템 v2
#   Branch: feature/auth-v2
#   Based on: abc123 (at checkpoint 6h4j2p)
```

### Session 비교

두 세션의 차이 분석:

```bash
# 세션 비교
entire session compare abc123 def456

# 출력:
# Comparing sessions:
#   abc123 - 사용자 인증 시스템
#   def456 - 결제 시스템
#
# Common Base: 9a7f3c2 (main)
#
# Divergence:
#   abc123: 12 commits, 23 files changed
#   def456: 8 commits, 15 files changed
#
# File Conflicts: 2
#   src/config/index.ts (both modified)
#   src/utils/api.ts (both modified)
#
# Unique to abc123:
#   src/auth/*
#   test/auth/*
#
# Unique to def456:
#   src/payment/*
#   test/payment/*
#
# Shared changes:
#   package.json (different versions)
#   src/config/index.ts (conflict)
```

### Session 메타데이터

#### 구조

```json
{
  "id": "abc123def456",
  "version": "1.0",
  "status": "active",
  "strategy": "manual-commit",

  "created": "2026-02-11T10:30:00+09:00",
  "updated": "2026-02-11T12:45:00+09:00",
  "duration": 8100,

  "message": "사용자 인증 시스템 구현",
  "description": "OAuth 2.0 기반 인증 시스템 개발",

  "git": {
    "baseBranch": "main",
    "baseCommit": "9a7f3c2",
    "currentBranch": "feature/auth",
    "shadowBranch": "entire/shadow/session-abc123def456",
    "metadataBranch": "entire/metadata/session-abc123def456"
  },

  "tags": ["auth", "security", "backend"],

  "customMetadata": {
    "priority": "high",
    "sprint": "Q1-2026",
    "assignee": "john@example.com",
    "reviewers": ["jane@example.com"]
  },

  "statistics": {
    "checkpoints": 5,
    "checkpointsCommitted": 3,
    "checkpointsTemporary": 2,
    "commits": 12,
    "filesChanged": 23,
    "linesAdded": 1245,
    "linesDeleted": 387,
    "aiInteractions": 18,
    "tokensUsed": 245320
  },

  "config": {
    "commitStyle": "conventional",
    "useWorktree": false,
    "autoCheckpoint": "every-5-commits",
    "batchCommits": null,
    "interactive": false
  }
}
```

#### 메타데이터 수정

```bash
# 1. 메시지 변경
entire session update abc123 \
  --message "OAuth 2.0 인증 시스템"

# 2. 설명 변경
entire session update abc123 \
  --description "Google, GitHub, Facebook OAuth 지원"

# 3. 태그 추가/제거
entire session update abc123 \
  --add-tags "oauth,google,github" \
  --remove-tags "backend"

# 4. 커스텀 메타데이터 수정
entire session update abc123 \
  --set-metadata "priority=critical" \
  --set-metadata "estimate=3d"

# 5. 여러 필드 동시 수정
entire session update abc123 \
  --message "새 메시지" \
  --add-tags "new-tag" \
  --set-metadata "key=value"
```

### Session 내보내기/가져오기

#### 내보내기

```bash
# 1. 기본 내보내기 (메타데이터만)
entire session export abc123 > session.json

# 2. 전체 내보내기 (코드 포함)
entire session export abc123 --full > session-full.tar.gz

# 3. 여러 세션 내보내기
entire session export --tag feature --since "1 week ago" \
  > recent-features.tar.gz

# 4. 특정 checkpoint까지만
entire session export abc123 --until 6h4j2p > session-partial.json
```

#### 가져오기

```bash
# 1. 메타데이터 가져오기
entire session import session.json

# 2. 전체 복원
entire session import session-full.tar.gz --restore-code

# 3. 새 ID로 가져오기
entire session import session.json --new-id

# 4. 다른 브랜치로 가져오기
entire session import session.json --branch feature/imported
```

### Session 통계

#### 개별 통계

```bash
# 세션 통계
entire session stats abc123

# 출력:
# Session: abc123def456
# Message: 사용자 인증 시스템 구현
#
# Time:
#   Created: 2026-02-11 10:30:00 +0900
#   Duration: 2h 15m
#   Active time: 1h 45m (idle: 30m)
#
# Code Changes:
#   Commits: 12
#   Files changed: 23 (18 added, 5 modified)
#   Lines added: 1,245
#   Lines deleted: 387
#   Net change: +858 lines
#
# Checkpoints:
#   Total: 5
#   Committed: 3 (60%)
#   Temporary: 2 (40%)
#   Average interval: 27m
#
# AI Interactions:
#   Total prompts: 18
#   Average response time: 8.3s
#   Tokens used: 245,320
#   Cost estimate: $1.23
#
# Files by Type:
#   TypeScript: 15 files (+1,023 lines)
#   Tests: 5 files (+180 lines)
#   Config: 3 files (+42 lines)
```

#### 전체 통계

```bash
# 모든 세션 통계
entire session stats --all

# 출력:
# Overall Statistics
#
# Sessions:
#   Total: 47
#   Active: 2
#   Paused: 1
#   Stopped: 44
#
# Time:
#   Total duration: 156h 30m
#   Average per session: 3h 20m
#
# Code:
#   Total commits: 342
#   Files changed: 1,247
#   Lines added: 45,678
#   Lines deleted: 12,345
#
# AI:
#   Total interactions: 892
#   Tokens used: 8,234,567
#   Estimated cost: $41.17
#
# Most Productive Day: 2026-02-08 (8 sessions, 4,567 lines)
# Most Used Strategy: auto-commit (68%)
# Most Common Tags: feature (23), bugfix (15), refactor (12)
```

### Session 자동화

#### Git Hooks 연동

```bash
# .git/hooks/pre-commit
#!/bin/bash
# Auto-start session if none active

if ! entire session current > /dev/null 2>&1; then
  echo "No active session. Starting one..."
  entire session start --auto --message "Auto-started session"
fi

# Validate session state
entire session validate || {
  echo "Session validation failed"
  exit 1
}
```

#### CI/CD 통합

```yaml
# .github/workflows/entire.yml
name: Entire Session Analysis

on: [pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install Entire CLI
        run: npm install -g @entire/cli

      - name: Extract session from PR
        run: |
          SESSION_ID=$(gh pr view ${{ github.event.pull_request.number }} \
            --json body --jq '.body | match("Session: ([a-f0-9]+)").captures[0].string')
          echo "SESSION_ID=$SESSION_ID" >> $GITHUB_ENV

      - name: Generate session report
        run: |
          entire session export $SESSION_ID --format markdown > session-report.md
          gh pr comment ${{ github.event.pull_request.number }} \
            --body-file session-report.md
```

### 베스트 프랙티스

```bash
# 1. 의미있는 메시지
entire session start --message "OAuth 2.0 인증 구현"
# 나쁜 예: --message "작업"

# 2. 적절한 태그 사용
--tags "feature,auth,backend"

# 3. 커스텀 메타데이터 활용
--metadata '{"jira":"PROJ-123","priority":"high"}'

# 4. 정기적인 세션 정리
entire session cleanup --older-than "30 days" --status stopped

# 5. 세션 문서화
entire session update abc123 \
  --description "상세한 작업 내용과 결정 사항 기록"

# 6. 백업
entire session export --all --since "1 month ago" > backup.tar.gz
```

### 다음 장 예고

다음 장에서는 **Checkpoint 시스템**을 상세히 다룹니다.

- Temporary vs Committed checkpoint
- Checkpoint 생성 및 복원
- Checkpoint 연결 (linking)
- Checkpoint 기반 시간 여행

---

**관련 문서**:
- [08. Auto-Commit Strategy](/2026/02/11/entire-cli-guide-08-auto-commit/)
- [10. Checkpoint 시스템](/2026/02/11/entire-cli-guide-10-checkpoint-system/)
- [12. Multi-Session 처리](/2026/02/11/entire-cli-guide-12-multi-session/)
