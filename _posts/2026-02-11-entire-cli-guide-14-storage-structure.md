---
layout: post
title: "Entire CLI 완벽 가이드 - 14. Storage 구조"
date: 2026-02-11
categories: [AI, Development Tools]
tags: [Entire, CLI, AI Coding, Storage, Git, Architecture]
description: "Entire CLI의 Shadow branch, Metadata branch 및 파일 시스템 저장 구조"
---

## 14. Storage 구조

Entire CLI는 모든 데이터를 Git 저장소 내에 저장합니다. 이 장에서는 shadow branch, metadata branch, 파일 시스템 구조를 상세히 다룹니다.

### 전체 구조 개요

```
Git 저장소 구조:

.git/
├── refs/
│   ├── heads/
│   │   ├── main                      (일반 브랜치)
│   │   ├── feature/auth
│   │   └── entire/
│   │       └── shadow/
│   │           ├── session-abc123   (Shadow branch)
│   │           └── session-def456
│   └── entire/
│       └── metadata/
│           ├── session-abc123       (Metadata branch)
│           └── session-def456
├── objects/                          (Git 객체 저장소)
└── worktrees/                        (Worktree 관리)

.entire/                              (로컬 캐시 및 설정)
├── config.json
├── cache/
├── logs/
└── worktrees/
    ├── session-abc123/
    └── session-def456/
```

### Shadow Branch 구조

#### 브랜치 네이밍

```
Shadow branch 패턴:
refs/heads/entire/shadow/session-<session-id>

예시:
refs/heads/entire/shadow/session-abc123def456
refs/heads/entire/shadow/session-4a7b3c9e2f1d
```

#### Shadow Branch 내용

```bash
# Shadow branch는 일반 코드 브랜치
git checkout entire/shadow/session-abc123

# 파일 구조는 메인 브랜치와 동일
ls -la
# src/
# package.json
# README.md
# ...

# Git 히스토리
git log --oneline
# 3d8k5m2 AI Assistant: Add OAuth integration
# 9a7f3c2 AI Assistant: Implement user authentication
# 7f4a2b1 Initial commit
```

#### Shadow Commit 형식

```bash
# Shadow branch의 커밋 메시지
git log --format=full

# 출력:
# commit 3d8k5m2a1b4c5e6f
# Author: Entire AI Assistant <ai@entire.dev>
# Commit: Entire AI Assistant <ai@entire.dev>
#
#     Add OAuth integration
#
#     Session: abc123def456
#     Checkpoint: 4a7b3c9e
#     Type: temporary
#     Timestamp: 2026-02-11T12:30:00+09:00
#
#     Prompt: "OAuth 프로바이더 추가해줘"
#     Tokens: 3,456
#     Model: claude-opus-4-6
```

### Metadata Branch 구조

#### Orphan Branch

Metadata branch는 코드와 독립적인 orphan branch입니다.

```bash
# Metadata branch 확인
git checkout entire/metadata/session-abc123

# 출력:
# Switched to branch 'entire/metadata/session-abc123'
# Note: This is an orphan branch (no common history with other branches)

# 부모 커밋 없음
git log --oneline
# 1a2b3c4 Update checkpoint 6h4j2p
# 5d6e7f8 Add checkpoint 4a7b3c
# 9g0h1i2 Initialize session abc123

git rev-list --max-parents=0 HEAD
# 9g0h1i2 (첫 커밋, 부모 없음)
```

#### 파일 구조

```bash
# Metadata branch 파일 구조
git ls-tree -r entire/metadata/session-abc123

# 출력:
100644 blob a1b2c3d4  session.json
100644 blob e5f6g7h8  checkpoints.json
100644 blob i9j0k1l2  config.json
100644 blob m3n4o5p6  statistics.json
100644 blob q7r8s9t0  prompts/001.txt
100644 blob u1v2w3x4  prompts/002.txt
100644 blob y5z6a7b8  prompts/003.txt
100644 blob c9d0e1f2  responses/001.txt
100644 blob g3h4i5j6  responses/002.txt
100644 blob k7l8m9n0  responses/003.txt
100644 blob o1p2q3r4  context/001.json
100644 blob s5t6u7v8  context/002.json
100644 blob w9x0y1z2  context/003.json
```

#### 각 파일 상세

**session.json** (세션 메타데이터):
```json
{
  "version": "1.0",
  "id": "abc123def456",
  "created": "2026-02-11T10:00:00+09:00",
  "updated": "2026-02-11T14:30:00+09:00",
  "status": "active",
  "strategy": "manual-commit",

  "message": "사용자 인증 시스템 구현",
  "description": "OAuth 2.0 기반 인증 시스템",
  "tags": ["auth", "security", "backend"],

  "git": {
    "baseBranch": "main",
    "baseCommit": "7f4a2b1c3d5e6f8g",
    "currentBranch": "main",
    "shadowBranch": "entire/shadow/session-abc123def456",
    "metadataBranch": "entire/metadata/session-abc123def456"
  },

  "author": {
    "name": "Developer",
    "email": "dev@example.com"
  },

  "customMetadata": {
    "jira": "PROJ-123",
    "priority": "high",
    "sprint": "2026-Q1"
  }
}
```

**checkpoints.json** (체크포인트 인덱스):
```json
{
  "version": "1.0",
  "sessionId": "abc123def456",
  "checkpoints": [
    {
      "id": "4a7b3c9e2f1d",
      "type": "committed",
      "created": "2026-02-11T12:00:00+09:00",
      "message": "기본 인증 완료",
      "description": "이메일/비밀번호 로그인 구현",

      "git": {
        "commit": "9a7f3c2d5e6f7g8h",
        "branch": "entire/shadow/session-abc123def456",
        "parent": "7f4a2b1c3d5e6f8g",
        "filesChanged": 12,
        "insertions": 456,
        "deletions": 23
      },

      "conversation": {
        "promptFile": "prompts/001.txt",
        "responseFile": "responses/001.txt",
        "contextFile": "context/001.json"
      },

      "tags": ["milestone", "auth", "feature"],

      "links": {
        "previous": null,
        "next": "6h4j2p3k8m2n",
        "related": [],
        "dependsOn": []
      },

      "metadata": {
        "tested": true,
        "reviewed": false
      }
    },
    {
      "id": "6h4j2p3k8m2n",
      "type": "committed",
      "created": "2026-02-11T13:00:00+09:00",
      "message": "OAuth 통합",
      "description": "Google, GitHub OAuth 프로바이더 추가",

      "git": {
        "commit": "3d8k5m2a1b4c5e6f",
        "branch": "entire/shadow/session-abc123def456",
        "parent": "9a7f3c2d5e6f7g8h",
        "filesChanged": 8,
        "insertions": 234,
        "deletions": 12
      },

      "conversation": {
        "promptFile": "prompts/002.txt",
        "responseFile": "responses/002.txt",
        "contextFile": "context/002.json"
      },

      "tags": ["milestone", "oauth", "integration"],

      "links": {
        "previous": "4a7b3c9e2f1d",
        "next": null,
        "related": ["4a7b3c9e2f1d"],
        "dependsOn": ["4a7b3c9e2f1d"]
      },

      "metadata": {
        "tested": true,
        "reviewed": true
      }
    }
  ]
}
```

**prompts/001.txt** (사용자 프롬프트):
```
사용자 인증 시스템을 구현해줘.

요구사항:
- 이메일/비밀번호 로그인
- JWT 토큰 기반 인증
- 비밀번호 해싱 (bcrypt)
- 토큰 갱신 메커니즘

참고:
- Express.js 사용
- TypeScript로 작성
- 단위 테스트 포함
```

**responses/001.txt** (AI 응답):
```
사용자 인증 시스템을 구현하겠습니다.

## 구현 계획

1. 인증 서비스 (src/auth/service.ts)
2. 인증 미들웨어 (src/auth/middleware.ts)
3. 인증 라우트 (src/routes/auth.ts)
4. 타입 정의 (types/auth.ts)

## 상세 구현

### 1. Auth Service

[... AI의 상세한 설명 ...]

### 2. Middleware

[... 코드 설명 ...]

### 3. Routes

[... 엔드포인트 설명 ...]

## 생성된 파일

- src/auth/service.ts (234 lines)
- src/auth/middleware.ts (67 lines)
- src/routes/auth.ts (123 lines)
- types/auth.ts (45 lines)
- test/auth.test.ts (89 lines)

## 다음 단계

OAuth 통합을 추가하시겠습니까?
```

**context/001.json** (실행 컨텍스트):
```json
{
  "timestamp": "2026-02-11T12:00:00+09:00",
  "checkpointId": "4a7b3c9e2f1d",

  "model": {
    "name": "claude-opus-4-6",
    "temperature": 0.7,
    "maxTokens": 4096
  },

  "tokens": {
    "prompt": 1234,
    "response": 2222,
    "total": 3456,
    "cached": 567
  },

  "timing": {
    "started": "2026-02-11T11:59:52+09:00",
    "completed": "2026-02-11T12:00:08+09:00",
    "duration": 8.3
  },

  "files": {
    "read": [
      "package.json",
      "src/config/index.ts",
      "types/user.ts"
    ],
    "created": [
      "src/auth/service.ts",
      "src/auth/middleware.ts",
      "src/routes/auth.ts",
      "types/auth.ts",
      "test/auth.test.ts"
    ],
    "modified": [],
    "deleted": []
  },

  "tools": [
    {
      "name": "Read",
      "calls": 3,
      "files": ["package.json", "src/config/index.ts", "types/user.ts"]
    },
    {
      "name": "Write",
      "calls": 5,
      "files": [
        "src/auth/service.ts",
        "src/auth/middleware.ts",
        "src/routes/auth.ts",
        "types/auth.ts",
        "test/auth.test.ts"
      ]
    }
  ],

  "environment": {
    "platform": "linux",
    "node": "v20.10.0",
    "npm": "10.2.3",
    "git": "2.40.0"
  }
}
```

**config.json** (세션 설정):
```json
{
  "version": "1.0",
  "sessionId": "abc123def456",

  "strategy": "manual-commit",
  "commitStyle": "conventional",
  "useWorktree": false,

  "autoCheckpoint": {
    "enabled": true,
    "interval": "every-5-commits"
  },

  "cleanup": {
    "enabled": true,
    "keepTemporary": 20,
    "keepCommitted": "all",
    "olderThan": "30d"
  },

  "sync": {
    "autoSync": false,
    "syncBranch": "main",
    "syncInterval": null
  }
}
```

**statistics.json** (세션 통계):
```json
{
  "version": "1.0",
  "sessionId": "abc123def456",
  "updated": "2026-02-11T14:30:00+09:00",

  "time": {
    "created": "2026-02-11T10:00:00+09:00",
    "totalDuration": 16200,
    "activeDuration": 14400,
    "idleDuration": 1800
  },

  "checkpoints": {
    "total": 15,
    "committed": 5,
    "temporary": 10,
    "averageInterval": 1080
  },

  "commits": {
    "total": 15,
    "additions": 2345,
    "deletions": 456,
    "filesChanged": 67
  },

  "conversations": {
    "prompts": 18,
    "averagePromptLength": 145,
    "averageResponseLength": 2345,
    "totalTokens": 62340
  },

  "files": {
    "created": 34,
    "modified": 23,
    "deleted": 2,
    "byType": {
      "ts": 28,
      "json": 3,
      "md": 3,
      "test.ts": 12
    }
  }
}
```

### 로컬 파일 시스템 구조

#### .entire 디렉토리

```bash
# 로컬 캐시 및 설정
.entire/
├── config.json              (전역 설정)
├── sessions.db              (세션 인덱스)
├── cache/
│   ├── checkpoints/         (Checkpoint 캐시)
│   ├── conversations/       (대화 캐시)
│   └── index/               (검색 인덱스)
├── logs/
│   ├── entire.log           (일반 로그)
│   ├── sessions.log         (세션 로그)
│   └── errors.log           (에러 로그)
├── temp/
│   └── [임시 파일]
└── worktrees/
    ├── session-abc123/      (Worktree 디렉토리)
    └── session-def456/
```

**config.json** (전역 설정):
```json
{
  "version": "1.0",
  "installation": {
    "date": "2026-02-10T15:00:00+09:00",
    "version": "1.0.0"
  },

  "defaults": {
    "strategy": "manual-commit",
    "commitStyle": "conventional",
    "useWorktree": false,
    "autoCheckpoint": "every-5-commits"
  },

  "git": {
    "shadowBranchPrefix": "entire/shadow",
    "metadataBranchPrefix": "entire/metadata",
    "author": {
      "name": "Entire AI Assistant",
      "email": "ai@entire.dev"
    }
  },

  "cache": {
    "enabled": true,
    "maxSize": "500MB",
    "ttl": 86400
  },

  "ai": {
    "model": "claude-opus-4-6",
    "temperature": 0.7,
    "maxTokens": 4096
  }
}
```

**sessions.db** (SQLite 데이터베이스):
```sql
-- 세션 인덱스
CREATE TABLE sessions (
  id TEXT PRIMARY KEY,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  status TEXT NOT NULL,
  strategy TEXT NOT NULL,
  message TEXT,
  base_branch TEXT,
  shadow_branch TEXT,
  metadata_branch TEXT
);

CREATE INDEX idx_sessions_status ON sessions(status);
CREATE INDEX idx_sessions_updated ON sessions(updated_at);

-- Checkpoint 인덱스
CREATE TABLE checkpoints (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  type TEXT NOT NULL,
  message TEXT,
  commit_sha TEXT,
  FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX idx_checkpoints_session ON checkpoints(session_id);
CREATE INDEX idx_checkpoints_type ON checkpoints(type);

-- 전문 검색 테이블
CREATE VIRTUAL TABLE conversations_fts USING fts5(
  checkpoint_id,
  prompt,
  response,
  content=''
);
```

### 객체 저장소

#### Git Objects

모든 데이터는 결국 Git 객체로 저장됩니다.

```bash
# Shadow branch 커밋 객체
git cat-file -p 3d8k5m2a1b4c5e6f

# 출력:
# tree 7a8b9c0d1e2f3g4h
# parent 9a7f3c2d5e6f7g8h
# author Entire AI Assistant <ai@entire.dev> 1707627000 +0900
# committer Entire AI Assistant <ai@entire.dev> 1707627000 +0900
#
# Add OAuth integration
#
# Session: abc123def456
# Checkpoint: 6h4j2p3k8m2n
# [...]

# Tree 객체
git cat-file -p 7a8b9c0d1e2f3g4h

# 출력:
# 040000 tree 1i2j3k4l5m6n  src
# 100644 blob 7o8p9q0r1s2t  package.json
# 100644 blob 3u4v5w6x7y8z  README.md

# Blob 객체 (파일 내용)
git cat-file -p 1a2b3c4d5e6f

# 출력:
# [파일 실제 내용]
```

#### Object Packing

```bash
# Git이 자동으로 객체를 pack
git count-objects -v

# 출력:
# count: 1234
# size: 45678
# in-pack: 5678
# packs: 2
# size-pack: 12345
# prune-packable: 0
# garbage: 0
# size-garbage: 0

# Pack 파일 확인
ls -lh .git/objects/pack/
# pack-abc123...pack  (12 MB)
# pack-abc123...idx   (45 KB)
# pack-def456...pack  (8 MB)
# pack-def456...idx   (32 KB)
```

### 데이터 크기 최적화

#### 압축

```bash
# Git이 자동 압축
# Blob 객체는 zlib으로 압축됨

# 예시: 100KB 파일 → 15KB 압축
echo "Large file content..." > large.txt
git add large.txt
git commit -m "Add large file"

# 압축된 크기 확인
git cat-file -s HEAD:large.txt
# 102400 (원본)

git cat-file blob HEAD:large.txt | wc -c
# 15234 (압축 후, 디스크)
```

#### Delta 압축

```bash
# 유사한 파일은 delta로 저장
# 파일을 여러 번 수정해도 전체를 다시 저장하지 않음

# 예시
echo "Version 1" > file.txt
git add file.txt && git commit -m "v1"

echo "Version 2" >> file.txt
git add file.txt && git commit -m "v2"

echo "Version 3" >> file.txt
git add file.txt && git commit -m "v3"

# Git이 delta chain 생성
git verify-pack -v .git/objects/pack/pack-*.idx | grep file.txt
# [객체 ID] blob 100 50 12345  # v3 (base)
# [객체 ID] blob 90 30 12395 1 [base ID]  # v2 (delta)
# [객체 ID] blob 80 20 12425 2 [base ID]  # v1 (delta)
```

### 스토리지 통계

```bash
# 전체 스토리지 사용량
entire storage stats

# 출력:
# Entire CLI Storage Statistics
#
# Git Repository:
#   Total size: 145.3 MB
#   Objects: 12,345
#   Packed: 9,876 (80%)
#   Loose: 2,469 (20%)
#
# Shadow Branches:
#   Count: 5
#   Total size: 23.4 MB
#   Average: 4.7 MB
#
# Metadata Branches:
#   Count: 5
#   Total size: 12.1 MB
#   Average: 2.4 MB
#
# Local Cache:
#   Checkpoints: 3.5 MB
#   Conversations: 8.9 MB
#   Index: 1.2 MB
#   Total: 13.6 MB
#
# Worktrees:
#   Count: 2
#   Total size: 89.2 MB
#   Average: 44.6 MB
#
# Breakdown by Session:
#   abc123: 45.2 MB (shadow: 12.3 MB, metadata: 5.4 MB, worktree: 27.5 MB)
#   def456: 38.7 MB (shadow: 11.1 MB, metadata: 6.9 MB, worktree: 20.7 MB)
#   [...]
```

### 정리 및 최적화

#### 자동 정리

```bash
# 주기적으로 자동 실행
git gc --auto

# 수동 실행
git gc --aggressive --prune=now

# 출력:
# Enumerating objects: 12345
# Counting objects: 100%
# Compressing objects: 100%
# Writing objects: 100%
# Removing duplicate objects: 100%
#
# Before: 145.3 MB
# After: 89.7 MB
# Saved: 55.6 MB (38%)
```

#### Entire 전용 정리

```bash
# 오래된 임시 데이터 정리
entire storage cleanup

# 출력:
# Analyzing storage...
#
# Cleanup candidates:
#   Old temporary checkpoints: 45 (7.8 MB)
#   Cached conversations: 123 (4.2 MB)
#   Expired indexes: 8 (1.1 MB)
#   Orphan worktrees: 1 (34.5 MB)
#
# Total recoverable: 47.6 MB
#
# Proceed? (y/N): y
#
# ✓ Deleted 45 temporary checkpoints
# ✓ Cleared 123 cached conversations
# ✓ Removed 8 expired indexes
# ✓ Deleted 1 orphan worktree
#
# Storage freed: 47.6 MB
```

### 백업 및 복원

#### 메타데이터 백업

```bash
# 모든 메타데이터 브랜치 백업
git push backup refs/entire/metadata/*:refs/entire/metadata/*

# 또는 Entire 명령 사용
entire backup create --destination backup.entire

# 출력:
# Creating backup...
#
# Sessions: 5
# Checkpoints: 67
# Conversations: 234
# Metadata size: 45.3 MB
#
# Compressing...
# ✓ Backup created: backup.entire (12.1 MB compressed)
```

#### 복원

```bash
# 백업에서 복원
entire backup restore backup.entire

# 출력:
# Restoring from backup.entire...
#
# Sessions: 5
# Checkpoints: 67
# Conversations: 234
#
# This will:
#   - Restore all sessions
#   - Recreate metadata branches
#   - Rebuild local cache
#
# Existing data will be preserved.
#
# Continue? (y/N): y
#
# ✓ Sessions restored: 5
# ✓ Checkpoints restored: 67
# ✓ Conversations restored: 234
# ✓ Cache rebuilt
#
# Restoration complete!
```

### 베스트 프랙티스

```bash
# 1. 정기적인 Git GC
git gc --auto  # 자동 실행되지만 확인

# 2. 메타데이터 백업
entire backup create --auto

# 3. 오래된 데이터 정리
entire storage cleanup --auto

# 4. 스토리지 모니터링
entire storage stats  # 주기적 확인

# 5. 대용량 파일은 Git LFS 사용
git lfs track "*.psd"

# 6. Worktree 정리
entire session stop --cleanup-worktree
```

### 다음 장 예고

다음 장에서는 **Claude Code Hooks**를 다룹니다.

- SessionStart hook
- UserPromptSubmit hook
- Stop hook
- 커스텀 hook 작성

---

**관련 문서**:
- [07. Manual-Commit Strategy](/2026/02/11/entire-cli-guide-07-manual-commit/)
- [10. Checkpoint 시스템](/2026/02/11/entire-cli-guide-10-checkpoint-system/)
- [13. Git 통합](/2026/02/11/entire-cli-guide-13-git-integration/)
