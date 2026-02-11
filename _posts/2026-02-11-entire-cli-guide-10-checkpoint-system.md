---
layout: post
title: "Entire CLI 완벽 가이드 - 10. Checkpoint 시스템"
date: 2026-02-11
categories: [AI, Development Tools]
tags: [Entire, CLI, AI Coding, Checkpoint, Version Control]
description: "Entire CLI Checkpoint의 생성, 관리, 복원 및 시간 여행 기능"
---

## 10. Checkpoint 시스템

Checkpoint는 Entire CLI에서 AI 작업의 특정 시점을 저장하고 추적하는 핵심 메커니즘입니다. Git 커밋과 유사하지만 AI 대화 컨텍스트까지 포함합니다.

### Checkpoint 개념

```
Checkpoint = Git Commit + AI Context

구성 요소:
┌────────────────────────────────────────┐
│ 1. Git Commit SHA                      │
│    코드 변경사항                       │
│                                        │
│ 2. AI Conversation                     │
│    - 사용자 프롬프트                   │
│    - AI 응답                           │
│    - 컨텍스트 데이터                   │
│                                        │
│ 3. Metadata                            │
│    - Checkpoint ID (12-hex)            │
│    - 타임스탬프                        │
│    - 타입 (temporary/committed)        │
│    - 메시지                            │
└────────────────────────────────────────┘
```

### Checkpoint 타입

#### Temporary Checkpoint

자동으로 생성되는 임시 저장점입니다.

```bash
# Auto-commit 전략에서 자동 생성
entire session start --strategy auto-commit

$ AI: 기능 추가해줘
# ✓ Auto-committed: 3d8k5m2
# ✓ Checkpoint: 4a7b3c (temporary)

특징:
- 매 AI 응답마다 자동 생성
- 경량 메타데이터
- 자동 정리 대상
- 빠른 실험용
```

#### Committed Checkpoint

명시적으로 생성하는 영구 저장점입니다.

```bash
# 수동으로 중요한 시점 표시
entire checkpoint create --message "핵심 기능 완성"

# ✓ Checkpoint created: 6h4j2p
# ✓ Type: committed
# ✓ This is a permanent checkpoint

특징:
- 명시적 생성
- 상세 메타데이터
- 영구 보존
- 마일스톤 표시
```

### Checkpoint 생성

#### 기본 생성

```bash
# 현재 상태를 checkpoint로 저장
entire checkpoint create

# 출력:
# ✓ Checkpoint created: 4a7b3c9e
# ✓ Type: committed
# ✓ Commit: 3d8k5m2
# ✓ Files changed: 5
#
# This checkpoint captures the current state.
```

#### 메시지와 함께 생성

```bash
# 의미있는 메시지 추가
entire checkpoint create \
  --message "사용자 인증 완료"

# ✓ Checkpoint created: 9e2f1d6h
# ✓ Message: 사용자 인증 완료
# ✓ Commit: 7f4a2b1
```

#### 상세 메타데이터 포함

```bash
# 모든 정보 포함
entire checkpoint create \
  --message "OAuth 통합 완료" \
  --description "Google, GitHub, Facebook OAuth 프로바이더 추가" \
  --tags "oauth,integration,milestone" \
  --metadata '{"tested":true,"reviewed":true}'

# 출력:
# ✓ Checkpoint created: 6h4j2p3k
# ✓ Message: OAuth 통합 완료
# ✓ Description: Google, GitHub, Facebook OAuth 프로바이더 추가
# ✓ Tags: oauth, integration, milestone
# ✓ Metadata: tested=true, reviewed=true
```

### Checkpoint 구조

#### 메타데이터 파일

```json
{
  "id": "4a7b3c9e",
  "type": "committed",
  "sessionId": "abc123def456",

  "created": "2026-02-11T12:30:00+09:00",
  "author": "user@example.com",

  "message": "사용자 인증 완료",
  "description": "기본 로그인, 회원가입, 비밀번호 재설정 구현",

  "git": {
    "commit": "3d8k5m2a1b4c",
    "branch": "feature/auth",
    "parent": "9a7f3c2",
    "filesChanged": 12,
    "linesAdded": 456,
    "linesDeleted": 23
  },

  "conversation": {
    "promptFile": "prompts/001.txt",
    "responseFile": "responses/001.txt",
    "tokensUsed": 3456,
    "duration": 8.3
  },

  "tags": ["auth", "feature", "milestone"],

  "customMetadata": {
    "tested": true,
    "reviewed": false,
    "priority": "high"
  },

  "links": {
    "previous": "9e2f1d6h",
    "next": "6h4j2p3k",
    "related": ["8r6t3v9p", "2q4w6e8r"]
  }
}
```

#### 대화 내용 저장

```
.git/refs/entire/metadata/session-abc123/
├── checkpoints.json           (인덱스)
├── prompts/
│   ├── 001.txt               (사용자 프롬프트)
│   ├── 002.txt
│   └── 003.txt
├── responses/
│   ├── 001.txt               (AI 응답)
│   ├── 002.txt
│   └── 003.txt
└── context/
    ├── 001.json              (컨텍스트 데이터)
    ├── 002.json
    └── 003.json
```

**prompts/001.txt**:
```
사용자 인증 시스템을 구현해줘.

요구사항:
- 이메일/비밀번호 로그인
- JWT 토큰 기반 인증
- 비밀번호 해싱 (bcrypt)
- 토큰 갱신 메커니즘
```

**responses/001.txt**:
```
사용자 인증 시스템을 구현했습니다.

구현 내용:
1. Auth Service (src/auth/service.ts)
   - 로그인, 회원가입, 토큰 갱신 로직
   - bcrypt를 사용한 비밀번호 해싱
   - JWT 토큰 생성 및 검증

2. Auth Middleware (src/auth/middleware.ts)
   - 토큰 검증 미들웨어
   - 권한 확인 미들웨어

3. Auth Routes (src/routes/auth.ts)
   - POST /auth/register
   - POST /auth/login
   - POST /auth/refresh

[... AI의 전체 설명 ...]

생성된 파일:
- src/auth/service.ts (234 lines)
- src/auth/middleware.ts (67 lines)
- src/routes/auth.ts (123 lines)
- types/auth.ts (45 lines)
```

**context/001.json**:
```json
{
  "timestamp": "2026-02-11T12:30:00+09:00",
  "model": "claude-opus-4-6",
  "temperature": 0.7,
  "maxTokens": 4096,
  "tokensUsed": {
    "prompt": 1234,
    "response": 2222,
    "total": 3456
  },
  "files": {
    "read": [
      "package.json",
      "src/config/index.ts"
    ],
    "created": [
      "src/auth/service.ts",
      "src/auth/middleware.ts",
      "src/routes/auth.ts",
      "types/auth.ts"
    ],
    "modified": []
  }
}
```

### Checkpoint 조회

#### 목록 보기

```bash
# 현재 세션의 모든 checkpoint
entire checkpoint list

# 출력:
# Session: abc123def456
#
# Committed Checkpoints (3):
#   ★ 6h4j2p - OAuth 통합 완료 (1h ago)
#   ★ 4a7b3c - 사용자 인증 완료 (2h ago)
#   ★ 9e2f1d - 프로젝트 구조 완성 (3h ago)
#
# Temporary Checkpoints (5):
#   3k8m2n - Add error handling (30m ago)
#   7f4a2b - Update UI components (45m ago)
#   8r6t3v - Add validation (1h 15m ago)
#   2q4w6e - Fix bug (1h 30m ago)
#   5t7y9u - Improve performance (2h 15m ago)
```

#### 필터링

```bash
# 1. 타입별 필터
entire checkpoint list --type committed
entire checkpoint list --type temporary

# 2. 태그별 필터
entire checkpoint list --tag milestone
entire checkpoint list --tag bugfix

# 3. 날짜 범위 필터
entire checkpoint list --since "2 hours ago"
entire checkpoint list --until "1 hour ago"

# 4. 메시지 검색
entire checkpoint list --message "auth"

# 5. 복합 필터
entire checkpoint list \
  --type committed \
  --tag milestone \
  --since "1 week ago"
```

#### 상세 정보

```bash
# Checkpoint 상세 정보
entire checkpoint info 4a7b3c

# 출력:
# Checkpoint: 4a7b3c9e2f1d
# Type: committed
# Session: abc123def456
#
# Created: 2026-02-11 12:30:00 +0900 (2 hours ago)
# Message: 사용자 인증 완료
# Description: 기본 로그인, 회원가입, 비밀번호 재설정 구현
#
# Git Information:
#   Commit: 3d8k5m2a1b4c
#   Branch: feature/auth
#   Parent: 9a7f3c2
#   Files changed: 12 (8 added, 4 modified)
#   Lines: +456 -23
#
# Conversation:
#   Prompt: prompts/001.txt (145 chars)
#   Response: responses/001.txt (2,345 chars)
#   Tokens used: 3,456
#   Duration: 8.3s
#
# Tags: auth, feature, milestone
#
# Custom Metadata:
#   tested: true
#   reviewed: false
#   priority: high
#
# Links:
#   Previous: 9e2f1d (프로젝트 구조 완성)
#   Next: 6h4j2p (OAuth 통합 완료)
```

#### 대화 내용 보기

```bash
# 프롬프트 보기
entire checkpoint show 4a7b3c --prompt

# 출력:
# Checkpoint: 4a7b3c
# Prompt (2026-02-11 12:29:45):
#
# 사용자 인증 시스템을 구현해줘.
#
# 요구사항:
# - 이메일/비밀번호 로그인
# - JWT 토큰 기반 인증
# - 비밀번호 해싱 (bcrypt)
# - 토큰 갱신 메커니즘

# 응답 보기
entire checkpoint show 4a7b3c --response

# 둘 다 보기
entire checkpoint show 4a7b3c --conversation
```

### Checkpoint 복원

#### 기본 복원

```bash
# Checkpoint로 되돌리기
entire checkpoint restore 4a7b3c

# 출력:
# Restoring checkpoint 4a7b3c...
#
# This will:
#   - Reset code to commit 3d8k5m2
#   - Restore conversation context
#   - Mark newer checkpoints as archived
#
# Current state:
#   Checkpoint: 6h4j2p (OAuth 통합 완료)
#   Commits ahead: 3
#
# Target state:
#   Checkpoint: 4a7b3c (사용자 인증 완료)
#
# You will lose uncommitted changes!
#
# Continue? (y/N): y
#
# ✓ Code restored to commit 3d8k5m2
# ✓ Conversation context restored
# ✓ 2 checkpoints archived
# ✓ Current checkpoint: 4a7b3c
```

#### 선택적 복원

```bash
# 1. 코드만 복원 (대화는 유지)
entire checkpoint restore 4a7b3c --code-only

# 2. 대화만 복원 (코드는 유지)
entire checkpoint restore 4a7b3c --context-only

# 3. 특정 파일만 복원
entire checkpoint restore 4a7b3c \
  --files "src/auth/*.ts"

# 4. 새 브랜치로 복원
entire checkpoint restore 4a7b3c \
  --branch feature/auth-restored
```

#### 미리보기

```bash
# 복원 전 변경사항 확인
entire checkpoint diff 6h4j2p 4a7b3c

# 출력:
# Comparing checkpoints:
#   Current: 6h4j2p (OAuth 통합 완료)
#   Target: 4a7b3c (사용자 인증 완료)
#
# Code changes (reverse):
#   src/auth/oauth.ts (deleted)
#   src/auth/providers/*.ts (deleted)
#   config/oauth.json (deleted)
#
# Conversation:
#   3 prompts will be archived
#   3 responses will be archived
#
# Statistics:
#   Files affected: 8
#   Lines removed: -234
#   Checkpoints archived: 2
```

### Checkpoint 비교

#### 두 checkpoint 비교

```bash
# 코드 변경 비교
entire checkpoint diff 4a7b3c 6h4j2p

# 출력:
# Comparing checkpoints:
#   From: 4a7b3c (사용자 인증 완료)
#   To: 6h4j2p (OAuth 통합 완료)
#
# Files changed: 8
#   Added: 5
#     src/auth/oauth.ts
#     src/auth/providers/google.ts
#     src/auth/providers/github.ts
#     src/auth/providers/facebook.ts
#     config/oauth.json
#   Modified: 3
#     src/auth/service.ts (+67 -12)
#     src/routes/auth.ts (+45 -5)
#     package.json (+3 -0)
#
# Total changes: +234 -17 lines
#
# Conversation:
#   Prompts between: 3
#   Key topics: OAuth, integration, providers
```

#### 상세 diff 보기

```bash
# Git diff 형식으로 보기
entire checkpoint diff 4a7b3c 6h4j2p --git-diff

# 특정 파일만 비교
entire checkpoint diff 4a7b3c 6h4j2p \
  --files "src/auth/service.ts"

# 대화 내용 비교
entire checkpoint diff 4a7b3c 6h4j2p \
  --conversation
```

### Checkpoint 태깅

#### 태그 추가/제거

```bash
# 태그 추가
entire checkpoint tag 4a7b3c add milestone feature-complete

# 태그 제거
entire checkpoint tag 4a7b3c remove draft

# 태그 목록 확인
entire checkpoint tag 4a7b3c list

# 출력:
# Checkpoint: 4a7b3c
# Tags: auth, feature, milestone, feature-complete
```

#### 태그로 검색

```bash
# 특정 태그가 있는 checkpoint 찾기
entire checkpoint list --tag milestone

# 여러 태그 조합
entire checkpoint list --tag milestone --tag feature-complete

# 태그 제외
entire checkpoint list --not-tag draft
```

### Checkpoint 연결 (Linking)

#### 관련 checkpoint 연결

```bash
# 현재 checkpoint와 관련된 checkpoint 연결
entire checkpoint link 6h4j2p --related 4a7b3c

# 설명과 함께 연결
entire checkpoint link 6h4j2p \
  --related 4a7b3c \
  --note "이전 인증 구현을 기반으로 OAuth 추가"

# 여러 checkpoint 연결
entire checkpoint link 6h4j2p \
  --related 4a7b3c \
  --related 9e2f1d \
  --related 3k8m2n
```

#### 연결 조회

```bash
# Checkpoint의 연결 확인
entire checkpoint links 6h4j2p

# 출력:
# Checkpoint: 6h4j2p (OAuth 통합 완료)
#
# Previous: 4a7b3c (사용자 인증 완료)
# Next: 8r6t3v (권한 관리 추가)
#
# Related:
#   4a7b3c - 사용자 인증 완료
#     Note: 이전 인증 구현을 기반으로 OAuth 추가
#   9e2f1d - 프로젝트 구조 완성
#     Note: 초기 구조 설계 참고
#
# Referenced by:
#   8r6t3v - 권한 관리 추가
#   2q4w6e - 세션 관리 구현
```

#### 그래프 시각화

```bash
# Checkpoint 의존성 그래프
entire checkpoint graph

# 출력:
# Checkpoint Graph for session abc123:
#
# 9e2f1d ─┬─ 4a7b3c ─── 6h4j2p ─┬─ 8r6t3v
#         │                      │
#         └─ 3k8m2n ─────────────┴─ 2q4w6e
#
# Legend:
#   ─── : Sequential (previous/next)
#   ┬┴ : Related checkpoints

# ASCII 아트 형식
entire checkpoint graph --ascii-art

# DOT 형식 (Graphviz)
entire checkpoint graph --dot > checkpoints.dot
dot -Tpng checkpoints.dot -o checkpoints.png
```

### Checkpoint 병합

여러 checkpoint를 하나로 통합:

```bash
# 연속된 checkpoint들을 하나로 병합
entire checkpoint squash 4a7b3c..6h4j2p \
  --message "인증 시스템 완성" \
  --description "기본 인증부터 OAuth 통합까지 모든 인증 기능"

# 출력:
# Squashing checkpoints:
#   4a7b3c - 사용자 인증 완료
#   5t7y9u - 비밀번호 검증 강화
#   6h4j2p - OAuth 통합 완료
#
# This will:
#   - Combine all conversations
#   - Create single git commit
#   - Archive original checkpoints
#   - Create new checkpoint: 7g5h3j
#
# Continue? (y/N): y
#
# ✓ Checkpoints squashed
# ✓ New checkpoint: 7g5h3j (인증 시스템 완성)
# ✓ Original checkpoints archived
```

### Checkpoint 정리

#### 자동 정리

```bash
# 자동 정리 설정
entire config set checkpoint.cleanup.enabled true
entire config set checkpoint.cleanup.strategy smart

# Smart 전략:
# - Committed는 모두 유지
# - Temporary는 최근 20개만 유지
# - 7일 이상 된 temporary는 자동 삭제
# - 단, tagged temporary는 유지

# 정리 실행
entire checkpoint cleanup

# 출력:
# Analyzing checkpoints...
#
# Temporary checkpoints: 45
#   Keep: 20 (recent)
#   Keep: 3 (tagged)
#   Delete: 22 (old, untagged)
#
# Committed checkpoints: 8
#   Keep: all
#
# Proceed? (y/N): y
# ✓ Deleted 22 temporary checkpoints
# ✓ Freed 15.3 MB
```

#### 수동 정리

```bash
# 특정 checkpoint 삭제
entire checkpoint delete 3k8m2n

# 날짜 기준 삭제
entire checkpoint cleanup --older-than "7 days"

# 타입 기준 삭제
entire checkpoint cleanup --type temporary --keep 10

# 안전 모드 (아카이브만)
entire checkpoint archive 3k8m2n
# 삭제는 안 하고 archived 상태로 변경
```

### Checkpoint Export/Import

#### 내보내기

```bash
# 단일 checkpoint 내보내기
entire checkpoint export 4a7b3c > checkpoint.json

# 대화 내용 포함
entire checkpoint export 4a7b3c --include-conversation \
  > checkpoint-full.json

# 코드 diff 포함
entire checkpoint export 4a7b3c --include-diff \
  > checkpoint-with-code.json

# 여러 checkpoint 내보내기
entire checkpoint export --range 4a7b3c..6h4j2p \
  > checkpoints-range.json
```

#### 가져오기

```bash
# Checkpoint 가져오기
entire checkpoint import checkpoint.json

# 다른 세션으로 가져오기
entire checkpoint import checkpoint.json \
  --session def456

# 새 ID로 가져오기
entire checkpoint import checkpoint.json \
  --new-id
```

### Checkpoint 검색

#### 내용 기반 검색

```bash
# 프롬프트 내용 검색
entire checkpoint search "OAuth" --in prompts

# 응답 내용 검색
entire checkpoint search "jwt" --in responses

# 코드 변경 검색
entire checkpoint search "class User" --in code

# 전체 검색
entire checkpoint search "authentication"
```

#### 고급 검색

```bash
# 메타데이터 검색
entire checkpoint search \
  --metadata "tested=true" \
  --metadata "priority=high"

# 파일 변경 기준
entire checkpoint search \
  --files-changed "src/auth/*.ts"

# 라인 변경 기준
entire checkpoint search \
  --lines-added ">100" \
  --lines-deleted "<50"

# 복합 검색
entire checkpoint search \
  --message "auth" \
  --tag milestone \
  --since "1 week ago" \
  --files-changed "*.ts"
```

### Checkpoint 통계

```bash
# Checkpoint 통계
entire checkpoint stats

# 출력:
# Checkpoint Statistics
#
# Total: 53
#   Committed: 8 (15%)
#   Temporary: 45 (85%)
#
# Timeline:
#   Today: 12
#   This week: 34
#   This month: 53
#
# Tags:
#   milestone: 5
#   feature: 12
#   bugfix: 8
#   refactor: 6
#
# Average interval: 15m 30s
# Most productive hour: 14:00-15:00 (8 checkpoints)
#
# Size:
#   Total metadata: 4.5 MB
#   Conversations: 2.1 MB
#   Context data: 1.8 MB
#   Indexes: 0.6 MB
```

### 실전 활용

#### 시간 여행 디버깅

```bash
# 버그가 언제 생겼는지 찾기

# 1. 현재 checkpoint 확인
entire checkpoint list

# 2. 과거 checkpoint로 이동
entire checkpoint restore 4a7b3c
npm test  # 통과

entire checkpoint restore 5t7y9u
npm test  # 통과

entire checkpoint restore 6h4j2p
npm test  # 실패! ← 이 지점에서 버그 발생

# 3. 해당 checkpoint의 변경사항 확인
entire checkpoint show 6h4j2p --conversation
git show $(entire checkpoint info 6h4j2p --commit)

# 4. 버그 수정
$ AI: 6h4j2p checkpoint에서 발생한 버그 고쳐줘

# 5. 최신으로 복귀
entire checkpoint restore 8r6t3v
```

#### 기능 개발 마일스톤

```bash
# 1. 설계 단계
$ AI: 사용자 인증 시스템 설계해줘
entire checkpoint create --message "설계 완료" --tag milestone

# 2. 기본 구현
$ AI: 기본 로그인 구현해줘
entire checkpoint create --message "기본 인증" --tag milestone

# 3. OAuth 추가
$ AI: OAuth 추가해줘
entire checkpoint create --message "OAuth 통합" --tag milestone

# 4. 권한 관리
$ AI: 권한 관리 추가해줘
entire checkpoint create --message "권한 시스템" --tag milestone

# 5. 마일스톤 확인
entire checkpoint list --tag milestone

# 6. 마일스톤 간 비교
entire checkpoint diff <설계> <권한시스템>
```

### 베스트 프랙티스

```bash
# 1. 의미있는 committed checkpoint 생성
entire checkpoint create --message "명확한 마일스톤 설명"

# 2. 적절한 태그 사용
--tags "milestone,feature,auth"

# 3. 메타데이터 활용
--metadata '{"tested":true,"reviewed":true}'

# 4. 정기적인 정리
entire checkpoint cleanup --smart

# 5. 중요 checkpoint 연결
entire checkpoint link <new> --related <base>

# 6. 백업
entire checkpoint export --all > backup.json
```

### 다음 장 예고

다음 장에서는 **Checkpoint ID 연결**을 상세히 다룹니다.

- 12-hex ID 체계
- Bidirectional linking
- 의존성 그래프
- 자동 연결 추론

---

**관련 문서**:
- [09. Session 관리](/2026/02/11/entire-cli-guide-09-session-management/)
- [11. Checkpoint ID 연결](/2026/02/11/entire-cli-guide-11-checkpoint-linking/)
- [14. Storage 구조](/2026/02/11/entire-cli-guide-14-storage-structure/)
