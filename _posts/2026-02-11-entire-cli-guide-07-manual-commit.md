---
layout: post
title: "Entire CLI 완벽 가이드 - 07. Manual-Commit Strategy"
date: 2026-02-11
categories: [AI, Development Tools]
tags: [Entire, CLI, AI Coding, Manual Commit, Git, Shadow Branch]
description: "Entire CLI Manual-Commit 전략의 Shadow Branch 메커니즘과 워크플로우"
---

## 07. Manual-Commit Strategy

Manual-Commit 전략은 AI의 변경사항을 별도의 shadow branch에서 격리하여 관리하는 Entire CLI의 핵심 기능입니다. 이 장에서는 shadow branch 메커니즘과 실전 워크플로우를 상세히 다룹니다.

### Shadow Branch 개념

Shadow branch는 AI 작업을 메인 코드베이스와 격리하기 위한 특수한 Git branch입니다.

```
개념적 구조:
┌────────────────────────────────────────┐
│ Main Branch (프로덕션 코드)            │
│ ─────o─────o─────o────── (안정적)     │
│                   │                    │
│                   └─ Shadow Branch     │
│                      (AI 실험 영역)    │
│                      ──x──x──x──x      │
│                        (자유로운 실험) │
└────────────────────────────────────────┘

특징:
1. Main과 독립적으로 발전
2. 언제든 삭제 가능 (안전)
3. 검증 후 병합 (제어)
4. 이력 분리 (깔끔)
```

### Shadow Branch 생성

#### 자동 생성

세션 시작 시 자동으로 생성됩니다.

```bash
# Manual-commit 세션 시작
entire session start --strategy manual-commit \
  --message "새로운 기능 개발"

# 출력:
# ✓ Session created: abc123def456
# ✓ Shadow branch: entire/shadow/session-abc123def456
# ✓ Base commit: 9a7f3c2 (main)
#
# You can now interact with AI.
# Changes will be isolated in the shadow branch.
```

#### 내부 동작

```bash
# Entire가 내부적으로 수행하는 작업:

# 1. 현재 브랜치와 커밋 기록
CURRENT_BRANCH=$(git branch --show-current)
BASE_COMMIT=$(git rev-parse HEAD)

# 2. Shadow branch 생성
SESSION_ID=$(generate_session_id)
SHADOW_BRANCH="entire/shadow/session-${SESSION_ID}"
git branch ${SHADOW_BRANCH} ${BASE_COMMIT}

# 3. 메타데이터 branch 생성
METADATA_BRANCH="entire/metadata/session-${SESSION_ID}"
git checkout --orphan ${METADATA_BRANCH}

# 4. 세션 정보 저장
cat > session.json <<EOF
{
  "id": "${SESSION_ID}",
  "strategy": "manual-commit",
  "created": "$(date -Iseconds)",
  "baseBranch": "${CURRENT_BRANCH}",
  "baseCommit": "${BASE_COMMIT}",
  "shadowBranch": "${SHADOW_BRANCH}",
  "metadataBranch": "${METADATA_BRANCH}"
}
EOF
git add session.json
git commit -m "Initialize session ${SESSION_ID}"

# 5. 원래 브랜치로 복귀
git checkout ${CURRENT_BRANCH}
```

### Shadow Branch 명명 규칙

```
패턴: entire/shadow/session-<session-id>

예시:
entire/shadow/session-abc123def456
entire/shadow/session-4a7b3c9e2f1d
entire/shadow/session-8k3m5n7p9q2r

이점:
1. 충돌 방지: 사용자 브랜치와 분리
2. 식별 용이: "entire/" 접두사로 즉시 인식
3. 정리 편의: git branch -D entire/shadow/* 로 일괄 삭제
4. Git 툴 호환: 표준 브랜치 명명 규칙 준수
```

### Shadow Branch에서의 작업

#### AI 상호작용

```bash
# AI와 대화하면 shadow branch가 업데이트됨
$ AI: 사용자 인증 시스템을 구현해줘

# AI 응답:
# 인증 시스템을 구현하겠습니다.
#
# [코드 변경 사항]
# - src/auth/index.ts 생성
# - src/auth/middleware.ts 생성
# - src/routes/auth.ts 추가
#
# ✓ Changes committed to shadow branch
#   Commit: 7f4a2b1 "Implement user authentication system"
```

#### 변경 추적

```bash
# 1. Shadow branch 상태 확인
entire session info

# 출력:
# Session: abc123def456
# Strategy: manual-commit
# Shadow Branch: entire/shadow/session-abc123def456
# Base: main @ 9a7f3c2
# Shadow: 7f4a2b1 (1 commit ahead)
#
# Checkpoints: 0
# Uncommitted changes: Yes

# 2. 변경 사항 비교
git diff main...entire/shadow/session-abc123def456

# 3. 커밋 로그 확인
git log main..entire/shadow/session-abc123def456

# 출력:
# commit 7f4a2b1
# Author: Entire AI Assistant <ai@entire.dev>
# Date:   Tue Feb 11 10:30:00 2026 +0900
#
#     Implement user authentication system
#
#     Session: abc123def456
#     Checkpoint: temporary
```

#### 변경 사항 검토

```bash
# 1. 파일별 변경 확인
git diff main...entire/shadow/session-abc123def456 --stat

# 출력:
# src/auth/index.ts       | 45 ++++++++++++++++++
# src/auth/middleware.ts  | 32 +++++++++++++
# src/routes/auth.ts      | 67 ++++++++++++++++++++++++
# 3 files changed, 144 insertions(+)

# 2. 특정 파일의 변경 상세
git diff main...entire/shadow/session-abc123def456 src/auth/index.ts

# 3. Shadow branch 체크아웃하여 테스트
git checkout entire/shadow/session-abc123def456
npm test
npm run build

# 4. 원래 브랜치로 복귀
git checkout main
```

### Git Worktree를 활용한 격리

Entire는 Git worktree를 사용하여 shadow branch를 물리적으로 격리할 수 있습니다.

#### Worktree 개념

```
프로젝트 구조:
/home/user/project/           (main 브랜치)
├── src/
├── package.json
└── .git/

/home/user/project/.entire/worktrees/session-abc123/
├── src/                      (shadow 브랜치)
├── package.json
└── .git -> ../../.git/worktrees/session-abc123
```

#### Worktree 사용

```bash
# 1. Worktree로 shadow branch 작업
entire session start --strategy manual-commit \
  --use-worktree \
  --message "독립된 작업 환경"

# 출력:
# ✓ Session created: abc123def456
# ✓ Shadow branch: entire/shadow/session-abc123def456
# ✓ Worktree created: .entire/worktrees/session-abc123def456
#
# You can work in the worktree directory:
#   cd .entire/worktrees/session-abc123def456

# 2. Worktree에서 작업
cd .entire/worktrees/session-abc123def456
npm install  # 독립적인 node_modules
npm test     # 독립적인 실행

# 3. 메인 디렉토리는 영향 없음
cd /home/user/project
npm test     # 여전히 main 브랜치 상태
```

#### Worktree 이점

```
장점:
1. 완전한 격리
   - Main과 shadow가 물리적으로 분리
   - node_modules, build 결과물 분리
   - 환경 변수, 설정 파일 독립

2. 동시 작업 가능
   - Main 브랜치에서 작업 계속
   - Shadow에서 AI 실험
   - 상호 간섭 없음

3. 안전한 테스트
   - Shadow에서 빌드 실패해도 main 영향 없음
   - 의존성 충돌 방지
   - 롤백 쉬움

단점:
1. 디스크 공간 사용
2. 초기 설정 시간 (npm install 등)
3. 관리 복잡도 증가
```

### Checkpoint 생성

Manual-commit에서는 명시적으로 checkpoint를 생성합니다.

#### 기본 생성

```bash
# 1. AI 작업 완료 후 checkpoint
$ AI: 인증 로직 구현 완료

entire checkpoint create --message "기본 인증 완료"

# 출력:
# ✓ Checkpoint created: 4a7b3c
# ✓ Commit: 7f4a2b1
# ✓ Message: 기본 인증 완료
#
# This checkpoint is now committed in the shadow branch.
# You can merge it to main when ready.
```

#### 내부 동작

```bash
# Checkpoint 생성 시 내부 동작:

# 1. Shadow branch에 현재 상태 커밋
git checkout entire/shadow/session-abc123
git add -A
git commit -m "User checkpoint: 기본 인증 완료

Session: abc123
Checkpoint: 4a7b3c
Type: manual
Timestamp: $(date -Iseconds)"

COMMIT_SHA=$(git rev-parse HEAD)

# 2. 메타데이터 업데이트
git checkout entire/metadata/session-abc123

# checkpoints.json에 추가
jq ".checkpoints += [{
  \"id\": \"4a7b3c\",
  \"type\": \"manual\",
  \"timestamp\": \"$(date -Iseconds)\",
  \"message\": \"기본 인증 완료\",
  \"commit\": \"${COMMIT_SHA}\",
  \"prompt\": \"prompts/$(next_number).txt\",
  \"response\": \"responses/$(next_number).txt\"
}]" checkpoints.json > tmp && mv tmp checkpoints.json

# 프롬프트와 응답 저장
cat > prompts/001.txt <<EOF
사용자 인증 시스템을 구현해줘
EOF

cat > responses/001.txt <<EOF
[AI의 전체 응답 내용]
EOF

git add .
git commit -m "Add checkpoint 4a7b3c"

# 3. 원래 브랜치로 복귀
git checkout main
```

### Shadow Branch 병합

#### 기본 병합

```bash
# 1. 최종 검토
git diff main...entire/shadow/session-abc123
npm test

# 2. 병합
git merge entire/shadow/session-abc123

# 출력:
# Updating 9a7f3c2..7f4a2b1
# Fast-forward
#  src/auth/index.ts       | 45 ++++++++++++++++++
#  src/auth/middleware.ts  | 32 +++++++++++++
#  src/routes/auth.ts      | 67 ++++++++++++++++++++++++
#  3 files changed, 144 insertions(+)

# 3. 세션 종료
entire session stop

# 출력:
# ✓ Session abc123 stopped
# ✓ Shadow branch entire/shadow/session-abc123 preserved
#
# To delete the shadow branch:
#   git branch -D entire/shadow/session-abc123
```

#### 선택적 병합 (Cherry-pick)

```bash
# 특정 checkpoint만 가져오기
# 1. Checkpoint 목록 확인
entire checkpoint list

# 출력:
# Session: abc123
#
# Checkpoints:
#   4a7b3c - 기본 인증 완료 (7f4a2b1)
#   9e2f1d - OAuth 추가 (3d8k5m2)
#   6h4j2p - 권한 관리 추가 (8r6t3v9)

# 2. 원하는 checkpoint만 cherry-pick
git cherry-pick 7f4a2b1  # 기본 인증만
git cherry-pick 8r6t3v9  # 권한 관리만
# OAuth는 건너뜀
```

#### Squash 병합

```bash
# 여러 checkpoint를 하나의 커밋으로 병합
git merge --squash entire/shadow/session-abc123

# 병합 커밋 작성
git commit -m "Add complete authentication system

Implemented:
- Basic username/password auth
- OAuth integration
- Role-based access control

Session: abc123
Checkpoints: 4a7b3c, 9e2f1d, 6h4j2p"
```

#### 충돌 해결

```bash
# 병합 시 충돌 발생
git merge entire/shadow/session-abc123

# 출력:
# Auto-merging src/auth/index.ts
# CONFLICT (content): Merge conflict in src/auth/index.ts
# Automatic merge failed; fix conflicts and then commit.

# 1. 충돌 파일 확인
git status

# 출력:
# Unmerged paths:
#   both modified:   src/auth/index.ts

# 2. 충돌 내용 확인
cat src/auth/index.ts

# <<<<<<< HEAD
# export function authenticate(user: User) {
#   return validateCredentials(user);
# }
# =======
# export async function authenticate(user: User) {
#   return await validateCredentialsAsync(user);
# }
# >>>>>>> entire/shadow/session-abc123

# 3. 수동 해결 또는 AI 도움 요청
$ AI: 이 충돌을 해결해줘. 비동기 버전이 더 나아.

# 4. 해결 후 커밋
git add src/auth/index.ts
git commit
```

### Shadow Branch 관리

#### 목록 조회

```bash
# 모든 shadow branch 확인
git branch -a | grep entire/shadow

# 출력:
# entire/shadow/session-abc123
# entire/shadow/session-def456
# entire/shadow/session-ghi789

# Entire 명령으로 조회
entire session list --show-shadow

# 출력:
# Active Sessions:
#   abc123 - 새로운 기능 개발
#     Shadow: entire/shadow/session-abc123 (3 commits)
#   def456 - 버그 수정
#     Shadow: entire/shadow/session-def456 (1 commit)
#
# Inactive Sessions:
#   ghi789 - 실험적 리팩토링
#     Shadow: entire/shadow/session-ghi789 (5 commits)
```

#### 정리

```bash
# 1. 특정 shadow branch 삭제
git branch -D entire/shadow/session-abc123

# 2. 모든 inactive shadow branch 삭제
entire session cleanup --shadows

# 출력:
# Found 3 inactive shadow branches:
#   entire/shadow/session-ghi789 (merged)
#   entire/shadow/session-jkl012 (not merged)
#   entire/shadow/session-mno345 (merged)
#
# Delete merged shadow branches? (y/N): y
# ✓ Deleted entire/shadow/session-ghi789
# ✓ Deleted entire/shadow/session-mno345
#
# Keep unmerged branch: entire/shadow/session-jkl012

# 3. 강제 삭제 (병합 여부 무시)
entire session cleanup --shadows --force
```

### 고급 워크플로우

#### Multi-Stage 병합

```bash
# 시나리오: 단계적 기능 통합

# 1. 기초 작업
entire session start --strategy manual-commit \
  --message "Stage 1: 기본 구조"

$ AI: 프로젝트 기본 구조 만들어줘
entire checkpoint create --message "프로젝트 구조"

# 2. 첫 번째 stage 병합
git merge entire/shadow/session-abc123
entire session stop

# 3. 다음 stage
entire session start --strategy manual-commit \
  --message "Stage 2: 핵심 기능"

$ AI: 이제 핵심 기능 구현해줘
entire checkpoint create --message "핵심 기능"

# 4. 두 번째 stage 병합
git merge entire/shadow/session-def456
entire session stop

# 결과: 단계별로 검증하며 안전하게 통합
```

#### 병렬 실험

```bash
# 시나리오: 여러 접근 방식 동시 시도

# 1. 접근 A
entire session start --strategy manual-commit \
  --message "Approach A: REST API"

$ AI: REST API로 구현해줘
entire checkpoint create --message "REST 완료"

# 2. 접근 B (새 세션)
entire session start --strategy manual-commit \
  --message "Approach B: GraphQL"

$ AI: GraphQL로 구현해줘
entire checkpoint create --message "GraphQL 완료"

# 3. 접근 C (또 다른 세션)
entire session start --strategy manual-commit \
  --message "Approach C: gRPC"

$ AI: gRPC로 구현해줘
entire checkpoint create --message "gRPC 완료"

# 4. 비교 및 선택
git diff main...entire/shadow/session-abc123 > approach-a.diff
git diff main...entire/shadow/session-def456 > approach-b.diff
git diff main...entire/shadow/session-ghi789 > approach-c.diff

# 5. 최적 선택 병합
git merge entire/shadow/session-def456  # GraphQL 선택

# 6. 나머지 세션 정리
entire session stop --all
git branch -D entire/shadow/session-abc123
git branch -D entire/shadow/session-ghi789
```

#### Rebase 활용

```bash
# Shadow branch를 최신 main 위에 rebase

# 1. Main이 업데이트됨
git checkout main
git pull origin main

# 2. Shadow branch rebase
git checkout entire/shadow/session-abc123
git rebase main

# 3. 충돌 해결 (필요시)
git add .
git rebase --continue

# 4. 깔끔한 이력으로 병합 준비 완료
git checkout main
git merge entire/shadow/session-abc123  # Fast-forward
```

### 실전 예시

#### 예시 1: 대규모 리팩토링

```bash
# 1. 리팩토링 세션 시작
entire session start --strategy manual-commit \
  --use-worktree \
  --message "전체 아키텍처 리팩토링"

# 2. Worktree에서 작업
cd .entire/worktrees/session-abc123

# 3. 단계별 리팩토링
$ AI: 먼저 타입 정의를 개선해줘
entire checkpoint create --message "타입 시스템 개선"

$ AI: 의존성 주입 패턴 적용해줘
entire checkpoint create --message "DI 적용"

$ AI: 모듈 구조 재구성해줘
entire checkpoint create --message "모듈 재구성"

# 4. 각 단계마다 테스트
npm test

# 5. 메인 디렉토리로 돌아가서 최종 검토
cd /home/user/project
git diff main...entire/shadow/session-abc123

# 6. 단계별 리뷰 및 병합
git log main..entire/shadow/session-abc123 --oneline
# 1a2b3c4 모듈 재구성
# 5d6e7f8 DI 적용
# 9g0h1i2 타입 시스템 개선

git merge entire/shadow/session-abc123

# 7. Worktree 정리
entire session stop
```

#### 예시 2: 실험적 기능

```bash
# 1. 실험 시작
entire session start --strategy manual-commit \
  --message "AI 기반 추천 시스템 실험"

# 2. 첫 시도
$ AI: 협업 필터링 알고리즘 구현해줘
entire checkpoint create --message "협업 필터링 v1"

# 3. 테스트 결과 부족
npm test
# Performance: 45% accuracy

# 4. 다른 접근
$ AI: 대신 딥러닝 기반으로 바꿔줘
entire checkpoint create --message "딥러닝 접근"

# 5. 여전히 부족
npm test
# Performance: 52% accuracy

# 6. 하이브리드 접근
$ AI: 두 방식을 결합한 하이브리드 시스템 만들어줘
entire checkpoint create --message "하이브리드 시스템"

# 7. 만족스러운 결과
npm test
# Performance: 78% accuracy

# 8. 병합
git merge entire/shadow/session-abc123

# 결과: 여러 시도가 모두 안전하게 shadow에 격리되어
#       main 브랜치는 영향받지 않음
```

### 베스트 프랙티스

```bash
# 1. 의미있는 checkpoint 생성
entire checkpoint create --message "사용자 인증 완료"
# 나쁜 예: entire checkpoint create --message "작업"

# 2. 정기적인 main 동기화
git checkout entire/shadow/session-abc123
git rebase main  # 주기적으로 최신화

# 3. Checkpoint 전 테스트
npm test
npm run lint
entire checkpoint create --message "기능 완료 (테스트 통과)"

# 4. 병합 전 최종 검토
git diff main...entire/shadow/session-abc123 --stat
git log main..entire/shadow/session-abc123
npm test  # Shadow branch에서

# 5. 사용 완료한 shadow 정리
entire session cleanup --shadows --merged-only
```

### 문제 해결

#### Shadow Branch가 보이지 않음

```bash
# 1. Branch 존재 확인
git branch -a | grep entire/shadow

# 2. 세션 정보 확인
entire session info

# 3. 수동 복구
entire session repair --session-id abc123
```

#### 병합 실패

```bash
# 1. 현재 상태 확인
git status

# 2. 병합 취소
git merge --abort

# 3. 충돌 미리 확인
git merge-tree $(git merge-base main entire/shadow/session-abc123) \
  main entire/shadow/session-abc123

# 4. AI에게 충돌 해결 요청
$ AI: 이 병합 충돌을 해결해줘
```

#### Worktree 정리 오류

```bash
# Worktree가 제대로 삭제 안 됨
git worktree prune
git worktree list
rm -rf .entire/worktrees/session-abc123
```

### 다음 장 예고

다음 장에서는 **Auto-Commit Strategy**를 다룹니다.

- 자동 커밋 메커니즘
- 즉시 반영 워크플로우
- Manual-commit과의 비교
- 사용 사례별 가이드

---

**관련 문서**:
- [06. Strategy 개요](/2026/02/11/entire-cli-guide-06-strategy-overview/)
- [08. Auto-Commit Strategy](/2026/02/11/entire-cli-guide-08-auto-commit/)
- [13. Git 통합](/2026/02/11/entire-cli-guide-13-git-integration/)
