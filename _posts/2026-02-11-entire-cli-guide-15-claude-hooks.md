---
layout: post
title: "Entire CLI 완벽 가이드 - 15. Claude Code Hooks"
date: 2026-02-11
categories: [AI, Development Tools]
tags: [Entire, CLI, AI Coding, Hooks, Claude Code, Automation]
description: "Entire CLI의 Claude Code Hooks을 활용한 자동화 및 워크플로우 커스터마이징"
---

## 15. Claude Code Hooks

Entire CLI는 Claude Code의 다양한 시점에 실행되는 hooks를 제공합니다. 이 장에서는 SessionStart, UserPromptSubmit, Stop 등의 hooks를 다룹니다.

### Hooks 개념

```
Hook Execution Flow:

Claude Code 시작
     ↓
SessionStart Hook ────────────────┐
     ↓                             │
사용자 프롬프트 입력               │ Entire CLI
     ↓                             │ Session 관리
UserPromptSubmit Hook ────────────┤
     ↓                             │
AI 처리                           │
     ↓                             │
AI 응답 완료                      │
     ↓                             │
ResponseComplete Hook ────────────┤
     ↓                             │
(반복: 프롬프트 → 응답)           │
     ↓                             │
세션 종료                         │
     ↓                             │
Stop Hook ────────────────────────┘
```

### Hook 타입

```
사용 가능한 Hooks:

1. SessionStart
   - 세션 시작 시 실행
   - 초기화 작업

2. UserPromptSubmit
   - 프롬프트 제출 시 실행
   - 프롬프트 전처리

3. ResponseComplete
   - AI 응답 완료 시 실행
   - 자동 커밋, 검증

4. CheckpointCreate
   - Checkpoint 생성 시 실행
   - 메타데이터 추가

5. Stop
   - 세션 종료 시 실행
   - 정리 작업

6. Error
   - 에러 발생 시 실행
   - 에러 처리, 로깅
```

### Hook 설정

#### 기본 설정

```bash
# Hooks 디렉토리 생성
entire hooks init

# 출력:
# Initializing hooks...
# ✓ Created .entire/hooks/
# ✓ Created .entire/hooks/session-start.sh
# ✓ Created .entire/hooks/prompt-submit.sh
# ✓ Created .entire/hooks/response-complete.sh
# ✓ Created .entire/hooks/checkpoint-create.sh
# ✓ Created .entire/hooks/stop.sh
# ✓ Created .entire/hooks/error.sh
#
# Edit hooks to customize behavior.
```

#### Hook 디렉토리 구조

```bash
.entire/
└── hooks/
    ├── session-start.sh          # SessionStart hook
    ├── prompt-submit.sh          # UserPromptSubmit hook
    ├── response-complete.sh      # ResponseComplete hook
    ├── checkpoint-create.sh      # CheckpointCreate hook
    ├── stop.sh                   # Stop hook
    ├── error.sh                  # Error hook
    ├── config.json               # Hook 설정
    └── lib/                      # 공통 라이브러리
        ├── utils.sh
        └── notify.sh
```

### SessionStart Hook

세션이 시작될 때 실행됩니다.

#### 기본 템플릿

```bash
# .entire/hooks/session-start.sh
#!/bin/bash

# SessionStart Hook
# Called when a new Entire session starts

# 환경 변수
# - SESSION_ID: 세션 ID
# - SESSION_MESSAGE: 세션 메시지
# - STRATEGY: 전략 (manual-commit/auto-commit)
# - BASE_BRANCH: 베이스 브랜치
# - SHADOW_BRANCH: Shadow 브랜치 (manual-commit만)

echo "Session starting: $SESSION_ID"
echo "Message: $SESSION_MESSAGE"
echo "Strategy: $STRATEGY"

# 초기화 작업 예시
# 1. 의존성 확인
if ! command -v npm &> /dev/null; then
  echo "Warning: npm not found"
fi

# 2. 테스트 실행
echo "Running initial tests..."
npm test

# 3. 알림 전송
if command -v notify-send &> /dev/null; then
  notify-send "Entire Session Started" "$SESSION_MESSAGE"
fi

# 4. 로깅
echo "[$(date)] Session $SESSION_ID started" >> .entire/logs/sessions.log

# 성공 시 0 반환
exit 0
```

#### 고급 예시

```bash
#!/bin/bash
# .entire/hooks/session-start.sh

# Session 시작 시 자동 작업

SESSION_ID="$1"
SESSION_MESSAGE="$2"
STRATEGY="$3"

# 1. Jira 티켓 자동 생성
if [[ "$SESSION_MESSAGE" =~ PROJ-[0-9]+ ]]; then
  TICKET="${BASH_REMATCH[0]}"
  echo "Linking to Jira ticket: $TICKET"

  # Jira API 호출
  curl -X POST "https://jira.company.com/api/ticket/$TICKET/comment" \
    -H "Authorization: Bearer $JIRA_TOKEN" \
    -d "{\"text\":\"Started Entire session: $SESSION_ID\"}"
fi

# 2. Slack 알림
if [ -n "$SLACK_WEBHOOK" ]; then
  curl -X POST "$SLACK_WEBHOOK" \
    -H "Content-Type: application/json" \
    -d "{
      \"text\": \"🚀 New Entire session started\",
      \"attachments\": [{
        \"fields\": [
          {\"title\": \"Session\", \"value\": \"$SESSION_ID\", \"short\": true},
          {\"title\": \"Message\", \"value\": \"$SESSION_MESSAGE\", \"short\": true},
          {\"title\": \"Strategy\", \"value\": \"$STRATEGY\", \"short\": true}
        ]
      }]
    }"
fi

# 3. 환경 설정
if [ "$STRATEGY" = "manual-commit" ]; then
  # Worktree 설정 권장
  echo "Tip: Use --use-worktree for better isolation"
fi

# 4. Git 상태 확인
if ! git diff-index --quiet HEAD --; then
  echo "⚠️  Warning: You have uncommitted changes"
  echo "Consider committing them before starting the session"
fi

exit 0
```

### UserPromptSubmit Hook

사용자가 프롬프트를 제출할 때 실행됩니다.

#### 기본 템플릿

```bash
# .entire/hooks/prompt-submit.sh
#!/bin/bash

# UserPromptSubmit Hook
# Called when user submits a prompt to AI

# 환경 변수
# - SESSION_ID: 세션 ID
# - PROMPT: 사용자 프롬프트
# - PROMPT_FILE: 프롬프트가 저장된 파일 경로

PROMPT_TEXT=$(cat "$PROMPT_FILE")

echo "Prompt submitted (${#PROMPT_TEXT} chars)"

# 프롬프트 전처리 예시
# 1. 민감 정보 체크
if echo "$PROMPT_TEXT" | grep -iE '(password|secret|token|api[_-]?key)'; then
  echo "⚠️  Warning: Prompt may contain sensitive information"
  read -p "Continue? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1  # Hook 실패 시 프롬프트 제출 중단
  fi
fi

# 2. 프롬프트 통계
WORD_COUNT=$(echo "$PROMPT_TEXT" | wc -w)
echo "Prompt stats: $WORD_COUNT words"

# 3. 로깅
echo "[$(date)] Prompt submitted: ${PROMPT_TEXT:0:50}..." \
  >> .entire/logs/prompts.log

exit 0
```

#### 프롬프트 증강

```bash
#!/bin/bash
# .entire/hooks/prompt-submit.sh

# 프롬프트 자동 증강

PROMPT_FILE="$1"
SESSION_ID="$2"

# 원본 프롬프트
ORIGINAL=$(cat "$PROMPT_FILE")

# 컨텍스트 추가
CONTEXT=""

# 1. 현재 브랜치 정보 추가
CURRENT_BRANCH=$(git branch --show-current)
CONTEXT="$CONTEXT\nCurrent branch: $CURRENT_BRANCH"

# 2. 최근 커밋 정보
RECENT_COMMIT=$(git log -1 --oneline)
CONTEXT="$CONTEXT\nLast commit: $RECENT_COMMIT"

# 3. 프로젝트 타입 감지
if [ -f "package.json" ]; then
  PROJECT_TYPE=$(jq -r '.type // "commonjs"' package.json)
  CONTEXT="$CONTEXT\nProject type: $PROJECT_TYPE (Node.js)"
elif [ -f "Cargo.toml" ]; then
  CONTEXT="$CONTEXT\nProject type: Rust"
elif [ -f "go.mod" ]; then
  CONTEXT="$CONTEXT\nProject type: Go"
fi

# 4. 테스트 상태
if npm test --silent 2>&1 | grep -q "passing"; then
  CONTEXT="$CONTEXT\nTests: passing"
else
  CONTEXT="$CONTEXT\nTests: failing (please fix)"
fi

# 증강된 프롬프트 작성
cat > "$PROMPT_FILE" <<EOF
$ORIGINAL

---
Context:
$CONTEXT
EOF

echo "✓ Prompt augmented with context"

exit 0
```

### ResponseComplete Hook

AI 응답이 완료되면 실행됩니다.

#### 기본 템플릿

```bash
# .entire/hooks/response-complete.sh
#!/bin/bash

# ResponseComplete Hook
# Called when AI completes a response

# 환경 변수
# - SESSION_ID: 세션 ID
# - RESPONSE_FILE: 응답이 저장된 파일
# - COMMIT_SHA: 생성된 커밋 SHA (있는 경우)
# - FILES_CHANGED: 변경된 파일 목록 (공백 구분)

echo "Response complete"
echo "Commit: $COMMIT_SHA"
echo "Files changed: $FILES_CHANGED"

# 자동 검증 예시
# 1. Lint 실행
if [ -n "$FILES_CHANGED" ]; then
  echo "Running lint on changed files..."
  for file in $FILES_CHANGED; do
    if [[ "$file" == *.ts ]]; then
      npx eslint "$file" --fix
    fi
  done
fi

# 2. 테스트 실행
echo "Running tests..."
if npm test; then
  echo "✓ Tests passed"
else
  echo "✗ Tests failed"
  # Checkpoint에 메타데이터 추가
  entire checkpoint update-metadata \
    --set "tested=false" \
    --set "test_status=failed"
fi

# 3. 빌드 확인
if npm run build; then
  echo "✓ Build successful"
else
  echo "✗ Build failed"
fi

exit 0
```

#### 자동 품질 검사

```bash
#!/bin/bash
# .entire/hooks/response-complete.sh

# AI 응답 후 자동 품질 검사

SESSION_ID="$1"
COMMIT_SHA="$2"
FILES_CHANGED="$3"

QUALITY_SCORE=0
ISSUES=()

# 1. Lint 검사
echo "Checking code quality..."
if npm run lint 2>&1 | grep -q "0 errors"; then
  QUALITY_SCORE=$((QUALITY_SCORE + 25))
else
  ISSUES+=("Lint errors found")
fi

# 2. 타입 체크
if npm run type-check 2>&1 | grep -q "0 errors"; then
  QUALITY_SCORE=$((QUALITY_SCORE + 25))
else
  ISSUES+=("Type errors found")
fi

# 3. 테스트 커버리지
COVERAGE=$(npm test -- --coverage 2>&1 | grep "Lines" | awk '{print $3}' | tr -d '%')
if [ "$COVERAGE" -gt 80 ]; then
  QUALITY_SCORE=$((QUALITY_SCORE + 25))
else
  ISSUES+=("Low test coverage: ${COVERAGE}%")
fi

# 4. 보안 검사
if npm audit 2>&1 | grep -q "0 vulnerabilities"; then
  QUALITY_SCORE=$((QUALITY_SCORE + 25))
else
  ISSUES+=("Security vulnerabilities found")
fi

# 결과 저장
entire checkpoint update-metadata \
  --set "quality_score=$QUALITY_SCORE" \
  --set "quality_issues=${ISSUES[*]}"

# 품질 점수에 따라 알림
if [ $QUALITY_SCORE -ge 75 ]; then
  notify-send "✓ High Quality" "Score: $QUALITY_SCORE/100"
elif [ $QUALITY_SCORE -ge 50 ]; then
  notify-send "⚠ Medium Quality" "Score: $QUALITY_SCORE/100"
else
  notify-send "✗ Low Quality" "Score: $QUALITY_SCORE/100\nIssues: ${ISSUES[*]}"
fi

exit 0
```

### CheckpointCreate Hook

Checkpoint가 생성될 때 실행됩니다.

#### 기본 템플릿

```bash
# .entire/hooks/checkpoint-create.sh
#!/bin/bash

# CheckpointCreate Hook
# Called when a checkpoint is created

# 환경 변수
# - SESSION_ID: 세션 ID
# - CHECKPOINT_ID: Checkpoint ID
# - CHECKPOINT_TYPE: committed/temporary
# - MESSAGE: Checkpoint 메시지
# - COMMIT_SHA: Git 커밋 SHA

echo "Checkpoint created: $CHECKPOINT_ID"
echo "Type: $CHECKPOINT_TYPE"
echo "Message: $MESSAGE"

# Committed checkpoint에 대한 특별 처리
if [ "$CHECKPOINT_TYPE" = "committed" ]; then
  echo "This is a committed checkpoint - important milestone"

  # 1. 스냅샷 생성
  entire checkpoint export "$CHECKPOINT_ID" \
    > ".entire/snapshots/checkpoint-$CHECKPOINT_ID.json"

  # 2. 알림
  notify-send "Checkpoint Created" "$MESSAGE"

  # 3. 백업
  git push origin "refs/entire/metadata/session-$SESSION_ID"
fi

exit 0
```

#### 자동 문서화

```bash
#!/bin/bash
# .entire/hooks/checkpoint-create.sh

# Checkpoint 생성 시 자동 문서 업데이트

CHECKPOINT_ID="$1"
CHECKPOINT_TYPE="$2"
MESSAGE="$3"
COMMIT_SHA="$4"

# Committed checkpoint만 처리
if [ "$CHECKPOINT_TYPE" != "committed" ]; then
  exit 0
fi

# CHANGELOG 업데이트
if [ ! -f "CHANGELOG.md" ]; then
  cat > CHANGELOG.md <<EOF
# Changelog

All notable changes to this project will be documented in this file.

EOF
fi

# 변경 사항 추가
DATE=$(date +"%Y-%m-%d")
cat > /tmp/changelog-entry.md <<EOF

## $MESSAGE - $DATE

Checkpoint: $CHECKPOINT_ID
Commit: $COMMIT_SHA

$(entire checkpoint show "$CHECKPOINT_ID" --conversation | sed 's/^/  /')

EOF

# CHANGELOG에 삽입 (첫 제목 뒤)
sed -i "/^# Changelog/r /tmp/changelog-entry.md" CHANGELOG.md

# 커밋
git add CHANGELOG.md
git commit -m "docs: Update CHANGELOG for checkpoint $CHECKPOINT_ID"

echo "✓ CHANGELOG updated"

exit 0
```

### Stop Hook

세션이 종료될 때 실행됩니다.

#### 기본 템플릿

```bash
# .entire/hooks/stop.sh
#!/bin/bash

# Stop Hook
# Called when a session stops

# 환경 변수
# - SESSION_ID: 세션 ID
# - DURATION: 세션 지속 시간 (초)
# - CHECKPOINTS: Checkpoint 수
# - COMMITS: 커밋 수

echo "Session stopping: $SESSION_ID"
echo "Duration: ${DURATION}s"
echo "Checkpoints: $CHECKPOINTS"
echo "Commits: $COMMITS"

# 정리 작업
# 1. 통계 생성
entire session stats "$SESSION_ID" > \
  ".entire/reports/session-$SESSION_ID-stats.txt"

# 2. 백업
entire session export "$SESSION_ID" > \
  ".entire/backups/session-$SESSION_ID.json"

# 3. 알림
notify-send "Session Ended" \
  "Duration: ${DURATION}s, Checkpoints: $CHECKPOINTS"

# 4. 정리
if [ -d ".entire/worktrees/session-$SESSION_ID" ]; then
  echo "Cleaning up worktree..."
  rm -rf ".entire/worktrees/session-$SESSION_ID"
fi

exit 0
```

#### 종합 리포트 생성

```bash
#!/bin/bash
# .entire/hooks/stop.sh

# 세션 종료 시 종합 리포트

SESSION_ID="$1"
DURATION="$2"
CHECKPOINTS="$3"

REPORT_DIR=".entire/reports"
mkdir -p "$REPORT_DIR"

REPORT_FILE="$REPORT_DIR/session-$SESSION_ID-$(date +%Y%m%d-%H%M%S).md"

# 리포트 생성
cat > "$REPORT_FILE" <<EOF
# Session Report: $SESSION_ID

Generated: $(date)

## Summary

- Session ID: $SESSION_ID
- Duration: $DURATION seconds ($(($DURATION / 60)) minutes)
- Checkpoints: $CHECKPOINTS

## Session Info

$(entire session info "$SESSION_ID")

## Checkpoints

$(entire checkpoint list "$SESSION_ID")

## Statistics

$(entire session stats "$SESSION_ID")

## Files Changed

\`\`\`
$(entire session files "$SESSION_ID")
\`\`\`

## Git History

\`\`\`
$(entire git log "$SESSION_ID" --oneline)
\`\`\`

## Code Quality

\`\`\`
Lint: $(npm run lint 2>&1 | tail -1)
Tests: $(npm test 2>&1 | grep -E "passing|failing")
Coverage: $(npm test -- --coverage 2>&1 | grep "Lines")
\`\`\`

## Recommendations

EOF

# 추천 사항 자동 생성
if [ "$CHECKPOINTS" -lt 3 ]; then
  echo "- Consider creating more checkpoints for better tracking" >> "$REPORT_FILE"
fi

if [ "$DURATION" -gt 7200 ]; then
  echo "- Long session (>2h). Consider breaking into smaller sessions" >> "$REPORT_FILE"
fi

# 테스트 실패 확인
if ! npm test > /dev/null 2>&1; then
  echo "- Tests are failing. Fix before merging" >> "$REPORT_FILE"
fi

echo "" >> "$REPORT_FILE"
echo "---" >> "$REPORT_FILE"
echo "Report generated by Entire CLI" >> "$REPORT_FILE"

# 리포트 표시
if command -v glow &> /dev/null; then
  glow "$REPORT_FILE"
else
  cat "$REPORT_FILE"
fi

# 이메일 전송 (설정된 경우)
if [ -n "$EMAIL_REPORT_TO" ]; then
  mail -s "Entire Session Report: $SESSION_ID" \
    "$EMAIL_REPORT_TO" < "$REPORT_FILE"
fi

echo "✓ Report saved: $REPORT_FILE"

exit 0
```

### Error Hook

에러 발생 시 실행됩니다.

#### 기본 템플릿

```bash
# .entire/hooks/error.sh
#!/bin/bash

# Error Hook
# Called when an error occurs

# 환경 변수
# - ERROR_TYPE: 에러 타입
# - ERROR_MESSAGE: 에러 메시지
# - SESSION_ID: 세션 ID (있는 경우)
# - STACK_TRACE: 스택 트레이스 (있는 경우)

echo "Error occurred: $ERROR_TYPE"
echo "Message: $ERROR_MESSAGE"

# 에러 로깅
cat >> .entire/logs/errors.log <<EOF
[$(date)] ERROR
Type: $ERROR_TYPE
Message: $ERROR_MESSAGE
Session: $SESSION_ID
Stack:
$STACK_TRACE
---
EOF

# 알림
notify-send "Entire CLI Error" "$ERROR_MESSAGE" --urgency=critical

exit 0
```

#### 자동 복구

```bash
#!/bin/bash
# .entire/hooks/error.sh

# 에러 처리 및 자동 복구

ERROR_TYPE="$1"
ERROR_MESSAGE="$2"
SESSION_ID="$3"

echo "⚠️  Error: $ERROR_TYPE"

# 에러 타입별 처리
case "$ERROR_TYPE" in
  "git_conflict")
    echo "Attempting auto-merge..."
    if entire git resolve-conflicts --auto; then
      echo "✓ Conflicts resolved automatically"
      exit 0
    else
      echo "✗ Manual intervention required"
      exit 1
    fi
    ;;

  "checkpoint_corrupt")
    echo "Attempting checkpoint repair..."
    if entire checkpoint repair "$CHECKPOINT_ID"; then
      echo "✓ Checkpoint repaired"
      exit 0
    else
      echo "✗ Checkpoint unrecoverable"
      # 이전 checkpoint로 복원
      PREVIOUS=$(entire checkpoint list --previous)
      entire checkpoint restore "$PREVIOUS"
      exit 1
    fi
    ;;

  "session_locked")
    echo "Session appears to be locked by another process"
    read -p "Force unlock? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      entire session unlock "$SESSION_ID" --force
      exit 0
    else
      exit 1
    fi
    ;;

  *)
    echo "Unknown error type: $ERROR_TYPE"
    echo "Manual intervention required"
    exit 1
    ;;
esac
```

### Hook 설정 파일

```json
// .entire/hooks/config.json
{
  "version": "1.0",

  "hooks": {
    "session-start": {
      "enabled": true,
      "timeout": 30,
      "async": false
    },
    "prompt-submit": {
      "enabled": true,
      "timeout": 10,
      "async": false
    },
    "response-complete": {
      "enabled": true,
      "timeout": 60,
      "async": true
    },
    "checkpoint-create": {
      "enabled": true,
      "timeout": 30,
      "async": true
    },
    "stop": {
      "enabled": true,
      "timeout": 60,
      "async": false
    },
    "error": {
      "enabled": true,
      "timeout": 10,
      "async": false
    }
  },

  "environment": {
    "JIRA_TOKEN": "${JIRA_TOKEN}",
    "SLACK_WEBHOOK": "${SLACK_WEBHOOK}",
    "EMAIL_REPORT_TO": "dev@example.com"
  },

  "options": {
    "abortOnHookFailure": false,
    "logHookOutput": true,
    "maxConcurrentAsyncHooks": 3
  }
}
```

### 실전 예시

#### CI/CD 통합

```bash
# .entire/hooks/response-complete.sh

#!/bin/bash
# 자동 CI/CD 트리거

FILES_CHANGED="$1"

# 특정 파일 변경 시 CI 트리거
if echo "$FILES_CHANGED" | grep -q "src/.*\.ts$"; then
  echo "Source files changed - triggering CI"

  # GitHub Actions 트리거
  gh workflow run ci.yml \
    --ref "$(git branch --show-current)" \
    --field "trigger=entire-cli" \
    --field "session=$SESSION_ID"
fi

# 배포 가능 상태 확인
if npm run build && npm test; then
  # 프로덕션 준비 태그
  entire checkpoint tag current add "production-ready"

  # Slack 알림
  curl -X POST "$SLACK_WEBHOOK" \
    -d "{\"text\": \"✓ Code is production-ready (Session: $SESSION_ID)\"}"
fi
```

#### 자동 코드 리뷰

```bash
# .entire/hooks/checkpoint-create.sh

#!/bin/bash
# AI 코드 리뷰 자동 요청

CHECKPOINT_ID="$1"
CHECKPOINT_TYPE="$2"

# Committed checkpoint만 리뷰
if [ "$CHECKPOINT_TYPE" != "committed" ]; then
  exit 0
fi

# 변경된 파일 가져오기
FILES=$(entire checkpoint info "$CHECKPOINT_ID" --files)

# AI에게 코드 리뷰 요청
echo "Requesting AI code review..."

REVIEW_PROMPT="다음 checkpoint의 코드를 리뷰해줘:

Checkpoint: $CHECKPOINT_ID
Files changed:
$FILES

다음을 확인해줘:
1. 코드 품질
2. 잠재적 버그
3. 보안 이슈
4. 성능 문제
5. 개선 제안"

# Claude Code에 리뷰 요청
entire ai-review "$CHECKPOINT_ID" "$REVIEW_PROMPT" \
  > ".entire/reviews/checkpoint-$CHECKPOINT_ID-review.md"

echo "✓ Review saved"
```

### 베스트 프랙티스

```bash
# 1. Hooks는 간단하게 유지
# 복잡한 로직은 별도 스크립트로 분리

# 2. 실패 처리
# Hook이 실패해도 워크플로우가 중단되지 않도록

# 3. 비동기 실행
# 오래 걸리는 작업은 async로 설정

# 4. 로깅
# 모든 Hook 실행을 로그로 기록

# 5. 환경 변수 활용
# 민감한 정보는 환경 변수로 관리

# 6. 테스트
# Hook을 별도로 테스트할 수 있도록 작성
```

### 정리

이것으로 Entire CLI 완벽 가이드 시리즈를 마무리합니다.

**다룬 주제**:
1. 소개 및 개요
2. 설치 및 설정
3. 기본 사용법
4. 고급 기능
5. 실전 워크플로우
6. Strategy 개요
7. Manual-Commit Strategy
8. Auto-Commit Strategy
9. Session 관리
10. Checkpoint 시스템
11. Checkpoint ID 연결
12. Multi-Session 처리
13. Git 통합
14. Storage 구조
15. Claude Code Hooks

**다음 단계**:
- 실제 프로젝트에 적용
- 팀과 워크플로우 공유
- 커뮤니티 참여

---

**관련 문서**:
- [01. 소개 및 개요](/2026/02/11/entire-cli-guide-01-introduction/)
- [13. Git 통합](/2026/02/11/entire-cli-guide-13-git-integration/)
- [14. Storage 구조](/2026/02/11/entire-cli-guide-14-storage-structure/)
