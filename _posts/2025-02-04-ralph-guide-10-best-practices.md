---
layout: post
title: "Ralph 가이드 10 - 베스트 프랙티스"
date: 2025-02-04
categories: [AI, Claude Code, Ralph]
tags: [ralph, best-practices, tips, troubleshooting]
permalink: /ralph-guide-10-best-practices/
---

# 베스트 프랙티스

## 효과적인 프롬프트 작성

### PROMPT.md 작성 원칙

**좋은 PROMPT.md:**
```markdown
# Ralph Development Instructions

## Context
You are Ralph, building a REST API for a bookstore inventory.
The API serves a React frontend (separate repo).

## Current Objectives
1. Implement CRUD endpoints for books and authors
2. Add JWT authentication
3. Implement pagination for list endpoints
4. Write comprehensive tests

## Key Principles
- Use FastAPI with async operations
- Follow REST conventions (proper status codes, HATEOAS links)
- Every endpoint must have tests
- Use Pydantic for validation
- Document with OpenAPI (auto-generated)

## Technical Constraints
- Python 3.11+
- PostgreSQL 15+
- Must support SQLite for testing
- Response time < 200ms for list operations
```

**나쁜 PROMPT.md:**
```markdown
# Instructions
Build a good API.
Make it fast.
Add tests.
```

### fix_plan.md 작성 원칙

**좋은 작업:**
```markdown
## Priority 1: Foundation
- [ ] Create Book model with fields: id, title, isbn, author_id, price, published_date
- [ ] Create Author model with fields: id, name, bio, birth_date
- [ ] Set up Alembic migrations
- [ ] Create database connection pool

## Priority 2: Book Endpoints
- [ ] POST /books - create book (validate ISBN format)
- [ ] GET /books - list with pagination (limit, offset, sort)
- [ ] GET /books/{id} - single book with author details
- [ ] PUT /books/{id} - update book (partial updates allowed)
- [ ] DELETE /books/{id} - soft delete (set deleted_at)
```

**나쁜 작업:**
```markdown
- [ ] Make the database
- [ ] Add endpoints
- [ ] Fix bugs
- [ ] Make it better
```

### 작업 세분화 기준

| 작업 크기 | 예시 | 권장 |
|----------|------|------|
| 너무 큼 | "Build the entire API" | 분할 필요 |
| 적절함 | "Create POST /books endpoint" | ✓ |
| 너무 작음 | "Add import statement" | 병합 필요 |

## 프로젝트 구조화

### 권장 디렉토리 구조

```
my-project/
├── .ralph/
│   ├── PROMPT.md           # 간결하게 유지
│   ├── fix_plan.md         # 구체적인 작업
│   ├── specs/              # 필요할 때만
│   │   ├── api.md          # API 스펙
│   │   └── stdlib/         # 표준 패턴
│   └── AGENT.md            # 자동 관리
├── src/                    # 소스 코드
├── tests/                  # 테스트
└── .ralphrc                # 프로젝트 설정
```

### specs/ 사용 시점

**specs/ 필요:**
- 복잡한 유효성 검사 규칙
- 외부 API 통합 요구사항
- 정확한 에러 응답 형식
- 비즈니스 로직 상세 정의

**specs/ 불필요:**
- 간단한 CRUD 작업
- 표준적인 구현
- PROMPT.md로 충분한 경우

## 효율적인 워크플로우

### 시작 전 체크리스트

```bash
# 1. 환경 확인
claude --version
ralph --version

# 2. Git 상태 확인
git status

# 3. 기존 세션 확인
ralph --status

# 4. 설정 검토
cat .ralphrc
```

### 실행 중 모니터링

```bash
# 모니터링 모드로 실행
ralph --monitor

# 또는 별도 터미널에서
watch -n 10 ralph --status

# 로그 스트리밍
tail -f .ralph/logs/session_latest.log
```

### 종료 후 검토

```bash
# 결과 확인
git diff
git status

# 테스트 실행
npm test  # 또는 해당 언어의 테스트 명령

# fix_plan.md 검토
cat .ralph/fix_plan.md
```

## 문제 해결

### 일반적인 문제와 해결책

#### 1. Ralph가 같은 작업을 반복함

**증상:** 같은 작업을 계속 시도하지만 완료하지 못함

**원인과 해결:**
| 원인 | 해결 |
|------|------|
| 작업이 너무 모호함 | 더 구체적으로 분할 |
| 테스트가 항상 실패 | 테스트 요구사항 확인 |
| 의존성 문제 | 선행 작업 먼저 완료 |

```markdown
# 나쁨
- [ ] Add authentication

# 좋음
- [ ] Install jsonwebtoken package
- [ ] Create JWT utility functions in src/utils/jwt.js
- [ ] Add /auth/login endpoint
- [ ] Add authentication middleware
- [ ] Protect /books endpoints with auth middleware
```

#### 2. 조기 종료

**증상:** 작업이 남았는데 Ralph가 종료함

**확인사항:**
```bash
# 상태 확인
ralph --status

# 가능한 원인들:
# - SESSION_TIMEOUT 도달
# - MAX_LOOPS_PER_SESSION 도달
# - MAX_CALLS_PER_HOUR 도달 후 쿨다운
# - 서킷 브레이커 OPEN
```

**해결:**
```bash
# 세션 재개
ralph --resume

# 또는 설정 조정 후 재실행
vim .ralphrc  # 타임아웃/한도 증가
ralph --monitor
```

#### 3. 무한 루프 감지

**증상:** 서킷 브레이커가 OPEN 상태

**확인:**
```bash
# 서킷 상태 확인
ralph --status

# 에러 로그 확인
grep "\[ERROR\]" .ralph/logs/session_latest.log | tail -20
```

**해결:**
```bash
# 문제 수정 후 서킷 리셋
ralph --reset-circuit

# 또는 수동으로
rm .ralph/status.json
```

#### 4. 테스트 실패 반복

**증상:** 테스트가 계속 실패하고 Ralph가 수정하지 못함

**해결 방법:**
1. 테스트 요구사항이 명확한지 확인
2. specs/에 테스트 기대값 명시
3. 테스트를 더 작은 단위로 분할

```markdown
# specs/testing.md
## Test Requirements

Each endpoint test must verify:
1. Success case (200/201)
2. Validation error (400)
3. Not found (404)
4. Auth required (401)

Example test structure:
describe('POST /books', () => {
  it('creates book with valid data')
  it('rejects invalid ISBN format')
  it('requires authentication')
})
```

### 디버깅 팁

#### 상세 로그 활성화

```bash
# 디버그 모드로 실행
RALPH_LOG_LEVEL=debug ralph --monitor

# 또는 .ralphrc에서
LOG_LEVEL="debug"
```

#### 단계별 실행

```bash
# dry-run으로 계획 확인
ralph --dry-run

# 한 루프만 실행
ralph --max-loops 1
```

#### 상태 검사

```bash
# 전체 상태 덤프
cat .ralph/status.json | jq '.'

# 진행 상황 확인
grep "\[x\]" .ralph/fix_plan.md | wc -l  # 완료된 작업 수
grep "\[ \]" .ralph/fix_plan.md | wc -l  # 남은 작업 수
```

## 성능 최적화

### API 호출 최소화

```bash
# 보수적인 설정
MAX_CALLS_PER_HOUR=50
COOLDOWN_MINUTES=10

# 작업을 더 구체적으로 = 더 적은 시도
```

### 효율적인 작업 구조

```markdown
# 비효율적 (많은 컨텍스트 스위칭)
- [ ] Add user model
- [ ] Add book model
- [ ] Add user routes
- [ ] Add book routes
- [ ] Add user tests
- [ ] Add book tests

# 효율적 (연관 작업 그룹화)
## User Feature
- [ ] Add user model with validation
- [ ] Add user CRUD routes
- [ ] Add user route tests

## Book Feature
- [ ] Add book model with validation
- [ ] Add book CRUD routes
- [ ] Add book route tests
```

### 리소스 관리

```bash
# 장시간 실행 시 로그 정리
find .ralph/logs -name "*.log" -mtime +3 -delete

# 불필요한 세션 정리
ralph --cleanup-sessions
```

## 안전한 사용

### 권한 제한

```bash
# 최소 권한 원칙
ALLOWED_TOOLS="Write,Read,Edit,Bash(git status),Bash(npm test)"

# 위험한 명령 차단
# Bash(rm -rf) - 절대 허용하지 말 것
# Bash(*) - 가능하면 피할 것
```

### 백업 전략

```bash
# Ralph 실행 전 커밋
git add -A && git commit -m "Before Ralph session"

# 또는 브랜치 사용
git checkout -b ralph-work
ralph --monitor
# 검토 후 머지
```

### 검토 프로세스

```bash
# Ralph 완료 후 검토
git diff HEAD~5..HEAD  # 최근 5 커밋 확인
npm test               # 테스트 확인
npm run lint           # 린트 확인

# 문제 있으면 롤백
git reset --hard HEAD~1  # 마지막 커밋 취소
```

## 요약 체크리스트

### 시작 전
- [ ] PROMPT.md가 명확한가?
- [ ] fix_plan.md 작업이 구체적인가?
- [ ] Git 상태가 깨끗한가?
- [ ] 설정(.ralphrc)이 적절한가?

### 실행 중
- [ ] 모니터링 대시보드 확인
- [ ] 진행 상황 주기적 검토
- [ ] 비정상 동작 시 개입

### 완료 후
- [ ] 변경사항 검토
- [ ] 테스트 통과 확인
- [ ] 코드 품질 검토
- [ ] 필요시 수정 후 재실행

---

**이전 장:** [모니터링](/ralph-guide-09-monitoring/) | **처음으로:** [소개](/ralph-guide-01-intro/)
