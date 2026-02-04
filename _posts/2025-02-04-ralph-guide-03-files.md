---
layout: post
title: "Ralph 가이드 03 - 파일 구조"
date: 2025-02-04
categories: [AI, Claude Code, Ralph]
tags: [ralph, files, prompt, fix-plan, configuration]
series: "ralph-guide"
permalink: /ralph-guide-03-files/
---

# 파일 구조

## .ralph/ 디렉토리 개요

```
.ralph/
├── PROMPT.md           # 프로젝트 비전과 지침
├── fix_plan.md         # 작업 체크리스트
├── AGENT.md            # 빌드/테스트 명령어
├── specs/              # 상세 스펙 문서
│   └── stdlib/         # 표준 패턴 문서
├── logs/               # 실행 로그
│   ├── session_001.log
│   └── session_002.log
└── status.json         # 런타임 상태
```

## 파일 참조 테이블

| 파일 | 자동 생성? | 작성자 | 읽는 주체 | 권장 행동 |
|------|-----------|--------|-----------|-----------|
| `PROMPT.md` | 기본 템플릿 | **사용자** | Ralph | 프로젝트 목표 정의 |
| `fix_plan.md` | 기본 템플릿 | **사용자** + Ralph | Ralph | 작업 추가/수정 |
| `AGENT.md` | 빌드 명령 감지 | Ralph | Ralph | 거의 편집 불필요 |
| `specs/` | 빈 디렉토리 | **사용자** | Ralph | 필요시 상세 스펙 추가 |
| `.ralphrc` | 프로젝트 인식 | 보통 그대로 | Ralph | 거의 편집 불필요 |
| `logs/` | 자동 생성 | Ralph | 사용자 | 읽기 전용 |
| `status.json` | 런타임 생성 | Ralph | 모니터링 툴 | 읽기 전용 |

## PROMPT.md - 프로젝트 비전

### 목적

프로젝트의 전체적인 방향과 원칙을 정의합니다. Ralph는 매 루프 시작 시 이 파일을 읽어 컨텍스트를 파악합니다.

### 포함할 내용

```markdown
# Ralph Development Instructions

## Context
You are Ralph, an autonomous AI agent building [프로젝트 설명].

## Current Objectives
1. [목표 1]
2. [목표 2]
3. [목표 3]

## Key Principles
- [원칙 1]
- [원칙 2]

## Technology Stack
- Language: TypeScript
- Framework: Express
- Database: PostgreSQL
- Testing: Jest

## Quality Standards
- All code must have tests
- Follow ESLint rules
- Document public APIs
```

### 포함하지 말아야 할 내용

| 포함하지 말 것 | 대신 사용할 곳 |
|---------------|---------------|
| 단계별 구현 작업 | `fix_plan.md` |
| 상세 API 스펙 | `specs/` |
| 빌드 명령어 | `AGENT.md` |

### 좋은 예시

```markdown
## Context
You are Ralph, building a REST API for a bookstore inventory system.

## Key Principles
- Use FastAPI with async database operations
- Follow REST conventions strictly
- Every endpoint needs tests
- Document all API endpoints with OpenAPI

## Constraints
- Must support PostgreSQL and SQLite
- Response time under 200ms for list operations
- Maximum 1000 records per page
```

## fix_plan.md - 작업 체크리스트

### 목적

Ralph가 수행할 구체적인 작업 목록입니다. Ralph는 체크되지 않은 작업을 찾아 구현하고, 완료 시 체크합니다.

### 형식

```markdown
# Fix Plan - [프로젝트명]

## Priority 1: [카테고리]
- [ ] 구체적인 작업 1
- [ ] 구체적인 작업 2
- [x] 완료된 작업

## Priority 2: [카테고리]
- [ ] 작업 3
- [ ] 작업 4
```

### 좋은 작업 vs 나쁜 작업

**좋은 작업 (구체적):**
```markdown
- [ ] Create POST /books endpoint that accepts {title, author, isbn}
- [ ] Add pagination to GET /books (limit, offset params)
- [ ] Write test for duplicate ISBN validation
```

**나쁜 작업 (모호함):**
```markdown
- [ ] Make the API work
- [ ] Add features
- [ ] Fix bugs
```

### Ralph의 작업 처리

```
fix_plan.md 읽기
    │
    ▼
[ ] 체크되지 않은 작업 찾기
    │
    ▼
작업 구현
    │
    ▼
테스트 실행
    │
    ▼
[x] 작업 완료 체크
    │
    ▼
(새 작업 발견 시 추가)
```

## specs/ - 상세 스펙

### 언제 사용하는가?

- PROMPT.md로 설명하기에 너무 상세한 요구사항
- 정확히 따라야 하는 API 계약
- 특정 유효성 검사 규칙이 있는 데이터 모델
- 외부 시스템 통합 요구사항

### 구조 예시

```
specs/
├── api-contracts.md      # API 엔드포인트 정의
├── data-models.md        # 엔티티 관계와 유효성 검사
├── third-party-auth.md   # OAuth 통합 요구사항
└── stdlib/
    ├── error-handling.md # 에러 처리 패턴
    └── logging.md        # 로깅 규칙
```

### specs/stdlib/ - 표준 패턴

프로젝트 전체에서 일관되게 사용할 패턴을 정의합니다.

```markdown
# Error Handling Standard

All API errors must return:
{
  "error": {
    "code": "BOOK_NOT_FOUND",
    "message": "No book with ID 123 exists",
    "details": {}
  }
}

Use HTTPException with these codes:
- 400: Validation errors
- 404: Resource not found
- 409: Conflict (duplicate)
- 500: Internal errors (log full trace)
```

## AGENT.md - 빌드 지침

### 목적

프로젝트를 빌드하고 테스트하는 방법을 기록합니다. Ralph가 자동으로 감지하고 유지합니다.

### 자동 생성 내용

```markdown
# Agent Instructions

## Build Commands
- Install: npm install
- Build: npm run build
- Test: npm test
- Lint: npm run lint

## Project Structure
- Source: src/
- Tests: tests/
- Config: package.json

## Environment
- Node.js version: 18
- Package manager: npm
```

### 편집이 필요한 경우

- 복잡한 환경 설정이 필요한 경우
- 특수한 빌드 단계가 있는 경우
- 배포 명령어 문서화

## .ralphrc - 프로젝트 설정

### 기본 설정

```bash
# 프로젝트 정보
PROJECT_NAME="my-project"
PROJECT_TYPE="typescript"

# 속도 제한
MAX_CALLS_PER_HOUR=100

# 허용된 도구
ALLOWED_TOOLS="Write,Read,Edit,Bash(git *),Bash(npm *),Bash(pytest)"

# 타임아웃
SESSION_TIMEOUT=3600
LOOP_TIMEOUT=300
```

### 설정 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `MAX_CALLS_PER_HOUR` | 100 | 시간당 최대 API 호출 |
| `SESSION_TIMEOUT` | 3600 | 세션 타임아웃 (초) |
| `LOOP_TIMEOUT` | 300 | 단일 루프 타임아웃 (초) |
| `ALLOWED_TOOLS` | 기본 도구 | 허용된 Claude 도구 |

## 파일 관계도

```
┌─────────────────────────────────────────────────────────────┐
│                         PROMPT.md                           │
│            (High-level goals and principles)                │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                      specs/                          │   │
│  │         (Detailed requirements when needed)          │   │
│  │                                                      │   │
│  │  specs/api.md ──────▶ Informs fix_plan.md tasks     │   │
│  │  specs/stdlib/ ─────▶ Conventions Ralph follows     │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    fix_plan.md                       │   │
│  │          (Concrete tasks Ralph executes)             │   │
│  │                                                      │   │
│  │  [ ] Task 1 ◄────── Ralph checks off when done      │   │
│  │  [x] Task 2                                         │   │
│  │  [ ] Task 3 ◄────── Ralph adds discovered tasks     │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                     AGENT.md                         │   │
│  │        (How to build/test - auto-maintained)         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 일반적인 시나리오

### 시나리오 1: 간단한 기능 추가

fix_plan.md만 편집:
```markdown
- [ ] Add a /health endpoint that returns {"status": "ok"}
```

### 시나리오 2: 복잡한 기능

1. specs/ 파일 먼저 생성:
```markdown
# specs/search-feature.md
## Requirements
- Full-text search on book titles
- Must support exact phrase matching
- Must support fuzzy matching
```

2. fix_plan.md에 참조 추가:
```markdown
- [ ] Implement search per specs/search-feature.md
```

---

**이전 장:** [설치 및 시작](/ralph-guide-02-installation/) | **다음 장:** [핵심 개념](/ralph-guide-04-concepts/)
