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
├── PROMPT.md      # 프로젝트 비전
├── fix_plan.md    # 작업 목록
├── AGENT.md       # 빌드 명령어
├── specs/         # 상세 스펙
├── logs/          # 실행 로그
└── status.json    # 런타임 상태
```

## 핵심 파일 요약

<div class="file-cards">

<div class="file-card">
<h4>📄 PROMPT.md</h4>
<p><strong>작성자:</strong> 사용자</p>
<p><strong>용도:</strong> 프로젝트 목표 정의</p>
<p>✏️ 직접 편집 필요</p>
</div>

<div class="file-card">
<h4>📋 fix_plan.md</h4>
<p><strong>작성자:</strong> 사용자 + Ralph</p>
<p><strong>용도:</strong> 작업 체크리스트</p>
<p>✏️ 작업 추가/수정</p>
</div>

<div class="file-card">
<h4>🔧 AGENT.md</h4>
<p><strong>작성자:</strong> Ralph (자동)</p>
<p><strong>용도:</strong> 빌드/테스트 명령</p>
<p>🔒 거의 편집 불필요</p>
</div>

<div class="file-card">
<h4>📁 specs/</h4>
<p><strong>작성자:</strong> 사용자</p>
<p><strong>용도:</strong> 상세 요구사항</p>
<p>✏️ 필요시 추가</p>
</div>

<div class="file-card">
<h4>⚙️ .ralphrc</h4>
<p><strong>작성자:</strong> 자동 생성</p>
<p><strong>용도:</strong> 프로젝트 설정</p>
<p>🔒 거의 편집 불필요</p>
</div>

<div class="file-card">
<h4>📊 logs/ & status.json</h4>
<p><strong>작성자:</strong> Ralph (자동)</p>
<p><strong>용도:</strong> 로그 및 상태</p>
<p>👁️ 읽기 전용</p>
</div>

</div>

---

## PROMPT.md - 프로젝트 비전

### 목적

프로젝트의 전체적인 방향과 원칙을 정의합니다. Ralph는 매 루프 시작 시 이 파일을 읽어 컨텍스트를 파악합니다.

### 포함할 내용

```markdown
# Ralph Development Instructions

## Context
You are Ralph, building [프로젝트].

## Current Objectives
1. [목표 1]
2. [목표 2]

## Key Principles
- [원칙 1]
- [원칙 2]

## Technology Stack
- Language: TypeScript
- Framework: Express
- Testing: Jest
```

### 포함하지 말 것

| 내용 | 대신 사용할 곳 |
|------|---------------|
| 단계별 작업 | `fix_plan.md` |
| API 스펙 | `specs/` |
| 빌드 명령 | `AGENT.md` |

### 좋은 예시

```markdown
## Context
You are Ralph, building a REST API
for a bookstore inventory.

## Key Principles
- Use FastAPI with async operations
- Follow REST conventions strictly
- Every endpoint needs tests

## Constraints
- Support PostgreSQL and SQLite
- Response time under 200ms
```

---

## fix_plan.md - 작업 체크리스트

### 목적

Ralph가 수행할 구체적인 작업 목록입니다. Ralph는 체크되지 않은 작업을 찾아 구현하고, 완료 시 체크합니다.

### 형식

```markdown
# Fix Plan

## Priority 1: 기초
- [ ] 구체적인 작업 1
- [ ] 구체적인 작업 2
- [x] 완료된 작업

## Priority 2: 기능
- [ ] 작업 3
- [ ] 작업 4
```

### 좋은 작업 vs 나쁜 작업

**✅ 좋은 작업 (구체적):**
```markdown
- [ ] Create POST /books endpoint
- [ ] Add pagination to GET /books
- [ ] Write test for ISBN validation
```

**❌ 나쁜 작업 (모호함):**
```markdown
- [ ] Make the API work
- [ ] Add features
- [ ] Fix bugs
```

### Ralph의 작업 흐름

1. **읽기** → fix_plan.md에서 `[ ]` 찾기
2. **구현** → 해당 작업 수행
3. **테스트** → 테스트 실행
4. **완료** → `[x]`로 체크
5. **반복** → 다음 작업으로

---

## specs/ - 상세 스펙

### 언제 사용하는가?

- PROMPT.md로 설명하기에 너무 상세할 때
- 정확한 API 계약이 필요할 때
- 특정 유효성 검사 규칙이 있을 때
- 외부 시스템 통합 요구사항

### 구조 예시

```
specs/
├── api-contracts.md
├── data-models.md
└── stdlib/
    ├── error-handling.md
    └── logging.md
```

### specs/stdlib/ - 표준 패턴

프로젝트 전체에서 일관되게 사용할 패턴:

```markdown
# Error Handling Standard

All API errors must return:
{
  "error": {
    "code": "BOOK_NOT_FOUND",
    "message": "No book exists"
  }
}

HTTP Status Codes:
- 400: Validation errors
- 404: Not found
- 409: Conflict
- 500: Internal errors
```

---

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

## Environment
- Node.js: 18
- Package manager: npm
```

### 편집이 필요한 경우

- 복잡한 환경 설정
- 특수한 빌드 단계
- 배포 명령어 문서화

---

## .ralphrc - 프로젝트 설정

### 기본 설정

```bash
# 프로젝트 정보
PROJECT_NAME="my-project"
PROJECT_TYPE="typescript"

# 속도 제한
MAX_CALLS_PER_HOUR=100

# 허용된 도구
ALLOWED_TOOLS="Write,Read,Edit"

# 타임아웃
SESSION_TIMEOUT=3600
LOOP_TIMEOUT=300
```

### 주요 설정 옵션

| 옵션 | 기본값 |
|------|--------|
| `MAX_CALLS_PER_HOUR` | 100 |
| `SESSION_TIMEOUT` | 3600초 |
| `LOOP_TIMEOUT` | 300초 |

---

## 파일 관계도

<div class="flow-diagram">
<div class="flow-item flow-top">
<strong>PROMPT.md</strong>
<span>프로젝트 목표와 원칙</span>
</div>
<div class="flow-arrow">↓</div>
<div class="flow-item">
<strong>specs/</strong>
<span>상세 요구사항 (필요시)</span>
</div>
<div class="flow-arrow">↓</div>
<div class="flow-item">
<strong>fix_plan.md</strong>
<span>구체적인 작업 목록</span>
</div>
<div class="flow-arrow">↓</div>
<div class="flow-item flow-bottom">
<strong>AGENT.md</strong>
<span>빌드/테스트 방법</span>
</div>
</div>

---

## 일반적인 시나리오

### 시나리오 1: 간단한 기능 추가

fix_plan.md만 편집:

```markdown
- [ ] Add /health endpoint
```

### 시나리오 2: 복잡한 기능

**Step 1:** specs/ 파일 먼저 생성

```markdown
# specs/search-feature.md

## Requirements
- Full-text search on titles
- Support exact phrase matching
- Support fuzzy matching
```

**Step 2:** fix_plan.md에 참조 추가

```markdown
- [ ] Implement search per
      specs/search-feature.md
```

---

**이전 장:** [설치 및 시작](/ralph-guide-02-installation/) | **다음 장:** [핵심 개념](/ralph-guide-04-concepts/)

<style>
.file-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin: 1.5rem 0;
}

.file-card {
  background: var(--card-bg, #f8f9fa);
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 8px;
  padding: 1rem;
}

.file-card h4 {
  margin: 0 0 0.5rem 0;
  font-size: 0.95rem;
}

.file-card p {
  margin: 0.25rem 0;
  font-size: 0.85rem;
  color: var(--text-muted, #666);
}

.flow-diagram {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  margin: 1.5rem 0;
  padding: 1rem;
}

.flow-item {
  background: var(--card-bg, #f0f7ff);
  border: 2px solid #059669;
  border-radius: 8px;
  padding: 0.75rem 1rem;
  text-align: center;
  width: 100%;
  max-width: 280px;
}

.flow-item strong {
  display: block;
  color: #059669;
}

.flow-item span {
  font-size: 0.85rem;
  color: var(--text-muted, #666);
}

.flow-arrow {
  font-size: 1.5rem;
  color: #059669;
}

@media (prefers-color-scheme: dark) {
  .file-card {
    --card-bg: #1e1e2e;
    --border-color: #333;
  }
  .flow-item {
    --card-bg: #1a2e1a;
  }
}
</style>
