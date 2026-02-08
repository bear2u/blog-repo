---
layout: post
title: "Mux 완벽 가이드 (04) - 에이전트 시스템"
date: 2026-02-08 00:00:00 +0900
categories: [AI 코딩, 개발 도구]
tags: [Mux, 에이전트, Plan모드, Exec모드, 서브에이전트, 워크플로우, 병렬처리]
author: cataclysm99
original_url: "https://github.com/coder/mux"
excerpt: "Plan/Exec 모드, 서브에이전트 위임, 병렬 워크플로우를 활용한 에이전트 시스템 완벽 가이드"
permalink: /mux-guide-04-agents/
toc: true
related_posts:
  - /blog-repo/2026-02-08-mux-guide-03-workspaces
  - /blog-repo/2026-02-08-mux-guide-05-multimodel
---

## 에이전트 개념

Mux의 에이전트는 **시스템 프롬프트**와 **도구 접근 정책**을 정의하는 Markdown 파일입니다.

### 에이전트 vs 모드

```
에이전트 = 시스템 프롬프트 + 도구 정책

과거 (레거시):
- UI 모드: Plan/Exec/Compact
- 서브에이전트: task 도구가 사용하는 프리셋

현재 (통합):
- 에이전트: 모든 것을 통합한 Markdown 정의
```

### 에이전트 구성 요소

```markdown
---
name: My Agent           # UI 표시 이름
description: ...         # 툴팁 설명
base: exec              # 상속받을 에이전트
tools:                  # 도구 정책 (정규식)
  add:
    - file_read
    - bash
  remove:
    - task_.*
---

시스템 프롬프트 내용 (Markdown)
```

---

## 빌트인 에이전트

### Plan (계획 수립)

**목적**: 코드 수정 전 계획 작성 및 검토

```yaml
name: Plan
description: Create a plan before coding
ui:
  color: var(--color-plan-mode)
tools:
  add:
    - .*  # 모든 도구 (MCP 포함)
  remove:
    - task_apply_git_patch  # 패치 적용 불가
```

#### 동작 흐름

```
1. 사용자 요청
   ↓
2. 저장소 조사 (file_read, bash)
   ↓
3. 플랜 파일 작성 (~/.mux/plans/<project>/<workspace>.md)
   ↓
4. propose_plan 호출 (UI 렌더링)
   ↓
5. 사용자 검토
   - 승인 → Exec 모드 전환
   - 수정 요청 → 플랜 수정
   - 외부 편집기로 수정 → 자동 감지
```

#### 플랜 파일 구조

```markdown
## Context

사용자가 요청한 작업과 목표를 간략히 설명

## Evidence

- 참조한 파일 경로
- 도구 출력 결과
- 사용자 제공 정보

## Implementation

### Step 1: 파일명 + 심볼명

변경 내용 설명

\`\`\`typescript
// 코드 스니펫 (필요 시)
function authenticate(token: string) {
  // ...
}
\`\`\`

### Step 2: ...

## Questions (선택사항)

- 불확실한 사항 리스트
```

#### 사용 예시

````
⌘+Shift+M → Plan 모드

사용자: "Add OAuth2 authentication with Google provider"

에이전트:
1. 기존 인증 코드 조사 (file_read src/auth/*)
2. 플랜 작성:
   - Google OAuth2 라이브러리 설치
   - src/auth/google.ts 생성
   - src/routes/auth.ts 수정
   - 환경 변수 추가 (.env.example)
   - 테스트 작성
3. propose_plan 호출

사용자:
[Edit] 버튼 클릭 → VS Code에서 플랜 수정
→ 채팅 입력 → 자동 diff 감지

에이전트:
수정사항 반영 → propose_plan 재호출

사용자:
"Looks good, proceed"

에이전트:
⌘+Shift+M → Exec 모드 자동 전환 → 구현 시작
````

### Exec (실행)

**목적**: 코드 수정 및 검증

```yaml
name: Exec
description: Implement changes in the repository
tools:
  add:
    - .*  # 모든 도구
  remove:
    - propose_plan
    - ask_user_question
```

#### 동작 흐름

```
1. Plan 모드에서 플랜 승인 (또는 직접 Exec 모드)
   ↓
2. 파일 수정 (file_edit_replace_string, file_edit_insert)
   ↓
3. 검증 (bash: npm test, make lint)
   ↓
4. 실패 시 재시도
   ↓
5. 성공 시 커밋 (선택사항)
   ↓
6. agent_report 호출 (완료 보고)
```

#### 사용 예시

```
⌘+Shift+M → Exec 모드

사용자: "Fix the failing Jest tests"

에이전트:
1. npm test 실행 → 실패 로그 분석
2. src/__tests__/auth.test.ts 수정
3. npm test 재실행 → 성공
4. agent_report: "Fixed 3 failing tests in auth.test.ts"

사용자: "Commit the changes"

에이전트:
1. git add src/__tests__/auth.test.ts
2. git commit -m "Fix failing auth tests"
3. agent_report: "Committed changes"
```

### Ask (질문 답변)

**목적**: 서브에이전트 위임 및 결과 종합

```yaml
name: Ask
description: Delegate questions to Explore sub-agents and synthesize an answer
base: exec
tools:
  remove:
    - file_edit_.*  # 읽기 전용
```

#### 동작 흐름

```
1. 사용자 질문
   ↓
2. 질문을 1-3개 연구 스레드로 분해
   ↓
3. Explore 서브에이전트 병렬 실행
   ↓
4. task_await로 결과 대기
   ↓
5. 결과 종합 및 답변
```

#### 사용 예시

```
⌘+Shift+M → Ask 모드

사용자: "Where is the database connection logic?"

에이전트:
1. task({ agentId: "explore", title: "Find DB connection", prompt: "..." })
2. task({ agentId: "explore", title: "Find DB config", prompt: "..." })
3. task_await([task1, task2])
4. 결과 종합:

"Database connection logic:

1. **Connection Factory**: `src/db/connection.ts`
   - createConnection() function
   - Uses environment variables from .env

2. **Configuration**: `src/config/database.ts`
   - Default pool size: 10
   - Timeout: 30s

3. **Initialization**: `src/server.ts:42`
   - Called during app startup"
```

### Explore (읽기 전용 탐색)

**목적**: 저장소 조사 및 정보 수집 (서브에이전트 전용)

```yaml
name: Explore
description: Read-only exploration of repository
base: exec
ui:
  hidden: true  # UI에서 숨김 (서브에이전트 전용)
subagent:
  runnable: true
  skip_init_hook: true
tools:
  remove:
    - file_edit_.*
    - task
    - task_apply_git_patch
    - task_.*
```

#### 사용 예시 (서브에이전트)

```javascript
// Exec 에이전트가 Explore 서브에이전트 실행

사용자: "Implement user profile page"

Exec 에이전트:
1. task({
     agentId: "explore",
     title: "Find existing profile code",
     prompt: "Locate user profile related components and API endpoints"
   })
2. task_await([task1])
3. Explore 보고서:
   - Components: src/components/UserProfile.tsx
   - API: src/api/users.ts:getUserProfile()
   - Tests: src/__tests__/profile.test.tsx
4. 위 정보 기반으로 구현
```

### Orchestrator (조정자)

**목적**: 서브에이전트 조정 및 패치 통합

```yaml
name: Orchestrator
description: Coordinate sub-agent implementation and apply patches
base: exec
ui:
  requires:
    - plan  # Plan 모드에서만 사용
subagent:
  runnable: false
tools:
  add:
    - ask_user_question
  remove:
    - propose_plan
```

#### 동작 흐름

```
1. Plan 승인
   ↓
2. 독립적 서브태스크 식별
   ↓
3. Exec 서브에이전트 병렬 실행 (run_in_background: true)
   ↓
4. task_await로 대기
   ↓
5. 각 서브에이전트 패치 적용
   - task_apply_git_patch (dry_run: true)
   - 충돌 없으면 실제 적용
   - 충돌 시 해결 또는 위임
   ↓
6. 통합 검증 (Explore 서브에이전트)
   ↓
7. 완료 보고
```

#### 사용 예시

```
Plan 모드에서 플랜 승인
→ Exec 모드 전환 (Orchestrator 자동 활성화)

Orchestrator:
1. 플랜 분석:
   - Task A: OAuth2 라이브러리 설치 및 설정
   - Task B: Google provider 구현
   - Task C: 라우트 추가
   - Task D: 테스트 작성

2. 의존성 분석:
   - Task A: 독립
   - Task B: Task A 의존 (라이브러리 필요)
   - Task C: Task B 의존 (provider 필요)
   - Task D: Task B, C 의존

3. 배치 1 (병렬):
   - task({ agentId: "exec", title: "Install OAuth2", run_in_background: true })

4. task_await([batch1])

5. 패치 적용:
   - task_apply_git_patch(taskId, dry_run: true)
   - task_apply_git_patch(taskId, dry_run: false)

6. 배치 2 (병렬):
   - task({ agentId: "exec", title: "Implement Google provider", ... })
   - task({ agentId: "exec", title: "Add routes", ... })

7. 반복...

8. 통합 검증:
   - task({ agentId: "explore", prompt: "Run tests and verify" })

9. agent_report: "All tasks completed successfully"
```

---

## 에이전트 루프 아키텍처

### 표준 루프

```
┌─────────────────────────────────────┐
│  사용자 메시지                       │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  LLM 추론                            │
│  - 시스템 프롬프트                   │
│  - 대화 히스토리                     │
│  - 도구 정의                         │
└──────────────┬──────────────────────┘
               ▼
       ┌──────────────┐
       │  도구 호출?   │
       └──────┬───────┘
              │
     ┌────────┴────────┐
     │                 │
    Yes               No
     │                 │
     ▼                 ▼
┌─────────────┐  ┌─────────────┐
│ 도구 실행    │  │ 최종 응답    │
└─────┬───────┘  └─────────────┘
      │
      ▼
   루프 반복
```

### Streaming 흐름

```
LLM 스트리밍 시작
    ↓
┌─────────────────────────┐
│  텍스트 청크            │ → UI 렌더링
├─────────────────────────┤
│  도구 호출 청크         │ → 도구 카드 표시
├─────────────────────────┤
│  도구 실행 결과         │ → 카드 업데이트
├─────────────────────────┤
│  계속 텍스트 청크...    │ → UI 렌더링
└─────────────────────────┘
    ↓
스트리밍 종료
```

---

## Plan 모드 상세

### ask_user_question (Plan 전용)

Plan 모드에서만 사용 가능한 대화형 질문 도구입니다.

```typescript
ask_user_question({
  questions: [
    {
      question: "어떤 OAuth 제공자를 사용하시겠습니까?",
      options: ["Google", "GitHub", "Facebook", "Other"]
    },
    {
      question: "세션 저장소는?",
      options: ["Redis", "PostgreSQL", "JWT only"]
    }
  ]
})
```

#### UI 렌더링

```
┌───────────────────────────────────────┐
│  Questions (2)                        │
├───────────────────────────────────────┤
│  1. 어떤 OAuth 제공자를 사용하시겠습니까?│
│  ○ Google                             │
│  ○ GitHub                             │
│  ○ Facebook                           │
│  ○ Other: [___________]               │
├───────────────────────────────────────┤
│  2. 세션 저장소는?                     │
│  ○ Redis                              │
│  ○ PostgreSQL                         │
│  ○ JWT only                           │
├───────────────────────────────────────┤
│  [Submit answers]                     │
└───────────────────────────────────────┘
```

#### 응답 처리

```
사용자 응답:
- 폼 제출 → 구조화된 답변
- 일반 채팅 → ask_user_question 취소, 자유 형식 메시지
```

### 외부 편집기 연동

```
1. propose_plan 호출 → 플랜 파일 저장 (~/.mux/plans/...)
2. [Edit] 버튼 클릭 → 외부 편집기 열기 (nvim, VS Code)
3. 사용자 수정 저장
4. Mux가 mtime 변경 감지
5. 다음 메시지 전송 시 diff 자동 주입

에이전트:
"I see you've updated the plan:
+ Added step for email verification
+ Changed database from PostgreSQL to MySQL
Updating the plan accordingly..."
```

---

## Exec 모드 상세

### Plan → Exec 핸드오프

```
Plan 모드:
1. 플랜 작성 및 승인
2. <plan> 블록에 플랜 내용 저장

Exec 모드 (또는 Orchestrator):
1. <plan> 블록 파싱
2. 플랜을 "진실의 원천"으로 취급
3. 추가 탐색 최소화 (플랜에 경로/심볼 명시)
4. 직접 구현
```

### Explore 서브에이전트 활용

```
Exec 에이전트:
"플랜에 인증 모듈 경로가 없음 → Explore 서브에이전트 실행"

task({
  agentId: "explore",
  title: "Find auth module",
  prompt: "Locate existing authentication module and list exported functions"
})

Explore 보고서:
"Authentication module: src/auth/index.ts
Exports:
- authenticateUser(username, password)
- validateToken(token)
- refreshToken(token)"

Exec 에이전트:
"보고서 기반으로 OAuth2 통합 구현"
```

### 실패 처리 루프

```
1. 파일 수정
2. npm test 실행
3. 실패 감지
   ↓
4. 오류 로그 분석
5. 수정 사항 식별
6. 파일 재수정
7. npm test 재실행
   ↓
   성공 → agent_report
   실패 → 4번으로 반복 (최대 N회)
```

---

## 서브에이전트 시스템

### 서브에이전트 생성

```javascript
// task 도구 호출
task({
  agentId: "explore",  // 또는 subagent_type (레거시)
  title: "Find database schema",
  prompt: "Locate the database schema files and list all tables",
  run_in_background: false  // 기본값: false (동기)
})
```

### 서브에이전트 워크스페이스

```
부모 워크스페이스: feature-auth-x7k2
    ├── 서브에이전트 1: explore-db-schema-s1
    ├── 서브에이전트 2: explore-api-routes-s2
    └── 서브에이전트 3: exec-implement-oauth-s3
```

#### 특징

- **독립 채팅 히스토리**: 각 서브에이전트 별도 세션
- **제한된 도구 접근**: propose_plan, ask_user_question 차단
- **재귀 제한**: 서브에이전트는 task 도구 사용 불가 (설정 가능)

### 병렬 실행

```javascript
// 병렬 서브에이전트 실행
const task1 = task({
  agentId: "explore",
  title: "Find frontend components",
  prompt: "...",
  run_in_background: true
});

const task2 = task({
  agentId: "explore",
  title: "Find backend API",
  prompt: "...",
  run_in_background: true
});

const task3 = task({
  agentId: "explore",
  title: "Find tests",
  prompt: "...",
  run_in_background: true
});

// 결과 대기
const results = task_await([task1, task2, task3]);

// 결과 활용
"Frontend components: " + results[0].report
"Backend API: " + results[1].report
"Tests: " + results[2].report
```

### agent_report (서브에이전트 필수)

```javascript
// 서브에이전트 완료 보고
agent_report({
  summary: "Found 5 database schema files",
  details: `
Tables:
- users (id, username, email, created_at)
- posts (id, user_id, title, content, created_at)
- comments (id, post_id, user_id, content, created_at)
- sessions (id, user_id, token, expires_at)
- oauth_providers (id, user_id, provider, provider_user_id)

Schema files:
- src/db/schema/users.ts
- src/db/schema/posts.ts
- src/db/schema/comments.ts
- src/db/schema/sessions.ts
- src/db/schema/oauth.ts
  `
})
```

> **중요**: 서브에이전트는 스트림 종료 전 반드시 `agent_report` 호출 필요

---

## 병렬 에이전트 워크플로우

### 패턴 1: 탐색 병렬화

```
사용자: "Refactor the authentication module"

Exec 에이전트:
1. 병렬 탐색 실행:
   - Explore 1: "Find all auth-related files"
   - Explore 2: "Find all callsites of auth functions"
   - Explore 3: "Find auth tests"

2. task_await([1, 2, 3])

3. 결과 종합 후 리팩토링 시작
```

### 패턴 2: 독립 작업 병렬화 (Orchestrator)

```
플랜:
- Task A: API 엔드포인트 추가
- Task B: 프론트엔드 컴포넌트 추가
- Task C: 테스트 추가

의존성 분석:
- Task A: 독립
- Task B: 독립
- Task C: Task A, B 의존

실행:
1. 배치 1 (병렬): [Task A, Task B]
2. task_await([A, B])
3. 패치 적용: A, B
4. 배치 2: [Task C]
5. task_await([C])
6. 패치 적용: C
```

### 패턴 3: 검증 병렬화

```
Exec 에이전트:
1. 파일 수정 완료

2. 병렬 검증:
   - Explore 1: "Run unit tests"
   - Explore 2: "Run integration tests"
   - Explore 3: "Run lint"
   - Explore 4: "Run type check"

3. task_await([1, 2, 3, 4])

4. 결과 분석:
   - 모두 성공 → 커밋
   - 일부 실패 → 수정 후 재검증
```

---

## 에이전트 상태 모니터링

### 사이드바 상태 표시

```
my-app
  ├── feature-auth-x7k2 [Plan ⏸]
  │   - Plan 모드, 플랜 작성 대기
  │
  ├── fix-bug-p3m9 [Exec 🔄]
  │   - Exec 모드, 파일 수정 중
  │   └── explore-tests-s1 [🔍]
  │       - Explore 서브에이전트 실행 중
  │
  └── deploy-staging-k1n4 [Exec ✓]
      - 완료
```

### 상태 아이콘

| 아이콘 | 의미 |
|-------|------|
| 🔄 | 스트리밍 중 |
| ⏸ | 사용자 입력 대기 |
| 🔍 | 서브에이전트 실행 중 |
| ✓ | 완료 |
| ⚠️ | 오류 |
| 💤 | 유휴 상태 |

### 실시간 로그

```
┌─────────────────────────────────────┐
│  Agent Status                       │
├─────────────────────────────────────┤
│  feature-auth-x7k2                  │
│  Mode: Exec                         │
│  Status: Running                    │
│                                     │
│  Current Action:                    │
│  ├─ file_edit_replace_string        │
│  │  File: src/auth/google.ts       │
│  │  Status: Completed               │
│  │                                  │
│  ├─ bash                            │
│  │  Command: npm test               │
│  │  Status: Running... (10s)        │
│  │                                  │
│  └─ Sub-agents:                     │
│     └─ explore-tests-s1 [Running]   │
└─────────────────────────────────────┘
```

---

## 슬래시 명령어

### /compact (컨텍스트 압축)

```
/compact

효과:
1. 대화 히스토리를 AI로 요약
2. 중요 정보 보존
3. 토큰 사용량 감소
4. 응답 품질 유지
```

#### 동작 원리

```
원본 히스토리 (10,000 토큰):
- 사용자: "Add OAuth2"
- 에이전트: [플랜 작성... 5,000 토큰]
- 사용자: "Looks good"
- 에이전트: [구현... 4,000 토큰]

압축 후 (2,000 토큰):
Summary:
- Implemented OAuth2 with Google provider
- Files modified: src/auth/google.ts, src/routes/auth.ts
- Tests added: src/__tests__/oauth.test.ts
- All tests passing
```

### /clear (전체 삭제)

```
/clear

효과:
- 대화 히스토리 완전 삭제
- 컨텍스트 완전 초기화
- 복구 불가
```

### /truncate (단순 잘라내기)

```
/truncate

효과:
- 최근 N개 메시지만 유지
- 즉시 실행 (AI 요약 불필요)
- 시간순 보존
```

### /model (모델 전환)

```
/model anthropic:claude-sonnet-4-5
/model openai:gpt-5.2-codex
/model ollama:llama3.1:70b
```

### /idle (유휴 압축 설정)

```
/idle 24    # 24시간 후 자동 압축
/idle 48    # 48시간 후 자동 압축
/idle off   # 자동 압축 비활성화
```

---

## 커스텀 에이전트 생성

### 파일 구조

```
프로젝트/.mux/agents/
├── review.md          # 코드 리뷰 에이전트
├── security.md        # 보안 감사 에이전트
└── docs.md            # 문서화 에이전트

~/.mux/agents/
├── terse.md           # 간결한 응답 에이전트
└── verbose.md         # 상세한 응답 에이전트
```

### 예시 1: 코드 리뷰 에이전트

```markdown
---
name: Review
description: Terse reviewer-style feedback
base: exec
ui:
  color: "#ff6b6b"
tools:
  remove:
    - file_edit_.*  # 읽기 전용
    - task
    - task_.*
---

You are a code reviewer.

- Focus on correctness, risks, and test coverage
- Prefer short, actionable comments
- Highlight security vulnerabilities
- Check for performance issues

## Review Checklist

- [ ] Input validation
- [ ] Error handling
- [ ] Edge cases
- [ ] Test coverage
- [ ] Documentation
```

### 예시 2: 보안 감사 에이전트

```markdown
---
name: Security Audit
description: Security-focused code review
base: exec
ui:
  color: "#ffa500"
tools:
  remove:
    - file_edit_.*
    - task
    - task_.*
---

You are a security auditor.

Analyze the codebase for:

- Authentication/authorization issues
- Injection vulnerabilities (SQL, XSS, Command)
- Data exposure risks
- Insecure dependencies
- Hardcoded secrets

Provide a structured report with severity levels:
- CRITICAL: Immediate fix required
- HIGH: Fix within 1 week
- MEDIUM: Fix within 1 month
- LOW: Nice to fix

Do not make changes, only report findings.
```

### 예시 3: 서브에이전트 전용

```markdown
---
name: Test Runner
description: Run tests and report results
base: exec
ui:
  hidden: true  # UI에서 숨김
subagent:
  runnable: true
  skip_init_hook: true
tools:
  remove:
    - file_edit_.*
    - task
    - task_.*
---

You are a test runner sub-agent.

1. Run the requested tests
2. Parse the output
3. Report results with:
   - Pass/fail count
   - Failed test details
   - Coverage percentage (if available)

Always call `agent_report` before stream end.
```

---

## 에이전트 우선순위

### Discovery 순서

```
1. .mux/agents/*.md         # 프로젝트 (최우선)
2. ~/.mux/agents/*.md        # 글로벌
3. Built-in agents           # 빌트인
```

### 덮어쓰기 예시

```markdown
<!-- 프로젝트/.mux/agents/exec.md -->
---
name: Exec
base: exec  # 빌트인 exec 상속
---

Additional project-specific instructions:

- Always run `make fmt` before committing
- Use `bun` instead of `npm`
- Run `make test` for verification
```

---

## 에이전트 설정 고급

### Mode Prompts (레거시 → AGENTS.md)

```markdown
<!-- AGENTS.md -->
## Model: sonnet

Be terse and to the point.

## Tool: bash

- Use `rg` instead of `grep`
- Use `fd` instead of `find`
```

### AI 기본값 설정

```yaml
---
name: Fast Exec
base: exec
ai:
  model: haiku  # 기본 모델
  thinkingLevel: low  # 사고 수준
---
```

### 도구 정책 패턴

```yaml
---
name: Read Only
base: exec
tools:
  add:
    - file_read
    - bash
    - web_fetch
  remove:
    - file_edit_.*  # 모든 편집 도구
    - task_.*       # 모든 태스크 도구
---
```

---

## Command Palette 통합

### 에이전트 전환

```
⌘+Shift+P / Ctrl+Shift+P
→ "Change Agent" 또는 "Switch Mode"
→ Plan / Exec / Ask / Review / ...
```

#### 단축키

```
⌘+Shift+M / Ctrl+Shift+M
→ 에이전트 순환 (Plan → Exec → Ask → ...)
```

### 슬래시 명령어 자동완성

```
채팅 입력창에 "/" 입력
→ 자동완성 목록:
  /compact  - Compress conversation history
  /clear    - Clear all history
  /truncate - Simple truncation
  /model    - Change model
  /idle     - Set idle compaction
```

---

## 문제 해결

### 서브에이전트가 응답하지 않음

```bash
# 원인: agent_report 누락

# 해결책: 서브에이전트 프롬프트에 명시
"Before stream end, you MUST call agent_report with summary and details"

# 또는 타임아웃 설정 (Settings → Agents → Task Settings)
```

### Plan 모드에서 플랜이 생성되지 않음

```bash
# 원인: file_edit_* 도구 차단

# 해결책: Plan 모드는 플랜 파일만 수정 가능 (내부 로직)
# 일반 파일 수정 시도 시 오류 발생

# Workaround: Exec 모드로 전환
```

### Orchestrator가 활성화되지 않음

```bash
# 원인: ui.requires: [plan] 조건

# 해결책:
# 1. Plan 모드에서 플랜 작성 및 승인
# 2. Exec 모드 전환 시 자동 활성화
```

---

## 성능 최적화

### 컨텍스트 크기 관리

```
자동 압축 활성화:
Settings → Costs → Auto-Compact: 70%

수동 압축:
/compact  (중요 정보 보존)
/truncate (빠르지만 단순)
```

### 서브에이전트 남용 방지

```
안티패턴:
- 간단한 작업에 서브에이전트 사용
- 과도한 병렬 실행 (10개 이상)

권장:
- 복잡한 탐색만 서브에이전트 위임
- 병렬 실행: 3-5개 이하
```

---

## 다음 단계

에이전트 시스템을 마스터했다면:

1. **[챕터 05: 멀티모델 지원](/blog-repo/mux-guide-05-multimodel)** - 모델별 특징 및 비용 최적화
2. **[챕터 07: 고급 기능](/blog-repo/mux-guide-07-advanced-features)** - Mode Prompts, Instruction Files
3. **[챕터 08: 개발 및 확장](/blog-repo/mux-guide-08-development)** - 커스텀 에이전트 개발

---

## 참고 자료

- [Agents 문서](https://mux.coder.com/agents/)
- [Plan Mode 문서](https://mux.coder.com/agents/plan-mode)
- [Instruction Files](https://mux.coder.com/agents/instruction-files)
- [Agent Skills](https://mux.coder.com/agents/agent-skills)
