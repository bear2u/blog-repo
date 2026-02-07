---
layout: post
title: "SpecWeave 완벽 가이드 (03) - 아키텍처 및 핵심 개념"
date: 2026-02-07
permalink: /specweave-guide-03-architecture/
author: Anton Abyzov
categories: [AI 코딩, 개발 도구]
tags: [SpecWeave, Architecture, Core Concepts, Design]
original_url: "https://github.com/anton-abyzov/specweave"
excerpt: "SpecWeave의 아키텍처, Spec-Driven Development, Increment 시스템 이해"
---

## 아키텍처 개요

SpecWeave는 **Spec-Driven Development** 철학을 기반으로 설계되었습니다.

### 핵심 원칙

1. **영구 문서** - 컨텍스트는 영원히 보존
2. **점진적 개발** - Increment 단위로 기능 구축
3. **품질 우선** - 자동 테스트 및 검증
4. **자율 실행** - AI가 수 시간 동안 작업

## 디렉토리 구조

```
.specweave/
├── config.json              # 프로젝트 설정
├── increments/              # 기능별 디렉토리
│   ├── 0001-oauth/
│   │   ├── spec.md         # WHAT: 무엇을 구축
│   │   ├── plan.md         # HOW: 어떻게 구축
│   │   └── tasks.md        # DO: 구체적 작업
│   └── 0002-dark-mode/
│       ├── spec.md
│       ├── plan.md
│       └── tasks.md
├── docs/                    # 리빙 문서
│   ├── internal/
│   │   ├── architecture/
│   │   │   └── adr/        # Architecture Decision Records
│   │   └── specs/          # 기능 명세서
│   └── public/             # 공개 문서
└── metrics/                 # DORA 메트릭
    └── dora-latest.json
```

## Increment 시스템

### Increment란?

**Increment**는 독립적으로 완성 가능한 기능 단위입니다.

특징:
- 고유 ID (0001, 0002, ...)
- 완전한 문서 (Spec, Plan, Tasks)
- 독립적 테스트 가능
- 외부 도구와 동기화

### Spec (무엇을)

```markdown
# Spec: User Authentication

## User Stories
- As a user, I want to log in with email/password
- As a user, I want to stay logged in (remember me)

## Acceptance Criteria
- [ ] User can register with email/password
- [ ] User can login with correct credentials
- [ ] Session persists across browser refresh

## Success Metrics
- Login success rate > 99%
- Password reset < 5 minutes
```

### Plan (어떻게)

```markdown
# Plan: User Authentication

## Architecture Decisions
- JWT for token-based auth
- bcrypt for password hashing
- Redis for session storage

## Tech Stack
- Express.js middleware
- Passport.js
- jsonwebtoken

## Implementation Strategy
1. Database schema for users
2. Registration endpoint
3. Login endpoint
4. Middleware for protected routes
```

### Tasks (구체적 작업)

```markdown
# Tasks: User Authentication

## T-001: Database Schema
- Create users table with email, password_hash
- Add indexes on email
- Test: Schema migration succeeds

## T-002: Registration Endpoint
- POST /api/auth/register
- Validate email format
- Hash password with bcrypt
- Test: Duplicate email rejected

## T-003: Login Endpoint
- POST /api/auth/login
- Verify credentials
- Generate JWT token
- Test: Invalid credentials return 401
```

## 68+ AI 에이전트 시스템

### 에이전트 역할 분담

```typescript
// PM 에이전트
- 요구사항 수집
- 사용자 스토리 작성
- 인수 기준 정의

// Architect 에이전트
- 시스템 설계
- ADR 작성
- 기술 스택 선택

// QA Lead 에이전트
- 테스트 전략 수립
- 테스트 케이스 작성
- 품질 게이트 정의

// Security 에이전트
- 보안 리뷰
- OWASP 체크리스트
- 취약점 스캔

// DevOps 에이전트
- CI/CD 파이프라인
- 인프라 설정
- 배포 전략
```

### 컨텍스트 기반 활성화

```bash
# "security" 언급 → Security 에이전트 활성화
/sw:increment "Add OAuth with PKCE security"

# "deploy" 언급 → DevOps 에이전트 활성화
/sw:increment "Setup Kubernetes deployment"

# "performance" 언급 → Performance 에이전트 활성화
/sw:increment "Optimize database queries"
```

## Lazy Plugin Loading

### 토큰 절약 메커니즘

```
일반 작업 (비-SpecWeave):
  Without Lazy: 60,000 tokens (모든 플러그인 로드)
  With Lazy:       500 tokens (기본만)
  절약: 99%

SpecWeave 작업:
  Without Lazy: 60,000 tokens
  With Lazy:    60,000 tokens (필요한 것만)
  절약: 0% (하지만 필요한 기능 모두 사용)
```

### 키워드 기반 로드

| 키워드 | 로드되는 플러그인 |
|--------|------------------|
| "React", "Vue", "Angular" | frontend-plugin |
| "Kubernetes", "Docker" | k8s-plugin |
| "TypeScript", ".ts" | typescript-lsp |
| "security", "OWASP" | security-plugin |

## LSP 통합 아키텍처

### Language Server Protocol

```
┌─────────────────────────────────────────┐
│         SpecWeave CLI                    │
├─────────────────────────────────────────┤
│   LSP Client (per language)             │
│   ├── TypeScript LSP                    │
│   ├── Python LSP (Pyright)              │
│   └── C# LSP                            │
├─────────────────────────────────────────┤
│   Semantic Code Intelligence            │
│   ├── Go to Definition                  │
│   ├── Find All References               │
│   ├── Get Diagnostics                   │
│   └── Hover Information                 │
└─────────────────────────────────────────┘
```

### 토큰 사용량 비교

| 작업 | 전통적 방식 | LSP 방식 |
|------|------------|----------|
| 참조 찾기 | Grep + 15파일 읽기 (10K) | 시맨틱 쿼리 (500) |
| 타입 에러 | 빌드 + 파싱 (5K) | getDiagnostics (1K) |
| 정의 탐색 | Grep + 검증 (8K) | goToDefinition (200) |

**총 절약**: 95% 이상

## Self-Improving Skills

### 학습 메커니즘

```markdown
## .specweave/docs/internal/skills/

### Testing Best Practices
<!-- Learned from correction on 2026-02-01 -->
- Use `vi.hoisted()` for ESM mocking in Vitest 4.x+
- Never mock in global scope

### Code Style
<!-- Learned from correction on 2026-02-03 -->
- Prefer native `fs` over `fs-extra`
- Use async/await, not callbacks
```

### Reflect 명령

```bash
/sw:reflect "Always check null before accessing properties"
```

다음번 작업에서 자동으로 적용됩니다.

## 품질 게이트

### 자동 검증

```typescript
// 모든 작업 완료 전 검증
- 모든 테스트 통과
- 코드 커버리지 > 80%
- ESLint/TSLint 통과
- 타입 체크 통과
- 보안 스캔 통과
```

### 수동 리뷰

```bash
/sw:grill 0001    # 코드 리뷰 요청
```

리뷰 항목:
- 코드 품질
- 테스트 완성도
- 문서 업데이트
- 보안 체크리스트

## 다음 단계

다음 챕터에서는 Increment 시스템의 실전 활용을 다룹니다.

---

## 시리즈 네비게이션

- **이전**: [(2) 설치 및 빠른 시작]({{ site.baseurl }}/specweave-guide-02-installation/)
- **현재**: (3) 아키텍처 및 핵심 개념
- **다음**: [(4) Increment 시스템]({{ site.baseurl }}/specweave-guide-04-increment-system/)

[📚 전체 목차로 돌아가기]({{ site.baseurl }}/specweave-guide/)
