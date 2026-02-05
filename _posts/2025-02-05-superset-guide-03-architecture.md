---
layout: post
title: "Superset 완벽 가이드 (3) - 아키텍처 분석"
date: 2025-02-05
permalink: /superset-guide-03-architecture/
author: Superset Team
categories: [AI 에이전트, Superset]
tags: [Superset, Architecture, Monorepo, Turborepo, Bun]
original_url: "https://github.com/superset-sh/superset"
excerpt: "Superset의 모노레포 구조와 아키텍처 설계 원칙을 분석합니다."
---

## 모노레포 구조

Superset은 **Bun + Turborepo** 기반의 모노레포로 구성됩니다. 여러 앱과 공유 패키지가 하나의 레포지토리에서 관리됩니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Superset Monorepo                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────────── apps/ ──────────────────────────┐    │
│   │                                                     │    │
│   │  desktop   web   api   marketing   admin   docs    │    │
│   │     │       │     │        │         │       │      │    │
│   │     └───────┴─────┴────────┴─────────┴───────┘      │    │
│   │                        │                             │    │
│   └────────────────────────┼────────────────────────────┘    │
│                            │                                  │
│   ┌─────────────── packages/ ──────────────────────────┐    │
│   │                        │                            │    │
│   │  ui    db    auth    trpc    mcp    shared         │    │
│   │                                                     │    │
│   │  local-db    ai-chat    email    scripts           │    │
│   │                                                     │    │
│   └─────────────────────────────────────────────────────┘    │
│                                                              │
│   ┌─────────────── tooling/ ───────────────────────────┐    │
│   │  typescript-config                                  │    │
│   └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 앱 구성

### apps/ 디렉토리

| 앱 | 설명 | 기술 |
|----|------|------|
| `desktop` | Electron 데스크탑 앱 (핵심) | Electron, React, Vite |
| `web` | 웹 애플리케이션 (app.superset.sh) | Next.js 16 |
| `api` | API 백엔드 | Next.js API Routes |
| `marketing` | 마케팅 사이트 (superset.sh) | Next.js |
| `admin` | 관리자 대시보드 | Next.js |
| `docs` | 문서 사이트 | Next.js |
| `cli` | CLI 도구 | TypeScript |
| `mobile` | 모바일 앱 | React Native |
| `streams` | 스트리밍 서비스 | - |

---

## 패키지 구성

### packages/ 디렉토리

| 패키지 | 용도 | 주요 의존성 |
|--------|------|------------|
| `ui` | 공유 UI 컴포넌트 | shadcn/ui, TailwindCSS v4 |
| `db` | Drizzle ORM 스키마 | Drizzle ORM, Neon |
| `auth` | 인증 모듈 | - |
| `trpc` | tRPC 설정 | tRPC |
| `mcp` | MCP 서버 | @modelcontextprotocol/sdk |
| `shared` | 공유 유틸리티 | - |
| `local-db` | 로컬 SQLite | better-sqlite3 |
| `ai-chat` | AI 채팅 모듈 | - |
| `email` | 이메일 템플릿 | - |
| `scripts` | CLI 스크립트 | - |

---

## 기술 스택 상세

### 패키지 매니저: Bun

```bash
# package.json
{
  "packageManager": "bun@1.3.0",
  "type": "module"
}
```

- npm/yarn/pnpm 대신 **Bun** 사용
- 빠른 설치 및 실행 속도
- 네이티브 TypeScript 지원

### 빌드 시스템: Turborepo

```jsonc
// turbo.jsonc
{
  "tasks": {
    "dev": {
      "cache": false,
      "persistent": true
    },
    "build": {
      "dependsOn": ["^build"],
      "outputs": [".next/**", "dist/**", "release/**"]
    },
    "typecheck": {
      "dependsOn": ["^typecheck"]
    }
  }
}
```

- **캐싱**: 빌드 결과 캐싱으로 빠른 재빌드
- **병렬 실행**: 의존성이 없는 태스크 동시 실행
- **의존성 그래프**: 패키지 간 빌드 순서 자동 관리

### 코드 품질: Biome

```jsonc
// biome.jsonc
{
  "formatter": {
    "enabled": true,
    "indentStyle": "tab"
  },
  "linter": {
    "enabled": true
  },
  "organizeImports": {
    "enabled": true
  }
}
```

- ESLint + Prettier 대체
- 단일 도구로 린팅, 포맷팅, import 정리
- 빠른 실행 속도

---

## 아키텍처 설계 원칙

### 1. 관심사 분리 (Separation of Concerns)

```
┌─────────────────────────────────────────────────────────────┐
│                      분리 계층 구조                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Transport Layer     →  Routes, API handlers                │
│         ↓                                                    │
│  Orchestration Layer →  tRPC procedures                     │
│         ↓                                                    │
│  Domain Layer        →  Business logic, utilities           │
│         ↓                                                    │
│  Data Layer          →  Database, external APIs             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

- **소유권 + 생명주기별 분리**: 트랜스포트, 오케스트레이션, 도메인 규칙을 별도 레이어로
- **생명주기별 코로케이션**: 기능별 코드를 함께 배치 (예: 모든 task 관련 코드는 `router/task/`에)
- **경계 레이어의 에러 핸들링**: 도메인 유틸리티는 데이터/특정 에러 반환, 경계 코드에서 `TRPCError`로 변환

### 2. 최소 결합 (Minimal Coupling)

```typescript
// ✅ Good - 의존성 주입
const createTask = ({ db, logger }: { db: DB; logger: Logger }) => {
  // ...
};

// ❌ Bad - 전역 싱글톤 임포트
import { db } from "@/lib/db";
import { logger } from "@/lib/logger";
```

- 모듈을 자급자족적으로 유지, 좁은 public API
- **데메테르 법칙**: 직접 협력자(주입된 의존성)에만 의존
- 복잡해지면 싱글톤 대신 **의존성 주입** 선호

### 3. 적절한 도구 선택 (Right Tool for the Job)

```typescript
// ✅ Good - 조건 분기 대신 lookup 객체
const handlers = {
  linear: handleLinear,
  github: handleGithub,
  jira: handleJira,
} as const;

const handle = handlers[provider];

// ❌ Bad - 반복적인 조건문
if (provider === "linear") { ... }
else if (provider === "github") { ... }
else if (provider === "jira") { ... }
```

- 새 추상화 전에 기존 프리미티브 확인 (`packages/ui`, `packages/constants`)
- 여러 케이스 처리 시 lookup 객체/맵 사용
- 정적 데이터는 코드로, 다중 테넌트 데이터는 Drizzle 사용

### 4. 안전 기본값 (Fail-Safe by Default)

```typescript
// 경계에서 검증 (Zod 스키마)
const input = inputSchema.parse(rawInput);

// 외부 API 데이터는 신뢰하지 않음
const data = externalData.field ?? defaultValue;

// 에러는 절대 삼키지 않음
try {
  // ...
} catch (error) {
  console.error("[context] Error:", error);
  throw new TRPCError({ code: "INTERNAL_SERVER_ERROR" });
}
```

### 5. 조기 추상화 방지 (Avoid Premature Abstraction)

```
┌─────────────────────────────────────────────────────────────┐
│                    세 번의 규칙                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  "같은 패턴을 세 번 보기 전까지는 추상화하지 마라"           │
│                                                              │
│  1번째: 그냥 작성                                            │
│  2번째: 복붙 허용                                            │
│  3번째: 이제 추상화 고려                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

- 가장 단순한 올바른 솔루션으로 시작
- 일회성 케이스에 프레임워크/DSL 도입 금지

### 6. 얇은 오케스트레이터 (Keep Orchestrators Thin)

```typescript
// ✅ Good - 얇은 tRPC procedure
export const createTask = protectedProcedure
  .input(createTaskSchema)
  .mutation(async ({ ctx, input }) => {
    // 검증 + 위임만
    return taskService.create({ db: ctx.db, input });
  });

// ❌ Bad - 두꺼운 procedure
export const createTask = protectedProcedure
  .input(createTaskSchema)
  .mutation(async ({ ctx, input }) => {
    // 50줄 이상의 비즈니스 로직...
  });
```

---

## 코딩 컨벤션

### 2개 이상 파라미터는 객체 시그니처

```typescript
// ✅ Good
const createTask = ({ title, userId, priority }: {
  title: string;
  userId: string;
  priority?: number;
}) => { ... };

// ❌ Bad - 위치 인자
const createTask = (title: string, userId: string, priority?: number) => { ... };
```

### TRPCError 에러 처리

```typescript
// NOT_FOUND - 리소스 없음
throw new TRPCError({ code: "NOT_FOUND", message: "Task not found" });

// UNAUTHORIZED - 로그인 안됨
throw new TRPCError({ code: "UNAUTHORIZED", message: "Must be logged in" });

// FORBIDDEN - 로그인됨, 권한 없음
throw new TRPCError({ code: "FORBIDDEN", message: "Not authorized" });

// BAD_REQUEST - 유효하지 않은 입력
throw new TRPCError({ code: "BAD_REQUEST", message: "Invalid state" });
```

### 로깅 컨벤션

```typescript
// 패턴: [domain/operation] message
console.log("[auth/refresh] Refreshing token for user:", userId);
console.error("[sync/linear] Failed to sync:", error);
console.warn("[task/archive] Task already archived:", taskId);
```

---

## 프로젝트 구조 표준

```
app/
├── page.tsx
├── dashboard/
│   ├── page.tsx
│   ├── components/              # dashboard 전용 컴포넌트
│   │   └── MetricsChart/
│   │       ├── MetricsChart.tsx
│   │       ├── MetricsChart.test.tsx
│   │       └── index.ts
│   ├── hooks/                   # dashboard 전용 훅
│   ├── utils/                   # dashboard 전용 유틸
│   ├── stores/                  # dashboard 전용 스토어
│   └── providers/               # dashboard 전용 프로바이더
│
└── components/                  # 2개 이상 페이지에서 사용
    └── Header/
```

**원칙:**
1. **컴포넌트당 하나의 폴더**: `ComponentName/ComponentName.tsx` + `index.ts`
2. **사용 위치에 코로케이션**: 한 곳에서만 사용하면 부모의 `components/`에 배치
3. **파일당 하나의 컴포넌트**: 멀티 컴포넌트 파일 금지
4. **의존성 코로케이션**: 유틸, 훅, 상수, 테스트는 사용하는 파일 옆에

---

## 코드 스멜 피하기

| 스멜 | 증상 | 권장 수정 |
|------|------|----------|
| 매직 넘버 | `100`, `3`, `"linear"` 하드코딩 | 명명된 상수로 추출 |
| 프로바이더 조건문 | 반복되는 `if (provider === ...)` | lookup 객체/맵 패턴 사용 |
| 갓 프로시저 | 검증+비즈니스+I/O+에러 모두 처리 | 유틸 함수로 추출 |
| 레이어 간 임포트 | UI에서 `packages/db` 내부 임포트 | 적절한 패키지 export 통해 접근 |
| 침묵의 에러 삼키기 | `catch(() => {})` | 최소한 로깅, 재throw 선호 |
| 깊은 중첩 | 4단계 이상 if/for/try | 조기 반환, 함수 추출 |
| Boolean 맹목 | `doThing(true, false, true)` | 명명된 프로퍼티 객체 사용 |

---

## 데이터베이스 규칙

```
⚠️ 중요: 명시적 요청 없이 프로덕션 DB를 절대 건드리지 말 것
```

- 스키마: `packages/db/src/`
- 모든 DB 작업에 Drizzle ORM 사용
- 마이그레이션 직접 실행 금지
- `packages/db/drizzle/` 파일 수동 편집 금지

---

*다음 글에서는 Electron 데스크탑 앱의 상세 구조를 분석합니다.*
