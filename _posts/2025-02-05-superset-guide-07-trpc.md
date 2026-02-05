---
layout: post
title: "Superset 완벽 가이드 (7) - tRPC 라우터"
date: 2025-02-05
permalink: /superset-guide-07-trpc/
author: Superset Team
categories: [AI 에이전트, Superset]
tags: [Superset, tRPC, API, TypeScript, Type-safe]
original_url: "https://github.com/superset-sh/superset"
excerpt: "Superset의 tRPC 기반 타입 안전 API 구조를 분석합니다."
---

## tRPC 개요

Superset은 내부 API 통신에 **tRPC**를 사용합니다. tRPC는 TypeScript 기반의 타입 안전 RPC 프레임워크로, API 스키마 없이 엔드-투-엔드 타입 안전성을 제공합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    tRPC 아키텍처                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────────────────────────────────────────────┐  │
│   │                   Renderer (Client)                   │  │
│   │                                                       │  │
│   │   const result = await trpc.workspace.create({...})  │  │
│   │                      │                                │  │
│   │                      │ TypeScript 타입 추론           │  │
│   │                      ▼                                │  │
│   └──────────────────────┼────────────────────────────────┘  │
│                          │ IPC / HTTP                        │
│   ┌──────────────────────┼────────────────────────────────┐  │
│   │                   Main (Server)                       │  │
│   │                      │                                │  │
│   │   export const workspaceRouter = router({            │  │
│   │     create: protectedProcedure                       │  │
│   │       .input(z.object({...}))                        │  │
│   │       .mutation(async ({ctx, input}) => {...})       │  │
│   │   })                                                  │  │
│   │                                                       │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 라우터 구조

```
lib/trpc/routers/
├── index.ts              # 루트 라우터 (모든 라우터 병합)
│
├── auth/                 # 인증
│   ├── index.ts
│   └── utils/
│       ├── auth-functions.ts
│       └── crypto-storage.ts
│
├── workspaces/           # 워크스페이스 관리
│   ├── index.ts
│   ├── workspaces.ts
│   ├── procedures/       # CRUD 프로시저
│   │   ├── create.ts
│   │   ├── delete.ts
│   │   ├── query.ts
│   │   ├── status.ts
│   │   ├── git-status.ts
│   │   ├── branch.ts
│   │   └── init.ts
│   └── utils/            # 유틸리티
│       ├── worktree.ts
│       ├── git.ts
│       ├── setup.ts
│       ├── teardown.ts
│       ├── shell-env.ts
│       └── github/
│
├── projects/             # 프로젝트 관리
│   ├── index.ts
│   ├── projects.ts
│   └── utils/
│
├── terminal/             # 터미널 관리
│   └── index.ts
│
├── changes/              # 변경사항 (diff)
│   ├── index.ts
│   ├── utils/
│   └── security/
│       ├── path-validation.ts
│       ├── git-commands.ts
│       └── secure-fs.ts
│
├── auto-update/          # 자동 업데이트
├── settings/             # 설정
├── hotkeys/              # 단축키
├── config/               # 앱 설정
├── filesystem/           # 파일 시스템
├── external/             # 외부 앱 실행
├── analytics/            # 분석
├── cache/                # 캐시
├── ports/                # 포트 관리
├── ringtone/             # 알림음
├── ui-state/             # UI 상태
├── window.ts             # 윈도우 관리
├── menu.ts               # 메뉴
└── notifications.ts      # 알림
```

---

## 루트 라우터

```typescript
// lib/trpc/routers/index.ts

import { router } from "../trpc";
import { authRouter } from "./auth";
import { workspacesRouter } from "./workspaces";
import { projectsRouter } from "./projects";
import { terminalRouter } from "./terminal";
import { changesRouter } from "./changes";
import { settingsRouter } from "./settings";
import { autoUpdateRouter } from "./auto-update";
// ... 더 많은 라우터

export const appRouter = router({
  auth: authRouter,
  workspaces: workspacesRouter,
  projects: projectsRouter,
  terminal: terminalRouter,
  changes: changesRouter,
  settings: settingsRouter,
  autoUpdate: autoUpdateRouter,
  // ...
});

// 타입 내보내기 (클라이언트에서 사용)
export type AppRouter = typeof appRouter;
```

---

## 프로시저 타입

### 공개 프로시저

```typescript
// 인증 없이 접근 가능
import { publicProcedure } from "../trpc";

export const healthCheck = publicProcedure
  .query(() => {
    return { status: "ok", timestamp: Date.now() };
  });
```

### 보호 프로시저

```typescript
// 인증 필요
import { protectedProcedure } from "../trpc";

export const getProfile = protectedProcedure
  .query(async ({ ctx }) => {
    // ctx.user가 보장됨
    const user = await db.query.users.findFirst({
      where: eq(users.id, ctx.user.id),
    });
    return user;
  });
```

---

## 프로시저 구조

### Query (읽기)

```typescript
// routers/workspaces/procedures/query.ts

export const list = protectedProcedure
  .input(z.object({
    projectId: z.string().uuid().optional(),
    includeDeleted: z.boolean().default(false),
  }))
  .query(async ({ ctx, input }) => {
    const workspaces = await ctx.db.query.workspaces.findMany({
      where: and(
        input.projectId
          ? eq(workspaces.projectId, input.projectId)
          : undefined,
        input.includeDeleted
          ? undefined
          : isNull(workspaces.deletedAt)
      ),
      orderBy: desc(workspaces.updatedAt),
    });

    return workspaces;
  });

export const getById = protectedProcedure
  .input(z.object({
    id: z.string().uuid(),
  }))
  .query(async ({ ctx, input }) => {
    const workspace = await ctx.db.query.workspaces.findFirst({
      where: eq(workspaces.id, input.id),
      with: {
        project: true,
      },
    });

    if (!workspace) {
      throw new TRPCError({
        code: "NOT_FOUND",
        message: "Workspace not found",
      });
    }

    return workspace;
  });
```

### Mutation (쓰기)

```typescript
// routers/workspaces/procedures/create.ts

export const create = protectedProcedure
  .input(z.object({
    projectId: z.string().uuid(),
    name: z.string().min(1).max(100).optional(),
    branchName: z.string().optional(),
    baseBranch: z.string().default("main"),
    taskId: z.string().uuid().optional(),
  }))
  .mutation(async ({ ctx, input }) => {
    // 1. 프로젝트 확인
    const project = await ctx.db.query.projects.findFirst({
      where: eq(projects.id, input.projectId),
    });

    if (!project) {
      throw new TRPCError({
        code: "NOT_FOUND",
        message: "Project not found",
      });
    }

    // 2. 워크스페이스 이름 생성
    const name = input.name || generateWorkspaceName();
    const branchName = input.branchName || name;

    // 3. 경로 계산
    const worktreePath = path.join(
      path.dirname(project.path),
      `${path.basename(project.path)}-wt`,
      name
    );

    // 4. Git worktree 생성
    await createGitWorktree({
      repoPath: project.path,
      worktreePath,
      branchName,
      baseBranch: input.baseBranch,
    });

    // 5. 설정 스크립트 실행
    await runSetupScripts(worktreePath);

    // 6. DB에 저장
    const [workspace] = await ctx.db.insert(workspaces).values({
      id: crypto.randomUUID(),
      projectId: input.projectId,
      name,
      path: worktreePath,
      branch: branchName,
      taskId: input.taskId,
    }).returning();

    return workspace;
  });
```

---

## 에러 처리

### TRPCError 코드

```typescript
import { TRPCError } from "@trpc/server";

// 리소스 없음
throw new TRPCError({
  code: "NOT_FOUND",
  message: "Workspace not found",
});

// 인증 필요
throw new TRPCError({
  code: "UNAUTHORIZED",
  message: "Must be logged in",
});

// 권한 없음
throw new TRPCError({
  code: "FORBIDDEN",
  message: "Not authorized to access this resource",
});

// 잘못된 요청
throw new TRPCError({
  code: "BAD_REQUEST",
  message: "Invalid input",
});

// 내부 에러
throw new TRPCError({
  code: "INTERNAL_SERVER_ERROR",
  message: "Something went wrong",
});

// 미구현
throw new TRPCError({
  code: "NOT_IMPLEMENTED",
  message: "Feature not yet available",
});
```

### 외부 서비스 에러 패턴

```typescript
export const syncWithGitHub = protectedProcedure
  .input(syncInput)
  .mutation(async ({ ctx, input }) => {
    try {
      const result = await github.sync(input);

      if (!result.ok) {
        console.error("[sync/github] GitHub sync failed:", result.error);
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "GitHub sync failed",
        });
      }

      return result.data;
    } catch (error) {
      if (error instanceof TRPCError) throw error;

      console.error("[sync/github] Unexpected error:", error);
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message: "Sync operation failed",
      });
    }
  });
```

---

## Context

### Context 정의

```typescript
// lib/trpc/context.ts

import type { inferAsyncReturnType } from "@trpc/server";
import { db } from "@superset/db";
import { localDb } from "@/main/lib/local-db";

export async function createContext() {
  return {
    db,       // 원격 DB (Neon PostgreSQL)
    localDb,  // 로컬 DB (SQLite)
  };
}

export type Context = inferAsyncReturnType<typeof createContext>;
```

### 인증 Context

```typescript
// lib/trpc/trpc.ts

import { initTRPC, TRPCError } from "@trpc/server";
import { Context } from "./context";

const t = initTRPC.context<Context>().create();

export const router = t.router;
export const publicProcedure = t.procedure;

// 인증 미들웨어
const isAuthed = t.middleware(async ({ ctx, next }) => {
  const user = await getCurrentUser();

  if (!user) {
    throw new TRPCError({
      code: "UNAUTHORIZED",
      message: "Not authenticated",
    });
  }

  return next({
    ctx: {
      ...ctx,
      user,
    },
  });
});

export const protectedProcedure = t.procedure.use(isAuthed);
```

---

## 클라이언트 사용

### React Query 통합

```typescript
// renderer/lib/trpc.ts

import { createTRPCReact } from "@trpc/react-query";
import type { AppRouter } from "@/lib/trpc/routers";

export const trpc = createTRPCReact<AppRouter>();
```

### Provider 설정

```typescript
// renderer/providers/trpc-provider.tsx

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { trpc } from "@/lib/trpc";
import { ipcLink } from "@/lib/trpc-ipc-link";

const queryClient = new QueryClient();

const trpcClient = trpc.createClient({
  links: [ipcLink()],
});

export function TRPCProvider({ children }: { children: React.ReactNode }) {
  return (
    <trpc.Provider client={trpcClient} queryClient={queryClient}>
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    </trpc.Provider>
  );
}
```

### Query 사용

```typescript
// renderer/components/WorkspaceList.tsx

import { trpc } from "@/lib/trpc";

export function WorkspaceList({ projectId }: { projectId: string }) {
  const { data, isLoading, error } = trpc.workspaces.list.useQuery({
    projectId,
  });

  if (isLoading) return <Spinner />;
  if (error) return <Error message={error.message} />;

  return (
    <ul>
      {data?.map((workspace) => (
        <li key={workspace.id}>{workspace.name}</li>
      ))}
    </ul>
  );
}
```

### Mutation 사용

```typescript
// renderer/components/CreateWorkspace.tsx

import { trpc } from "@/lib/trpc";
import { useNewWorkspaceModal } from "@/stores/new-workspace-modal";

export function CreateWorkspaceButton({ projectId }: { projectId: string }) {
  const utils = trpc.useUtils();

  const createMutation = trpc.workspaces.create.useMutation({
    onSuccess: () => {
      // 목록 갱신
      utils.workspaces.list.invalidate({ projectId });
    },
  });

  const handleCreate = () => {
    createMutation.mutate({
      projectId,
      name: "new-feature",
      baseBranch: "main",
    });
  };

  return (
    <button
      onClick={handleCreate}
      disabled={createMutation.isPending}
    >
      {createMutation.isPending ? "Creating..." : "Create Workspace"}
    </button>
  );
}
```

---

## IPC 링크

Electron 환경에서는 HTTP 대신 IPC로 통신합니다.

```typescript
// renderer/lib/trpc-ipc-link.ts

import { TRPCLink } from "@trpc/client";
import { observable } from "@trpc/server/observable";
import type { AppRouter } from "@/lib/trpc/routers";

export function ipcLink(): TRPCLink<AppRouter> {
  return () =>
    ({ op }) =>
      observable((observer) => {
        const { type, path, input } = op;

        window.ipcRenderer
          .invoke("trpc", { type, path, input })
          .then((result) => {
            observer.next({
              result: {
                type: "data",
                data: result,
              },
            });
            observer.complete();
          })
          .catch((error) => {
            observer.error(error);
          });
      });
}
```

### Main Process 핸들러

```typescript
// main/lib/trpc-handler.ts

import { ipcMain } from "electron";
import { appRouter } from "@/lib/trpc/routers";
import { createContext } from "@/lib/trpc/context";

ipcMain.handle("trpc", async (_event, { type, path, input }) => {
  const ctx = await createContext();
  const caller = appRouter.createCaller(ctx);

  // path를 따라 procedure 호출
  const pathParts = path.split(".");
  let procedure: any = caller;

  for (const part of pathParts) {
    procedure = procedure[part];
  }

  if (type === "query") {
    return procedure(input);
  } else if (type === "mutation") {
    return procedure(input);
  }
});
```

---

## 베스트 프랙티스

### 1. 프로시저는 얇게 유지

```typescript
// ✅ Good - 로직을 유틸리티로 분리
export const create = protectedProcedure
  .input(createInput)
  .mutation(async ({ ctx, input }) => {
    return workspaceService.create({ db: ctx.db, input });
  });

// ❌ Bad - 두꺼운 프로시저
export const create = protectedProcedure
  .input(createInput)
  .mutation(async ({ ctx, input }) => {
    // 50줄 이상의 비즈니스 로직...
  });
```

### 2. 입력 검증은 Zod로

```typescript
const createInput = z.object({
  name: z.string()
    .min(1, "Name is required")
    .max(100, "Name too long")
    .regex(/^[a-z0-9-]+$/, "Only lowercase letters, numbers, and hyphens"),
  priority: z.enum(["low", "medium", "high"]).default("medium"),
});
```

### 3. 에러는 명시적으로

```typescript
// ✅ Good
if (!workspace) {
  throw new TRPCError({
    code: "NOT_FOUND",
    message: `Workspace ${id} not found`,
  });
}

// ❌ Bad - 에러 삼키기
if (!workspace) {
  return null;
}
```

---

*다음 글에서는 MCP 서버 통합을 분석합니다.*
