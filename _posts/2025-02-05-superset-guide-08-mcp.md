---
layout: post
title: "Superset 완벽 가이드 (8) - MCP 서버"
date: 2025-02-05
permalink: /superset-guide-08-mcp/
author: Superset Team
categories: [AI 에이전트, Superset]
tags: [Superset, MCP, Model Context Protocol, API, Tasks]
original_url: "https://github.com/superset-sh/superset"
excerpt: "Superset의 MCP 서버 통합과 태스크/디바이스 관리 API를 분석합니다."
---

## MCP (Model Context Protocol) 개요

Superset은 **MCP 서버**를 제공하여 AI 에이전트가 태스크와 디바이스를 관리할 수 있게 합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP 통합 아키텍처                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Claude Code / 기타 에이전트              │   │
│   │                                                      │   │
│   │  "새 태스크 만들어줘: 로그인 버그 수정"               │   │
│   │                      │                               │   │
│   └──────────────────────┼───────────────────────────────┘   │
│                          │ MCP Protocol                      │
│   ┌──────────────────────┼───────────────────────────────┐   │
│   │              Superset MCP Server                      │   │
│   │                      │                               │   │
│   │   Tools:                                             │   │
│   │   • create_task    • list_tasks                      │   │
│   │   • update_task    • get_task                        │   │
│   │   • list_devices   • create_worktree                 │   │
│   │   • switch_workspace                                 │   │
│   │                                                      │   │
│   └──────────────────────┼───────────────────────────────┘   │
│                          │                                   │
│   ┌──────────────────────┼───────────────────────────────┐   │
│   │                 Database / Desktop                    │   │
│   └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## MCP 패키지 구조

```
packages/mcp/
├── src/
│   ├── index.ts          # 메인 엔트리
│   ├── server.ts         # MCP 서버 설정
│   ├── auth.ts           # 인증 유틸리티
│   ├── in-memory.ts      # 인메모리 저장소
│   └── tools/            # MCP 도구들
│       ├── tasks/        # 태스크 관리
│       ├── devices/      # 디바이스 관리
│       ├── workspaces/   # 워크스페이스 관리
│       └── organization/ # 조직 관리
│
└── package.json
```

---

## 인증

### API 키 구조

```typescript
// auth.ts

interface ApiKeyPayload {
  userId: string;         // 요청자 ID
  organizationId: string; // 조직 컨텍스트
  defaultDeviceId: string; // 기본 타겟 디바이스
}

export function decodeApiKey(apiKey: string): ApiKeyPayload {
  // API 키 디코딩 로직
  const decoded = Buffer.from(apiKey, "base64").toString("utf-8");
  return JSON.parse(decoded);
}
```

### 헤더 인증

```typescript
// 요청 시 X-API-Key 헤더 사용
const headers = {
  "X-API-Key": "base64_encoded_api_key",
  "Content-Type": "application/json",
};
```

---

## 도구 카테고리

### 1. 태스크 도구 (Cloud - 즉시 실행)

#### create_task

```typescript
const createTaskInput = z.object({
  title: z.string().min(1).describe("Task title"),
  description: z.string().optional().describe("Task description (markdown)"),
  priority: z.enum(["urgent", "high", "medium", "low", "none"]).default("none"),
  assigneeId: z.string().uuid().optional().describe("User ID to assign"),
  statusId: z.string().uuid().optional().describe("Status ID"),
  labels: z.array(z.string()).optional().describe("Label strings"),
  dueDate: z.string().datetime().optional().describe("Due date (ISO)"),
  estimate: z.number().int().positive().optional().describe("Estimate"),
});

const createTaskOutput = z.object({
  id: z.string().uuid(),
  slug: z.string(),
  title: z.string(),
  // ... 전체 태스크 객체
});
```

**사용 예:**

```json
{
  "tool": "create_task",
  "arguments": {
    "title": "Fix login bug",
    "description": "Users can't login with SSO",
    "priority": "high",
    "labels": ["bug", "auth"]
  }
}
```

#### update_task

```typescript
const updateTaskInput = z.object({
  taskId: z.string().describe("Task ID or slug"),
  title: z.string().min(1).optional(),
  description: z.string().optional(),
  priority: z.enum(["urgent", "high", "medium", "low", "none"]).optional(),
  assigneeId: z.string().uuid().nullable().optional(),
  statusId: z.string().uuid().optional(),
  labels: z.array(z.string()).optional(),
  dueDate: z.string().datetime().nullable().optional(),
  estimate: z.number().int().positive().nullable().optional(),
});
```

#### list_tasks

```typescript
const listTasksInput = z.object({
  statusId: z.string().uuid().optional(),
  statusType: z.enum([
    "backlog", "unstarted", "started", "completed", "canceled"
  ]).optional(),
  assigneeId: z.string().uuid().optional(),
  assignedToMe: z.boolean().optional(),
  priority: z.enum(["urgent", "high", "medium", "low", "none"]).optional(),
  search: z.string().optional(),
  limit: z.number().int().min(1).max(100).default(50),
  offset: z.number().int().min(0).default(0),
});

const listTasksOutput = z.object({
  tasks: z.array(taskSchema),
  total: z.number(),
  hasMore: z.boolean(),
});
```

#### get_task

```typescript
const getTaskInput = z.object({
  taskId: z.string().describe("Task ID (uuid) or slug"),
});

// 반환: 전체 태스크 정보 + 관계
```

#### delete_task

```typescript
const deleteTaskInput = z.object({
  taskId: z.string().describe("Task ID or slug"),
});

const deleteTaskOutput = z.object({
  success: z.boolean(),
  deletedAt: z.string().datetime(),
});
```

---

### 2. 조직 도구 (Cloud - 즉시 실행)

#### list_members

```typescript
const listMembersInput = z.object({
  search: z.string().optional(),
  limit: z.number().int().min(1).max(100).default(50),
});

const listMembersOutput = z.object({
  members: z.array(z.object({
    id: z.string().uuid(),
    name: z.string(),
    email: z.string().email(),
    image: z.string().url().nullable(),
    role: z.enum(["owner", "admin", "member"]),
  })),
});
```

#### list_task_statuses

```typescript
const listTaskStatusesOutput = z.object({
  statuses: z.array(z.object({
    id: z.string().uuid(),
    name: z.string(),
    color: z.string(),
    type: z.enum(["backlog", "unstarted", "started", "completed", "canceled"]),
    position: z.number(),
  })),
});
```

---

### 3. 디바이스 도구 (Desktop으로 라우팅)

디바이스 도구는 `agent_commands` 테이블에 명령을 쓰고 결과를 폴링합니다.

#### list_devices

```typescript
const listDevicesInput = z.object({
  includeOffline: z.boolean().default(false),
});

const listDevicesOutput = z.object({
  devices: z.array(z.object({
    deviceId: z.string(),
    deviceName: z.string(),
    deviceType: z.enum(["desktop", "mobile", "web"]),
    ownerId: z.string().uuid(),
    ownerName: z.string(),
    lastSeenAt: z.string().datetime(),
    isOnline: z.boolean(),
  })),
});
```

#### list_workspaces

```typescript
const listWorkspacesInput = z.object({
  deviceId: z.string().optional(), // 기본: 호출자 디바이스
});

const listWorkspacesOutput = z.object({
  workspaces: z.array(z.object({
    id: z.string().uuid(),
    name: z.string(),
    path: z.string(),
    branch: z.string(),
    isActive: z.boolean(),
    repositoryId: z.string().uuid().nullable(),
  })),
});
```

#### get_current_workspace

```typescript
const getCurrentWorkspaceInput = z.object({
  deviceId: z.string().optional(),
});

const getCurrentWorkspaceOutput = z.object({
  workspace: z.object({
    id: z.string().uuid(),
    name: z.string(),
    path: z.string(),
    branch: z.string(),
    repositoryId: z.string().uuid().nullable(),
    uncommittedChanges: z.number().int(),
    currentTask: taskSchema.nullable(),
  }).nullable(),
});
```

#### create_worktree

```typescript
const createWorktreeInput = z.object({
  deviceId: z.string().optional(),
  name: z.string().optional(),         // 자동 생성 가능
  branchName: z.string().optional(),   // 자동 생성 가능
  baseBranch: z.string().optional(),   // 기본: main
  taskId: z.string().optional(),       // 연결할 태스크
});

const createWorktreeOutput = z.object({
  workspace: z.object({
    id: z.string().uuid(),
    name: z.string(),
    path: z.string(),
    branch: z.string(),
  }),
});
```

#### switch_workspace

```typescript
const switchWorkspaceInput = z.object({
  deviceId: z.string().optional(),
  workspaceId: z.string().uuid().optional(),
  workspaceName: z.string().optional(),
});

const switchWorkspaceOutput = z.object({
  success: z.boolean(),
  workspace: z.object({
    id: z.string().uuid(),
    name: z.string(),
    path: z.string(),
    branch: z.string(),
  }),
});
```

---

## 디바이스 타겟팅

디바이스 명령은 **조직 내 모든 디바이스**를 타겟팅할 수 있습니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    디바이스 타겟팅                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  deviceId 미지정 → API 키의 defaultDeviceId 사용            │
│                                                              │
│  deviceId 지정   → 해당 디바이스로 명령 전송                 │
│                                                              │
│  조건:                                                       │
│  • 디바이스가 온라인 (60초 내 하트비트)                     │
│  • 조직 멤버만 명령 전송 가능                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 공유 스키마

### Task 스키마

```typescript
const taskSchema = z.object({
  id: z.string().uuid(),
  slug: z.string(),
  title: z.string(),
  description: z.string().nullable(),
  priority: z.enum(["urgent", "high", "medium", "low", "none"]),

  status: z.object({
    id: z.string().uuid(),
    name: z.string(),
    color: z.string(),
    type: z.enum(["backlog", "unstarted", "started", "completed", "canceled"]),
  }),

  assignee: z.object({
    id: z.string().uuid(),
    name: z.string(),
    email: z.string(),
    image: z.string().nullable(),
  }).nullable(),

  labels: z.array(z.string()),
  estimate: z.number().nullable(),
  dueDate: z.string().datetime().nullable(),

  branch: z.string().nullable(),
  prUrl: z.string().url().nullable(),

  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
});
```

---

## MCP 서버 구현

### 서버 설정

```typescript
// packages/mcp/src/server.ts

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

export function createMCPServer() {
  const server = new Server(
    {
      name: "superset",
      version: "1.0.0",
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // 도구 등록
  registerTaskTools(server);
  registerDeviceTools(server);
  registerOrganizationTools(server);

  return server;
}
```

### 도구 등록

```typescript
// packages/mcp/src/tools/tasks/create.ts

import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

export function registerCreateTask(server: Server) {
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
      tools: [
        {
          name: "create_task",
          description: "Create a new task in the organization",
          inputSchema: zodToJsonSchema(createTaskInput),
        },
      ],
    };
  });

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    if (request.params.name === "create_task") {
      const input = createTaskInput.parse(request.params.arguments);
      const task = await createTask(input);
      return { content: [{ type: "text", text: JSON.stringify(task) }] };
    }
  });
}
```

---

## 구현 노트

### 1. 검증

모든 입력은 **Zod**로 검증되고 MCP용 **JSON Schema**로 변환됩니다.

```typescript
import { zodToJsonSchema } from "zod-to-json-schema";

const jsonSchema = zodToJsonSchema(createTaskInput);
```

### 2. 디바이스 라우팅

디바이스 도구는 실행 전에 `canRunTool(deviceType, toolName)`를 체크합니다.

```typescript
// 디바이스 타입별 허용 도구
const deviceToolPermissions = {
  desktop: ["list_workspaces", "create_worktree", "switch_workspace"],
  mobile: ["list_workspaces"],
  web: [],
};
```

### 3. 타임아웃

디바이스 명령은 기본 **30초** 타임아웃이며 호출별로 설정 가능합니다.

```typescript
const result = await executeDeviceCommand({
  deviceId,
  command: "create_worktree",
  args: { name: "feature-x" },
  timeout: 60000, // 60초
});
```

### 4. 인증

API 키는 필수이며 user/org/device 컨텍스트를 인코딩합니다.

---

## Claude Code에서 사용

### MCP 설정

```json
// .mcp.json
{
  "mcpServers": {
    "superset": {
      "command": "npx",
      "args": ["@superset/mcp"],
      "env": {
        "SUPERSET_API_KEY": "your_api_key"
      }
    }
  }
}
```

### 에이전트 프롬프트 예시

```
사용자: "새 태스크 만들어줘: 결제 버그 수정, 우선순위 높음"

Claude: create_task 도구를 사용하여 태스크를 생성합니다.

{
  "tool": "create_task",
  "arguments": {
    "title": "결제 버그 수정",
    "priority": "high",
    "labels": ["bug", "payment"]
  }
}

결과: 태스크 PAY-123이 생성되었습니다.
```

---

## 확장 가이드

### 새 도구 추가

1. **스키마 정의** (`packages/mcp/src/tools/my-tool/schema.ts`)

```typescript
export const myToolInput = z.object({
  param1: z.string(),
  param2: z.number().optional(),
});

export const myToolOutput = z.object({
  result: z.string(),
});
```

2. **도구 구현** (`packages/mcp/src/tools/my-tool/index.ts`)

```typescript
export async function myTool(input: MyToolInput): Promise<MyToolOutput> {
  // 구현
  return { result: "success" };
}
```

3. **서버에 등록** (`packages/mcp/src/server.ts`)

```typescript
registerMyTool(server);
```

---

*다음 글에서는 UI 컴포넌트와 상태 관리를 분석합니다.*
