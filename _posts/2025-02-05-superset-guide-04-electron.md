---
layout: post
title: "Superset 완벽 가이드 (4) - Electron 데스크탑 앱"
date: 2025-02-05
permalink: /superset-guide-04-electron/
author: Superset Team
categories: [AI 에이전트, Superset]
tags: [Superset, Electron, Desktop, IPC, node-pty]
original_url: "https://github.com/superset-sh/superset"
excerpt: "Superset의 Electron 데스크탑 앱 아키텍처와 IPC 시스템을 분석합니다."
---

## Electron 앱 개요

Superset의 핵심은 **Electron 기반 데스크탑 앱**입니다. 터미널 관리, 워크스페이스 격리, 에이전트 모니터링 등 주요 기능이 여기서 구현됩니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Electron 아키텍처                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                  Main Process                        │   │
│   │                 (Node.js 환경)                       │   │
│   │                                                      │   │
│   │  • 터미널 관리 (node-pty)                           │   │
│   │  • 워크스페이스/Worktree 관리                        │   │
│   │  • 파일 시스템 접근                                  │   │
│   │  • 시스템 트레이                                     │   │
│   │  • 자동 업데이트                                     │   │
│   └────────────────────┬────────────────────────────────┘   │
│                        │ IPC                                │
│   ┌────────────────────┴────────────────────────────────┐   │
│   │                 Preload Script                       │   │
│   │              (Context Bridge)                        │   │
│   └────────────────────┬────────────────────────────────┘   │
│                        │                                    │
│   ┌────────────────────┴────────────────────────────────┐   │
│   │                Renderer Process                      │   │
│   │                (브라우저 환경)                        │   │
│   │                                                      │   │
│   │  • React UI                                          │   │
│   │  • 상태 관리 (Zustand)                              │   │
│   │  • tRPC 클라이언트                                   │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 디렉토리 구조

```
apps/desktop/src/
├── main/                    # Main Process (Node.js)
│   ├── index.ts            # 앱 진입점
│   ├── lib/                # 핵심 라이브러리
│   │   ├── terminal.ts     # 터미널 관리
│   │   ├── auto-updater.ts # 자동 업데이트
│   │   ├── tray.ts         # 시스템 트레이
│   │   ├── sentry.ts       # 에러 리포팅
│   │   ├── agent-setup.ts  # 에이전트 훅 설정
│   │   └── local-db.ts     # 로컬 SQLite
│   ├── windows/            # 윈도우 관리
│   └── terminal-host/      # 터미널 호스트
│
├── renderer/               # Renderer Process (Browser)
│   ├── index.tsx          # React 진입점
│   ├── index.html         # HTML 템플릿
│   ├── components/        # UI 컴포넌트
│   ├── routes/            # 라우트 정의
│   ├── screens/           # 화면 컴포넌트
│   ├── stores/            # Zustand 스토어
│   ├── hooks/             # React 훅
│   ├── providers/         # Context 프로바이더
│   ├── lib/               # 유틸리티
│   └── react-query/       # React Query 설정
│
├── preload/               # Preload Scripts
│   └── index.ts          # IPC 브릿지
│
├── shared/                # 공유 타입/상수
│   ├── types.ts          # 데이터 모델
│   ├── constants.ts      # 상수 정의
│   └── ipc-channels.ts   # IPC 타입 정의
│
├── lib/                   # 공용 라이브러리
│   ├── electron-app/     # Electron 앱 팩토리
│   └── trpc/             # tRPC 라우터
│       └── routers/      # tRPC 라우터들
│
├── resources/            # 앱 리소스
└── types/                # TypeScript 타입
```

---

## Main Process

### 앱 진입점 (`main/index.ts`)

```typescript
// 싱글 인스턴스 락
const gotTheLock = app.requestSingleInstanceLock();

if (!gotTheLock) {
  app.exit(0);
} else {
  app.on("second-instance", async (_event, argv) => {
    focusMainWindow();
    const url = findDeepLinkInArgv(argv);
    if (url) await processDeepLink(url);
  });

  (async () => {
    await app.whenReady();

    initSentry();                        // 에러 리포팅
    await initAppState();                // 앱 상태 초기화
    await reconcileDaemonSessions();     // 데몬 세션 정리
    setupAgentHooks();                   // 에이전트 훅 설정
    await makeAppSetup(() => MainWindow());
    setupAutoUpdater();                  // 자동 업데이트
    initTray();                          // 시스템 트레이
  })();
}
```

### 딥 링크 처리

```typescript
// superset:// 프로토콜 등록
app.setAsDefaultProtocolClient(PROTOCOL_SCHEME);

// 딥 링크 처리
async function processDeepLink(url: string): Promise<void> {
  // 인증 딥 링크
  const authParams = parseAuthDeepLink(url);
  if (authParams) {
    await handleAuthCallback(authParams);
    return;
  }

  // 일반 네비게이션
  // superset://tasks/my-slug -> /tasks/my-slug
  const path = `/${url.split("://")[1]}`;
  mainWindow.webContents.send("deep-link-navigate", path);
}
```

### 종료 처리

```typescript
app.on("before-quit", async (event) => {
  if (shouldConfirm) {
    event.preventDefault();

    const { response } = await dialog.showMessageBox({
      type: "question",
      buttons: ["Quit", "Cancel"],
      message: "Are you sure you want to quit?",
    });

    if (response === 1) return; // 취소
  }

  disposeTray();
  app.exit(0);
});
```

---

## 타입 안전 IPC 시스템

### IPC 채널 정의

```typescript
// shared/ipc-channels.ts
export interface IpcChannels {
  // 워크스페이스 관리
  "workspace:create": {
    request: { name: string; baseBranch?: string };
    response: { id: string; path: string };
  };

  "workspace:delete": {
    request: { workspaceId: string };
    response: { success: boolean };
  };

  // 터미널 관리
  "terminal:spawn": {
    request: { workspaceId: string; shell?: string };
    response: { sessionId: string };
  };

  "terminal:write": {
    request: { sessionId: string; data: string };
    response: void;
  };
}
```

### Main Process 핸들러

```typescript
// ✅ 올바른 방식 - 객체 파라미터
ipcMain.handle("workspace:create", async (_event, input: {
  name: string;
  baseBranch?: string;
}) => {
  const workspace = await workspaceManager.create(input);
  return { id: workspace.id, path: workspace.path };
});

// ❌ 잘못된 방식 - 위치 파라미터
ipcMain.handle("workspace:create", async (_event, name, baseBranch) => {
  // 타입 추론이 안됨!
});
```

### Preload Script

```typescript
// preload/index.ts
import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("ipcRenderer", {
  invoke: <K extends keyof IpcChannels>(
    channel: K,
    data: IpcChannels[K]["request"]
  ): Promise<IpcChannels[K]["response"]> => {
    return ipcRenderer.invoke(channel, data);
  },

  on: (channel: string, callback: (...args: any[]) => void) => {
    ipcRenderer.on(channel, callback);
  },

  off: (channel: string, callback: (...args: any[]) => void) => {
    ipcRenderer.removeListener(channel, callback);
  },
});
```

### Renderer에서 호출

```typescript
// renderer/hooks/useWorkspace.ts
export function useCreateWorkspace() {
  return useMutation({
    mutationFn: async (input: { name: string; baseBranch?: string }) => {
      // 타입 안전 - 자동 추론!
      const result = await window.ipcRenderer.invoke("workspace:create", input);
      return result;
    },
  });
}
```

---

## 중요 아키텍처 규칙

### Node.js 모듈 임포트 제한

```
⚠️ Renderer나 공유 코드에서 Node.js 모듈을 절대 임포트하지 마세요!
```

```typescript
// ❌ renderer에서 금지
import fs from "node:fs";
import path from "node:path";
// Error: Module "node:fs" has been externalized for browser compatibility

// ✅ main process에서만 허용
// main/lib/file-utils.ts
import fs from "node:fs";
import path from "node:path";
```

**환경별 접근:**

| 환경 | Node.js 모듈 | 위치 |
|------|-------------|------|
| Main Process | ✅ 허용 | `src/main/` |
| Renderer Process | ❌ 금지 | `src/renderer/` |
| Shared Code | ❌ 금지 | `src/shared/`, `src/lib/` |

**Node.js 기능이 필요한 경우:**
1. 코드를 `src/main/lib/`로 이동
2. IPC로 main-renderer 통신
3. preload 또는 환경 변수로 데이터 전달

### 린트 체크

```bash
# Node.js 임포트 위반 검사
bun run lint:check-node-imports

# typecheck에도 포함됨
bun run typecheck
```

---

## 환경 변수 로딩

### 로딩 순서

```
1. main/index.ts
   └─ dotenv.config({ override: true })  # .env 로드 (main process)

2. electron.vite.config.ts
   └─ dotenv.config({ override: true })  # 빌드 타임 로드
```

### 중요 사항

```typescript
// main/index.ts - 최상단에서 로드
import dotenv from "dotenv";
dotenv.config({ path: "../../.env", override: true });

// ⚠️ override: true 필수!
// 상속된 환경 변수보다 .env 값이 우선되도록
```

---

## tRPC 라우터

데스크탑 앱은 내부 API로 tRPC를 사용합니다.

```
lib/trpc/routers/
├── index.ts              # 루트 라우터
├── auth/                 # 인증
├── workspaces/           # 워크스페이스 관리
│   ├── procedures/       # CRUD 프로시저
│   │   ├── create.ts
│   │   ├── delete.ts
│   │   ├── query.ts
│   │   └── ...
│   └── utils/            # 유틸리티
│       ├── worktree.ts   # git worktree
│       ├── git.ts        # git 작업
│       ├── setup.ts      # 설정 스크립트
│       └── teardown.ts   # 정리 스크립트
├── projects/             # 프로젝트 관리
├── terminal/             # 터미널 관리
├── changes/              # 변경사항 관리
│   └── security/         # 보안 검증
├── auto-update/          # 자동 업데이트
├── settings/             # 설정
├── hotkeys/              # 단축키
└── ...
```

### 라우터 예시

```typescript
// routers/workspaces/procedures/create.ts
export const create = protectedProcedure
  .input(z.object({
    projectId: z.string().uuid(),
    name: z.string().optional(),
    branchName: z.string().optional(),
    baseBranch: z.string().optional(),
  }))
  .mutation(async ({ ctx, input }) => {
    // 1. 워크스페이스 생성
    const workspace = await createWorkspace(input);

    // 2. Git worktree 생성
    await createWorktree({
      repoPath: project.path,
      workspacePath: workspace.path,
      branch: input.branchName,
    });

    // 3. 설정 스크립트 실행
    await runSetupScripts(workspace.path);

    return workspace;
  });
```

---

## 보안 고려사항

### 경로 검증

```typescript
// changes/security/path-validation.ts
export function validatePath(userPath: string, basePath: string): boolean {
  const resolved = path.resolve(basePath, userPath);

  // basePath 외부로 탈출 방지
  if (!resolved.startsWith(basePath)) {
    throw new Error("Path traversal detected");
  }

  return true;
}
```

### Git 명령 보안

```typescript
// changes/security/git-commands.ts
const ALLOWED_GIT_COMMANDS = [
  "status",
  "diff",
  "log",
  "branch",
  "checkout",
  // ...
];

export function validateGitCommand(command: string): boolean {
  const [gitCmd, ...args] = command.split(" ");

  if (!ALLOWED_GIT_COMMANDS.includes(args[0])) {
    throw new Error(`Git command not allowed: ${args[0]}`);
  }

  return true;
}
```

### 파일 시스템 보안

```typescript
// changes/security/secure-fs.ts
export async function secureReadFile(
  filePath: string,
  basePath: string
): Promise<string> {
  validatePath(filePath, basePath);
  return fs.readFile(filePath, "utf-8");
}
```

---

## 디버깅 팁

### DevTools 열기

```typescript
// 개발 모드에서 자동 열림
if (process.env.NODE_ENV === "development") {
  mainWindow.webContents.openDevTools();
}
```

### IPC 로깅

```typescript
// main process
ipcMain.handle("my-channel", async (_event, input) => {
  console.log("[ipc/my-channel] Request:", input);
  const result = await doSomething(input);
  console.log("[ipc/my-channel] Response:", result);
  return result;
});
```

### 에러 추적

```bash
# Sentry DSN 설정
SENTRY_DSN="https://xxx@sentry.io/xxx"
```

---

*다음 글에서는 Workspace와 Git Worktree 시스템을 분석합니다.*
