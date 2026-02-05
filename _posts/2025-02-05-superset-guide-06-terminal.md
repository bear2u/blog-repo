---
layout: post
title: "Superset 완벽 가이드 (6) - 터미널 관리"
date: 2025-02-05
permalink: /superset-guide-06-terminal/
author: Superset Team
categories: [AI 에이전트, Superset]
tags: [Superset, Terminal, node-pty, PTY, xterm.js]
original_url: "https://github.com/superset-sh/superset"
excerpt: "Superset의 터미널 관리 시스템과 node-pty 통합을 분석합니다."
---

## 터미널 시스템 개요

Superset의 터미널은 **node-pty**와 **xterm.js**를 결합하여 네이티브급 터미널 경험을 제공합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    터미널 아키텍처                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌────────────────────────────────────────────────────┐    │
│   │              Renderer Process                       │    │
│   │                                                     │    │
│   │   ┌─────────┐    ┌─────────┐    ┌─────────┐       │    │
│   │   │ xterm.js│    │ xterm.js│    │ xterm.js│       │    │
│   │   │ Terminal│    │ Terminal│    │ Terminal│       │    │
│   │   └────┬────┘    └────┬────┘    └────┬────┘       │    │
│   │        │              │              │             │    │
│   └────────┼──────────────┼──────────────┼─────────────┘    │
│            │              │              │                   │
│            └──────────────┼──────────────┘                   │
│                           │ IPC                              │
│   ┌───────────────────────┼────────────────────────────┐    │
│   │              Main Process                           │    │
│   │                       │                             │    │
│   │   ┌───────────────────┴───────────────────────┐    │    │
│   │   │           Terminal Manager                 │    │    │
│   │   └───────────────────┬───────────────────────┘    │    │
│   │                       │                             │    │
│   │   ┌─────────┐    ┌────┴────┐    ┌─────────┐       │    │
│   │   │ node-pty│    │ node-pty│    │ node-pty│       │    │
│   │   │   PTY   │    │   PTY   │    │   PTY   │       │    │
│   │   └────┬────┘    └────┬────┘    └────┬────┘       │    │
│   │        │              │              │             │    │
│   └────────┼──────────────┼──────────────┼─────────────┘    │
│            │              │              │                   │
│            ▼              ▼              ▼                   │
│        /bin/zsh       /bin/zsh       /bin/zsh               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## node-pty 기본

### PTY (Pseudo Terminal)란?

PTY는 **가상 터미널 장치**로, 프로그램이 실제 터미널처럼 동작할 수 있게 합니다.

```
┌──────────────┐      ┌──────────────┐
│   Master     │◄────►│    Slave     │
│   (앱 측)    │      │  (셸 측)     │
└──────────────┘      └──────────────┘
       │                     │
       │                     ▼
   입력/출력              /bin/zsh
   처리
```

### node-pty 설치

```bash
# node-pty는 네이티브 모듈이므로 빌드 필요
bun add node-pty
```

---

## 터미널 관리 구현

### 터미널 세션 생성

```typescript
// main/lib/terminal.ts

import * as pty from "node-pty";

interface TerminalSession {
  id: string;
  pty: pty.IPty;
  workspaceId: string;
}

const sessions = new Map<string, TerminalSession>();

export function createTerminalSession({
  workspaceId,
  cwd,
  shell = process.env.SHELL || "/bin/zsh",
  env = process.env,
}: CreateOptions): string {
  const sessionId = crypto.randomUUID();

  // PTY 프로세스 생성
  const ptyProcess = pty.spawn(shell, [], {
    name: "xterm-256color",
    cols: 80,
    rows: 24,
    cwd,
    env: {
      ...env,
      TERM: "xterm-256color",
      COLORTERM: "truecolor",
      // Superset 환경 변수
      SUPERSET_WORKSPACE_ID: workspaceId,
      SUPERSET_SESSION_ID: sessionId,
    },
  });

  // 세션 저장
  sessions.set(sessionId, {
    id: sessionId,
    pty: ptyProcess,
    workspaceId,
  });

  return sessionId;
}
```

### 데이터 처리

```typescript
// main/lib/terminal.ts

export function setupTerminalDataHandler(
  sessionId: string,
  onData: (data: string) => void
) {
  const session = sessions.get(sessionId);
  if (!session) throw new Error("Session not found");

  // PTY에서 데이터 수신
  session.pty.onData((data) => {
    onData(data);
  });
}

export function writeToTerminal(sessionId: string, data: string) {
  const session = sessions.get(sessionId);
  if (!session) throw new Error("Session not found");

  // PTY로 데이터 전송
  session.pty.write(data);
}
```

### 크기 조정

```typescript
// main/lib/terminal.ts

export function resizeTerminal(
  sessionId: string,
  cols: number,
  rows: number
) {
  const session = sessions.get(sessionId);
  if (!session) return;

  session.pty.resize(cols, rows);
}
```

### 세션 종료

```typescript
// main/lib/terminal.ts

export function destroyTerminalSession(sessionId: string) {
  const session = sessions.get(sessionId);
  if (!session) return;

  // PTY 프로세스 종료
  session.pty.kill();

  // 세션 제거
  sessions.delete(sessionId);
}
```

---

## tRPC 터미널 라우터

```typescript
// lib/trpc/routers/terminal/index.ts

import { router, protectedProcedure } from "@/lib/trpc";
import { z } from "zod";
import * as terminal from "@/main/lib/terminal";

export const terminalRouter = router({
  // 터미널 생성
  spawn: protectedProcedure
    .input(z.object({
      workspaceId: z.string().uuid(),
      shell: z.string().optional(),
    }))
    .mutation(async ({ input }) => {
      const workspace = await getWorkspace(input.workspaceId);

      const sessionId = terminal.createTerminalSession({
        workspaceId: input.workspaceId,
        cwd: workspace.path,
        shell: input.shell,
      });

      return { sessionId };
    }),

  // 데이터 쓰기
  write: protectedProcedure
    .input(z.object({
      sessionId: z.string(),
      data: z.string(),
    }))
    .mutation(async ({ input }) => {
      terminal.writeToTerminal(input.sessionId, input.data);
    }),

  // 크기 조정
  resize: protectedProcedure
    .input(z.object({
      sessionId: z.string(),
      cols: z.number(),
      rows: z.number(),
    }))
    .mutation(async ({ input }) => {
      terminal.resizeTerminal(input.sessionId, input.cols, input.rows);
    }),

  // 세션 종료
  destroy: protectedProcedure
    .input(z.object({
      sessionId: z.string(),
    }))
    .mutation(async ({ input }) => {
      terminal.destroyTerminalSession(input.sessionId);
    }),
});
```

---

## Renderer: xterm.js 통합

### Terminal 컴포넌트

```typescript
// renderer/components/Terminal/Terminal.tsx

import { Terminal as XTerm } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import { WebLinksAddon } from "@xterm/addon-web-links";
import { useEffect, useRef } from "react";

interface TerminalProps {
  sessionId: string;
  onData?: (data: string) => void;
}

export function Terminal({ sessionId, onData }: TerminalProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const terminalRef = useRef<XTerm | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // xterm.js 인스턴스 생성
    const terminal = new XTerm({
      theme: {
        background: "#1e1e1e",
        foreground: "#d4d4d4",
        cursor: "#d4d4d4",
        selectionBackground: "#264f78",
      },
      fontFamily: "JetBrains Mono, monospace",
      fontSize: 14,
      cursorBlink: true,
      cursorStyle: "bar",
      allowProposedApi: true,
    });

    // 애드온 로드
    const fitAddon = new FitAddon();
    terminal.loadAddon(fitAddon);
    terminal.loadAddon(new WebLinksAddon());

    // DOM에 연결
    terminal.open(containerRef.current);
    fitAddon.fit();

    // 참조 저장
    terminalRef.current = terminal;
    fitAddonRef.current = fitAddon;

    // 사용자 입력 처리
    terminal.onData((data) => {
      window.ipcRenderer.invoke("terminal:write", { sessionId, data });
      onData?.(data);
    });

    // PTY 출력 수신
    window.ipcRenderer.on(`terminal:data:${sessionId}`, (_event, data) => {
      terminal.write(data);
    });

    // 크기 변경 감지
    const resizeObserver = new ResizeObserver(() => {
      fitAddon.fit();
      window.ipcRenderer.invoke("terminal:resize", {
        sessionId,
        cols: terminal.cols,
        rows: terminal.rows,
      });
    });
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
      terminal.dispose();
      window.ipcRenderer.off(`terminal:data:${sessionId}`);
    };
  }, [sessionId]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full bg-[#1e1e1e]"
    />
  );
}
```

### 터미널 스타일

```css
/* renderer/globals.css */

/* xterm.js 기본 스타일 */
@import "@xterm/xterm/css/xterm.css";

/* 커스텀 스타일 */
.xterm {
  padding: 8px;
}

.xterm-viewport::-webkit-scrollbar {
  width: 10px;
}

.xterm-viewport::-webkit-scrollbar-thumb {
  background: #3c3c3c;
  border-radius: 5px;
}
```

---

## 탭 & 패인 관리

### 탭 스토어

```typescript
// renderer/stores/tabs/index.ts

interface Tab {
  id: string;
  workspaceId: string;
  sessionId: string;
  title: string;
}

interface TabsState {
  tabs: Tab[];
  activeTabId: string | null;

  addTab: (workspaceId: string) => Promise<string>;
  removeTab: (tabId: string) => void;
  setActiveTab: (tabId: string) => void;
}

export const useTabsStore = create<TabsState>((set, get) => ({
  tabs: [],
  activeTabId: null,

  addTab: async (workspaceId) => {
    // 새 터미널 세션 생성
    const { sessionId } = await trpc.terminal.spawn.mutate({
      workspaceId,
    });

    const tabId = crypto.randomUUID();
    const newTab: Tab = {
      id: tabId,
      workspaceId,
      sessionId,
      title: "Terminal",
    };

    set((state) => ({
      tabs: [...state.tabs, newTab],
      activeTabId: tabId,
    }));

    return tabId;
  },

  removeTab: (tabId) => {
    const tab = get().tabs.find((t) => t.id === tabId);
    if (tab) {
      // 터미널 세션 종료
      trpc.terminal.destroy.mutate({ sessionId: tab.sessionId });
    }

    set((state) => {
      const newTabs = state.tabs.filter((t) => t.id !== tabId);
      const newActiveId = state.activeTabId === tabId
        ? newTabs[newTabs.length - 1]?.id || null
        : state.activeTabId;

      return { tabs: newTabs, activeTabId: newActiveId };
    });
  },

  setActiveTab: (tabId) => set({ activeTabId: tabId }),
}));
```

### 분할 패인

```typescript
// renderer/stores/tabs/split-panes.ts

type SplitDirection = "horizontal" | "vertical";

interface Pane {
  id: string;
  tabId: string;
  size: number; // 퍼센트
}

interface SplitState {
  direction: SplitDirection;
  panes: Pane[];

  split: (direction: SplitDirection, tabId: string) => void;
  removePane: (paneId: string) => void;
  resizePane: (paneId: string, size: number) => void;
}

export const useSplitPanes = create<SplitState>((set) => ({
  direction: "horizontal",
  panes: [],

  split: (direction, tabId) => {
    set((state) => ({
      direction,
      panes: [
        ...state.panes,
        {
          id: crypto.randomUUID(),
          tabId,
          size: 100 / (state.panes.length + 1),
        },
      ].map((p) => ({ ...p, size: 100 / (state.panes.length + 1) })),
    }));
  },

  removePane: (paneId) => {
    set((state) => {
      const newPanes = state.panes.filter((p) => p.id !== paneId);
      const newSize = 100 / newPanes.length;
      return {
        panes: newPanes.map((p) => ({ ...p, size: newSize })),
      };
    });
  },

  resizePane: (paneId, size) => {
    set((state) => ({
      panes: state.panes.map((p) =>
        p.id === paneId ? { ...p, size } : p
      ),
    }));
  },
}));
```

---

## 데몬 모드 & 세션 복구

### 데몬 세션

```typescript
// main/lib/terminal.ts

interface DaemonSession extends TerminalSession {
  startedAt: Date;
  lastOutput: string;
  isRunning: boolean;
}

const daemonSessions = new Map<string, DaemonSession>();

export function createDaemonSession(options: CreateOptions) {
  const sessionId = createTerminalSession(options);
  const session = sessions.get(sessionId)!;

  // 데몬 세션으로 승격
  const daemonSession: DaemonSession = {
    ...session,
    startedAt: new Date(),
    lastOutput: "",
    isRunning: true,
  };

  // 출력 버퍼링
  session.pty.onData((data) => {
    daemonSession.lastOutput += data;
    // 최근 10KB만 유지
    if (daemonSession.lastOutput.length > 10240) {
      daemonSession.lastOutput = daemonSession.lastOutput.slice(-10240);
    }
  });

  // 종료 감지
  session.pty.onExit(() => {
    daemonSession.isRunning = false;
  });

  daemonSessions.set(sessionId, daemonSession);

  return sessionId;
}
```

### 세션 복구

```typescript
// main/lib/terminal.ts

export async function reconcileDaemonSessions() {
  // 로컬 DB에서 이전 세션 로드
  const savedSessions = await localDb
    .select()
    .from(terminalSessions)
    .where(eq(terminalSessions.isDaemon, true));

  for (const saved of savedSessions) {
    const session = daemonSessions.get(saved.sessionId);

    if (!session || !session.isRunning) {
      // 세션이 없거나 종료됨 - DB에서 제거
      await localDb
        .delete(terminalSessions)
        .where(eq(terminalSessions.sessionId, saved.sessionId));
    }
  }

  console.log(`[terminal] Reconciled ${savedSessions.length} daemon sessions`);
}
```

---

## 프리셋

워크스페이스마다 자주 사용하는 명령어를 프리셋으로 저장할 수 있습니다.

```typescript
// 프리셋 실행 (Ctrl+1-9)
export async function executePreset(
  workspaceId: string,
  presetIndex: number
) {
  const workspace = await getWorkspace(workspaceId);
  const config = await loadSupersetConfig(workspace.path);

  const preset = config.presets?.[presetIndex];
  if (!preset) return;

  // 새 터미널에서 프리셋 실행
  const sessionId = createTerminalSession({
    workspaceId,
    cwd: workspace.path,
  });

  writeToTerminal(sessionId, preset.command + "\n");

  return sessionId;
}
```

### 프리셋 설정

```json
// .superset/config.json
{
  "setup": ["./.superset/setup.sh"],
  "presets": [
    {
      "name": "Dev Server",
      "command": "bun run dev"
    },
    {
      "name": "Tests",
      "command": "bun test"
    },
    {
      "name": "Build",
      "command": "bun run build"
    }
  ]
}
```

---

## 단축키

| 단축키 | 동작 |
|--------|------|
| `⌘T` | 새 탭 |
| `⌘W` | 탭/패인 닫기 |
| `⌘D` | 오른쪽 분할 |
| `⌘⇧D` | 아래 분할 |
| `⌘K` | 터미널 클리어 |
| `⌘F` | 터미널 내 검색 |
| `⌘⌥←/→` | 이전/다음 탭 |
| `Ctrl+1-9` | 프리셋 1-9 실행 |

---

*다음 글에서는 tRPC 라우터 구조를 분석합니다.*
