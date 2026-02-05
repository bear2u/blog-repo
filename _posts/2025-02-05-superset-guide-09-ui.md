---
layout: post
title: "Superset 완벽 가이드 (9) - UI 컴포넌트"
date: 2025-02-05
permalink: /superset-guide-09-ui/
author: Superset Team
categories: [AI 에이전트, Superset]
tags: [Superset, React, UI, Zustand, TailwindCSS, shadcn/ui]
original_url: "https://github.com/superset-sh/superset"
excerpt: "Superset의 React UI 컴포넌트와 상태 관리 시스템을 분석합니다."
---

## UI 스택 개요

Superset은 현대적인 React UI 스택을 사용합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                      UI 기술 스택                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  React            - UI 프레임워크                    │   │
│   │  TailwindCSS v4   - 유틸리티 기반 스타일링           │   │
│   │  shadcn/ui        - UI 컴포넌트 라이브러리           │   │
│   │  Zustand          - 상태 관리                        │   │
│   │  React Query      - 서버 상태 관리                   │   │
│   │  Radix UI         - 헤드리스 UI 프리미티브           │   │
│   │  Lucide           - 아이콘                           │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Renderer 디렉토리 구조

```
apps/desktop/src/renderer/
├── index.tsx              # React 진입점
├── index.html             # HTML 템플릿
├── globals.css            # 글로벌 스타일
│
├── components/            # UI 컴포넌트
│   ├── ui/               # shadcn/ui 컴포넌트
│   ├── Terminal/         # 터미널 컴포넌트
│   ├── Sidebar/          # 사이드바
│   ├── DiffViewer/       # Diff 뷰어
│   └── ...
│
├── screens/               # 화면 컴포넌트
│   ├── Home/             # 홈 화면
│   ├── Workspace/        # 워크스페이스 화면
│   └── Settings/         # 설정 화면
│
├── routes/                # 라우트 정의
│   └── index.tsx
│
├── stores/                # Zustand 스토어
│   ├── tabs/             # 탭 상태
│   ├── changes/          # 변경사항 상태
│   ├── hotkeys/          # 단축키 상태
│   ├── theme/            # 테마 상태
│   └── ...
│
├── hooks/                 # React 훅
│   └── useTerminal.ts
│
├── providers/             # Context 프로바이더
│   ├── ThemeProvider/
│   ├── TRPCProvider/
│   └── HotkeysProvider/
│
├── lib/                   # 유틸리티
│   ├── trpc.ts
│   └── utils.ts
│
└── react-query/           # React Query 설정
    └── queryClient.ts
```

---

## shadcn/ui 컴포넌트

### 컴포넌트 추가

```bash
# packages/ui/ 디렉토리에서 실행
cd packages/ui
npx shadcn@latest add button
npx shadcn@latest add dialog
npx shadcn@latest add dropdown-menu
```

### 사용 예시

```typescript
// renderer/components/CreateWorkspaceDialog.tsx

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@superset/ui/dialog";
import { Button } from "@superset/ui/button";
import { Input } from "@superset/ui/input";
import { Label } from "@superset/ui/label";

export function CreateWorkspaceDialog() {
  const [name, setName] = useState("");
  const createMutation = trpc.workspaces.create.useMutation();

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button>New Workspace</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Create Workspace</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name">Name</Label>
            <Input
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="feature-login"
            />
          </div>

          <Button
            onClick={() => createMutation.mutate({ name })}
            disabled={createMutation.isPending}
          >
            {createMutation.isPending ? "Creating..." : "Create"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
```

---

## Zustand 상태 관리

### 스토어 구조

```
stores/
├── index.ts                    # 스토어 내보내기
├── sidebar-state.ts            # 사이드바 상태
├── workspace-sidebar-state.ts  # 워크스페이스 사이드바
├── settings-state.ts           # 설정 상태
├── new-workspace-modal.ts      # 모달 상태
├── file-explorer.ts            # 파일 탐색기
├── chat-panel-state.ts         # 채팅 패널
├── config-modal.ts             # 설정 모달
├── workspace-init.ts           # 워크스페이스 초기화
├── drag-pane-store.ts          # 드래그 패인
│
├── tabs/                       # 탭 관련
│   └── index.ts
│
├── changes/                    # 변경사항 관련
│   └── index.ts
│
├── hotkeys/                    # 단축키 관련
│   └── index.ts
│
├── theme/                      # 테마 관련
│   └── index.ts
│
├── ports/                      # 포트 관리
│   └── index.ts
│
├── ringtone/                   # 알림음
│   └── index.ts
│
└── markdown-preferences/       # 마크다운 설정
    └── index.ts
```

### 스토어 예시: 사이드바 상태

```typescript
// stores/sidebar-state.ts

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SidebarState {
  // 상태
  isOpen: boolean;
  width: number;
  activeSection: "workspaces" | "files" | "changes";

  // 액션
  toggle: () => void;
  setWidth: (width: number) => void;
  setActiveSection: (section: SidebarState["activeSection"]) => void;
}

export const useSidebarState = create<SidebarState>()(
  persist(
    (set) => ({
      // 초기 상태
      isOpen: true,
      width: 280,
      activeSection: "workspaces",

      // 액션
      toggle: () => set((state) => ({ isOpen: !state.isOpen })),
      setWidth: (width) => set({ width }),
      setActiveSection: (section) => set({ activeSection: section }),
    }),
    {
      name: "sidebar-state", // localStorage 키
    }
  )
);
```

### 스토어 예시: 탭 상태

```typescript
// stores/tabs/index.ts

import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

interface Tab {
  id: string;
  type: "terminal" | "editor" | "diff";
  title: string;
  workspaceId: string;
  sessionId?: string;
  filePath?: string;
}

interface TabsState {
  tabs: Tab[];
  activeTabId: string | null;

  addTab: (tab: Omit<Tab, "id">) => string;
  removeTab: (id: string) => void;
  setActiveTab: (id: string) => void;
  updateTab: (id: string, updates: Partial<Tab>) => void;
  reorderTabs: (fromIndex: number, toIndex: number) => void;
}

export const useTabsState = create<TabsState>()(
  immer((set, get) => ({
    tabs: [],
    activeTabId: null,

    addTab: (tabData) => {
      const id = crypto.randomUUID();
      set((state) => {
        state.tabs.push({ ...tabData, id });
        state.activeTabId = id;
      });
      return id;
    },

    removeTab: (id) => {
      set((state) => {
        const index = state.tabs.findIndex((t) => t.id === id);
        if (index === -1) return;

        state.tabs.splice(index, 1);

        // 다음 탭 선택
        if (state.activeTabId === id) {
          state.activeTabId = state.tabs[index - 1]?.id
            || state.tabs[index]?.id
            || null;
        }
      });
    },

    setActiveTab: (id) => set({ activeTabId: id }),

    updateTab: (id, updates) => {
      set((state) => {
        const tab = state.tabs.find((t) => t.id === id);
        if (tab) {
          Object.assign(tab, updates);
        }
      });
    },

    reorderTabs: (fromIndex, toIndex) => {
      set((state) => {
        const [tab] = state.tabs.splice(fromIndex, 1);
        state.tabs.splice(toIndex, 0, tab);
      });
    },
  }))
);
```

---

## 테마 시스템

### 테마 스토어

```typescript
// stores/theme/index.ts

import { create } from "zustand";
import { persist } from "zustand/middleware";

type Theme = "light" | "dark" | "system";

interface ThemeState {
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

export const useTheme = create<ThemeState>()(
  persist(
    (set) => ({
      theme: "system",
      setTheme: (theme) => {
        set({ theme });
        applyTheme(theme);
      },
    }),
    { name: "theme" }
  )
);

function applyTheme(theme: Theme) {
  const root = document.documentElement;

  if (theme === "system") {
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.classList.toggle("dark", prefersDark);
  } else {
    root.classList.toggle("dark", theme === "dark");
  }
}
```

### 테마 프로바이더

```typescript
// providers/ThemeProvider/index.tsx

import { useEffect } from "react";
import { useTheme } from "@/stores/theme";

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const { theme } = useTheme();

  useEffect(() => {
    const root = document.documentElement;

    if (theme === "system") {
      const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

      const handler = (e: MediaQueryListEvent) => {
        root.classList.toggle("dark", e.matches);
      };

      root.classList.toggle("dark", mediaQuery.matches);
      mediaQuery.addEventListener("change", handler);

      return () => mediaQuery.removeEventListener("change", handler);
    }

    root.classList.toggle("dark", theme === "dark");
  }, [theme]);

  return <>{children}</>;
}
```

---

## 단축키 시스템

### 단축키 스토어

```typescript
// stores/hotkeys/index.ts

import { create } from "zustand";

interface HotkeyBinding {
  id: string;
  key: string;
  modifiers: ("ctrl" | "shift" | "alt" | "meta")[];
  action: () => void;
  description: string;
  scope?: string;
}

interface HotkeysState {
  bindings: Map<string, HotkeyBinding>;
  activeScope: string | null;

  register: (binding: Omit<HotkeyBinding, "id">) => string;
  unregister: (id: string) => void;
  setScope: (scope: string | null) => void;
}

export const useHotkeys = create<HotkeysState>((set, get) => ({
  bindings: new Map(),
  activeScope: null,

  register: (binding) => {
    const id = crypto.randomUUID();
    set((state) => {
      const newBindings = new Map(state.bindings);
      newBindings.set(id, { ...binding, id });
      return { bindings: newBindings };
    });
    return id;
  },

  unregister: (id) => {
    set((state) => {
      const newBindings = new Map(state.bindings);
      newBindings.delete(id);
      return { bindings: newBindings };
    });
  },

  setScope: (scope) => set({ activeScope: scope }),
}));
```

### 단축키 프로바이더

```typescript
// providers/HotkeysProvider/index.tsx

import { useEffect } from "react";
import { useHotkeys } from "@/stores/hotkeys";

export function HotkeysProvider({ children }: { children: React.ReactNode }) {
  const { bindings, activeScope } = useHotkeys();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      for (const binding of bindings.values()) {
        // 스코프 체크
        if (binding.scope && binding.scope !== activeScope) continue;

        // 키 매칭
        const keyMatch = e.key.toLowerCase() === binding.key.toLowerCase();
        const ctrlMatch = binding.modifiers.includes("ctrl") === e.ctrlKey;
        const shiftMatch = binding.modifiers.includes("shift") === e.shiftKey;
        const altMatch = binding.modifiers.includes("alt") === e.altKey;
        const metaMatch = binding.modifiers.includes("meta") === e.metaKey;

        if (keyMatch && ctrlMatch && shiftMatch && altMatch && metaMatch) {
          e.preventDefault();
          binding.action();
          break;
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [bindings, activeScope]);

  return <>{children}</>;
}
```

### 단축키 훅

```typescript
// hooks/useHotkey.ts

import { useEffect } from "react";
import { useHotkeys } from "@/stores/hotkeys";

export function useHotkey(
  key: string,
  modifiers: ("ctrl" | "shift" | "alt" | "meta")[],
  action: () => void,
  deps: any[] = []
) {
  const { register, unregister } = useHotkeys();

  useEffect(() => {
    const id = register({
      key,
      modifiers,
      action,
      description: "",
    });

    return () => unregister(id);
  }, deps);
}
```

---

## 파일 탐색기

```typescript
// stores/file-explorer.ts

import { create } from "zustand";

interface FileNode {
  name: string;
  path: string;
  type: "file" | "directory";
  children?: FileNode[];
}

interface FileExplorerState {
  root: FileNode | null;
  expandedPaths: Set<string>;
  selectedPath: string | null;

  setRoot: (node: FileNode) => void;
  toggleExpanded: (path: string) => void;
  setSelected: (path: string) => void;
  expandPath: (path: string) => void;
}

export const useFileExplorer = create<FileExplorerState>((set) => ({
  root: null,
  expandedPaths: new Set(),
  selectedPath: null,

  setRoot: (node) => set({ root: node }),

  toggleExpanded: (path) =>
    set((state) => {
      const newExpanded = new Set(state.expandedPaths);
      if (newExpanded.has(path)) {
        newExpanded.delete(path);
      } else {
        newExpanded.add(path);
      }
      return { expandedPaths: newExpanded };
    }),

  setSelected: (path) => set({ selectedPath: path }),

  expandPath: (path) =>
    set((state) => {
      const newExpanded = new Set(state.expandedPaths);
      // 경로의 모든 상위 디렉토리 확장
      const parts = path.split("/");
      let current = "";
      for (const part of parts.slice(0, -1)) {
        current += "/" + part;
        newExpanded.add(current);
      }
      return { expandedPaths: newExpanded };
    }),
}));
```

---

## 컴포넌트 패턴

### 1. 컴포넌트당 하나의 폴더

```
components/
└── WorkspaceCard/
    ├── WorkspaceCard.tsx     # 메인 컴포넌트
    ├── WorkspaceCard.test.tsx # 테스트
    ├── index.ts              # 배럴 export
    └── components/           # 서브 컴포넌트
        └── StatusBadge/
```

### 2. 의존성 코로케이션

```
components/
└── DiffViewer/
    ├── DiffViewer.tsx
    ├── useDiffParser.ts      # 전용 훅
    ├── constants.ts          # 전용 상수
    └── types.ts              # 전용 타입
```

### 3. 조건부 렌더링

```typescript
// ✅ Good - 조기 반환
function WorkspaceView({ workspace }: Props) {
  if (!workspace) return <EmptyState />;
  if (workspace.isLoading) return <Skeleton />;

  return <WorkspaceContent workspace={workspace} />;
}

// ❌ Bad - 깊은 중첩
function WorkspaceView({ workspace }: Props) {
  return (
    <div>
      {workspace ? (
        workspace.isLoading ? (
          <Skeleton />
        ) : (
          <WorkspaceContent workspace={workspace} />
        )
      ) : (
        <EmptyState />
      )}
    </div>
  );
}
```

---

*다음 글에서는 확장 및 커스터마이징 방법을 분석합니다.*
