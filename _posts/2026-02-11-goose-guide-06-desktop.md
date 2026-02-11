---
layout: post
title: "Goose 완벽 가이드 (06) - Desktop 앱"
date: 2026-02-11
permalink: /goose-guide-06-desktop/
author: Block
categories: [AI 에이전트, 개발 도구]
tags: [Goose, Desktop, Electron, React, UI]
original_url: "https://github.com/block/goose"
excerpt: "Goose Desktop 앱의 UI/UX와 Electron 아키텍처"
---

## Desktop 앱 개요

Goose Desktop은 **Electron** 기반의 크로스 플랫폼 데스크톱 애플리케이션으로, 직관적인 GUI를 통해 AI 에이전트를 사용할 수 있습니다.

---

## 기술 스택

```
┌─────────────────────────────────────────────────────┐
│              Goose Desktop Stack                     │
├─────────────────────────────────────────────────────┤
│ Framework      │ Electron 28+                       │
│ UI Library     │ React 18                           │
│ Language       │ TypeScript                         │
│ Build Tool     │ Vite                               │
│ Components     │ Shadcn UI                          │
│ State          │ React Hooks                        │
│ API Client     │ OpenAPI Generated                  │
└─────────────────────────────────────────────────────┘
```

---

## 프로젝트 구조

```
ui/desktop/
├── src/
│   ├── main.ts                  # Electron 메인 프로세스
│   ├── preload.ts               # Preload 스크립트
│   ├── renderer/                # React 앱
│   │   ├── App.tsx              # 메인 컴포넌트
│   │   ├── components/          # UI 컴포넌트
│   │   │   ├── Chat/
│   │   │   ├── Sidebar/
│   │   │   ├── Settings/
│   │   │   └── Extensions/
│   │   ├── hooks/               # React Hooks
│   │   │   ├── useSession.ts
│   │   │   ├── useProvider.ts
│   │   │   └── useExtensions.ts
│   │   ├── lib/                 # 유틸리티
│   │   │   ├── api.ts           # API 클라이언트
│   │   │   └── storage.ts       # 로컬 스토리지
│   │   └── styles/              # CSS
│   └── assets/                  # 이미지, 아이콘
├── openapi.json                 # API 스펙 (자동 생성)
├── package.json
├── forge.config.ts              # Electron Forge 설정
└── vite.config.mts              # Vite 설정
```

---

## 주요 기능

### 1. 세션 관리

**화면:**
```
┌────────────────────────────────────────────────────┐
│ ≡  Goose                                   ⚙ 👤   │
├────────────────────────────────────────────────────┤
│                                                    │
│  Sessions                                          │
│  ┌──────────────────────────────────────────────┐ │
│  │  📁 My Web App Project                       │ │
│  │     Last active: 2 hours ago                 │ │
│  ├──────────────────────────────────────────────┤ │
│  │  📁 Python Data Analysis                     │ │
│  │     Last active: Yesterday                   │ │
│  ├──────────────────────────────────────────────┤ │
│  │  📁 Rust CLI Tool                            │ │
│  │     Last active: 2 days ago                  │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  [+ New Session]                                   │
│                                                    │
└────────────────────────────────────────────────────┘
```

**기능:**
- 세션 목록 보기
- 새 세션 생성
- 기존 세션 재개
- 세션 검색
- 세션 삭제

### 2. 채팅 인터페이스

```
┌────────────────────────────────────────────────────┐
│ ← Sessions    My Web App Project           ⚙      │
├────────────────────────────────────────────────────┤
│                                                    │
│  You: Create a login form component               │
│  10:30 AM                                          │
│                                                    │
│  Goose: I'll create a React login form component  │
│  with validation. Here's my plan:                  │
│                                                    │
│  1. Create LoginForm.tsx                          │
│  2. Add form validation with Zod                  │
│  3. Style with Tailwind                           │
│  4. Add tests                                      │
│                                                    │
│  [✓] Creating LoginForm.tsx                       │
│  [✓] Installing dependencies                      │
│  [⟳] Writing component code...                    │
│                                                    │
├────────────────────────────────────────────────────┤
│  Type a message...                            [↑]  │
└────────────────────────────────────────────────────┘
```

**기능:**
- 실시간 스트리밍 응답
- 코드 하이라이팅
- 파일 미리보기
- 작업 진행 상태 표시
- 메시지 편집/삭제
- 메시지 복사

### 3. 파일 브라우저

```
┌────────────────────────────────────────────────────┐
│ Files                                   [Refresh]  │
├────────────────────────────────────────────────────┤
│                                                    │
│  📁 src/                                           │
│    📁 components/                                  │
│      📄 LoginForm.tsx                    (new)     │
│      📄 Button.tsx                                 │
│    📁 hooks/                                       │
│    📄 App.tsx                           (modified) │
│  📁 tests/                                         │
│  📄 package.json                        (modified) │
│  📄 README.md                                      │
│                                                    │
│  Changes: 3 files modified, 1 file added           │
│                                                    │
└────────────────────────────────────────────────────┘
```

**기능:**
- 작업 디렉토리 탐색
- 파일 변경 사항 추적
- 파일 미리보기
- 파일 열기 (기본 에디터)
- 변경사항 되돌리기

### 4. Extensions 관리

```
┌────────────────────────────────────────────────────┐
│ Extensions                                         │
├────────────────────────────────────────────────────┤
│                                                    │
│  Built-in Extensions                               │
│  ┌──────────────────────────────────────────────┐ │
│  │ [✓] Developer                                │ │
│  │     Shell commands, file operations          │ │
│  │     [Settings]                                │ │
│  ├──────────────────────────────────────────────┤ │
│  │ [✓] Computer Controller                      │ │
│  │     Browser automation, web scraping         │ │
│  │     Timeout: 300s  [Settings]                │ │
│  ├──────────────────────────────────────────────┤ │
│  │ [ ] Custom MCP Server                        │ │
│  │     Your custom extension                    │ │
│  │     [Configure]                               │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  [+ Add Extension]                                 │
│                                                    │
└────────────────────────────────────────────────────┘
```

**기능:**
- Extension 활성화/비활성화
- Extension 설정
- 커스텀 MCP 서버 추가
- Extension 상태 확인

### 5. 설정

```
┌────────────────────────────────────────────────────┐
│ Settings                                           │
├────────────────────────────────────────────────────┤
│                                                    │
│  Provider                                          │
│  ┌──────────────────────────────────────────────┐ │
│  │ Current: Anthropic (Claude Sonnet 4.5)       │ │
│  │ [Change Provider]                             │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  Behavior                                          │
│  ┌──────────────────────────────────────────────┐ │
│  │ Execution Mode: [Smart Approval ▼]           │ │
│  │ Auto-save sessions: [✓]                      │ │
│  │ Show file changes: [✓]                       │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  Appearance                                        │
│  ┌──────────────────────────────────────────────┐ │
│  │ Theme: [Dark ▼]                              │ │
│  │ Font size: [14px ▼]                          │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## Electron 아키텍처

### Main Process (main.ts)

```typescript
// src/main.ts
import { app, BrowserWindow, ipcMain } from 'electron';
import { spawn } from 'child_process';

let mainWindow: BrowserWindow | null = null;
let gooseServer: ChildProcess | null = null;

app.on('ready', async () => {
  // 1. Goose 서버 시작
  gooseServer = spawn('goosed', ['--port', '8080']);

  // 2. 메인 윈도우 생성
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
    },
  });

  // 3. React 앱 로드
  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
  } else {
    mainWindow.loadFile('dist/index.html');
  }
});

// IPC 핸들러
ipcMain.handle('get-config', async () => {
  // 설정 읽기
  return readConfig();
});

ipcMain.handle('save-config', async (event, config) => {
  // 설정 저장
  return saveConfig(config);
});

// 앱 종료 시 서버 종료
app.on('quit', () => {
  if (gooseServer) {
    gooseServer.kill();
  }
});
```

### Preload Script (preload.ts)

```typescript
// src/preload.ts
import { contextBridge, ipcRenderer } from 'electron';

// 안전한 API 노출
contextBridge.exposeInMainWorld('electron', {
  // 설정 API
  getConfig: () => ipcRenderer.invoke('get-config'),
  saveConfig: (config: any) => ipcRenderer.invoke('save-config', config),

  // 파일 API
  openFile: (path: string) => ipcRenderer.invoke('open-file', path),
  selectDirectory: () => ipcRenderer.invoke('select-directory'),

  // 시스템 API
  platform: process.platform,
  version: app.getVersion(),
});
```

### Renderer Process (React)

```typescript
// src/renderer/App.tsx
import { useState, useEffect } from 'react';
import { Chat } from './components/Chat';
import { Sidebar } from './components/Sidebar';
import { useSession } from './hooks/useSession';

export function App() {
  const { sessions, currentSession, createSession } = useSession();

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        onNewSession={createSession}
      />
      <Chat session={currentSession} />
    </div>
  );
}
```

---

## API 통신

### OpenAPI 클라이언트 생성

```bash
# openapi.json에서 TypeScript 클라이언트 생성
npm run openapi-ts

# 생성된 파일:
# src/renderer/lib/api/
```

### API 사용 예시

```typescript
// src/renderer/hooks/useSession.ts
import { api } from '../lib/api';

export function useSession() {
  const [sessions, setSessions] = useState([]);

  const fetchSessions = async () => {
    const response = await api.sessions.list();
    setSessions(response.data);
  };

  const createSession = async (name: string) => {
    const response = await api.sessions.create({ name });
    return response.data;
  };

  const sendMessage = async (sessionId: string, message: string) => {
    const response = await api.messages.send({
      sessionId,
      content: message,
    });
    return response.data;
  };

  return { sessions, createSession, sendMessage };
}
```

---

## 스트리밍 응답

### Server-Sent Events

```typescript
// src/renderer/hooks/useStreamingMessage.ts
export function useStreamingMessage(sessionId: string) {
  const [message, setMessage] = useState('');

  const sendMessage = (content: string) => {
    const eventSource = new EventSource(
      `/api/sessions/${sessionId}/messages/stream`
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMessage((prev) => prev + data.content);
    };

    eventSource.onerror = () => {
      eventSource.close();
    };

    // 메시지 전송
    fetch(`/api/sessions/${sessionId}/messages`, {
      method: 'POST',
      body: JSON.stringify({ content }),
    });
  };

  return { message, sendMessage };
}
```

---

## 빌드 및 배포

### 개발 모드

```bash
cd ui/desktop

# 의존성 설치
npm install

# 개발 서버 시작
npm run dev

# Electron 앱 실행
npm start
```

### 프로덕션 빌드

```bash
# 빌드
npm run build

# Electron 패키징
npm run make

# 생성된 파일:
# out/
#   ├── goose-desktop-darwin-x64.zip       (macOS Intel)
#   ├── goose-desktop-darwin-arm64.zip     (macOS ARM)
#   ├── goose-desktop-linux-x64.deb        (Linux)
#   └── goose-desktop-win32-x64.exe        (Windows)
```

### Electron Forge 설정

```typescript
// forge.config.ts
import { MakerDeb } from '@electron-forge/maker-deb';
import { MakerZIP } from '@electron-forge/maker-zip';

const config = {
  packagerConfig: {
    asar: true,
    icon: './assets/icon',
  },
  makers: [
    new MakerZIP({}, ['darwin']),
    new MakerDeb({
      options: {
        maintainer: 'Block',
        homepage: 'https://github.com/block/goose',
      },
    }),
  ],
};

export default config;
```

---

## UI 컴포넌트

### Shadcn UI 사용

```bash
# 컴포넌트 추가
npx shadcn-ui@latest add button
npx shadcn-ui@latest add input
npx shadcn-ui@latest add dialog
```

```typescript
// src/renderer/components/MessageInput.tsx
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

export function MessageInput({ onSend }: Props) {
  const [message, setMessage] = useState('');

  const handleSubmit = () => {
    if (message.trim()) {
      onSend(message);
      setMessage('');
    }
  };

  return (
    <div className="flex gap-2">
      <Input
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
        placeholder="Type a message..."
      />
      <Button onClick={handleSubmit}>
        Send ↑
      </Button>
    </div>
  );
}
```

---

## 다음 단계

Desktop 앱을 이해했다면, 다음 장에서는 MCP 통합을 살펴봅니다.

*다음 글에서는 Model Context Protocol과 확장 시스템을 상세히 분석합니다.*
