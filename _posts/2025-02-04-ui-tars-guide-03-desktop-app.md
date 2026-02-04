---
layout: post
title: "UI-TARS 완벽 가이드 (3) - Desktop 앱 분석"
date: 2025-02-04
permalink: /ui-tars-guide-03-desktop-app/
author: ByteDance
category: AI
tags: [UI-TARS, Electron, Desktop App, IPC, React]
series: ui-tars-guide
part: 3
original_url: "https://github.com/bytedance/UI-TARS-desktop"
excerpt: "UI-TARS Desktop Electron 앱의 구조를 분석합니다. 메인/렌더러 프로세스, IPC 통신, 상태 관리를 살펴봅니다."
---

## UI-TARS Desktop 앱 개요

UI-TARS Desktop은 **Electron 기반의 AI 자동화 애플리케이션**으로, Vision-Language Model을 활용하여 자연어로 컴퓨터를 제어합니다.

```
apps/ui-tars/
├── src/
│   ├── main/          # Electron 메인 프로세스
│   ├── preload/       # 프리로드 스크립트
│   └── renderer/      # React UI (렌더러 프로세스)
├── resources/         # 아이콘, 언어 파일
├── static/            # 정적 파일
├── build/             # 빌드 설정
└── e2e/               # E2E 테스트
```

---

## 메인/렌더러 프로세스 분리

### 메인 프로세스 (`src/main/main.ts`)

```typescript
// 주요 책임
├── Electron 앱 생명주기 관리
├── 윈도우 생성 및 관리
├── IPC 핸들러 등록
├── 시스템 권한 관리
└── GUI 에이전트 실행

// 초기화 흐름
async function createWindow() {
  // 1. 메인 윈도우 생성 (1200x700)
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 700,
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  // 2. 트레이 아이콘 생성
  createTray();

  // 3. 데스크탑 캡처 설정
  desktopCapturer.getSources({ types: ['screen'] });

  // 4. 접근성 지원 활성화
  app.commandLine.appendSwitch('force-renderer-accessibility');
}
```

### 렌더러 프로세스 (`src/renderer/`)

```
renderer/src/
├── pages/              # 라우트 페이지
│   ├── home/           # 홈 화면
│   ├── local/          # 로컬 Operator
│   ├── remote/         # 원격 Operator
│   ├── widget/         # 위젯
│   └── settings/       # 설정
├── components/         # UI 컴포넌트
├── hooks/              # React 훅
├── store/              # Zustand 상태
├── db/                 # IndexedDB
└── layouts/            # 레이아웃
```

### 프리로드 스크립트 (`src/preload/index.ts`)

메인과 렌더러 프로세스 간의 **안전한 브릿지**를 제공합니다.

```typescript
import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electron', {
  ipcRenderer: {
    invoke: (channel: string, ...args: any[]) =>
      ipcRenderer.invoke(channel, ...args),

    sendMessage: (channel: string, ...args: any[]) =>
      ipcRenderer.send(channel, ...args),

    on: (channel: string, func: (...args: any[]) => void) => {
      ipcRenderer.on(channel, (_event, ...args) => func(...args));
    }
  },

  setting: {
    getSetting: () => ipcRenderer.invoke('setting:get'),
    updateSetting: (data: any) => ipcRenderer.invoke('setting:update', data)
  },

  utio: {
    shareReport: (params: any) => ipcRenderer.invoke('utio:share', params)
  }
});

// 플랫폼 정보 노출
contextBridge.exposeInMainWorld('platform', process.platform);
```

---

## IPC 라우팅 시스템

### IPC 라우트 구조

```
src/main/ipcRoutes/
├── agent.ts           # GUI 에이전트 제어
├── screen.ts          # 스크린샷, 화면 정보
├── window.ts          # 윈도우 관리
├── permission.ts      # 시스템 권한
├── browser.ts         # 브라우저 제어
├── setting.ts         # 설정 조회/수정
└── remoteResource.ts  # 원격 리소스
```

### 에이전트 IPC 핸들러 (`agent.ts`)

```typescript
ipcMain.handle('agent:run', async (event, instruction: string) => {
  const settings = store.get('settings');

  // 1. Operator 선택
  const operator = getOperator(settings.operator);

  // 2. VLM 모델 설정
  const model = getModel(settings.vlmProvider, settings.modelName);

  // 3. GUI Agent 실행
  const agent = new GUIAgent({
    operator,
    model,
    maxLoopCount: settings.maxLoopCount,
    loopIntervalInMs: settings.loopInterval
  });

  // 4. 이벤트 전달
  agent.on('status', (status) => {
    event.sender.send('agent:status', status);
  });

  await agent.run(instruction);
});

ipcMain.handle('agent:stop', async () => {
  abortController.abort();
});
```

---

## 상태 관리

### 메인 프로세스 상태 (`AppState`)

```typescript
interface AppState {
  theme: 'dark' | 'light';

  ensurePermissions: {
    screenCapture?: boolean;
    accessibility?: boolean;
  };

  instructions: string | null;
  restUserData: GUIAgentData | null;
  status: GUIAgentData['status'];
  errorMsg: string | null;

  sessionHistoryMessages: Message[];
  messages: ConversationWithSoM[];

  abortController: AbortController;
  thinking: boolean;
  browserAvailable: boolean;
}
```

### 렌더러 상태 (Zustand)

```typescript
// src/renderer/src/store/index.ts
import { create } from 'zustand';

interface UIState {
  currentSession: string | null;
  messages: Message[];
  isRunning: boolean;

  setCurrentSession: (id: string) => void;
  addMessage: (msg: Message) => void;
  setRunning: (running: boolean) => void;
}

export const useStore = create<UIState>((set) => ({
  currentSession: null,
  messages: [],
  isRunning: false,

  setCurrentSession: (id) => set({ currentSession: id }),
  addMessage: (msg) => set((state) => ({
    messages: [...state.messages, msg]
  })),
  setRunning: (running) => set({ isRunning: running })
}));
```

### Zustand Bridge (메인-렌더러 동기화)

```typescript
// 메인 프로세스에서 상태 변경
appState.status = 'running';

// Zustand Bridge가 자동으로 렌더러에 동기화
window.zustandBridge.subscribe((state) => {
  // 상태 업데이트
});
```

---

## GUI 에이전트 실행 시스템

### `runAgent.ts` 핵심 로직

```typescript
export async function runAgent(
  instruction: string,
  settings: Settings
): Promise<void> {
  // 1. Operator 초기화
  const operator = await createOperator(settings.operator);

  // 2. VLM 클라이언트 생성
  const llmClient = createLLMClient({
    provider: settings.vlmProvider,
    apiKey: settings.apiKey,
    baseURL: settings.apiBaseUrl,
    model: settings.modelName
  });

  // 3. GUIAgent 생성
  const agent = new GUIAgent({
    operator,
    model: llmClient,
    systemPrompt: getSystemPrompt(settings.language),
    maxLoopCount: settings.maxLoopCount,
    loopIntervalInMs: settings.loopInterval
  });

  // 4. 실행
  try {
    await agent.run(instruction, {
      signal: abortController.signal
    });
  } catch (error) {
    if (error.name === 'AbortError') {
      console.log('Agent stopped by user');
    } else {
      throw error;
    }
  } finally {
    await operator.cleanup();
  }
}
```

### Operator 생성

```typescript
function createOperator(type: OperatorType): Operator {
  switch (type) {
    case 'local-computer':
      return new NutJSOperator();

    case 'local-browser':
      return new BrowserOperator({
        headless: false,
        searchEngine: settings.searchEngine
      });

    case 'remote-computer':
      return new RemoteComputerOperator({
        proxyClient: createProxyClient()
      });

    case 'remote-browser':
      return new RemoteBrowserOperator({
        cdpEndpoint: settings.cdpEndpoint
      });

    default:
      throw new Error(`Unknown operator type: ${type}`);
  }
}
```

---

## 설정 관리 시스템

### 설정 스키마

```typescript
interface Settings {
  // VLM 설정
  vlmProvider: 'huggingface' | 'volcengine' | 'openai';
  apiBaseUrl: string;
  apiKey: string;
  modelName: string;

  // 작업 설정
  operator: 'local-computer' | 'local-browser' | 'remote-computer' | 'remote-browser';
  language: 'en' | 'zh';
  screenshotResize: boolean;
  maxLoopCount: number;       // 25~200
  loopInterval: number;       // 0~3000ms

  // 원격 설정
  reportStorageUrl: string;
  utioBaseUrl: string;
  presetSource: 'local' | 'remote';

  // 브라우저 설정
  searchEngine: 'google' | 'baidu' | 'bing';
}
```

### 설정 저장소 (`electron-store`)

```typescript
import Store from 'electron-store';

const store = new Store<{
  settings: Settings;
  sessions: Session[];
}>({
  defaults: {
    settings: defaultSettings,
    sessions: []
  }
});

// 설정 조회
export function getSetting(): Settings {
  return store.get('settings');
}

// 설정 업데이트
export function updateSetting(data: Partial<Settings>): void {
  const current = store.get('settings');
  store.set('settings', { ...current, ...data });
}
```

---

## 윈도우 관리

### 메인 윈도우 설정

```typescript
const mainWindow = new BrowserWindow({
  width: 1200,
  height: 700,
  minWidth: 800,
  minHeight: 600,

  titleBarStyle: 'hiddenInset',  // macOS 스타일
  trafficLightPosition: { x: 16, y: 16 },

  webPreferences: {
    preload: path.join(__dirname, '../preload/index.js'),
    contextIsolation: true,
    nodeIntegration: false,
    sandbox: true
  }
});

// 콘텐츠 보호 (스크린샷 방지)
mainWindow.setContentProtection(true);

// 마우스 이벤트 제어
mainWindow.setIgnoreMouseEvents(false);
```

### 트레이 아이콘

```typescript
function createTray() {
  const tray = new Tray(path.join(__dirname, 'icon.png'));

  const contextMenu = Menu.buildFromTemplate([
    { label: 'Show', click: () => mainWindow.show() },
    { label: 'Hide', click: () => mainWindow.hide() },
    { type: 'separator' },
    { label: 'Quit', click: () => app.quit() }
  ]);

  tray.setContextMenu(contextMenu);
  tray.on('click', () => mainWindow.show());
}
```

---

## 권한 관리

### macOS 권한 요청

```typescript
import { systemPreferences } from 'electron';

async function ensurePermissions() {
  // 스크린 캡처 권한
  const screenCaptureAccess = systemPreferences
    .getMediaAccessStatus('screen');

  if (screenCaptureAccess !== 'granted') {
    await systemPreferences.askForMediaAccess('screen');
  }

  // 접근성 권한 (NutJS에 필요)
  const accessibilityAccess = systemPreferences
    .isTrustedAccessibilityClient(false);

  if (!accessibilityAccess) {
    // 시스템 환경설정으로 안내
    shell.openExternal(
      'x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility'
    );
  }
}
```

---

## 빌드 및 패키징

### Electron Forge 설정 (`forge.config.ts`)

```typescript
const config: ForgeConfig = {
  packagerConfig: {
    asar: {
      unpack: '**/*.node'  // 네이티브 모듈 제외
    },
    icon: './resources/icon',
    appBundleId: 'com.bytedance.ui-tars'
  },

  makers: [
    {
      name: '@electron-forge/maker-dmg',
      config: { format: 'ULFO' }
    },
    {
      name: '@electron-forge/maker-squirrel',
      config: {}
    },
    {
      name: '@electron-forge/maker-zip',
      platforms: ['darwin', 'linux']
    }
  ],

  plugins: [
    {
      name: '@electron-forge/plugin-vite',
      config: {
        build: [
          { entry: 'src/main/main.ts', config: 'vite.main.config.ts' },
          { entry: 'src/preload/index.ts', config: 'vite.preload.config.ts' }
        ],
        renderer: [
          { name: 'main_window', config: 'vite.renderer.config.ts' }
        ]
      }
    }
  ]
};
```

### 보안 설정 (Electron Fuses)

```typescript
// 보안 퓨즈 활성화
FuseV1Options.RunAsNode: false,           // Node.js 실행 비활성화
FuseV1Options.EnableCookieEncryption: true,  // 쿠키 암호화
FuseV1Options.EnableNodeOptionsEnvironmentVariable: false,
FuseV1Options.EnableNodeCliInspectArguments: false
```

---

*다음 글에서는 Agent TARS Core를 분석합니다.*
