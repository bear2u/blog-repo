---
layout: post
title: "Mux 완벽 가이드 (08) - 개발 및 확장"
date: 2026-02-08 00:00:00 +0900
categories: [AI 코딩, 개발 도구]
tags: [Mux, 개발, Electron, React, Bun, TypeScript, 빌드, 테스트, 커스터마이징]
author: cataclysm99
original_url: "https://github.com/coder/mux"
excerpt: "Mux 개발 환경 설정, 프로젝트 구조, 빌드/테스트, 커스터마이징 및 기여 가이드"
permalink: /mux-guide-08-development/
toc: true
related_posts:
  - /blog-repo/2026-02-08-mux-guide-07-advanced-features
  - /blog-repo/2026-02-08-mux-guide-01-introduction
---

## 개발 환경 설정

### 요구사항

```yaml
필수:
  - Node.js: v20+
  - Bun: 최신 버전
  - Git: 2.30+

선택사항:
  - Electron: 38+ (자동 설치)
  - Make: GNU Make (빌드 도구)
  - Docker: 컨테이너 테스트
```

### Bun 설치

```bash
# macOS/Linux
curl -fsSL https://bun.sh/install | bash

# Windows
# https://bun.sh/docs/installation#windows

# 설치 확인
bun --version
```

### Node.js 버전 확인

```bash
node --version
# v20.x.x 이상 필요

# 업그레이드 (n 버전 매니저)
npm install -g n
sudo n 20
```

---

## 저장소 클론 및 의존성 설치

```bash
# 클론
git clone https://github.com/coder/mux.git
cd mux

# 의존성 설치
bun install

# 네이티브 모듈 리빌드 (node-pty for Electron)
make rebuild-native

# 또는
bun run rebuild-native
```

---

## 프로젝트 구조

### 디렉토리 레이아웃

```
mux/
├── src/
│   ├── browser/          # React 프론트엔드
│   │   ├── components/   # UI 컴포넌트
│   │   ├── hooks/        # React 훅
│   │   ├── styles/       # CSS/Tailwind
│   │   └── App.tsx       # 메인 앱
│   ├── main.ts           # Electron 메인 프로세스
│   ├── preload.ts        # Electron 프리로드
│   ├── common/           # 공유 코드 (타입, 유틸)
│   ├── cli/              # CLI 도구
│   └── config.ts         # 설정 관리
├── docs/                 # 문서 (Markdown)
├── tests/
│   ├── ui/               # React 컴포넌트 테스트 (Jest)
│   ├── integration/      # 통합 테스트
│   └── e2e/              # E2E 테스트 (Playwright)
├── build/                # 빌드 리소스 (아이콘 등)
├── dist/                 # 컴파일 출력
├── vite.config.ts        # Vite 설정
├── tsconfig.json         # TypeScript 설정
├── Makefile              # 빌드 명령
└── package.json          # 의존성 및 스크립트
```

### 핵심 파일

```
src/main.ts           # Electron 메인 프로세스 (IPC 서버)
src/preload.ts        # IPC 브릿지 (안전한 노출)
src/App.tsx           # React 앱 루트
src/browser/App.tsx   # 브라우저 메인 컴포넌트
src/common/knownModels.ts  # 모델 정의
src/config.ts         # 설정 로드/저장
```

---

## 빌드 및 실행

### 개발 모드

```bash
# 추천 방법 (Makefile)
make dev

# 또는 npm 스크립트
bun run dev

# 내부 동작:
# 1. Nodemon (메인 프로세스 watcher)
# 2. esbuild (CLI API watcher)
# 3. Vite (React 렌더러)
```

#### 개발 모드 특징

```
핫 리로드:
- React 컴포넌트: 즉시 반영 (Vite HMR)
- 메인 프로세스: 재시작 (nodemon)
- CLI: 재빌드 (esbuild watch)

텔레메트리:
- 기본 활성화 (프로덕션과 동일)
- 비활성화: MUX_DISABLE_TELEMETRY=1
```

### 프로덕션 빌드

```bash
# 전체 빌드
make build

# 또는
bun run build

# 출력:
dist/
├── main.js           # Electron 메인
├── preload.js        # Preload 스크립트
├── renderer/         # Vite 빌드 출력
└── cli/              # CLI 번들
```

### 앱 시작 (빌드 후)

```bash
# Electron 앱 실행
make start

# 또는
electron .

# CLI 실행
node dist/cli/index.js
```

---

## Makefile 명령어

### 주요 타겟

```bash
make help          # 모든 명령어 표시

# 개발
make dev           # 개발 서버 (핫 리로드)
make build         # 프로덕션 빌드
make start         # 빌드 후 앱 실행
make clean         # 빌드 아티팩트 삭제

# 정적 검사
make lint          # ESLint 검사
make lint-fix      # ESLint 자동 수정
make typecheck     # TypeScript 타입 검사
make fmt           # Prettier 포맷팅
make fmt-check     # 포맷 검증
make static-check  # 모든 정적 검사

# 테스트
make test          # Jest 단위 테스트
make test-watch    # Jest watch 모드
make test-coverage # 커버리지 리포트
make test-integration  # 통합 테스트
make test-e2e      # Playwright E2E

# 배포
make dist          # 플랫폼별 배포 패키지
make dist-mac      # macOS DMG
make dist-linux    # Linux AppImage
make dist-win      # Windows 설치 파일

# 문서
make docs          # Mintlify 문서 빌드
make docs-server   # 문서 로컬 서버

# 기타
make rebuild-native  # node-pty 리빌드
make storybook     # Storybook 실행
```

### Bun vs Make

```bash
# Make (권장)
make dev

# Bun (동일)
bun run dev

# Make가 실행 관리
# package.json 스크립트는 Make 프록시
```

---

## 테스트

### Jest (단위/통합 테스트)

```bash
# 모든 테스트 실행
make test

# Watch 모드
make test-watch

# 커버리지
make test-coverage

# 특정 파일
bun test src/browser/components/Chat.test.tsx
```

#### 테스트 작성 예시

```typescript
// src/browser/components/Chat.test.tsx
import { render, screen } from '@testing-library/react';
import { Chat } from './Chat';

describe('Chat Component', () => {
  it('renders message input', () => {
    render(<Chat workspaceId="test-ws" />);
    expect(screen.getByPlaceholderText(/type a message/i)).toBeInTheDocument();
  });

  it('sends message on submit', async () => {
    const onSend = jest.fn();
    render(<Chat workspaceId="test-ws" onSend={onSend} />);

    const input = screen.getByPlaceholderText(/type a message/i);
    fireEvent.change(input, { target: { value: 'Hello' } });
    fireEvent.submit(input.closest('form'));

    expect(onSend).toHaveBeenCalledWith('Hello');
  });
});
```

### Playwright (E2E 테스트)

```bash
# E2E 테스트 실행
make test-e2e

# Headless 모드 (CI)
bun run test:e2e --headed=false

# 특정 테스트
bun playwright test tests/e2e/workspace-creation.spec.ts
```

#### E2E 테스트 예시

```typescript
// tests/e2e/workspace-creation.spec.ts
import { test, expect, _electron as electron } from '@playwright/test';

test('create new workspace', async () => {
  const app = await electron.launch({ args: ['.'] });
  const window = await app.firstWindow();

  // 프로젝트 추가
  await window.click('button:has-text("Add Project")');
  await window.fill('input[name="path"]', '/path/to/project');
  await window.click('button:has-text("Add")');

  // 워크스페이스 생성
  await window.click('button:has-text("New Workspace")');
  await window.fill('input[name="name"]', 'test-workspace');
  await window.click('button:has-text("Create")');

  // 검증
  await expect(window.locator('text=test-workspace')).toBeVisible();

  await app.close();
});
```

### Storybook (컴포넌트 개발)

```bash
# Storybook 실행
make storybook

# 빌드
make storybook-build

# 테스트
make test-storybook
```

---

## TypeScript 설정

### tsconfig.json

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "lib": ["ES2020", "DOM"],
    "jsx": "react-jsx",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "resolveJsonModule": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### 경로 별칭

```typescript
// 절대 경로 임포트
import { Button } from '@/browser/components/Button';
import { useWorkspace } from '@/browser/hooks/useWorkspace';

// 상대 경로 (지양)
// import { Button } from '../../components/Button';
```

---

## 커스터마이징

### 테마 (CSS 변수)

```css
/* src/browser/styles/globals.css */
:root {
  --color-plan-mode: #6b5bff;
  --color-exec-mode: #00d4aa;
  --color-ask-mode: #ff6b6b;

  --color-background: #1e1e1e;
  --color-foreground: #d4d4d4;
  --color-primary: #6b5bff;
  --color-secondary: #00d4aa;
}
```

### 키바인딩

```typescript
// src/browser/hooks/useKeyboard.ts
export function useKeyboard() {
  useEffect(() => {
    const handleKeydown = (e: KeyboardEvent) => {
      // ⌘+Shift+P: Command Palette
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'P') {
        e.preventDefault();
        openCommandPalette();
      }

      // ⌘+Shift+M: Change Agent
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'M') {
        e.preventDefault();
        cycleAgent();
      }

      // ⌘+/: Change Model
      if ((e.metaKey || e.ctrlKey) && e.key === '/') {
        e.preventDefault();
        cycleModel();
      }
    };

    window.addEventListener('keydown', handleKeydown);
    return () => window.removeEventListener('keydown', handleKeydown);
  }, []);
}
```

### 커스텀 모델 추가

```typescript
// src/common/knownModels.ts
export const KNOWN_MODELS = {
  // 기존 모델...

  // 커스텀 모델 추가
  'custom-model': {
    id: 'custom-provider:custom-model-id',
    name: 'My Custom Model',
    provider: 'custom-provider',
    aliases: ['custom'],
    contextWindow: 128000,
    maxOutputTokens: 4096,
  },
};
```

### 커스텀 에이전트 (빌트인)

```typescript
// src/common/agents/builtin/custom.md
export const CUSTOM_AGENT = `---
name: Custom Agent
description: My custom agent
base: exec
ui:
  color: "#ff00ff"
tools:
  add:
    - file_read
    - bash
  remove:
    - file_edit_.*
---

You are a custom agent with specific instructions.
`;
```

---

## 기여 가이드

### 개발 워크플로우

```bash
# 1. Fork 및 클론
git clone https://github.com/YOUR_USERNAME/mux.git
cd mux

# 2. 의존성 설치
bun install
make rebuild-native

# 3. 브랜치 생성
git checkout -b feature/my-feature

# 4. 개발
make dev

# 5. 테스트
make test
make test-e2e
make static-check

# 6. 커밋
git add .
git commit -m "Add my feature"

# 7. 푸시
git push origin feature/my-feature

# 8. PR 생성
# GitHub에서 Pull Request 열기
```

### 코드 스타일

```typescript
// 금지: as any
const foo = bar as any; // ❌

// 권장: 타입 가드
function isString(value: unknown): value is string {
  return typeof value === 'string';
}
const foo = isString(bar) ? bar : ''; // ✓

// 금지: void 비동기 호출
void asyncFn(); // ❌

// 권장: await 또는 명시적 처리
await asyncFn(); // ✓
asyncFn().catch(handleError); // ✓

// 금지: 하드코딩된 색상
<div style={{ color: '#6b5bff' }} /> // ❌

// 권장: CSS 변수
<div style={{ color: 'var(--color-primary)' }} /> // ✓
```

### 커밋 메시지

```
feat: Add OAuth2 authentication
fix: Resolve workspace creation race condition
docs: Update installation guide
refactor: Simplify agent loop logic
test: Add tests for compaction feature
chore: Update dependencies
```

### PR 체크리스트

```
- [ ] 모든 테스트 통과 (make test, make test-e2e)
- [ ] 정적 검사 통과 (make static-check)
- [ ] 타입 에러 없음 (make typecheck)
- [ ] 문서 업데이트 (필요 시)
- [ ] AGENTS.md 읽음 (기여 규칙)
- [ ] Codex 리뷰 해결 (있다면)
```

---

## 디버깅

### Electron 개발자 도구

```bash
# 개발 모드에서 자동 활성화
make dev

# 윈도우에서 ⌘+Option+I (macOS) / Ctrl+Shift+I (Windows/Linux)
```

### 로그 확인

```bash
# macOS
tail -f ~/Library/Logs/Mux/main.log

# Linux
tail -f ~/.config/Mux/logs/main.log

# Windows
Get-Content "$env:APPDATA\Mux\logs\main.log" -Wait
```

### IPC 디버깅

```typescript
// src/main.ts
import { ipcMain } from 'electron';

ipcMain.handle('my-channel', async (event, ...args) => {
  console.log('IPC called:', args);
  // ...
});
```

### LLM 요청 디버깅

```bash
# 전체 LLM 요청 로그
MUX_DEBUG_LLM_REQUEST=1 make dev

# 또는 CLI
MUX_DEBUG_LLM_REQUEST=1 node dist/cli/index.js run "Hello"
```

---

## 배포

### 로컬 빌드

```bash
# 모든 플랫폼 (현재 OS)
make dist

# macOS만
make dist-mac

# Linux만
make dist-linux

# Windows만
make dist-win
```

### CI/CD (GitHub Actions)

```yaml
# .github/workflows/build.yml
name: Build

on:
  push:
    branches: [main]
  pull_request:

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [macos-latest, ubuntu-latest, windows-latest]

    steps:
      - uses: actions/checkout@v3
      - uses: oven-sh/setup-bun@v1

      - run: bun install
      - run: make rebuild-native
      - run: make build
      - run: make test
      - run: make static-check
      - run: make dist
```

### 서명 및 공증 (macOS)

```bash
# Apple Developer 계정 필요
# build/entitlements.mac.plist 설정

# 환경 변수
export APPLE_ID="your-apple-id@example.com"
export APPLE_ID_PASSWORD="app-specific-password"
export APPLE_TEAM_ID="TEAM_ID"

# 빌드 (자동 서명/공증)
make dist-mac
```

---

## 문서 기여

### Mintlify 문서 빌드

```bash
# 로컬 서버
make docs-server

# 브라우저: http://localhost:3000

# 문서 수정
vim docs/getting-started/installation.mdx
```

### 문서 구조

```
docs/
├── index.mdx              # 홈페이지
├── install.mdx            # 설치 가이드
├── getting-started/       # 시작하기
├── workspaces/            # 워크스페이스
├── agents/                # 에이전트
├── config/                # 설정
├── integrations/          # 통합
├── hooks/                 # 훅
├── reference/             # 참조
└── docs.json              # 네비게이션 설정
```

### 문서 링크 검증

```bash
# 깨진 링크 확인
make check-docs-links
```

---

## 성능 최적화

### React Compiler

```
Mux는 React Compiler 활성화

금지:
- 수동 React.memo()
- 수동 useMemo() (메모이제이션 목적)
- 수동 useCallback() (메모이제이션 목적)

대신:
- 불안정한 객체 참조 수정
  - 예: new Set() in state setter
  - 예: inline object literals as props
```

### 번들 크기 최적화

```bash
# 번들 크기 분석
bun run build
du -h dist/

# Eager imports 확인
make check-eager-imports

# 번들 크기 확인
make check-bundle-size
```

### 시작 성능

```bash
# 시작 시간 측정
make check-startup

# 느린 초기화 찾기
# src/main.ts에서 try-catch로 감싸기
```

---

## 고급 개발 팁

### IPC 타입 안전성

```typescript
// src/common/ipc/types.ts
export interface IpcChannels {
  'get-workspace': (id: string) => Workspace;
  'create-workspace': (data: CreateWorkspaceData) => Workspace;
  'delete-workspace': (id: string) => void;
}

// src/main.ts
import type { IpcChannels } from './common/ipc/types';

for (const [channel, handler] of Object.entries(ipcHandlers)) {
  ipcMain.handle(channel as keyof IpcChannels, handler);
}

// src/browser/hooks/useIpc.ts
export function useIpc() {
  const invoke = <K extends keyof IpcChannels>(
    channel: K,
    ...args: Parameters<IpcChannels[K]>
  ): Promise<ReturnType<IpcChannels[K]>> => {
    return window.ipc.invoke(channel, ...args);
  };

  return { invoke };
}
```

### State Management

```typescript
// Colocate subscriptions with consumers
// 안티패턴: 부모가 구독 후 props로 전달
function Parent() {
  const stats = useWorkspaceStats(workspaceId);
  return <Child stats={stats} />; // ❌ 불필요한 리렌더
}

// 권장: 자식이 직접 구독
function Child({ workspaceId }: Props) {
  const stats = useWorkspaceStats(workspaceId); // ✓
  return <div>{stats.tokenCount}</div>;
}
```

---

## 다음 단계

Mux 개발을 마스터했다면:

1. **실전 기여** - GitHub Issues에서 Good First Issue 찾기
2. **커스텀 에이전트 개발** - 팀/프로젝트 특화 에이전트
3. **플러그인 시스템** - MCP 서버 통합 실험

---

## 참고 자료

- [AGENTS.md](https://github.com/coder/mux/blob/main/AGENTS.md) - 개발 가이드라인
- [기여 가이드](https://github.com/coder/mux/blob/main/CONTRIBUTING.md)
- [GitHub Issues](https://github.com/coder/mux/issues)
- [Discord 커뮤니티](https://discord.gg/thkEdtwm8c)
- [Electron 문서](https://www.electronjs.org/docs)
- [React 문서](https://react.dev/)
- [Bun 문서](https://bun.sh/docs)
