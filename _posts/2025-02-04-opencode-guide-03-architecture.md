---
layout: post
title: "OpenCode 가이드 - 아키텍처"
date: 2025-02-04
category: AI
tags: [opencode, architecture, monorepo, client-server, bun]
series: opencode-guide
part: 3
author: anomalyco
original_url: https://github.com/anomalyco/opencode
---

## 아키텍처 개요

OpenCode는 **클라이언트-서버 아키텍처**를 채택한 **모노레포** 프로젝트입니다. Bun을 패키지 매니저 및 런타임으로 사용하며, Turbo로 빌드를 관리합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenCode Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│  │  TUI App  │  │  Desktop  │  │  Mobile   │  ← Clients    │
│  │ (Terminal)│  │  (Tauri)  │  │  (Future) │               │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘               │
│        │              │              │                      │
│        └──────────────┼──────────────┘                      │
│                       │                                     │
│                       ▼                                     │
│        ┌──────────────────────────────┐                    │
│        │      OpenCode Server         │                    │
│        │  (WebSocket / HTTP API)      │                    │
│        └──────────────┬───────────────┘                    │
│                       │                                     │
│        ┌──────────────┼──────────────┐                     │
│        ▼              ▼              ▼                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │  Agent   │  │  Tools   │  │ Provider │                  │
│  │  System  │  │  (Edit,  │  │  (AI     │                  │
│  │          │  │  Bash..) │  │  APIs)   │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 모노레포 구조

프로젝트는 `packages/` 디렉토리 아래 여러 패키지로 구성됩니다:

```
opencode/
├── packages/
│   ├── opencode/     # 핵심 라이브러리 (에이전트, 도구, 프로바이더)
│   ├── app/          # TUI 앱 (SolidJS 기반)
│   ├── desktop/      # 데스크톱 앱 (Tauri v2)
│   ├── console/      # 웹 콘솔
│   ├── docs/         # 문서
│   ├── sdk/          # JavaScript SDK
│   ├── plugin/       # 플러그인 시스템
│   ├── ui/           # UI 컴포넌트
│   ├── util/         # 유틸리티
│   └── ...
├── package.json      # 루트 워크스페이스 설정
├── turbo.json        # Turbo 빌드 설정
└── AGENTS.md         # 에이전트 설정 가이드
```

## 핵심 패키지 분석

### packages/opencode (핵심 라이브러리)

```
packages/opencode/src/
├── agent/           # 에이전트 시스템
│   ├── agent.ts     # 에이전트 정의
│   └── prompt/      # 프롬프트 템플릿
├── provider/        # AI 프로바이더
│   ├── provider.ts  # 프로바이더 관리
│   ├── transform.ts # 메시지 변환
│   └── sdk/         # 커스텀 SDK (Copilot 등)
├── tool/            # 내장 도구
│   ├── edit.ts      # 파일 편집
│   ├── bash.ts      # 쉘 명령
│   ├── read.ts      # 파일 읽기
│   ├── grep.ts      # 텍스트 검색
│   └── ...
├── config/          # 설정 관리
├── lsp/             # LSP 클라이언트
├── mcp/             # MCP 클라이언트
├── skill/           # 스킬 시스템
├── permission/      # 권한 시스템
├── session/         # 세션 관리
└── cli/             # CLI 명령어
```

### packages/app (TUI 앱)

SolidJS로 작성된 터미널 UI 앱:

```
packages/app/src/
├── app.tsx          # 앱 엔트리
├── components/      # UI 컴포넌트
├── pages/           # 페이지
├── hooks/           # 리액트 훅
├── context/         # 컨텍스트
└── i18n/            # 국제화
```

### packages/desktop (데스크톱 앱)

Tauri v2 기반 네이티브 앱:

```
packages/desktop/
├── src/             # 프론트엔드 (Vite)
├── src-tauri/       # Rust 백엔드
│   ├── Cargo.toml
│   └── src/
└── tauri.conf.json  # Tauri 설정
```

## 클라이언트-서버 모델

### 서버 (Backend)

OpenCode 서버는 여러 가지 역할을 수행합니다:

```typescript
// 서버 주요 책임
const server = {
  // 세션 관리
  sessions: new Map<string, Session>(),

  // AI 프로바이더 통신
  providers: await Provider.list(),

  // 도구 실행
  tools: await Tool.registry(),

  // MCP 서버 관리
  mcp: await MCP.clients(),

  // LSP 서버 관리
  lsp: await LSP.status()
}
```

### 클라이언트 통신

클라이언트는 WebSocket 또는 HTTP API로 서버와 통신합니다:

```typescript
// 이벤트 기반 통신
const events = {
  // 메시지 스트리밍
  "message.delta": (delta: MessageDelta) => void,

  // 도구 실행
  "tool.start": (tool: ToolCall) => void,
  "tool.end": (result: ToolResult) => void,

  // 상태 변경
  "session.state": (state: SessionState) => void
}
```

## 빌드 시스템

### Bun + Turbo

```json
// package.json
{
  "packageManager": "bun@1.3.5",
  "workspaces": {
    "packages": [
      "packages/*",
      "packages/console/*",
      "packages/sdk/js"
    ]
  }
}
```

### 빌드 명령

```bash
# 개발 모드
bun run dev

# 타입 체크
bun run typecheck

# 테스트
bun test
```

### 카탈로그 의존성

워크스페이스 패키지는 공통 의존성을 카탈로그로 관리합니다:

```json
{
  "workspaces": {
    "catalog": {
      "typescript": "5.8.2",
      "zod": "4.1.8",
      "solid-js": "1.9.10",
      "hono": "4.10.7"
    }
  }
}
```

## 코딩 스타일 가이드

프로젝트는 AGENTS.md에 정의된 스타일 가이드를 따릅니다:

### 일반 원칙

```typescript
// ✅ 좋은 예: 단일 함수에 유지
function process(data: Data) {
  const result = transform(data)
  return validate(result)
}

// ❌ 나쁜 예: 불필요한 분리
function prepareProcess(data: Data) { /* ... */ }
function executeProcess(data: Data) { /* ... */ }
function finalizeProcess(data: Data) { /* ... */ }
```

### 네이밍 규칙

```typescript
// ✅ 좋은 예: 짧은 단어
const foo = 1
function journal(dir: string) {}

// ❌ 나쁜 예: 여러 단어 조합
const fooBarBaz = 1
function prepareJournal(dir: string) {}
```

### 변수 인라인

```typescript
// ✅ 좋은 예: 인라인
const journal = await Bun.file(path.join(dir, "journal.json")).json()

// ❌ 나쁜 예: 불필요한 변수
const journalPath = path.join(dir, "journal.json")
const journal = await Bun.file(journalPath).json()
```

### 제어 흐름

```typescript
// ✅ 좋은 예: 조기 반환
function foo() {
  if (condition) return 1
  return 2
}

// ❌ 나쁜 예: else 사용
function foo() {
  if (condition) return 1
  else return 2
}
```

## 데이터 흐름

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│   User     │ ──▶ │   Agent    │ ──▶ │  Provider  │
│   Input    │     │   System   │     │   (AI)     │
└────────────┘     └─────┬──────┘     └─────┬──────┘
                         │                   │
                         ▼                   ▼
                  ┌────────────┐     ┌────────────┐
                  │   Tool     │ ◀── │  Response  │
                  │   Calls    │     │  (Stream)  │
                  └─────┬──────┘     └────────────┘
                         │
                         ▼
                  ┌────────────┐
                  │   File     │
                  │   System   │
                  └────────────┘
```

## 확장 지점

### 플러그인

```typescript
// packages/plugin 인터페이스
interface Plugin {
  name: string
  version: string

  // 훅
  onInit?(): Promise<void>
  onMessage?(msg: Message): Promise<void>
  onTool?(call: ToolCall): Promise<ToolResult>
}
```

### 커스텀 에이전트

```json
// .opencode/agents/custom.json
{
  "name": "custom",
  "description": "커스텀 에이전트",
  "mode": "primary",
  "prompt": "특별한 지시사항..."
}
```

### MCP 서버

```json
// opencode.json
{
  "mcp": {
    "my-server": {
      "type": "local",
      "command": ["node", "my-mcp-server.js"]
    }
  }
}
```

## 다음 단계

다음 챕터에서는 에이전트 시스템을 자세히 살펴봅니다.

---

**이전 글**: [설치 가이드](/2025/02/04/opencode-guide-02-installation)

**다음 글**: [에이전트 시스템](/2025/02/04/opencode-guide-04-agents)
