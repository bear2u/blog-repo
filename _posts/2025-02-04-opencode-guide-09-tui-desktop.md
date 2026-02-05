---
layout: post
title: "OpenCode 가이드 - TUI & 데스크톱 앱"
date: 2025-02-04
categories: [AI 코딩 에이전트, OpenCode]
tags: [opencode, tui, terminal, desktop, tauri, solidjs]
author: anomalyco
original_url: https://github.com/anomalyco/opencode
---

## TUI (Terminal User Interface)

OpenCode의 핵심 인터페이스는 터미널 기반 TUI입니다. Neovim 사용자들이 만든 프로젝트답게, 터미널에서 최적의 경험을 제공합니다.

## TUI 실행

```bash
# 기본 실행
opencode

# 특정 디렉토리에서 실행
opencode --cwd /path/to/project

# 초기 프롬프트와 함께 실행
opencode "이 프로젝트의 구조를 설명해줘"
```

## TUI 인터페이스

```
┌─────────────────────────────────────────────────────────┐
│ OpenCode                             build │ claude-4  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  User: 이 프로젝트의 구조를 설명해줘                    │
│                                                         │
│  Assistant: 이 프로젝트는 다음과 같은 구조입니다:       │
│                                                         │
│  ┌─ src/                                                │
│  │  ├── components/                                     │
│  │  ├── hooks/                                          │
│  │  └── utils/                                          │
│  └─ package.json                                        │
│                                                         │
│  [Tool: glob **/*.tsx] ✓                                │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ > 메시지를 입력하세요...                          [Tab] │
└─────────────────────────────────────────────────────────┘
```

## 키보드 단축키

### 기본 조작

| 키 | 동작 |
|----|------|
| `Enter` | 메시지 전송 |
| `Shift+Enter` | 줄바꿈 |
| `Tab` | 에이전트 전환 |
| `Ctrl+C` | 현재 작업 취소 / 종료 |
| `Ctrl+L` | 화면 클리어 |
| `Esc` | 취소 / 닫기 |

### 네비게이션

| 키 | 동작 |
|----|------|
| `↑` / `↓` | 이전/다음 메시지 |
| `Page Up` / `Page Down` | 페이지 스크롤 |
| `Home` / `End` | 처음/끝으로 이동 |

### 도구 권한

| 키 | 동작 |
|----|------|
| `y` | 허용 |
| `n` | 거부 |
| `a` | 항상 허용 |

## TUI 구성 요소

### 헤더

```
┌──────────────────────────────────────────────────┐
│ OpenCode                    build │ claude-4-opus │
└──────────────────────────────────────────────────┘
      │                         │          │
      │                         │          └── 현재 모델
      │                         └── 현재 에이전트
      └── 프로젝트 이름
```

### 메시지 영역

대화 내역이 표시됩니다:
- **User**: 사용자 입력
- **Assistant**: AI 응답
- **Tool**: 도구 실행 결과

### 도구 실행 표시

```
[Tool: edit src/utils.ts]
 ├─ old_string: "function foo("
 ├─ new_string: "function bar("
 └─ ✓ Success
```

### 입력 영역

```
┌────────────────────────────────────────────────────┐
│ > 메시지를 입력하세요...                    [Tab]  │
└────────────────────────────────────────────────────┘
```

## 에이전트 선택

Tab 키를 누르면 에이전트 선택 메뉴가 표시됩니다:

```
┌─────────────────────────────┐
│ Select Agent                │
├─────────────────────────────┤
│ > build  (current)          │
│   plan                      │
│   custom-agent              │
└─────────────────────────────┘
```

## 권한 요청 다이얼로그

```
┌─────────────────────────────────────────────────────┐
│ Permission Required                                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Tool: bash                                          │
│ Command: npm install lodash                         │
│                                                     │
│ ┌─────┐  ┌─────┐  ┌──────────────┐                │
│ │  y  │  │  n  │  │  a (always)  │                │
│ └─────┘  └─────┘  └──────────────┘                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 토스트 알림

중요 알림이 토스트로 표시됩니다:

```
┌─────────────────────────────────────┐
│ ⚠️ MCP Authentication Required      │
│                                     │
│ Server "github" requires auth.      │
│ Run: opencode mcp auth github       │
└─────────────────────────────────────┘
```

## 데스크톱 앱

### 개요

OpenCode 데스크톱 앱은 **Tauri v2** 기반의 네이티브 애플리케이션입니다. 터미널 환경이 아닌 GUI 환경을 선호하는 사용자를 위해 제공됩니다.

### 기술 스택

```
┌─────────────────────────────────────────┐
│          Desktop App                    │
├─────────────────────────────────────────┤
│  Frontend: SolidJS + Vite               │
│  Backend:  Rust (Tauri)                 │
│  UI:       TailwindCSS                  │
│  Build:    Tauri Bundler                │
└─────────────────────────────────────────┘
```

### 설치

```bash
# macOS (Homebrew Cask)
brew install --cask opencode-desktop

# Windows (Scoop)
scoop bucket add extras
scoop install extras/opencode-desktop
```

또는 [GitHub Releases](https://github.com/anomalyco/opencode/releases)에서 직접 다운로드:

| 플랫폼 | 파일 |
|--------|------|
| macOS (Apple Silicon) | `opencode-desktop-darwin-aarch64.dmg` |
| macOS (Intel) | `opencode-desktop-darwin-x64.dmg` |
| Windows | `opencode-desktop-windows-x64.exe` |
| Linux | `.deb`, `.rpm`, AppImage |

### 개발 환경 설정

데스크톱 앱을 개발하려면:

```bash
# 저장소 클론
git clone https://github.com/anomalyco/opencode
cd opencode

# 의존성 설치
bun install

# 개발 모드 실행
bun run --cwd packages/desktop tauri dev
```

### 빌드

```bash
# 프로덕션 빌드
bun run --cwd packages/desktop tauri build
```

### 데스크톱 앱 아키텍처

```
packages/desktop/
├── src/                    # 프론트엔드 (SolidJS)
│   ├── App.tsx
│   ├── components/
│   └── ...
├── src-tauri/              # 백엔드 (Rust)
│   ├── Cargo.toml
│   ├── src/
│   │   └── main.rs
│   └── tauri.conf.json
├── package.json
└── vite.config.ts
```

### Tauri 설정

```json
// tauri.conf.json
{
  "productName": "OpenCode Desktop",
  "mainBinaryName": "opencode-desktop",
  "version": "0.1.0",
  "identifier": "ai.opencode.desktop",
  "build": {
    "devPath": "http://localhost:1420",
    "distDir": "../dist"
  },
  "app": {
    "windows": [{
      "title": "OpenCode",
      "width": 1200,
      "height": 800
    }]
  }
}
```

## 웹 콘솔

OpenCode는 웹 기반 콘솔도 제공합니다.

### 패키지 위치

```
packages/console/
└── app/
    ├── src/
    │   ├── components/
    │   ├── pages/
    │   └── ...
    └── package.json
```

### 기능

- 원격 OpenCode 서버 접속
- 브라우저에서 AI 에이전트 사용
- 세션 관리
- 프로젝트 관리

## 클라이언트-서버 모드

OpenCode의 클라이언트-서버 아키텍처 덕분에:

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│  ┌─────────┐     ┌──────────────────┐               │
│  │   TUI   │────▶│                  │               │
│  └─────────┘     │                  │               │
│                  │  OpenCode Server │               │
│  ┌─────────┐     │  (localhost)     │               │
│  │ Desktop │────▶│                  │               │
│  └─────────┘     │                  │               │
│                  │                  │               │
│  ┌─────────┐     │                  │               │
│  │  Web    │────▶│                  │               │
│  └─────────┘     └──────────────────┘               │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 원격 접속

서버를 로컬에서 실행하고 다른 장치에서 접속 가능:

```bash
# 서버 시작 (포트 지정)
opencode serve --port 4096

# 원격에서 접속
opencode connect --host remote-machine:4096
```

## 국제화 (i18n)

TUI와 데스크톱 앱 모두 다국어를 지원합니다:

```
packages/app/src/i18n/
├── en.json
├── ko.json
├── ja.json
└── ...
```

## UI 컴포넌트

공통 UI 컴포넌트는 별도 패키지로 관리됩니다:

```
packages/ui/
├── src/
│   ├── Button.tsx
│   ├── Input.tsx
│   ├── Modal.tsx
│   └── ...
└── package.json
```

## 다음 단계

다음 챕터에서는 LSP 지원과 스킬 시스템을 알아봅니다.

---

**이전 글**: [MCP 통합](/opencode-guide-08-mcp/)

**다음 글**: [LSP & 스킬 시스템](/opencode-guide-10-lsp-skills/)
