---
layout: post
title: "Tauri 완벽 가이드 (03) - 아키텍처 분석"
date: 2026-02-07
permalink: /tauri-guide-03-architecture/
author: Tauri Programme
categories: [웹 개발, 데스크톱]
tags: [Tauri, Architecture, TAO, WRY, Rust, WebView]
original_url: "https://github.com/tauri-apps/tauri"
excerpt: "Tauri의 전체 아키텍처와 핵심 컴포넌트 이해하기"
---

## Tauri 아키텍처 개요

Tauri는 **폴리글롯(Polyglot)** 도구킷으로, Rust와 Web 기술을 결합하여 데스크톱 애플리케이션을 구축합니다. Electron과 달리 Chromium을 번들링하지 않고 OS의 네이티브 WebView를 사용합니다.

---

## 전체 레이어 구조

```
┌──────────────────────────────────────────────────────────┐
│                  Application Layer                        │
│  ┌────────────────────┐  ┌──────────────────────────┐   │
│  │   Frontend (Web)    │  │   Backend (Rust)         │   │
│  │                     │  │                          │   │
│  │  React/Vue/Svelte  │  │  tauri::Builder          │   │
│  │  HTML/CSS/JS       │  │  Commands / Events       │   │
│  │                     │  │  State Management        │   │
│  │  @tauri-apps/api   │◄─┼─►Plugin System           │   │
│  └────────────────────┘  └──────────────────────────┘   │
└────────────┬─────────────────────────┬───────────────────┘
             │      IPC Bridge         │
┌────────────┴─────────────────────────┴───────────────────┐
│                  Tauri Core Layer                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              tauri-runtime                        │   │
│  │         (Runtime Abstraction)                     │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │          tauri-runtime-wry                        │   │
│  │        (WRY Runtime Implementation)               │   │
│  └──────────────────────────────────────────────────┘   │
└────────────┬─────────────────────────┬───────────────────┘
             │                         │
┌────────────┴──────────┐  ┌──────────┴───────────────────┐
│   WRY (WebView)        │  │   TAO (Window)               │
│                        │  │                              │
│  ┌─────────────────┐  │  │  ┌────────────────────────┐ │
│  │ Platform WebView│  │  │  │ Window Management       │ │
│  ├─────────────────┤  │  │  ├────────────────────────┤ │
│  │ • WKWebView     │  │  │  │ • Create/Close         │ │
│  │ • WebView2      │  │  │  │ • Resize/Move          │ │
│  │ • WebKitGTK     │  │  │  │ • Menu/Tray            │ │
│  │ • Android WV    │  │  │  │ • Events               │ │
│  └─────────────────┘  │  │  └────────────────────────┘ │
└───────────────────────┘  └──────────────────────────────┘
             │                         │
┌────────────┴─────────────────────────┴───────────────────┐
│              Operating System                             │
│     Windows / macOS / Linux / iOS / Android              │
└──────────────────────────────────────────────────────────┘
```

---

## 핵심 컴포넌트

### 1. TAO - 윈도우 관리

**GitHub**: https://github.com/tauri-apps/tao

TAO는 크로스 플랫폼 애플리케이션 윈도우 생성 라이브러리입니다.

#### 주요 기능

- 윈도우 생성 및 관리
- 메뉴바 (Menu Bar)
- 시스템 트레이 (System Tray)
- 이벤트 루프 (Event Loop)
- 키보드/마우스 입력 처리

#### winit과의 차이점

TAO는 [winit](https://github.com/rust-windowing/winit)의 포크로, 다음 기능이 추가되었습니다:

| 기능 | winit | TAO |
|-----|-------|-----|
| 윈도우 생성 | ✅ | ✅ |
| 메뉴바 | ❌ | ✅ |
| 시스템 트레이 | ❌ | ✅ |
| 모바일 지원 | 부분 | ✅ |

#### 사용 예시

```rust
use tao::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Hello TAO")
        .build(&event_loop)
        .unwrap();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => (),
        }
    });
}
```

---

### 2. WRY - WebView 렌더링

**GitHub**: https://github.com/tauri-apps/wry

WRY는 크로스 플랫폼 WebView 렌더링 라이브러리입니다.

#### 플랫폼별 WebView

| 플랫폼 | WebView 엔진 | 버전 |
|--------|-------------|------|
| Windows | WebView2 (Edge) | Chromium 기반 |
| macOS | WKWebView | Safari 기반 |
| Linux | WebKitGTK | WebKit 기반 |
| iOS | WKWebView | Safari 기반 |
| Android | System WebView | Chromium 기반 |

#### 주요 기능

- HTML/CSS/JS 렌더링
- JavaScript ↔ Rust 통신
- 커스텀 프로토콜 (`tauri://`)
- DevTools 통합

#### 사용 예시

```rust
use wry::{
    application::{
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::WindowBuilder,
    },
    webview::WebViewBuilder,
};

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let _webview = WebViewBuilder::new(window)
        .unwrap()
        .with_url("https://tauri.app")?
        .build()?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        if let Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } = event
        {
            *control_flow = ControlFlow::Exit;
        }
    });
}
```

---

### 3. Tauri Core - 핵심 프레임워크

Tauri Core는 여러 crate로 구성됩니다:

#### 3.1 tauri

**위치**: `crates/tauri`

메인 crate로, 모든 것을 하나로 묶습니다.

```rust
use tauri::Builder;

fn main() {
    Builder::default()
        .invoke_handler(tauri::generate_handler![my_command])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

**주요 모듈:**

```rust
tauri::
├── api/              // 프론트엔드 API 매핑
├── command/          // 명령어 시스템
├── event/            // 이벤트 시스템
├── plugin/           // 플러그인 시스템
├── window/           // 윈도우 관리
├── menu/             // 메뉴 빌더
├── tray/             // 시스템 트레이
└── state/            // 상태 관리
```

#### 3.2 tauri-runtime

**위치**: `crates/tauri-runtime`

Tauri와 하위 웹뷰 라이브러리 간의 접착층입니다.

```rust
pub trait Runtime: Sized + 'static {
    type Dispatcher: Dispatch;
    type Handle: RuntimeHandle;
    // ...
}
```

#### 3.3 tauri-runtime-wry

**위치**: `crates/tauri-runtime-wry`

WRY를 위한 `tauri-runtime` 구현체입니다.

#### 3.4 tauri-utils

**위치**: `crates/tauri-utils`

재사용 가능한 유틸리티:

- 설정 파싱 (`tauri.conf.json`)
- 플랫폼 감지
- CSP 주입
- 에셋 관리

#### 3.5 tauri-build

**위치**: `crates/tauri-build`

빌드 타임에 매크로 적용:

```rust
// build.rs
fn main() {
    tauri_build::build()
}
```

#### 3.6 tauri-codegen

**위치**: `crates/tauri-codegen`

- 에셋 임베드, 해시, 압축
- 아이콘 처리
- `tauri.conf.json` 파싱

#### 3.7 tauri-macros

**위치**: `crates/tauri-macros`

편리한 매크로 제공:

```rust
#[tauri::command]
fn my_command() {}

tauri::generate_handler![my_command]
tauri::generate_context!()
```

---

### 4. Tauri Tooling

#### 4.1 @tauri-apps/api (TypeScript)

**위치**: `packages/api`

프론트엔드 JavaScript API:

```typescript
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/api/dialog';
import { readTextFile } from '@tauri-apps/api/fs';
```

**빌드 출력:**

- CJS (CommonJS)
- ESM (ES Module)
- TypeScript 타입 정의

#### 4.2 @tauri-apps/cli (JavaScript)

**위치**: `packages/cli`

Rust `tauri-cli`의 Node.js 래퍼:

```bash
npm install @tauri-apps/cli
npx tauri dev
npx tauri build
```

[napi-rs](https://github.com/napi-rs/napi-rs)를 사용하여 Rust를 Node.js 네이티브 모듈로 패키징합니다.

#### 4.3 tauri-cli (Rust)

**위치**: `crates/tauri-cli`

Rust로 작성된 CLI 도구:

```bash
cargo install tauri-cli
cargo tauri dev
cargo tauri build
```

**주요 명령어:**

| 명령어 | 설명 |
|-------|------|
| `tauri init` | 기존 프로젝트에 Tauri 추가 |
| `tauri dev` | 개발 서버 실행 |
| `tauri build` | 프로덕션 빌드 |
| `tauri info` | 환경 정보 출력 |
| `tauri plugin` | 플러그인 관리 |

#### 4.4 tauri-bundler

**위치**: `crates/tauri-bundler`

플랫폼별 번들 생성:

```rust
pub enum PackageType {
    MacOsBundle,
    IosBundle,
    WindowsMsi,
    Deb,
    Rpm,
    AppImage,
    Dmg,
    Nsis,
}
```

#### 4.5 create-tauri-app

**GitHub**: https://github.com/tauri-apps/create-tauri-app

프로젝트 스캐폴딩 도구:

```bash
npm create tauri-app@latest
```

---

## IPC (Inter-Process Communication)

### 메시지 흐름

```
┌─────────────────┐
│  Frontend (JS)  │
│                 │
│  invoke('cmd')  │
└────────┬────────┘
         │ JSON
         ▼
┌─────────────────┐
│  @tauri-apps/api│
│  (TypeScript)   │
└────────┬────────┘
         │ postMessage
         ▼
┌─────────────────┐
│   WebView IPC   │
│   (WRY)         │
└────────┬────────┘
         │ Native Call
         ▼
┌─────────────────┐
│  Tauri Runtime  │
│  (Rust)         │
└────────┬────────┘
         │ Deserialize
         ▼
┌─────────────────┐
│  Command Handler│
│  #[tauri::command]│
└────────┬────────┘
         │ Result
         ▼
┌─────────────────┐
│  Response (JSON)│
│  ↓ Frontend     │
└─────────────────┘
```

### 프론트엔드 → 백엔드

```typescript
// Frontend
import { invoke } from '@tauri-apps/api/tauri';

const result = await invoke('my_command', {
    arg1: 'value',
    arg2: 123
});
```

```rust
// Backend
#[tauri::command]
fn my_command(arg1: String, arg2: i32) -> Result<String, String> {
    Ok(format!("Got {} and {}", arg1, arg2))
}
```

### 백엔드 → 프론트엔드 (이벤트)

```rust
// Backend
use tauri::Manager;

app.emit_all("my-event", Payload {
    message: "Hello from Rust!".into()
})?;
```

```typescript
// Frontend
import { listen } from '@tauri-apps/api/event';

await listen('my-event', (event) => {
    console.log(event.payload.message);
});
```

---

## 컴파일 과정

### 개발 빌드 (dev)

```
1. npm run dev (Vite 개발 서버 시작)
   ├─→ http://localhost:1420
   └─→ HMR 활성화

2. cargo tauri dev
   ├─→ Rust 컴파일 (증분)
   ├─→ tauri-runtime-wry 링크
   ├─→ WebView에 http://localhost:1420 로드
   └─→ 앱 윈도우 표시
```

### 프로덕션 빌드 (build)

```
1. npm run build (Vite 프로덕션 빌드)
   ├─→ dist/ 생성
   ├─→ 코드 최소화/번들링
   └─→ 에셋 최적화

2. cargo tauri build
   ├─→ Rust 릴리스 컴파일 (최적화)
   ├─→ dist/ 에셋 임베드
   ├─→ 바이너리 생성
   └─→ 플랫폼별 번들 생성
       ├─→ .app / .dmg (macOS)
       ├─→ .exe / .msi (Windows)
       └─→ .deb / .appimage (Linux)
```

---

## 보안 모델

### Allowlist 시스템

`tauri.conf.json`:

```json
{
  "tauri": {
    "allowlist": {
      "all": false,
      "fs": {
        "all": false,
        "readFile": true,
        "writeFile": true,
        "scope": ["$APPDATA/*"]
      },
      "http": {
        "all": false,
        "request": true,
        "scope": ["https://api.example.com/*"]
      }
    }
  }
}
```

### CSP (Content Security Policy)

```json
{
  "tauri": {
    "security": {
      "csp": "default-src 'self'; script-src 'self' 'unsafe-inline'"
    }
  }
}
```

---

## 디렉토리 구조 (모노레포)

```
tauri/
├── crates/                    # Rust crates
│   ├── tauri/                 # 메인 crate
│   ├── tauri-runtime/
│   ├── tauri-runtime-wry/
│   ├── tauri-cli/
│   ├── tauri-bundler/
│   └── ...
├── packages/                  # JavaScript packages
│   ├── api/                   # @tauri-apps/api
│   └── cli/                   # @tauri-apps/cli
├── examples/                  # 예제 앱
└── bench/                     # 벤치마크
```

---

*다음 글에서는 Tauri의 Rust Backend와 명령어 시스템을 자세히 살펴봅니다.*
