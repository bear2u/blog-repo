---
layout: post
title: "Tauri 완벽 가이드 (01) - 소개 및 개요"
date: 2026-02-07
permalink: /tauri-guide-01-intro/
author: Tauri Programme
categories: [웹 개발, 데스크톱]
tags: [Tauri, Rust, Desktop App, WebView, Electron Alternative]
original_url: "https://github.com/tauri-apps/tauri"
excerpt: "Rust 기반 초경량 데스크톱 애플리케이션 프레임워크 Tauri 소개"
---

## Tauri란?

**Tauri**는 Rust로 작성된 차세대 데스크톱 애플리케이션 프레임워크입니다. 모든 주요 데스크톱 플랫폼(Windows, macOS, Linux)과 모바일 플랫폼(iOS, Android)을 위한 작고 빠른 바이너리를 빌드할 수 있습니다.

개발자는 HTML, JS, CSS로 컴파일되는 모든 프론트엔드 프레임워크(React, Vue, Svelte 등)를 사용하여 UI를 구축하고, Rust로 작성된 백엔드 API와 통신할 수 있습니다.

---

## 핵심 특징

### 1. **초경량 바이너리**

Tauri 앱은 OS의 네이티브 WebView를 사용하므로 Chromium을 번들링할 필요가 없습니다.

| 프레임워크 | Hello World 앱 크기 |
|-----------|-------------------|
| Electron | ~120MB |
| **Tauri** | **~3-10MB** |

### 2. **보안 우선**

- 기본적으로 안전한 IPC(Inter-Process Communication)
- CSP(Content Security Policy) 내장
- 프로덕션 빌드에서 개발자 도구 비활성화
- 코드 서명 및 앱 공증 지원

### 3. **네이티브 성능**

- Rust 백엔드로 빠른 시스템 호출
- 메모리 안전성 보장
- 멀티스레딩 지원
- Webview와 백엔드 간 최적화된 메시지 패싱

### 4. **멀티 플랫폼**

| 플랫폼 | 지원 버전 |
|--------|----------|
| Windows | 7 이상 |
| macOS | 10.15 이상 |
| Linux | webkit2gtk 4.1+ (Ubuntu 22.04+) |
| iOS/iPadOS | 9 이상 |
| Android | 7 이상 |

---

## Tauri vs Electron

| 항목 | Tauri | Electron |
|------|-------|----------|
| **언어** | Rust + Web | Node.js + Web |
| **WebView** | 시스템 네이티브 | Chromium 번들 |
| **앱 크기** | 3-10MB | 120MB+ |
| **메모리** | 낮음 (50-100MB) | 높음 (200MB+) |
| **시작 속도** | 빠름 | 느림 |
| **보안** | 매우 강함 | 보통 |
| **생태계** | 성장 중 | 성숙 |

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────┐
│                     Tauri Application                    │
├──────────────────────────┬──────────────────────────────┤
│      Frontend (Web)       │      Backend (Rust)         │
│                          │                              │
│  React / Vue / Svelte    │    tauri-core               │
│  HTML / CSS / JS         │    Commands / Events        │
│                          │    System API               │
│  @tauri-apps/api         │    Plugin System            │
│  (TypeScript)            │                              │
├──────────────────────────┼──────────────────────────────┤
│           Message Passing (IPC)                          │
├──────────────────────────┴──────────────────────────────┤
│                    WRY (WebView)                        │
│  ┌──────────┬───────────────┬───────────────┐         │
│  │ WKWebView│  WebView2     │ WebKitGTK     │         │
│  │ (macOS)  │  (Windows)    │ (Linux)       │         │
│  └──────────┴───────────────┴───────────────┘         │
├──────────────────────────────────────────────────────────┤
│                    TAO (Window)                         │
│              크로스 플랫폼 윈도우 관리                      │
├──────────────────────────────────────────────────────────┤
│                Operating System                          │
│        Windows / macOS / Linux / iOS / Android          │
└──────────────────────────────────────────────────────────┘
```

---

## 주요 컴포넌트

### 1. **TAO** (윈도우 관리)

- 크로스 플랫폼 윈도우 생성 라이브러리
- winit의 포크로 메뉴바, 시스템 트레이 지원 추가
- 모든 주요 플랫폼 지원

### 2. **WRY** (WebView 렌더링)

- 크로스 플랫폼 WebView 렌더링 라이브러리
- 플랫폼별 WebView 추상화:
  - macOS: WKWebView
  - Windows: WebView2
  - Linux: WebKitGTK
  - Android: Android System WebView

### 3. **tauri-core** (핵심 프레임워크)

- Rust 백엔드 로직
- 명령어 시스템 (Commands)
- 이벤트 시스템 (Events)
- 플러그인 시스템
- 빌드 타임 설정 (`tauri.conf.json`)

### 4. **@tauri-apps/api** (프론트엔드 API)

- TypeScript로 작성된 JavaScript 라이브러리
- 프론트엔드에서 백엔드 호출
- 이벤트 리스닝
- CJS, ESM 모두 지원

---

## 빠른 시작 예제

### 프로젝트 생성

```bash
# npm 사용
npm create tauri-app@latest

# pnpm 사용
pnpm create tauri-app

# yarn 사용
yarn create tauri-app
```

### 디렉토리 구조

```
my-tauri-app/
├── src/              # 프론트엔드 소스 (React, Vue 등)
├── src-tauri/        # Rust 백엔드
│   ├── src/
│   │   └── main.rs   # 메인 엔트리
│   ├── Cargo.toml    # Rust 의존성
│   ├── tauri.conf.json  # Tauri 설정
│   └── build.rs
├── package.json
└── index.html
```

### 간단한 Rust 명령어

```rust
// src-tauri/src/main.rs
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### 프론트엔드에서 호출

```typescript
// src/App.tsx
import { invoke } from '@tauri-apps/api/tauri';

async function greet() {
    const message = await invoke('greet', { name: 'World' });
    console.log(message); // "Hello, World!"
}
```

---

## 내장 기능

Tauri는 다양한 네이티브 기능을 기본 제공합니다:

- **앱 번들러**: `.app`, `.dmg`, `.deb`, `.rpm`, `.AppImage`, `.exe`, `.msi` 생성
- **자동 업데이트**: 데스크톱 앱 자동 업데이트 (Self Updater)
- **시스템 트레이**: 트레이 아이콘 및 메뉴
- **네이티브 알림**: OS 알림 표시
- **WebView Protocol**: localhost 서버 없이 네이티브 프로토콜 사용
- **GitHub Actions**: CI/CD 통합
- **VS Code 확장**: 개발 도구

---

## 플러그인 생태계

공식 및 커뮤니티 플러그인:

| 플러그인 | 기능 |
|---------|------|
| `tauri-plugin-fs` | 파일 시스템 API |
| `tauri-plugin-shell` | 셸 명령 실행 |
| `tauri-plugin-http` | HTTP 클라이언트 |
| `tauri-plugin-sql` | SQL 데이터베이스 |
| `tauri-plugin-store` | 키-값 저장소 |
| `tauri-plugin-window-state` | 윈도우 상태 저장 |

---

## 사용 사례

Tauri는 다음과 같은 앱에 적합합니다:

1. **생산성 도구**: 노트 앱, 태스크 매니저
2. **개발 도구**: IDE, Git 클라이언트, API 테스터
3. **크리에이티브 앱**: 이미지 편집기, 음악 플레이어
4. **비즈니스 앱**: CRM, ERP, 대시보드
5. **게임**: 2D 인디 게임, 퍼즐 게임

---

## 학습 자료

- **공식 문서**: https://tauri.app
- **GitHub**: https://github.com/tauri-apps/tauri
- **예제**: https://github.com/tauri-apps/tauri/tree/dev/examples
- **Discord**: https://discord.com/invite/tauri
- **플러그인 카탈로그**: https://tauri.app/plugins

---

*다음 글에서는 Tauri 개발 환경 설정과 첫 앱 생성 방법을 살펴봅니다.*
