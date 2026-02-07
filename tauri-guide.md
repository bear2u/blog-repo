---
layout: page
title: Tauri 가이드
permalink: /tauri-guide/
icon: fas fa-rocket
---

# Tauri 완벽 가이드

> **Rust 기반 초경량 크로스 플랫폼 데스크톱 & 모바일 애플리케이션 프레임워크**

**Tauri**는 Rust로 작성된 차세대 데스크톱 및 모바일 애플리케이션 프레임워크입니다. Electron의 대안으로, OS 네이티브 WebView를 사용하여 **3-10MB**의 초경량 바이너리를 생성합니다. HTML/CSS/JavaScript로 UI를 작성하고, Rust 백엔드로 시스템 API에 접근할 수 있습니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요]({{ site.baseurl }}/tauri-guide-01-intro/) | Tauri란? 특징, Electron과 비교 |
| 02 | [설치 및 빠른 시작]({{ site.baseurl }}/tauri-guide-02-installation/) | 개발 환경 설정, 첫 앱 생성 |
| 03 | [아키텍처 분석]({{ site.baseurl }}/tauri-guide-03-architecture/) | TAO, WRY, Core 컴포넌트 |
| 04 | [Rust Backend]({{ site.baseurl }}/tauri-guide-04-rust-backend/) | 명령어 시스템, 상태 관리 |
| 05 | [프론트엔드 통합]({{ site.baseurl }}/tauri-guide-05-frontend-integration/) | @tauri-apps/api 사용법 |
| 06 | [플러그인 시스템]({{ site.baseurl }}/tauri-guide-06-plugin-system/) | 공식/커스텀 플러그인 |
| 07 | [번들링 및 배포]({{ site.baseurl }}/tauri-guide-07-bundling/) | 멀티 플랫폼 빌드, 코드 서명 |
| 08 | [시스템 통합]({{ site.baseurl }}/tauri-guide-08-system-integration/) | 트레이, 메뉴, 알림 |
| 09 | [모바일 지원]({{ site.baseurl }}/tauri-guide-09-mobile/) | iOS/Android 개발 |
| 10 | [실전 활용 및 팁]({{ site.baseurl }}/tauri-guide-10-best-practices/) | 성능 최적화, 보안, 베스트 프랙티스 |

---

## 주요 특징

- **초경량**: Electron 대비 1/10 크기 (3-10MB vs 120MB+)
- **빠른 성능**: Rust 백엔드, 네이티브 WebView
- **멀티 플랫폼**: Windows, macOS, Linux, iOS, Android
- **보안 우선**: 기본 보안 설정, CSP, Allowlist
- **풍부한 생태계**: 공식 플러그인, 커뮤니티 지원

---

## Tauri vs Electron

| 항목 | Tauri | Electron |
|------|-------|----------|
| **언어** | Rust + Web | Node.js + Web |
| **WebView** | 시스템 네이티브 | Chromium 번들 |
| **앱 크기** | 3-10MB | 120MB+ |
| **메모리** | 50-100MB | 200MB+ |
| **시작 속도** | 빠름 | 느림 |
| **보안** | 매우 강함 | 보통 |
| **모바일** | ✅ iOS/Android | ❌ |

---

## 빠른 시작

### 1. 프로젝트 생성

```bash
npm create tauri-app@latest
```

### 2. 개발 서버 실행

```bash
npm install
npm run tauri dev
```

### 3. 프로덕션 빌드

```bash
npm run tauri build
```

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────┐
│              Tauri Application                      │
├──────────────────┬──────────────────────────────────┤
│  Frontend (Web)  │      Backend (Rust)              │
│                  │                                  │
│  React/Vue/      │  tauri-core                      │
│  Svelte          │  Commands / Events               │
│                  │  Plugin System                   │
│  @tauri-apps/api │  State Management                │
└────────┬─────────┴──────────┬───────────────────────┘
         │   IPC (Message)     │
┌────────┴─────────────────────┴───────────────────────┐
│                WRY (WebView)                          │
│  WKWebView (macOS) | WebView2 (Win) | WebKitGTK (Linux) │
└───────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────┐
│               TAO (Window Management)                 │
└───────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────┐
│              Operating System                         │
│   Windows / macOS / Linux / iOS / Android            │
└───────────────────────────────────────────────────────┘
```

---

## 핵심 컴포넌트

### TAO (Window)
크로스 플랫폼 윈도우 생성 및 관리 라이브러리

### WRY (WebView)
크로스 플랫폼 WebView 렌더링 라이브러리

### tauri-core
Rust 백엔드 프레임워크, 명령어/이벤트 시스템

### @tauri-apps/api
TypeScript/JavaScript 프론트엔드 API

---

## 간단한 예제

### Rust 명령어

```rust
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

### 프론트엔드 호출

```typescript
import { invoke } from '@tauri-apps/api/tauri';

const message = await invoke('greet', { name: 'World' });
console.log(message); // "Hello, World!"
```

---

## 플러그인 생태계

| 플러그인 | 기능 |
|---------|------|
| `tauri-plugin-fs` | 파일 시스템 API |
| `tauri-plugin-shell` | 셸 명령 실행 |
| `tauri-plugin-http` | HTTP 클라이언트 |
| `tauri-plugin-sql` | SQL 데이터베이스 (SQLite, MySQL, PostgreSQL) |
| `tauri-plugin-store` | 키-값 저장소 |
| `tauri-plugin-window-state` | 윈도우 상태 저장 |
| `tauri-plugin-log` | 로깅 시스템 |
| `tauri-plugin-updater` | 자동 업데이트 |

---

## 지원 플랫폼

| 플랫폼 | 지원 버전 | WebView |
|--------|----------|---------|
| Windows | 7+ | WebView2 (Edge) |
| macOS | 10.15+ | WKWebView |
| Linux | Ubuntu 22.04+ | WebKitGTK 4.1 |
| iOS | 9+ | WKWebView |
| Android | 7+ | System WebView |

---

## 사용 사례

Tauri는 다음과 같은 앱에 적합합니다:

1. **생산성 도구**: 노트 앱, 태스크 매니저, 타임 트래커
2. **개발 도구**: Git 클라이언트, API 테스터, 데이터베이스 관리 도구
3. **크리에이티브 앱**: 이미지/비디오 편집기, 음악 플레이어
4. **비즈니스 앱**: CRM, ERP, 대시보드, 재고 관리
5. **게임**: 2D 인디 게임, 퍼즐 게임, 캐주얼 게임

---

## 기술 스택

| 레이어 | 기술 |
|--------|------|
| **Frontend** | React, Vue, Svelte, Angular, Solid, Yew |
| **Backend** | Rust (1.77.2+) |
| **Window** | TAO (winit fork) |
| **WebView** | WRY (WKWebView/WebView2/WebKitGTK) |
| **Bundler** | tauri-bundler (MSI, DMG, DEB, RPM, AppImage) |
| **CLI** | tauri-cli (Rust), @tauri-apps/cli (Node.js) |

---

## 학습 자료

- **공식 문서**: [https://tauri.app](https://tauri.app)
- **GitHub**: [https://github.com/tauri-apps/tauri](https://github.com/tauri-apps/tauri)
- **예제**: [https://github.com/tauri-apps/tauri/tree/dev/examples](https://github.com/tauri-apps/tauri/tree/dev/examples)
- **Discord**: [https://discord.com/invite/tauri](https://discord.com/invite/tauri)
- **Awesome Tauri**: [https://github.com/tauri-apps/awesome-tauri](https://github.com/tauri-apps/awesome-tauri)

---

## 시작하기

Tauri의 강력함을 직접 경험해보세요! [01. 소개 및 개요]({{ site.baseurl }}/tauri-guide-01-intro/)부터 시작하여 단계별로 학습할 수 있습니다.

---

*Rust의 성능과 안전성, 웹의 유연성을 결합한 Tauri로 차세대 데스크톱 앱을 만들어보세요!*
