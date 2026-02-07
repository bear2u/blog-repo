---
layout: post
title: "Tauri 완벽 가이드 (02) - 설치 및 빠른 시작"
date: 2026-02-07
permalink: /tauri-guide-02-installation/
author: Tauri Programme
categories: [웹 개발, 데스크톱]
tags: [Tauri, Rust, Installation, Setup, Getting Started]
original_url: "https://github.com/tauri-apps/tauri"
excerpt: "Tauri 개발 환경 설정 및 첫 애플리케이션 생성하기"
---

## 시스템 요구사항

Tauri 개발을 위해서는 다음이 필요합니다:

### 공통 요구사항

- **Node.js**: 16.14 이상 (최신 LTS 권장)
- **Rust**: 1.77.2 이상
- **패키지 매니저**: npm, pnpm, 또는 yarn

### 플랫폼별 추가 요구사항

#### Windows

- **Microsoft Visual Studio C++ Build Tools**
- **WebView2** (Windows 11은 기본 탑재)

#### macOS

- **Xcode Command Line Tools**
- **macOS 10.15+**

```bash
xcode-select --install
```

#### Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install libwebkit2gtk-4.1-dev \
  build-essential \
  curl \
  wget \
  file \
  libxdo-dev \
  libssl-dev \
  libayatana-appindicator3-dev \
  librsvg2-dev
```

#### Linux (Fedora)

```bash
sudo dnf check-update
sudo dnf install webkit2gtk4.1-devel \
  openssl-devel \
  curl \
  wget \
  file \
  libappindicator-gtk3-devel \
  librsvg2-devel
sudo dnf group install "C Development Tools and Libraries"
```

---

## Rust 설치

### rustup으로 Rust 설치

```bash
# Unix/Linux/macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Windows
# https://rustup.rs 에서 rustup-init.exe 다운로드 후 실행
```

### Rust 버전 확인

```bash
rustc --version
# rustc 1.77.2 (25ef9e3d8 2024-04-09)

cargo --version
# cargo 1.77.2 (e52e36006 2024-03-26)
```

### Rust 업데이트

```bash
rustup update
```

---

## Tauri CLI 설치

Tauri CLI는 두 가지 방법으로 설치할 수 있습니다:

### 방법 1: npm을 통한 설치 (권장)

```bash
# npm
npm install --save-dev @tauri-apps/cli

# pnpm
pnpm add -D @tauri-apps/cli

# yarn
yarn add -D @tauri-apps/cli
```

**package.json에 스크립트 추가:**

```json
{
  "scripts": {
    "tauri": "tauri"
  }
}
```

**사용:**

```bash
npm run tauri dev
npm run tauri build
```

### 방법 2: Cargo를 통한 글로벌 설치

```bash
cargo install tauri-cli
```

**사용:**

```bash
cargo tauri dev
cargo tauri build
```

---

## 새 프로젝트 생성

### create-tauri-app 사용

가장 빠른 방법은 공식 스캐폴딩 도구를 사용하는 것입니다:

```bash
# npm
npm create tauri-app@latest

# pnpm
pnpm create tauri-app

# yarn
yarn create tauri-app
```

### 대화형 프롬프트

```
✔ Project name · my-tauri-app
✔ Choose your package manager · pnpm
✔ Choose your UI template · React
✔ Choose your UI flavor · TypeScript
```

**지원되는 프론트엔드 템플릿:**

- Vanilla (HTML/CSS/JS)
- React
- Vue
- Svelte
- SolidJS
- Preact
- Angular
- Yew (Rust)

---

## 프로젝트 구조

생성된 프로젝트 구조:

```
my-tauri-app/
├── src/                      # 프론트엔드 소스
│   ├── App.tsx
│   ├── main.tsx
│   └── styles.css
├── src-tauri/                # Rust 백엔드
│   ├── src/
│   │   └── main.rs          # 메인 엔트리포인트
│   ├── target/              # Rust 빌드 출력
│   ├── icons/               # 앱 아이콘
│   ├── Cargo.toml           # Rust 의존성
│   ├── Cargo.lock
│   ├── tauri.conf.json      # Tauri 설정
│   └── build.rs             # 빌드 스크립트
├── public/                  # 정적 파일
├── node_modules/
├── package.json
├── vite.config.ts           # Vite 설정
└── index.html
```

---

## 핵심 파일 살펴보기

### 1. src-tauri/src/main.rs

```rust
// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### 2. src-tauri/tauri.conf.json

```json
{
  "build": {
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build",
    "devPath": "http://localhost:1420",
    "distDir": "../dist"
  },
  "package": {
    "productName": "my-tauri-app",
    "version": "0.1.0"
  },
  "tauri": {
    "allowlist": {
      "all": false
    },
    "windows": [
      {
        "title": "My Tauri App",
        "width": 800,
        "height": 600
      }
    ]
  }
}
```

### 3. src-tauri/Cargo.toml

```toml
[package]
name = "my-tauri-app"
version = "0.1.0"
edition = "2021"

[build-dependencies]
tauri-build = { version = "2.0", features = [] }

[dependencies]
tauri = { version = "2.0", features = ["devtools"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[features]
default = ["custom-protocol"]
custom-protocol = ["tauri/custom-protocol"]
```

---

## 개발 서버 실행

### 개발 모드 시작

```bash
# npm을 통한 CLI
npm run tauri dev

# Cargo를 통한 CLI
cargo tauri dev
```

**첫 실행 시:**
- Rust 의존성 다운로드 (시간 소요)
- 프론트엔드 번들링
- 네이티브 바이너리 컴파일
- 앱 윈도우 표시

**이후 실행:**
- 증분 컴파일로 빠른 시작
- Hot Module Replacement (HMR) 지원

### 개발 도구

개발 빌드에서는 WebView DevTools가 자동 활성화됩니다:

- **Windows/Linux**: `F12` 또는 `Ctrl+Shift+I`
- **macOS**: `Cmd+Option+I`

---

## 첫 번째 명령어 추가

### 1. Rust 명령어 작성

`src-tauri/src/main.rs` 수정:

```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// 명령어 정의
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! Welcome to Tauri!", name)
}

fn main() {
    tauri::Builder::default()
        // 명령어 등록
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### 2. 프론트엔드에서 호출

`src/App.tsx`:

```typescript
import { useState } from "react";
import { invoke } from "@tauri-apps/api/tauri";

function App() {
  const [name, setName] = useState("");
  const [greetMsg, setGreetMsg] = useState("");

  async function greet() {
    // Rust 명령어 호출
    const message = await invoke<string>("greet", { name });
    setGreetMsg(message);
  }

  return (
    <div>
      <h1>Tauri Greet App</h1>
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Enter your name"
      />
      <button onClick={greet}>Greet</button>
      <p>{greetMsg}</p>
    </div>
  );
}

export default App;
```

### 3. 테스트

```bash
npm run tauri dev
```

- 이름 입력
- "Greet" 버튼 클릭
- "Hello, [이름]! Welcome to Tauri!" 메시지 표시

---

## 프로덕션 빌드

### 릴리스 빌드 생성

```bash
# npm
npm run tauri build

# Cargo
cargo tauri build
```

### 빌드 출력

빌드가 완료되면 다음 위치에 번들이 생성됩니다:

```
src-tauri/target/release/
├── my-tauri-app         # 실행 파일 (Linux/macOS)
├── my-tauri-app.exe     # 실행 파일 (Windows)
└── bundle/              # 플랫폼별 인스톨러
    ├── dmg/            # macOS
    ├── deb/            # Debian/Ubuntu
    ├── rpm/            # Fedora/RHEL
    ├── appimage/       # Linux AppImage
    ├── msi/            # Windows Installer (WiX)
    └── nsis/           # Windows Installer (NSIS)
```

### 빌드 최적화

**tauri.conf.json**에서 최적화 설정:

```json
{
  "tauri": {
    "bundle": {
      "active": true,
      "targets": ["deb", "appimage", "msi"],
      "identifier": "com.mycompany.myapp",
      "icon": [
        "icons/32x32.png",
        "icons/128x128.png",
        "icons/icon.icns",
        "icons/icon.ico"
      ]
    }
  }
}
```

---

## 디버깅 팁

### Rust 로그 출력

```rust
#[tauri::command]
fn my_command() {
    println!("Debug message");  // stdout에 출력
    eprintln!("Error message"); // stderr에 출력
}
```

**로그 확인:**

```bash
# 개발 모드에서 터미널에 출력
npm run tauri dev
```

### 프론트엔드 디버깅

```typescript
import { invoke } from "@tauri-apps/api/tauri";

try {
  const result = await invoke("my_command", { param: value });
  console.log("Success:", result);
} catch (error) {
  console.error("Error:", error);
}
```

### 빌드 오류 해결

**Cargo 캐시 정리:**

```bash
cd src-tauri
cargo clean
cd ..
npm run tauri dev
```

**Node 모듈 재설치:**

```bash
rm -rf node_modules package-lock.json
npm install
```

---

## 환경 변수

### 개발 환경 변수

`.env` 파일 생성:

```env
VITE_API_URL=https://api.example.com
```

**Vite에서 사용:**

```typescript
const apiUrl = import.meta.env.VITE_API_URL;
```

### Rust 빌드 타임 변수

`tauri.conf.json`:

```json
{
  "build": {
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "TAURI_PROFILE=release npm run build"
  }
}
```

---

## VS Code 설정

### 추천 확장

- **rust-analyzer**: Rust 언어 서버
- **Tauri**: Tauri 개발 도구
- **ESLint**: JavaScript/TypeScript 린팅
- **Prettier**: 코드 포매팅

### .vscode/settings.json

```json
{
  "rust-analyzer.linkedProjects": ["src-tauri/Cargo.toml"],
  "rust-analyzer.cargo.features": ["custom-protocol"],
  "files.watcherExclude": {
    "**/target/**": true
  }
}
```

---

## 다음 단계

첫 Tauri 앱을 성공적으로 생성했습니다! 이제 다음을 배울 준비가 되었습니다:

1. Tauri 아키텍처 이해
2. 고급 명령어 시스템
3. 이벤트 시스템
4. 플러그인 통합
5. 네이티브 기능 (파일 시스템, HTTP, SQL 등)

---

*다음 글에서는 Tauri의 전체 아키텍처와 핵심 컴포넌트를 자세히 살펴봅니다.*
