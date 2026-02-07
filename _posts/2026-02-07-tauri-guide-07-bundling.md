---
layout: post
title: "Tauri 완벽 가이드 (07) - 번들링 및 배포"
date: 2026-02-07
permalink: /tauri-guide-07-bundling/
author: Tauri Programme
categories: [웹 개발, 데스크톱]
tags: [Tauri, Bundling, Distribution, Deployment, Cross-Platform]
original_url: "https://github.com/tauri-apps/tauri"
excerpt: "Tauri 앱을 모든 플랫폼에 배포하기"
---

## 번들링 개요

Tauri는 플랫폼별 네이티브 인스톨러를 자동으로 생성합니다.

---

## 지원되는 번들 형식

| 플랫폼 | 번들 타입 | 설명 |
|--------|----------|------|
| Windows | `.msi` | Windows Installer (WiX) |
| Windows | `.exe` | NSIS 설치 프로그램 |
| macOS | `.app` | 앱 번들 |
| macOS | `.dmg` | 디스크 이미지 |
| Linux | `.deb` | Debian/Ubuntu 패키지 |
| Linux | `.rpm` | Fedora/RHEL 패키지 |
| Linux | `.AppImage` | 이식 가능한 실행 파일 |

---

## 번들 설정

### tauri.conf.json

```json
{
  "package": {
    "productName": "MyApp",
    "version": "1.0.0"
  },
  "tauri": {
    "bundle": {
      "active": true,
      "targets": ["deb", "appimage", "msi", "dmg"],
      "identifier": "com.mycompany.myapp",
      "icon": [
        "icons/32x32.png",
        "icons/128x128.png",
        "icons/128x128@2x.png",
        "icons/icon.icns",
        "icons/icon.ico"
      ],
      "resources": ["assets/*"],
      "externalBin": [],
      "copyright": "Copyright © 2026",
      "category": "Utility",
      "shortDescription": "A short description",
      "longDescription": "A longer description..."
    }
  }
}
```

---

## 아이콘 생성

### 필요한 아이콘 크기

```
icons/
├── 32x32.png
├── 128x128.png
├── 128x128@2x.png
├── icon.icns      # macOS
└── icon.ico       # Windows
```

### 자동 생성 도구

```bash
npm install --save-dev @tauri-apps/tauricon

npx tauricon path/to/icon.png
```

---

## 플랫폼별 빌드

### Windows

```bash
npm run tauri build -- --target x86_64-pc-windows-msvc
```

### macOS

```bash
npm run tauri build -- --target aarch64-apple-darwin  # Apple Silicon
npm run tauri build -- --target x86_64-apple-darwin   # Intel
npm run tauri build -- --target universal-apple-darwin # Universal
```

### Linux

```bash
npm run tauri build -- --target x86_64-unknown-linux-gnu
```

---

## 코드 서명

### Windows (Code Signing)

```json
{
  "tauri": {
    "bundle": {
      "windows": {
        "certificateThumbprint": "THUMBPRINT",
        "digestAlgorithm": "sha256",
        "timestampUrl": "http://timestamp.digicert.com"
      }
    }
  }
}
```

### macOS (App Notarization)

```bash
export APPLE_CERTIFICATE="Developer ID Application: ..."
export APPLE_CERTIFICATE_PASSWORD="password"
export APPLE_ID="apple-id@email.com"
export APPLE_PASSWORD="app-specific-password"
export APPLE_TEAM_ID="TEAM_ID"

npm run tauri build
```

---

## 자동 업데이트

### tauri.conf.json

```json
{
  "tauri": {
    "updater": {
      "active": true,
      "endpoints": [
        "https://releases.myapp.com/{{target}}/{{current_version}}"
      ],
      "dialog": true,
      "pubkey": "YOUR_PUBLIC_KEY"
    }
  }
}
```

### 프론트엔드 업데이트 체크

```typescript
import { checkUpdate, installUpdate } from '@tauri-apps/api/updater';
import { relaunch } from '@tauri-apps/api/process';

const { shouldUpdate, manifest } = await checkUpdate();

if (shouldUpdate) {
    console.log(`Update to ${manifest?.version} available`);

    await installUpdate();
    await relaunch();
}
```

---

*다음 글에서는 시스템 통합 기능을 살펴봅니다.*
