---
layout: post
title: "Tauri 완벽 가이드 (09) - 모바일 지원"
date: 2026-02-07
permalink: /tauri-guide-09-mobile/
author: Tauri Programme
categories: [웹 개발, 모바일]
tags: [Tauri, Mobile, iOS, Android, Cross-Platform]
original_url: "https://github.com/tauri-apps/tauri"
excerpt: "Tauri로 iOS 및 Android 앱 개발하기"
---

## Tauri Mobile 개요

Tauri 2.0부터 iOS와 Android를 공식 지원합니다. 동일한 코드베이스로 데스크톱과 모바일 앱을 모두 빌드할 수 있습니다.

---

## 모바일 설정

### iOS 요구사항

- **macOS** (필수)
- **Xcode** 13.0 이상
- **CocoaPods**

```bash
# CocoaPods 설치
sudo gem install cocoapods
```

### Android 요구사항

- **Android Studio**
- **Android SDK** (API 24 이상)
- **Java Development Kit** (JDK 11 이상)

```bash
# Android SDK 경로 설정
export ANDROID_HOME=$HOME/Library/Android/sdk
export PATH=$PATH:$ANDROID_HOME/tools
export PATH=$PATH:$ANDROID_HOME/platform-tools
```

---

## 모바일 타겟 추가

```bash
# iOS 타겟 추가
npm run tauri ios init

# Android 타겟 추가
npm run tauri android init
```

---

## 디렉토리 구조

```
my-tauri-app/
├── src/              # 프론트엔드
├── src-tauri/        # Rust 백엔드
├── gen/              # 생성된 모바일 프로젝트
│   ├── android/      # Android 프로젝트
│   │   ├── app/
│   │   ├── build.gradle
│   │   └── settings.gradle
│   └── apple/        # iOS/macOS 프로젝트
│       ├── MyApp.xcodeproj/
│       └── Podfile
└── tauri.conf.json
```

---

## 모바일 빌드 및 실행

### iOS

```bash
# 개발 모드
npm run tauri ios dev

# 프로덕션 빌드
npm run tauri ios build

# 특정 시뮬레이터에서 실행
npm run tauri ios dev --simulator "iPhone 15 Pro"
```

### Android

```bash
# 개발 모드
npm run tauri android dev

# 프로덕션 빌드
npm run tauri android build

# 특정 디바이스에서 실행
npm run tauri android dev --device "Pixel_7_API_33"
```

---

## 플랫폼별 코드

### Rust 조건부 컴파일

```rust
#[tauri::command]
fn platform_specific() -> String {
    #[cfg(target_os = "ios")]
    {
        "Running on iOS".to_string()
    }

    #[cfg(target_os = "android")]
    {
        "Running on Android".to_string()
    }

    #[cfg(desktop)]
    {
        "Running on Desktop".to_string()
    }
}
```

### TypeScript 플랫폼 감지

```typescript
import { platform } from '@tauri-apps/api/os';

const platformName = await platform();

if (platformName === 'ios') {
    // iOS 전용 코드
} else if (platformName === 'android') {
    // Android 전용 코드
} else {
    // 데스크톱 코드
}
```

---

## 모바일 권한

### iOS (Info.plist)

```xml
<key>NSCameraUsageDescription</key>
<string>We need camera access for taking photos</string>

<key>NSLocationWhenInUseUsageDescription</key>
<string>We need location access for...</string>

<key>NSPhotoLibraryUsageDescription</key>
<string>We need photo library access for...</string>
```

### Android (AndroidManifest.xml)

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
```

---

## 모바일 플러그인

### tauri-plugin-barcode-scanner

```bash
npm install tauri-plugin-barcode-scanner-api
```

```rust
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_barcode_scanner::init())
        .run(tauri::generate_context!())
        .unwrap();
}
```

```typescript
import { scan } from 'tauri-plugin-barcode-scanner-api';

const result = await scan();
console.log('Scanned:', result);
```

---

## 반응형 UI

```css
/* 모바일 대응 */
@media (max-width: 768px) {
    .sidebar {
        display: none;
    }

    .main-content {
        width: 100%;
    }
}

/* 터치 디바이스 */
@media (pointer: coarse) {
    button {
        min-height: 44px;
        min-width: 44px;
    }
}
```

---

## 배포

### iOS App Store

```bash
# Archive 생성
xcodebuild archive \
    -workspace gen/apple/MyApp.xcworkspace \
    -scheme MyApp \
    -archivePath MyApp.xcarchive

# IPA 생성
xcodebuild -exportArchive \
    -archivePath MyApp.xcarchive \
    -exportPath MyApp.ipa \
    -exportOptionsPlist ExportOptions.plist
```

### Google Play

```bash
# APK 생성
npm run tauri android build --release

# AAB 생성 (Google Play)
cd gen/android
./gradlew bundleRelease
```

---

*다음 글에서는 실전 활용 팁과 베스트 프랙티스를 살펴봅니다.*
