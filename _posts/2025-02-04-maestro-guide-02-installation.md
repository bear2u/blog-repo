---
layout: post
title: "Maestro 가이드 - 설치 및 설정"
date: 2025-02-04
category: AI
tags: [maestro, installation, setup, android, ios]
series: maestro-guide
part: 2
author: mobile-dev-inc
original_url: https://github.com/mobile-dev-inc/Maestro
---

## 요구사항

### 공통 요구사항

```bash
# Java 17 이상 필요
java -version
# openjdk version "17.0.x" 이상
```

### 플랫폼별 요구사항

| 플랫폼 | 요구사항 |
|--------|----------|
| **Android** | Android SDK, ADB, 에뮬레이터 또는 실제 디바이스 |
| **iOS** | macOS, Xcode, 시뮬레이터 또는 실제 디바이스 |
| **Web** | Chrome, Safari, 또는 Firefox |

## CLI 설치

### macOS / Linux / Windows (WSL)

```bash
# 자동 설치 스크립트
curl -fsSL "https://get.maestro.mobile.dev" | bash
```

### Windows (네이티브)

```powershell
# Chocolatey 사용
choco install maestro

# 또는 Scoop 사용
scoop bucket add ACooper81_scoop-bucket https://github.com/ACooper81/ACooper81_scoop-bucket
scoop install maestro
```

### 설치 확인

```bash
maestro --version
# Maestro version: 1.x.x
```

## Android 설정

### 1. Android SDK 설치

```bash
# Android Studio 설치 또는 커맨드라인 도구
# https://developer.android.com/studio

# SDK 경로 설정
export ANDROID_HOME=$HOME/Android/Sdk
export PATH=$PATH:$ANDROID_HOME/platform-tools
```

### 2. 에뮬레이터 생성

```bash
# AVD Manager로 에뮬레이터 생성
# 또는 커맨드라인:
sdkmanager "system-images;android-34;google_apis;x86_64"
avdmanager create avd -n test_device -k "system-images;android-34;google_apis;x86_64"
```

### 3. 에뮬레이터 시작

```bash
# 에뮬레이터 시작
emulator -avd test_device

# 또는 Android Studio에서 시작
```

### 4. 디바이스 확인

```bash
# 연결된 디바이스 확인
adb devices
# List of devices attached
# emulator-5554    device
```

### 5. 앱 설치

```bash
# APK 설치
adb install path/to/your/app.apk
```

## iOS 설정 (macOS)

### 1. Xcode 설치

```bash
# App Store에서 Xcode 설치
# 또는 커맨드라인 도구만 설치
xcode-select --install
```

### 2. 시뮬레이터 시작

```bash
# 사용 가능한 시뮬레이터 목록
xcrun simctl list devices

# 시뮬레이터 부팅
xcrun simctl boot "iPhone 15 Pro"

# 시뮬레이터 앱 열기
open -a Simulator
```

### 3. 앱 설치

```bash
# .app 파일 설치 (시뮬레이터)
xcrun simctl install booted path/to/YourApp.app

# 실제 디바이스는 Xcode 또는 ios-deploy 사용
```

## Web 설정

### Chrome (기본)

```bash
# Chrome 설치 확인
google-chrome --version

# Maestro가 자동으로 ChromeDriver 관리
```

### 다른 브라우저

```yaml
# flow.yaml에서 브라우저 지정
browser: firefox  # 또는 safari
---
- launchBrowser
```

## 프로젝트 구조

### 권장 디렉토리 구조

```
my-app/
├── app/                    # 앱 소스 코드
├── maestro/                # Maestro 테스트
│   ├── flows/              # 메인 플로우
│   │   ├── login.yaml
│   │   ├── signup.yaml
│   │   └── checkout.yaml
│   ├── subflows/           # 재사용 가능한 서브플로우
│   │   ├── common_login.yaml
│   │   └── common_logout.yaml
│   └── config.yaml         # 공통 설정
└── README.md
```

### config.yaml 예시

```yaml
# maestro/config.yaml
appId: com.example.myapp
env:
  BASE_URL: https://api.example.com
  TEST_USER: test@example.com
```

## 첫 번째 테스트 작성

### 1. 플로우 파일 생성

```yaml
# maestro/flows/hello.yaml
appId: com.android.settings
---
- launchApp
- assertVisible: "Settings"
```

### 2. 테스트 실행

```bash
maestro test maestro/flows/hello.yaml
```

### 3. 결과 확인

```
Running flow: hello.yaml

 ✅ Launch app "com.android.settings"
 ✅ Assert "Settings" is visible

Flow completed successfully in 2.3s
```

## 유용한 CLI 명령어

### 기본 명령어

```bash
# 단일 플로우 실행
maestro test flow.yaml

# 디렉토리 내 모든 플로우 실행
maestro test flows/

# 디바이스 지정
maestro test --device emulator-5554 flow.yaml

# 환경 변수 전달
maestro test -e USERNAME=test flow.yaml
```

### 디버깅

```bash
# 요소 계층 구조 출력
maestro hierarchy

# 실시간 인스펙터 (Maestro Studio)
maestro studio

# 상세 로그
maestro test flow.yaml --debug-output ./debug
```

### 녹화 및 스크린샷

```bash
# 테스트 실행 녹화
maestro record flow.yaml

# 스크린샷 저장
maestro test flow.yaml --screenshots ./screenshots
```

## 환경 변수

### 시스템 환경 변수

```bash
# Android SDK 경로
export ANDROID_HOME=$HOME/Android/Sdk

# Java 경로
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
```

### Maestro 환경 변수

```bash
# 플로우에서 사용할 변수
export MAESTRO_APP_ID=com.example.app
export MAESTRO_USERNAME=test@example.com
```

### 플로우 내 환경 변수 사용

```yaml
appId: ${MAESTRO_APP_ID}
---
- launchApp
- inputText: ${MAESTRO_USERNAME}
```

## 문제 해결

### "No devices found"

```bash
# Android
adb devices  # 디바이스 목록 확인
adb kill-server && adb start-server  # ADB 재시작

# iOS
xcrun simctl list devices  # 시뮬레이터 목록
xcrun simctl boot "iPhone 15 Pro"  # 시뮬레이터 시작
```

### "Java version not supported"

```bash
# Java 버전 확인
java -version

# Java 17 설치 (Ubuntu)
sudo apt install openjdk-17-jdk

# macOS (Homebrew)
brew install openjdk@17
```

### 앱이 설치되지 않음

```bash
# Android
adb install -r app.apk  # -r: 재설치 허용

# iOS 시뮬레이터
xcrun simctl install booted app.app
```

## 로그 위치

```bash
# CLI 로그
~/.maestro/tests/*/maestro.log

# iOS XCTest 러너 로그
~/Library/Logs/maestro/xctest_runner_logs
```

## 다음 단계

다음 챕터에서는 YAML 플로우 문법을 자세히 다룹니다.

---

**이전 글**: [소개](/maestro-guide-01-intro/)

**다음 글**: [YAML 플로우](/maestro-guide-03-yaml-flows/)
