---
layout: post
title: "Maestro 가이드 - 플랫폼별 테스트"
date: 2025-02-04
categories: [개발 도구, Maestro]
tags: [maestro, android, ios, web, react-native, flutter]
author: mobile-dev-inc
original_url: https://github.com/mobile-dev-inc/Maestro
---

## 플랫폼 지원 개요

```
┌─────────────────────────────────────────────────────────────┐
│                  Supported Platforms                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   📱 Mobile                │  🌐 Web                        │
│   ─────────────────────    │  ─────────────────             │
│   • Android (Native)       │  • Chrome                      │
│   • iOS (Native)           │  • Safari                      │
│   • React Native           │  • Firefox                     │
│   • Flutter                │  • Edge                        │
│   • Ionic/Cordova          │                                │
│   • Xamarin                │                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Android 테스트

### 기본 설정

```yaml
# Android 앱 테스트
appId: com.example.android.app
---
- launchApp
- tapOn: "시작하기"
```

### 에뮬레이터 vs 실제 디바이스

```bash
# 연결된 디바이스 확인
adb devices

# 특정 디바이스 지정
maestro test --device emulator-5554 flow.yaml
maestro test --device 192.168.1.100:5555 flow.yaml  # Wi-Fi ADB
```

### Android 전용 기능

```yaml
# 권한 자동 허용
- launchApp:
    permissions:
      android.permission.CAMERA: allow
      android.permission.LOCATION: deny

# 시스템 설정 접근
- launchApp:
    appId: com.android.settings
- tapOn: "Display"

# 알림 패널 열기
- swipe:
    start: "50%,0%"
    end: "50%,50%"
```

### Android 리소스 ID

```yaml
# 리소스 ID로 요소 선택
- tapOn:
    id: "com.example.app:id/login_button"

# 짧은 형식 (앱 패키지 생략)
- tapOn:
    id: "login_button"
```

### Intent로 앱 실행

```yaml
- launchApp:
    arguments:
      url: "myapp://product/123"
      extra_string: "test_value"
```

## iOS 테스트

### 기본 설정

```yaml
# iOS 앱 테스트
appId: com.example.ios.app
---
- launchApp
- tapOn: "시작하기"
```

### 시뮬레이터 vs 실제 디바이스

```bash
# 사용 가능한 시뮬레이터 목록
xcrun simctl list devices

# 시뮬레이터 시작
xcrun simctl boot "iPhone 15 Pro"

# 특정 시뮬레이터 지정
maestro test --device "iPhone 15 Pro" flow.yaml
```

### iOS 전용 기능

```yaml
# 권한 처리
- launchApp:
    permissions:
      notifications: allow
      photos: allow
      camera: deny

# Face ID / Touch ID 시뮬레이션
- evalScript: |
    // 시뮬레이터에서 Face ID 성공 시뮬레이션

# 시스템 다이얼로그 처리
- tapOn: "Allow"  # "앱이 위치에 접근하도록 허용하시겠습니까?"
```

### Accessibility ID

```yaml
# iOS Accessibility ID로 요소 선택
- tapOn:
    id: "loginButton"

# Accessibility Label
- tapOn:
    text: "로그인 버튼"
```

### 딥링크 (Universal Links)

```yaml
- openLink: "https://example.com/app/product/123"
# 또는
- openLink: "myapp://product/123"
```

## Web 테스트

### 기본 설정

```yaml
# 웹 앱 테스트
url: https://example.com
---
- launchBrowser
- tapOn: "로그인"
```

### 브라우저 선택

```yaml
# Chrome (기본)
url: https://example.com
---
- launchBrowser

# Firefox
url: https://example.com
browser: firefox
---
- launchBrowser

# Safari
url: https://example.com
browser: safari
---
- launchBrowser
```

### 웹 전용 명령어

```yaml
# URL 직접 이동
- openLink: "https://example.com/products"

# 페이지 새로고침
- evalScript: |
    location.reload()

# JavaScript 실행
- evalScript: |
    document.querySelector('#hidden-button').click()
```

### 웹 요소 선택

```yaml
# 텍스트로 선택
- tapOn: "Submit"

# CSS 선택자 (evalScript 사용)
- evalScript: |
    document.querySelector('.btn-primary').click()

# 링크 텍스트
- tapOn: "자세히 보기"
```

### 반응형 테스트

```yaml
# 모바일 뷰포트 시뮬레이션
url: https://example.com
browser: chrome
---
- launchBrowser
# 브라우저 개발자 도구로 모바일 뷰포트 설정
```

## React Native

### 기본 설정

```yaml
# React Native 앱 (Android)
appId: com.example.reactnative
---
- launchApp
- tapOn: "Welcome to React Native"
```

### testID 사용 (권장)

```jsx
// React Native 코드
<TouchableOpacity testID="login-button">
  <Text>Login</Text>
</TouchableOpacity>
```

```yaml
# Maestro 플로우
- tapOn:
    id: "login-button"
```

### 네비게이션 처리

```yaml
# React Navigation 딥링크
- openLink: "myapp://home/profile"

# 탭 네비게이터
- tapOn: "Settings"
- assertVisible: "Settings Screen"
```

## Flutter

### 기본 설정

```yaml
# Flutter 앱
appId: com.example.flutter_app
---
- launchApp
- tapOn: "Increment"
```

### Key 사용 (권장)

```dart
// Flutter 코드
ElevatedButton(
  key: Key('login_button'),
  onPressed: () {},
  child: Text('Login'),
)
```

```yaml
# Maestro 플로우
- tapOn:
    id: "login_button"
```

### Semantics Label

```dart
// Flutter 코드
Semantics(
  label: 'Submit button',
  child: ElevatedButton(...),
)
```

```yaml
# Maestro 플로우
- tapOn: "Submit button"
```

## 크로스 플랫폼 플로우

### 플랫폼 공통 플로우

```yaml
# common_login.yaml
# Android와 iOS 모두에서 동작
appId: ${PLATFORM_APP_ID}
---
- launchApp:
    clearState: true
- tapOn: "Sign In"
- inputText: ${USERNAME}
- tapOn: "Password"
- inputText: ${PASSWORD}
- tapOn: "Log In"
- assertVisible: "Welcome"
```

### 실행

```bash
# Android
maestro test -e PLATFORM_APP_ID=com.example.android flow.yaml

# iOS
maestro test -e PLATFORM_APP_ID=com.example.ios flow.yaml
```

### 플랫폼별 분기

```yaml
# 플랫폼에 따라 다른 동작
- runFlow:
    when:
      platform: "android"
    commands:
      - tapOn: "Android 전용 버튼"

- runFlow:
    when:
      platform: "ios"
    commands:
      - tapOn: "iOS 전용 버튼"
```

## 실제 사례: Now in Android 앱

```yaml
# recipes/nowinandroid/pick_interests.yaml
appId: com.google.samples.apps.nowinandroid.demo.debug
name: Pick Interests
---
- launchApp:
    clearState: true
- tapOn: Headlines
- tapOn: Testing
- tapOn: Done
- assertVisible: "For you"
```

## 실제 사례: 웹 쇼핑

```yaml
# recipes/web/shopping.yaml
url: https://amazon.com
---
- launchBrowser
- tapOn: "Search Amazon"
- inputText: "Wireless Headphones"
- pressKey: "Enter"
- assertVisible: "results"
- tapOn:
    text: ".*Sony.*"
    index: 0
- assertVisible: "Add to Cart"
```

## 문제 해결

### Android: 요소를 찾을 수 없음

```bash
# 요소 계층 구조 확인
maestro hierarchy

# Layout Inspector 사용 (Android Studio)
```

### iOS: 권한 다이얼로그

```yaml
# 시스템 다이얼로그 처리
- tapOn:
    text: "Allow"
    optional: true
```

### Web: 동적 콘텐츠

```yaml
# 로딩 완료 대기
- extendedWaitUntil:
    visible: "Content Loaded"
    timeout: 10000
```

## 다음 단계

다음 챕터에서는 고급 기능을 다룹니다.

---

**이전 글**: [핵심 명령어](/maestro-guide-04-commands/)

**다음 글**: [고급 기능](/maestro-guide-06-advanced/)
