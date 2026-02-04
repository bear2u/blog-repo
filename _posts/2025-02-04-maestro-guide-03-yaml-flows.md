---
layout: post
title: "Maestro 가이드 - YAML 플로우"
date: 2025-02-04
category: AI
tags: [maestro, yaml, flow, test-definition, syntax]
series: maestro-guide
part: 3
author: mobile-dev-inc
original_url: https://github.com/mobile-dev-inc/Maestro
---

## 플로우 기본 구조

Maestro 플로우는 YAML 파일로 작성됩니다. 헤더와 명령어 목록으로 구성됩니다.

```yaml
# 헤더 섹션 (메타데이터)
appId: com.example.app
name: Login Flow
tags:
  - smoke
  - login

---
# 명령어 섹션 (테스트 단계)
- launchApp
- tapOn: "Login"
- inputText: "user@example.com"
- assertVisible: "Welcome"
```

## 헤더 섹션

### 필수 필드

```yaml
# 앱 패키지 ID (Android) 또는 Bundle ID (iOS)
appId: com.example.app
```

### 선택 필드

```yaml
appId: com.example.app
name: "사용자 로그인 테스트"
tags:
  - smoke
  - regression
  - login
env:
  USERNAME: test@example.com
  PASSWORD: secret123
onFlowStart:
  - clearState
onFlowComplete:
  - stopApp
```

### 헤더 필드 설명

| 필드 | 설명 |
|------|------|
| `appId` | 테스트할 앱의 패키지 ID |
| `name` | 플로우 이름 (리포트용) |
| `tags` | 필터링용 태그 |
| `env` | 환경 변수 정의 |
| `onFlowStart` | 플로우 시작 시 실행할 명령 |
| `onFlowComplete` | 플로우 완료 시 실행할 명령 |

## 명령어 문법

### 단순 명령어

```yaml
# 파라미터 없는 명령어
- launchApp
- back
- hideKeyboard
```

### 단일 파라미터

```yaml
# 콜론 뒤에 값 직접 지정
- tapOn: "Login"
- inputText: "Hello World"
- assertVisible: "Welcome"
```

### 다중 파라미터

```yaml
# 중첩 구조로 여러 파라미터 지정
- tapOn:
    text: "Submit"
    index: 0
    retryTapIfNoChange: true

- inputText:
    text: "user@example.com"
    clearText: true
```

## 요소 선택자

### 텍스트로 선택

```yaml
# 정확한 텍스트 매칭
- tapOn: "Login"

# 정규식 매칭
- tapOn: ".*Login.*"
```

### ID로 선택

```yaml
# 리소스 ID (Android)
- tapOn:
    id: "com.example.app:id/login_button"

# Accessibility ID
- tapOn:
    id: "loginButton"
```

### 인덱스로 선택

```yaml
# 같은 텍스트의 여러 요소 중 선택
- tapOn:
    text: "Item"
    index: 2  # 세 번째 요소 (0부터 시작)
```

### 위치로 선택

```yaml
# 좌표로 탭
- tapOn:
    point: "50%,50%"  # 화면 중앙

- tapOn:
    point: "100,200"  # 절대 좌표
```

### 복합 선택자

```yaml
# 여러 조건 조합
- tapOn:
    text: "Submit"
    below: "Email"
    above: "Cancel"
```

## 앱 제어

### 앱 실행

```yaml
# 기본 실행
- launchApp

# 상태 초기화 후 실행
- launchApp:
    clearState: true

# 딥링크로 실행
- launchApp:
    arguments:
      url: "myapp://product/123"
```

### 앱 중지

```yaml
# 앱 중지
- stopApp

# 특정 앱 중지
- stopApp:
    appId: com.other.app
```

### 앱 전환

```yaml
# 다른 앱 실행
- launchApp:
    appId: com.android.settings

# 원래 앱으로 돌아가기
- launchApp:
    appId: ${appId}
```

## 입력 처리

### 텍스트 입력

```yaml
# 기본 입력
- inputText: "Hello World"

# 현재 텍스트 지우고 입력
- inputText:
    text: "new@email.com"
    clearText: true

# 랜덤 텍스트 생성
- inputRandomText:
    length: 10
```

### 키 입력

```yaml
# 키보드 키 입력
- pressKey: "Enter"
- pressKey: "Backspace"
- pressKey: "Tab"

# 여러 번 입력
- pressKey:
    key: "Backspace"
    repeat: 5
```

### 키보드 제어

```yaml
# 키보드 숨기기
- hideKeyboard

# 키보드가 열릴 때까지 대기
- waitForKeyboard
```

## 네비게이션

### 뒤로가기

```yaml
# 시스템 뒤로가기
- back
```

### 홈 화면

```yaml
# 홈 버튼
- pressKey: "Home"
```

### 스크롤

```yaml
# 아래로 스크롤
- scroll

# 방향 지정
- scroll:
    direction: "up"

# 요소가 보일 때까지 스크롤
- scrollUntilVisible:
    element:
      text: "마지막 항목"
```

### 스와이프

```yaml
# 기본 스와이프 (위로)
- swipe:
    direction: "up"

# 좌표로 스와이프
- swipe:
    start: "50%,80%"
    end: "50%,20%"
    duration: 500
```

## 어서션

### 가시성 확인

```yaml
# 요소가 보이는지 확인
- assertVisible: "Welcome"

# 요소가 보이지 않는지 확인
- assertNotVisible: "Error"
```

### 활성화 상태 확인

```yaml
# 요소가 활성화되어 있는지
- assertTrue:
    id: "submit_button"
    enabled: true
```

### 텍스트 내용 확인

```yaml
# 특정 텍스트 포함 확인
- assertVisible:
    text: ".*총.*\\$[0-9]+.*"
```

## 대기

### 고정 대기

```yaml
# 밀리초 단위 대기 (필요한 경우에만 사용)
- extendedWaitUntil:
    visible: "Loading Complete"
    timeout: 10000
```

### 조건부 대기

```yaml
# 요소가 나타날 때까지 대기
- waitForAnimationToEnd

# 특정 요소가 나타날 때까지
- extendedWaitUntil:
    visible:
      text: "Ready"
    timeout: 30000
```

## 주석

```yaml
# 한 줄 주석
- launchApp  # 인라인 주석

# 여러 줄 설명
# 이 플로우는 로그인 기능을 테스트합니다.
# 사전 조건: 테스트 계정이 존재해야 합니다.
- tapOn: "Login"
```

## 변수 사용

### 환경 변수

```yaml
appId: com.example.app
env:
  USERNAME: test@example.com
  PASSWORD: secret123
---
- inputText: ${USERNAME}
- inputText: ${PASSWORD}
```

### 외부 변수

```bash
# 실행 시 변수 전달
maestro test -e USERNAME=admin -e PASSWORD=admin123 flow.yaml
```

```yaml
# 플로우에서 사용
- inputText: ${USERNAME}
- inputText: ${PASSWORD}
```

## 예제: 완전한 로그인 플로우

```yaml
# flows/login.yaml
appId: com.example.app
name: User Login Flow
tags:
  - smoke
  - auth
env:
  TEST_EMAIL: test@example.com
  TEST_PASSWORD: password123

---
# 앱 시작 (새 상태로)
- launchApp:
    clearState: true

# 온보딩 스킵 (있는 경우)
- runFlow:
    when:
      visible: "Skip"
    commands:
      - tapOn: "Skip"

# 로그인 화면으로 이동
- tapOn: "Already have an account?"

# 이메일 입력
- tapOn:
    id: "email_input"
- inputText: ${TEST_EMAIL}

# 비밀번호 입력
- tapOn:
    id: "password_input"
- inputText: ${TEST_PASSWORD}

# 로그인 버튼 클릭
- tapOn: "Log In"

# 성공 확인
- assertVisible: "Welcome back"
```

## 다음 단계

다음 챕터에서는 핵심 명령어를 자세히 다룹니다.

---

**이전 글**: [설치 및 설정](/maestro-guide-02-installation/)

**다음 글**: [핵심 명령어](/maestro-guide-04-commands/)
