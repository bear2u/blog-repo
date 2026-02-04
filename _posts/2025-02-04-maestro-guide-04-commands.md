---
layout: post
title: "Maestro 가이드 - 핵심 명령어"
date: 2025-02-04
category: AI
tags: [maestro, commands, tapOn, inputText, assertVisible]
series: maestro-guide
part: 4
author: mobile-dev-inc
original_url: https://github.com/mobile-dev-inc/Maestro
---

## 명령어 분류

```
┌─────────────────────────────────────────────────────────────┐
│                    Maestro Commands                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   🚀 App Control     │ 👆 Interaction   │ ✅ Assertion      │
│   ─────────────────  │ ───────────────  │ ───────────────   │
│   launchApp          │ tapOn            │ assertVisible     │
│   stopApp            │ doubleTapOn      │ assertNotVisible  │
│   clearState         │ longPressOn      │ assertTrue        │
│   killApp            │ inputText        │ assertFalse       │
│                      │ pressKey         │                   │
│   📱 Navigation      │ swipe            │ 📷 Capture        │
│   ─────────────────  │ scroll           │ ───────────────   │
│   back               │ hideKeyboard     │ takeScreenshot    │
│   scrollUntilVisible │                  │ startRecording    │
│   openLink           │                  │ stopRecording     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## tapOn - 탭 상호작용

### 기본 사용

```yaml
# 텍스트로 탭
- tapOn: "Login"

# ID로 탭
- tapOn:
    id: "login_button"
```

### 고급 옵션

```yaml
- tapOn:
    text: "Submit"
    index: 0                    # 여러 요소 중 선택
    retryTapIfNoChange: true    # 변화 없으면 재시도
    waitToSettleTimeoutMs: 3000 # 안정화 대기 시간
    point: "50%,50%"            # 좌표로 탭
```

### 상대 위치로 탭

```yaml
# "Email" 아래에 있는 텍스트 필드 탭
- tapOn:
    below: "Email"

# "Submit" 위에 있는 체크박스 탭
- tapOn:
    text: "I agree"
    above: "Submit"
```

### 탭 후 확인

```yaml
- tapOn:
    text: "Load More"
    retryTapIfNoChange: true
    # 화면 변화 없으면 자동 재시도
```

## doubleTapOn - 더블 탭

```yaml
# 더블 탭
- doubleTapOn: "Zoom In"

- doubleTapOn:
    id: "image_view"
```

## longPressOn - 길게 누르기

```yaml
# 길게 누르기
- longPressOn: "Item to Delete"

- longPressOn:
    text: "Hold Me"
    duration: 2000  # 2초
```

## inputText - 텍스트 입력

### 기본 사용

```yaml
# 현재 포커스된 필드에 입력
- inputText: "Hello World"
```

### 고급 옵션

```yaml
- inputText:
    text: "user@example.com"
    clearText: true  # 기존 텍스트 지우기
```

### 특수 문자 입력

```yaml
# 줄바꿈 포함
- inputText: "Line 1\nLine 2"

# 탭 문자
- inputText: "Column1\tColumn2"
```

### 랜덤 텍스트

```yaml
# 랜덤 이메일 생성
- inputRandomEmail

# 랜덤 숫자
- inputRandomNumber:
    length: 6

# 랜덤 텍스트
- inputRandomText:
    length: 10
```

## pressKey - 키 입력

### 지원 키 목록

```yaml
# 기본 키
- pressKey: "Enter"
- pressKey: "Backspace"
- pressKey: "Delete"
- pressKey: "Tab"

# 네비게이션 키
- pressKey: "Home"
- pressKey: "Back"

# 볼륨 키
- pressKey: "Volume Up"
- pressKey: "Volume Down"
```

### 반복 입력

```yaml
- pressKey:
    key: "Backspace"
    repeat: 10  # 10번 삭제
```

## swipe - 스와이프

### 방향으로 스와이프

```yaml
# 위로 스와이프
- swipe:
    direction: "up"

# 아래로 스와이프
- swipe:
    direction: "down"

# 좌우 스와이프
- swipe:
    direction: "left"
- swipe:
    direction: "right"
```

### 좌표로 스와이프

```yaml
# 시작점에서 끝점으로
- swipe:
    start: "50%,80%"
    end: "50%,20%"
    duration: 500  # 밀리초
```

### 요소 기준 스와이프

```yaml
# 특정 요소에서 스와이프
- swipe:
    from:
      id: "carousel"
    direction: "left"
```

## scroll - 스크롤

### 기본 스크롤

```yaml
# 기본 (아래로)
- scroll

# 방향 지정
- scroll:
    direction: "up"
```

### 요소까지 스크롤

```yaml
# 요소가 보일 때까지 스크롤
- scrollUntilVisible:
    element:
      text: "Footer"
    direction: "down"
    timeout: 30000
```

## assertVisible - 가시성 확인

### 기본 사용

```yaml
# 텍스트 확인
- assertVisible: "Welcome"

# ID로 확인
- assertVisible:
    id: "success_message"
```

### 정규식 매칭

```yaml
# 패턴 매칭
- assertVisible:
    text: "Order #[0-9]+"
```

### 부분 매칭

```yaml
# 포함 여부 확인
- assertVisible:
    text: ".*success.*"
```

## assertNotVisible - 비가시성 확인

```yaml
# 요소가 없는지 확인
- assertNotVisible: "Error"

# 로딩 완료 확인
- assertNotVisible: "Loading..."
```

## assertTrue / assertFalse

```yaml
# 조건 확인
- assertTrue:
    id: "checkbox"
    checked: true

- assertTrue:
    id: "submit_button"
    enabled: true

- assertFalse:
    id: "premium_badge"
    visible: true
```

## takeScreenshot - 스크린샷

```yaml
# 스크린샷 저장
- takeScreenshot: "login_screen"

# 경로 지정
- takeScreenshot:
    path: "./screenshots/step1.png"
```

## 녹화

```yaml
# 녹화 시작
- startRecording: "test_video"

# ... 테스트 명령어들 ...

# 녹화 중지
- stopRecording
```

## openLink - 딥링크/URL

```yaml
# 딥링크 열기
- openLink: "myapp://product/123"

# 웹 URL 열기
- openLink: "https://example.com/page"
```

## back - 뒤로가기

```yaml
# 시스템 뒤로가기
- back

# 여러 번 뒤로가기
- repeat:
    times: 3
    commands:
      - back
```

## hideKeyboard - 키보드 숨기기

```yaml
# 키보드 닫기
- hideKeyboard
```

## clearState - 상태 초기화

```yaml
# 앱 데이터 초기화
- clearState

# 특정 앱 초기화
- clearState:
    appId: com.example.app
```

## 조건부 실행

```yaml
# 요소가 보이면 실행
- runFlow:
    when:
      visible: "Skip Tutorial"
    commands:
      - tapOn: "Skip Tutorial"
```

## 대기

```yaml
# 요소가 나타날 때까지 대기
- extendedWaitUntil:
    visible: "Content Loaded"
    timeout: 10000

# 애니메이션 완료 대기
- waitForAnimationToEnd
```

## 명령어 조합 예제

### 로그인 플로우

```yaml
- launchApp:
    clearState: true
- tapOn: "Sign In"
- tapOn:
    id: "email_field"
- inputText: "user@example.com"
- tapOn:
    id: "password_field"
- inputText: "password123"
- hideKeyboard
- tapOn: "Log In"
- assertVisible: "Welcome back"
```

### 상품 구매 플로우

```yaml
- launchApp
- tapOn: "Shop"
- scrollUntilVisible:
    element:
      text: "Special Offer"
- tapOn: "Special Offer"
- tapOn: "Add to Cart"
- assertVisible: "Added to cart"
- tapOn: "Checkout"
- assertVisible: "Order Summary"
- takeScreenshot: "checkout_summary"
```

## 다음 단계

다음 챕터에서는 플랫폼별 테스트 방법을 다룹니다.

---

**이전 글**: [YAML 플로우](/maestro-guide-03-yaml-flows/)

**다음 글**: [플랫폼별 테스트](/maestro-guide-05-platforms/)
