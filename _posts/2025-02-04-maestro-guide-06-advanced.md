---
layout: post
title: "Maestro 가이드 - 고급 기능"
date: 2025-02-04
categories: [AI]
tags: [maestro, advanced, conditions, loops, subflows, variables]
author: mobile-dev-inc
original_url: https://github.com/mobile-dev-inc/Maestro
---

## 변수 시스템

### 환경 변수 정의

```yaml
appId: com.example.app
env:
  USERNAME: test@example.com
  PASSWORD: secret123
  BASE_URL: https://api.example.com
---
- launchApp
- inputText: ${USERNAME}
```

### 외부 변수 전달

```bash
# CLI에서 변수 전달
maestro test -e USERNAME=admin -e PASSWORD=admin123 flow.yaml

# 여러 변수
maestro test \
  -e USERNAME=admin \
  -e PASSWORD=admin123 \
  -e ENV=staging \
  flow.yaml
```

### 출력 변수 캡처

```yaml
# 화면에서 텍스트 추출
- copyTextFrom:
    id: "order_number"
- assertVisible: ${maestro.copiedText}

# JavaScript로 변수 조작
- evalScript: |
    const orderNum = maestro.copiedText;
    output.orderNumber = orderNum.replace('Order #', '');
```

## 조건부 실행

### runFlow when 조건

```yaml
# 요소가 보이면 실행
- runFlow:
    when:
      visible: "Skip Tutorial"
    commands:
      - tapOn: "Skip Tutorial"

# 요소가 없으면 실행
- runFlow:
    when:
      notVisible: "Welcome Back"
    commands:
      - tapOn: "Sign Up"
```

### 복합 조건

```yaml
# AND 조건
- runFlow:
    when:
      visible: "Login"
      platform: "android"
    commands:
      - tapOn: "Login with Google"

# 플랫폼별 분기
- runFlow:
    when:
      platform: "ios"
    commands:
      - tapOn: "Sign in with Apple"
```

### optional 명령어

```yaml
# 실패해도 계속 진행
- tapOn:
    text: "Dismiss"
    optional: true

# 권한 다이얼로그 (나타나지 않을 수도 있음)
- tapOn:
    text: "Allow"
    optional: true
```

## 반복문

### repeat 기본

```yaml
# 고정 횟수 반복
- repeat:
    times: 5
    commands:
      - tapOn: "Next"
```

### 조건부 반복

```yaml
# 조건이 충족될 때까지 반복
- repeat:
    while:
      visible: "Load More"
    commands:
      - tapOn: "Load More"
      - scroll
```

### 인덱스 사용

```yaml
# 반복 인덱스 활용
- repeat:
    times: 3
    commands:
      - tapOn: "Item ${index}"  # Item 0, Item 1, Item 2
```

## 서브플로우

### 서브플로우 정의

```yaml
# subflows/login.yaml
appId: com.example.app
---
- tapOn: "Login"
- inputText: ${USERNAME}
- tapOn: "Password"
- inputText: ${PASSWORD}
- tapOn: "Submit"
- assertVisible: "Dashboard"
```

### 서브플로우 호출

```yaml
# main_flow.yaml
appId: com.example.app
env:
  USERNAME: test@example.com
  PASSWORD: secret123
---
- launchApp
- runFlow: subflows/login.yaml
- tapOn: "Settings"
```

### 조건부 서브플로우

```yaml
# 로그인 상태에 따라 분기
- runFlow:
    when:
      notVisible: "Dashboard"
    file: subflows/login.yaml
```

### 파라미터 전달

```yaml
# 서브플로우 호출 시 변수 전달
- runFlow:
    file: subflows/add_item.yaml
    env:
      ITEM_NAME: "Product A"
      QUANTITY: "3"
```

## JavaScript 스크립트

### evalScript 기본

```yaml
# JavaScript 실행
- evalScript: |
    console.log('Current state:', JSON.stringify(maestro));
```

### 변수 조작

```yaml
- evalScript: |
    // 랜덤 이메일 생성
    const random = Math.random().toString(36).substring(7);
    output.randomEmail = `test_${random}@example.com`;

- inputText: ${output.randomEmail}
```

### 날짜/시간 처리

```yaml
- evalScript: |
    const now = new Date();
    output.today = now.toISOString().split('T')[0];
    output.timestamp = now.getTime();

- assertVisible: ${output.today}
```

### 조건부 로직

```yaml
- evalScript: |
    const hour = new Date().getHours();
    output.greeting = hour < 12 ? 'Good morning' : 'Good afternoon';

- assertVisible: ${output.greeting}
```

## 외부 파일 사용

### 테스트 데이터 파일

```yaml
# data.yaml 참조
appId: com.example.app
---
- runFlow:
    file: flows/checkout.yaml
    env:
      CARD_NUMBER: "4242424242424242"
```

### JSON 데이터

```yaml
- evalScript: |
    const testData = {
      users: [
        { name: "Alice", email: "alice@example.com" },
        { name: "Bob", email: "bob@example.com" }
      ]
    };
    output.firstUser = testData.users[0].email;

- inputText: ${output.firstUser}
```

## 에러 처리

### onFlowError

```yaml
appId: com.example.app
onFlowError:
  - takeScreenshot: "error_state"
  - evalScript: |
      console.log('Error occurred at:', new Date().toISOString());
---
- launchApp
- tapOn: "Might Fail"
```

### 재시도 로직

```yaml
# 최대 3번 재시도
- repeat:
    times: 3
    commands:
      - tapOn:
          text: "Flaky Button"
          optional: true
      - runFlow:
          when:
            visible: "Success"
          commands:
            - evalScript: |
                output.breakLoop = true
```

## 대기 전략

### 스마트 대기 (기본)

```yaml
# Maestro가 자동으로 요소 대기
- tapOn: "Dynamic Content"
```

### 명시적 대기

```yaml
# 특정 요소 대기
- extendedWaitUntil:
    visible: "Loading Complete"
    timeout: 30000

# 요소 사라질 때까지 대기
- extendedWaitUntil:
    notVisible: "Spinner"
    timeout: 10000
```

### 애니메이션 대기

```yaml
# 애니메이션 완료 대기
- waitForAnimationToEnd
```

## 복잡한 예제

### E2E 쇼핑 플로우

```yaml
appId: com.example.shop
name: Complete Purchase Flow
env:
  PRODUCT: "Wireless Headphones"
  CARD_NUMBER: "4242424242424242"

onFlowError:
  - takeScreenshot: "error"

---
# 로그인 (필요한 경우)
- launchApp
- runFlow:
    when:
      visible: "Sign In"
    file: subflows/login.yaml

# 상품 검색
- tapOn: "Search"
- inputText: ${PRODUCT}
- pressKey: "Enter"

# 결과 대기
- extendedWaitUntil:
    visible: "results"
    timeout: 10000

# 첫 번째 상품 선택
- tapOn:
    text: ".*${PRODUCT}.*"
    index: 0

# 장바구니 추가
- tapOn: "Add to Cart"
- assertVisible: "Added"

# 결제 진행
- tapOn: "Checkout"

# 결제 정보 입력
- tapOn:
    id: "card_number"
- inputText: ${CARD_NUMBER}

# 주문 완료
- tapOn: "Place Order"

# 확인
- extendedWaitUntil:
    visible: "Order Confirmed"
    timeout: 30000

# 주문 번호 캡처
- copyTextFrom:
    id: "order_number"
- evalScript: |
    console.log('Order completed:', maestro.copiedText);

- takeScreenshot: "order_confirmation"
```

## 다음 단계

다음 챕터에서는 AI 통합 기능을 다룹니다.

---

**이전 글**: [플랫폼별 테스트](/maestro-guide-05-platforms/)

**다음 글**: [AI 통합](/maestro-guide-07-ai/)
