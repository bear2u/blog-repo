---
layout: post
title: "Maestro 가이드 - AI 통합"
date: 2025-02-04
categories: [AI]
tags: [maestro, ai, gpt, assertWithAI, extractTextWithAI]
author: mobile-dev-inc
original_url: https://github.com/mobile-dev-inc/Maestro
---

## AI 기능 개요

Maestro는 GPT 기반 AI 기능을 제공하여 복잡한 UI 검증과 데이터 추출을 자연어로 수행할 수 있습니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Features                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   🤖 assertWithAI                                           │
│   └── 자연어로 UI 상태 검증                                  │
│       "모든 상품에 가격이 표시되어 있는지 확인"             │
│                                                             │
│   📝 extractTextWithAI                                      │
│   └── 화면에서 특정 정보 추출                               │
│       "주문 번호를 추출해줘"                                │
│                                                             │
│   🎨 MaestroGPT (Studio)                                    │
│   └── 테스트 작성 도우미                                    │
│       "로그인 플로우 테스트 만들어줘"                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 설정

### API 키 설정

```bash
# OpenAI API 키
export MAESTRO_CLI_AI_KEY=sk-...

# 또는 Anthropic API 키
export MAESTRO_CLI_AI_KEY=sk-ant-api-...
```

### 모델 선택

```bash
# 기본: gpt-4o
# 다른 모델 사용
export MAESTRO_CLI_AI_MODEL=gpt-4o-2024-08-06
```

## assertWithAI - AI 어서션

### 기본 사용

```yaml
# 자연어로 UI 검증
- assertWithAI:
    assertion: "로그인 버튼이 화면에 보인다"

- assertWithAI:
    assertion: "모든 상품에 가격이 표시되어 있다"
```

### 복잡한 검증

```yaml
# 레이아웃 검증
- assertWithAI:
    assertion: "네비게이션 바가 화면 상단에 있고 메뉴 아이콘이 왼쪽에 있다"

# 콘텐츠 검증
- assertWithAI:
    assertion: "검색 결과가 최소 5개 이상 표시되어 있다"

# 시각적 검증
- assertWithAI:
    assertion: "에러 메시지가 빨간색으로 표시되어 있다"
```

### 조건부 검증

```yaml
# 특정 상황에서만 AI 검증
- runFlow:
    when:
      visible: "Search Results"
    commands:
      - assertWithAI:
          assertion: "모든 검색 결과에 썸네일 이미지가 있다"
```

### 상세한 어서션

```yaml
# 여러 조건 한번에 검증
- assertWithAI:
    assertion: |
      다음 조건들을 모두 만족하는지 확인:
      1. 상품 목록이 표시되어 있다
      2. 각 상품에 이름, 가격, 이미지가 있다
      3. "장바구니에 담기" 버튼이 있다
      4. 가격은 원화(₩) 형식이다
```

## extractTextWithAI - AI 텍스트 추출

### 기본 사용

```yaml
# 특정 정보 추출
- extractTextWithAI: "주문 번호"
- inputText: ${aiOutput}  # 추출된 값 사용

# 가격 추출
- extractTextWithAI: "총 결제 금액 (숫자만)"
- evalScript: |
    output.totalPrice = parseInt(aiOutput);
```

### 복잡한 추출

```yaml
# 여러 정보 추출
- extractTextWithAI: "배송 예정일 (날짜 형식으로)"
- evalScript: |
    output.deliveryDate = aiOutput;

# 조건부 추출
- extractTextWithAI: "할인 전 원래 가격 (없으면 '없음')"
```

### 추출 후 사용

```yaml
# CAPTCHA 처리 예시
- extractTextWithAI: "CAPTCHA에 표시된 텍스트"
- tapOn: "Enter CAPTCHA"
- inputText: ${aiOutput}
- tapOn: "Submit"
```

### 실제 사례: 웹 쇼핑

```yaml
# recipes/web/xmas.yaml
url: https://amazon.com
---
- launchBrowser

# CAPTCHA 처리
- extractTextWithAI: CAPTCHA value
- tapOn: Type characters
- inputText: ${aiOutput}
- tapOn: Continue shopping

# 팝업 닫기
- tapOn: .*Dismiss.*

# 검색
- tapOn: "Search Amazon"
- inputText: "Ugly Christmas Sweater With Darth Vader"
- pressKey: "Enter"

# AI 검증
- assertWithAI:
    assertion: All sweaters have Darth Vader's mask on them

- assertWithAI:
    assertion: At least one result is Star Wars themed

# 가격 추출
- extractTextWithAI: Dollar price without cents and currency of the first item
- tapOn: ${aiOutput}

# 상품 페이지 검증
- assertWithAI:
    assertion: User is shown a product detail page that fits in the screen

# 장바구니 추가
- swipe:
    start: 50%,50%
    end: 20%,50%
- tapOn: "Add to Cart"
- tapOn: "Proceed to checkout"

# 로그인 요청 확인
- assertWithAI:
    assertion: User is asked to sign in
```

## MaestroGPT (Maestro Studio)

Maestro Studio에서 AI 어시스턴트를 사용하여 테스트를 작성할 수 있습니다.

### 사용 방법

```bash
# Maestro Studio 실행
maestro studio
```

### AI 어시스턴트 기능

1. **자연어로 명령 생성**
   - "로그인 버튼을 탭해줘" → `- tapOn: "Login"`

2. **플로우 자동 생성**
   - "회원가입 플로우 만들어줘" → 전체 플로우 생성

3. **에러 해결 도움**
   - "이 요소를 찾을 수 없어" → 대안 선택자 제안

## AI 기능 모범 사례

### 명확한 어서션 작성

```yaml
# 좋은 예 ✅
- assertWithAI:
    assertion: "화면에 '로그인 성공' 메시지가 표시되어 있다"

# 나쁜 예 ❌
- assertWithAI:
    assertion: "로그인이 됐다"
```

### 구체적인 추출 요청

```yaml
# 좋은 예 ✅
- extractTextWithAI: "주문 번호 (# 기호 제외, 숫자만)"

# 나쁜 예 ❌
- extractTextWithAI: "번호"
```

### AI와 전통적 어서션 조합

```yaml
# 기본 검증은 전통적 방식
- assertVisible: "Order Confirmation"

# 복잡한 검증만 AI 사용
- assertWithAI:
    assertion: "주문 요약에 배송 주소, 결제 수단, 총액이 모두 표시되어 있다"
```

## 비용 고려사항

AI 기능은 LLM API 호출을 사용하므로 비용이 발생합니다:

```yaml
# 비용 최적화: 필요한 경우에만 AI 사용
- assertVisible: "Login"  # 무료

# 복잡한 검증에만 AI 사용
- assertWithAI:           # API 비용 발생
    assertion: "폼 유효성 검사 에러 메시지가 적절히 표시된다"
```

## 제한사항

- 스크린샷 기반 분석 (실시간 화면 상태)
- 네트워크 지연 가능
- API 비용 발생
- 100% 정확도 보장 불가

## 디버깅

### AI 응답 확인

```bash
# 상세 로그 출력
maestro test flow.yaml --debug-output ./debug

# AI 프롬프트와 응답 확인
cat ./debug/ai_interactions.log
```

### 실패 시 스크린샷

```yaml
onFlowError:
  - takeScreenshot: "ai_assertion_failed"
```

## 다음 단계

다음 챕터에서는 Maestro Studio를 다룹니다.

---

**이전 글**: [고급 기능](/maestro-guide-06-advanced/)

**다음 글**: [Maestro Studio](/maestro-guide-08-studio/)
