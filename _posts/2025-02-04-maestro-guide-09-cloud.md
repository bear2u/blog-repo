---
layout: post
title: "Maestro 가이드 - Maestro Cloud"
date: 2025-02-04
categories: [개발 도구, Maestro]
tags: [maestro, cloud, ci-cd, parallel-testing, scaling]
author: mobile-dev-inc
original_url: https://github.com/mobile-dev-inc/Maestro
---

## Maestro Cloud 소개

**Maestro Cloud**는 테스트를 클라우드에서 병렬로 실행하여 실행 시간을 최대 90% 단축시키는 서비스입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Maestro Cloud                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Local                         Cloud                       │
│   ─────                         ─────                       │
│   100 tests                     100 tests                   │
│   Sequential                    Parallel (50 devices)       │
│   ~2 hours                      ~3 minutes                  │
│                                                             │
│   Features:                                                 │
│   • 병렬 실행 (최대 수백 개 디바이스)                        │
│   • 결정론적 환경 (항상 동일한 결과)                         │
│   • 상세 리포트 및 스크린샷                                  │
│   • Slack/Teams 알림                                        │
│   • CI/CD 통합                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 시작하기

### 1. 계정 생성

[Maestro Cloud 가입](https://maestro.dev/cloud) - 7일 무료 체험

### 2. CLI 로그인

```bash
maestro login

# 브라우저가 열리고 인증
# 또는 API 키로 로그인
maestro login --api-key YOUR_API_KEY
```

### 3. 앱 업로드 및 테스트

```bash
# Android
maestro cloud --app app.apk flows/

# iOS
maestro cloud --app app.ipa flows/

# iOS 시뮬레이터 빌드
maestro cloud --app app.app flows/
```

## 명령어 옵션

### 기본 옵션

```bash
maestro cloud \
  --app app.apk \
  --name "Release 1.2.3 Tests" \
  --device-locale "ko_KR" \
  flows/
```

### 전체 옵션

```bash
maestro cloud \
  --app app.apk \                    # 앱 파일
  --name "Nightly Tests" \           # 실행 이름
  --device-locale "ko_KR" \          # 디바이스 언어
  --ios-version "17" \               # iOS 버전
  --android-api-level "34" \         # Android API 레벨
  --include-tags "smoke" \           # 포함할 태그
  --exclude-tags "slow" \            # 제외할 태그
  --env USERNAME=test \              # 환경 변수
  --env PASSWORD=secret \
  --async \                          # 비동기 실행
  flows/
```

### 태그 필터링

```yaml
# flows/login.yaml
appId: com.example.app
tags:
  - smoke
  - auth
---
- launchApp
```

```bash
# smoke 태그만 실행
maestro cloud --app app.apk --include-tags smoke flows/

# slow 태그 제외
maestro cloud --app app.apk --exclude-tags slow flows/
```

## CI/CD 통합

### GitHub Actions

```yaml
# .github/workflows/maestro.yml
name: Maestro Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build App
        run: ./gradlew assembleDebug

      - name: Install Maestro
        run: curl -fsSL "https://get.maestro.mobile.dev" | bash

      - name: Run Maestro Tests
        env:
          MAESTRO_CLOUD_API_KEY: ${{ secrets.MAESTRO_CLOUD_API_KEY }}
        run: |
          maestro cloud \
            --app app/build/outputs/apk/debug/app-debug.apk \
            --name "PR #${{ github.event.number }}" \
            flows/
```

### GitLab CI

```yaml
# .gitlab-ci.yml
maestro-tests:
  stage: test
  image: openjdk:17
  script:
    - curl -fsSL "https://get.maestro.mobile.dev" | bash
    - export PATH="$PATH:$HOME/.maestro/bin"
    - maestro cloud --app app.apk flows/
  variables:
    MAESTRO_CLOUD_API_KEY: $MAESTRO_CLOUD_API_KEY
```

### Bitrise

```yaml
# bitrise.yml
workflows:
  primary:
    steps:
      - script:
          title: Run Maestro Cloud
          inputs:
            - content: |
                curl -fsSL "https://get.maestro.mobile.dev" | bash
                export PATH="$PATH:$HOME/.maestro/bin"
                maestro cloud --app $BITRISE_APK_PATH flows/
```

### CircleCI

```yaml
# .circleci/config.yml
version: 2.1

jobs:
  test:
    docker:
      - image: cimg/openjdk:17.0
    steps:
      - checkout
      - run:
          name: Install Maestro
          command: curl -fsSL "https://get.maestro.mobile.dev" | bash
      - run:
          name: Run Tests
          command: |
            export PATH="$PATH:$HOME/.maestro/bin"
            maestro cloud --app app.apk flows/
```

## 리포트 및 결과

### 웹 대시보드

실행 완료 후 상세 리포트 URL 제공:

```
✅ Flow completed: login.yaml
✅ Flow completed: signup.yaml
❌ Flow failed: checkout.yaml

View results: https://cloud.maestro.dev/runs/abc123
```

### 리포트 내용

- **실행 요약**: 성공/실패 수, 총 시간
- **플로우별 결과**: 각 플로우의 상세 결과
- **스크린샷**: 각 단계별 스크린샷
- **비디오**: 테스트 실행 녹화
- **로그**: 상세 실행 로그

### 실패 분석

```
Flow: checkout.yaml
Status: ❌ Failed

Step 5: assertVisible: "Order Confirmed"
  └── Timeout: Element not found after 10000ms

Screenshot: [실패 시점 스크린샷]
Video: [전체 실행 비디오]
```

## 알림 설정

### Slack 통합

```bash
# 프로젝트 설정에서 Slack Webhook 추가
# https://cloud.maestro.dev/settings/notifications
```

알림 예시:
```
🎭 Maestro Cloud
━━━━━━━━━━━━━━━━━
Run: Release 1.2.3 Tests
Status: ✅ Passed (48/50)
Duration: 3m 24s
━━━━━━━━━━━━━━━━━
View Report →
```

### Email 알림

- 실패 시 알림
- 일일 요약
- 주간 리포트

## 병렬화 전략

### 자동 병렬화

```bash
# 100개 플로우를 자동으로 분배
maestro cloud --app app.apk flows/
# → 50개 디바이스에서 동시 실행
# → 기존 2시간 → 3분
```

### 샤딩

```bash
# 수동 샤딩 (대규모 테스트 스위트)
maestro cloud --app app.apk --shard-count 10 --shard-index 0 flows/
maestro cloud --app app.apk --shard-count 10 --shard-index 1 flows/
# ...
```

## 비용 및 가격

### 가격 모델

| 플랜 | 테스트/월 | 가격 |
|------|----------|------|
| **Free** | 100 | $0 |
| **Starter** | 1,000 | $99/월 |
| **Pro** | 10,000 | $499/월 |
| **Enterprise** | 무제한 | 문의 |

[가격 페이지](https://maestro.dev/pricing)에서 상세 확인

### 무료 체험

- 7일 무료 체험
- 신용카드 불필요
- 모든 기능 사용 가능

## 베스트 프랙티스

### 테스트 조직화

```
flows/
├── smoke/          # 빠른 기본 테스트
│   ├── login.yaml
│   └── home.yaml
├── regression/     # 전체 회귀 테스트
│   ├── checkout.yaml
│   └── settings.yaml
└── e2e/            # 엔드투엔드 시나리오
    └── full_purchase.yaml
```

### PR별 테스트

```bash
# PR에서는 smoke 테스트만
maestro cloud --app app.apk --include-tags smoke flows/

# 머지 후 전체 테스트
maestro cloud --app app.apk flows/
```

### 환경별 설정

```bash
# 스테이징
maestro cloud --app staging.apk \
  --env BASE_URL=https://staging.api.com \
  flows/

# 프로덕션
maestro cloud --app production.apk \
  --env BASE_URL=https://api.com \
  flows/
```

## 문제 해결

### 타임아웃

```bash
# 타임아웃 증가
maestro cloud --app app.apk --timeout 600000 flows/
```

### 로그 확인

```bash
# 상세 로그 출력
maestro cloud --app app.apk --debug flows/
```

## 다음 단계

다음 챕터에서는 내부 아키텍처와 MCP 통합을 다룹니다.

---

**이전 글**: [Maestro Studio](/maestro-guide-08-studio/)

**다음 글**: [아키텍처 & MCP](/maestro-guide-10-architecture/)
