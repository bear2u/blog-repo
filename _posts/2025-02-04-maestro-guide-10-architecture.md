---
layout: post
title: "Maestro 가이드 - 아키텍처 & MCP"
date: 2025-02-04
categories: [AI]
tags: [maestro, architecture, mcp, llm, internal]
author: mobile-dev-inc
original_url: https://github.com/mobile-dev-inc/Maestro
---

## Maestro 내부 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                  Maestro Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                    Maestro CLI                       │  │
│   │   App.kt, TestCommand, StudioCommand, etc.          │  │
│   └───────────────────────┬─────────────────────────────┘  │
│                           │                                 │
│   ┌───────────────────────▼─────────────────────────────┐  │
│   │                    Orchestra                         │  │
│   │   MaestroCommand → Maestro API 변환                  │  │
│   └───────────────────────┬─────────────────────────────┘  │
│                           │                                 │
│   ┌───────────────────────▼─────────────────────────────┐  │
│   │                     Maestro                          │  │
│   │   플랫폼 독립적 API (tapOn, inputText, etc.)        │  │
│   └───────────────────────┬─────────────────────────────┘  │
│                           │                                 │
│   ┌───────────────────────▼─────────────────────────────┐  │
│   │                     Driver                           │  │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐            │  │
│   │   │ Android │  │   iOS   │  │   Web   │            │  │
│   │   │ Driver  │  │ Driver  │  │ Driver  │            │  │
│   │   └─────────┘  └─────────┘  └─────────┘            │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 핵심 모듈

### 프로젝트 구조

```
maestro/
├── maestro-cli/           # CLI 진입점
│   └── command/           # 서브커맨드들
│       ├── TestCommand.kt
│       ├── StudioCommand.kt
│       └── McpCommand.kt
│
├── maestro-orchestra/     # 명령어 해석 및 실행
│   ├── Orchestra.kt       # 메인 오케스트레이터
│   └── yaml/              # YAML 파서
│
├── maestro-client/        # 플랫폼 독립 API
│   └── Maestro.kt         # 핵심 API
│
├── maestro-android/       # Android 드라이버
│   └── AndroidDriver.kt
│
├── maestro-ios/           # iOS 드라이버
│   └── IOSDriver.kt
│
├── maestro-ios-driver/    # iOS XCTest 러너
├── maestro-web/           # 웹 드라이버
├── maestro-ai/            # AI 기능
├── maestro-studio/        # Studio 웹 UI
└── maestro-proto/         # gRPC 프로토콜
```

### 계층 구조 설명

| 계층 | 역할 |
|------|------|
| **CLI** | 사용자 인터페이스, 커맨드 파싱 |
| **Orchestra** | YAML 명령어를 Maestro API로 변환 |
| **Maestro** | 플랫폼 독립적 자동화 API |
| **Driver** | 플랫폼별 구현 (Android/iOS/Web) |

## MCP (Model Context Protocol) 서버

Maestro는 LLM이 직접 디바이스를 제어할 수 있는 MCP 서버를 제공합니다.

### MCP 서버 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Integration                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐│
│   │     LLM      │◀──▶│  MCP Server  │◀──▶│   Device     ││
│   │  (Claude,    │STDIO│  (maestro    │    │  (Android,   ││
│   │   GPT, etc)  │    │    mcp)      │    │   iOS, Web)  ││
│   └──────────────┘    └──────────────┘    └──────────────┘│
│                                                             │
│   MCP Tools:                                                │
│   • 디바이스 목록 조회                                      │
│   • 앱 실행/중지                                            │
│   • UI 상호작용 (tap, type, swipe)                         │
│   • 플로우 실행 및 검증                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### MCP 서버 실행

```bash
# MCP 서버 시작
maestro mcp

# Claude Desktop 또는 다른 MCP 클라이언트에서 사용
```

### MCP 도구 목록

| 도구 | 설명 |
|------|------|
| `list_devices` | 연결된 디바이스 목록 |
| `launch_app` | 앱 실행 |
| `tap` | 요소 탭 |
| `type_text` | 텍스트 입력 |
| `swipe` | 스와이프 |
| `run_flow` | YAML 플로우 실행 |
| `check_syntax` | 플로우 문법 검증 |

### LLM에서 사용 예시

```
사용자: "앱을 실행하고 로그인 버튼을 탭해줘"

LLM → MCP:
1. launch_app(appId: "com.example.app")
2. tap(text: "Login")

MCP → 디바이스:
실제 앱 실행 및 버튼 탭
```

## Android 드라이버

### 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                   Android Driver                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   maestro-cli                                               │
│       │                                                     │
│       ▼                                                     │
│   AndroidDriver (maestro-android)                           │
│       │                                                     │
│       ├── ADB 연결                                          │
│       │                                                     │
│       ▼                                                     │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              Android Device/Emulator                 │  │
│   │  ┌─────────────────┐  ┌─────────────────┐          │  │
│   │  │ maestro-app.apk │  │maestro-server.apk│          │  │
│   │  │   (Host App)    │  │  (UIAutomator)   │          │  │
│   │  └─────────────────┘  └─────────────────┘          │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 필요한 아티팩트

- `maestro-app.apk`: 호스트 앱 (아무 것도 안 함)
- `maestro-server.apk`: UIAutomator 기반 HTTP 서버

### 빌드

```bash
./gradlew :maestro-android:assemble
./gradlew :maestro-android:assembleAndroidTest
```

## iOS 드라이버

### 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     iOS Driver                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   maestro-cli                                               │
│       │                                                     │
│       ▼                                                     │
│   IOSDriver (maestro-ios)                                   │
│       │                                                     │
│       ▼                                                     │
│   ┌─────────────────────────────────────────────────────┐  │
│   │            iOS Simulator/Device                      │  │
│   │  ┌─────────────────────────────────────────────┐    │  │
│   │  │    maestro-driver-iosUITests-Runner.app     │    │  │
│   │  │           (XCTest Runner)                    │    │  │
│   │  │      HTTP Server on port 22087              │    │  │
│   │  └─────────────────────────────────────────────┘    │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 필요한 아티팩트

- `maestro-driver-ios`: 호스트 앱
- `maestro-driver-iosUITests-Runner.app`: XCTest 러너
- `maestro-driver-ios-config.xctestrun`: 설정 파일

### 빌드

```bash
./maestro-ios-xctest-runner/build-maestro-ios-runner.sh
```

### 독립 실행 테스트

```bash
# XCTest 러너만 실행
./maestro-ios-xctest-runner/run-maestro-ios-runner.sh

# HTTP API 테스트
curl -fsSL -X GET localhost:22087/deviceInfo | jq
curl -fsSL -X POST localhost:22087/touch -d '{"x": 150, "y": 150, "duration": 0.2}'
```

## 새 명령어 추가하기

### 1. Command 정의

```kotlin
// Commands.kt
data class MyNewCommand(
    val param1: String,
    val param2: Int = 0
) : Command
```

### 2. MaestroCommand에 추가

```kotlin
// MaestroCommand.kt
data class MaestroCommand(
    // ... 기존 필드들
    val myNewCommand: MyNewCommand? = null
)
```

### 3. YAML 매핑

```kotlin
// YamlFluentCommand.kt
data class YamlFluentCommand(
    // ... 기존 필드들
    val myNewCommand: YamlMyNewCommand? = null
)
```

### 4. Orchestra 처리

```kotlin
// Orchestra.kt
when (command) {
    is MyNewCommand -> {
        maestro.performMyNewAction(command.param1, command.param2)
    }
}
```

### 5. 테스트 추가

```kotlin
// IntegrationTest.kt
@Test
fun `test myNewCommand`() {
    // ...
}
```

## 테스트 전략

### 테스트 유형

```
├── Unit Tests
│   └── ./gradlew test
│
├── Integration Tests
│   └── ./gradlew :maestro-test:test
│   └── FakeDriver 사용
│
└── Manual Tests
    └── ./maestro test flow.yaml
```

### FakeDriver

```kotlin
// 실제 디바이스 없이 테스트
class FakeDriver : Driver {
    override fun tap(x: Int, y: Int) {
        // 시뮬레이션
    }
}
```

## 기여 가이드

### Type A: 간단한 수정

```bash
# 버그 수정, 오타 수정
# → 직접 PR 생성
```

### Type B: 기능 추가

```bash
# 새 기능, 대규모 리팩토링
# → 먼저 이슈 생성하여 논의
```

### 빌드 및 테스트

```bash
# CLI 빌드
./gradlew :maestro-cli:installDist

# 테스트
./gradlew test

# 로컬 CLI 사용
./maestro-cli/build/install/maestro/bin/maestro test flow.yaml
```

## 리소스

- **GitHub**: [github.com/mobile-dev-inc/Maestro](https://github.com/mobile-dev-inc/Maestro)
- **공식 문서**: [docs.maestro.dev](https://docs.maestro.dev)
- **Slack 커뮤니티**: [가입](https://maestrodev.typeform.com/to/FelIEe8A)
- **라이선스**: Apache 2.0

---

**이전 글**: [Maestro Cloud](/maestro-guide-09-cloud/)

**시리즈 처음으로**: [소개](/maestro-guide-01-intro/)
