---
layout: post
title: "Goose 완벽 가이드 (03) - 아키텍처 분석"
date: 2026-02-11
permalink: /goose-guide-03-architecture/
author: Block
categories: [AI 에이전트, 개발 도구]
tags: [Goose, Architecture, Rust, Workspace, Crates]
original_url: "https://github.com/block/goose"
excerpt: "Goose의 내부 아키텍처와 Workspace 구조 심층 분석"
---

## 아키텍처 개요

Goose는 Rust의 **Workspace** 시스템을 활용한 모듈화된 아키텍처를 가지고 있습니다.

```
┌────────────────────────────────────────────────────────────────┐
│                      Goose Ecosystem                            │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────┐              ┌─────────────────┐        │
│   │  Desktop UI     │              │   CLI           │        │
│   │  (Electron)     │              │   (Terminal)    │        │
│   │  ui/desktop/    │              │   goose-cli     │        │
│   └────────┬────────┘              └────────┬────────┘        │
│            │                                │                  │
│            └───────────┬────────────────────┘                  │
│                        │                                       │
│             ┌──────────▼──────────┐                            │
│             │  Goose Server       │                            │
│             │  (Backend API)      │                            │
│             │  goose-server       │                            │
│             └──────────┬──────────┘                            │
│                        │                                       │
│             ┌──────────▼──────────┐                            │
│             │  Goose Core         │                            │
│             │  (Agent Engine)     │                            │
│             │  goose              │                            │
│             └──────────┬──────────┘                            │
│                        │                                       │
│      ┌─────────────────┼─────────────────┐                    │
│      │                 │                 │                    │
│  ┌───▼────┐      ┌────▼─────┐     ┌────▼─────┐              │
│  │ LLM    │      │ MCP      │     │ ACP      │              │
│  │Provider│      │ Servers  │     │ Protocol │              │
│  └────────┘      │goose-mcp │     │goose-acp │              │
│                  └──────────┘     └──────────┘              │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Workspace 구조

Goose는 Cargo Workspace로 구성된 여러 crate들로 이루어져 있습니다.

### Cargo.toml (Workspace Root)

```toml
[workspace]
members = ["crates/*"]
resolver = "2"

[workspace.package]
edition = "2021"
version = "1.23.0"
authors = ["Block <ai-oss-tools@block.xyz>"]
license = "Apache-2.0"
repository = "https://github.com/block/goose"
description = "An AI agent"
```

---

## 핵심 Crate 분석

### 1. goose (Core)

**위치:** `crates/goose/`

**역할:** 에이전트의 핵심 로직

```
crates/goose/
├── src/
│   ├── agents/           # 에이전트 구현
│   │   └── agent.rs      # 메인 에이전트
│   ├── providers/        # LLM 제공자
│   │   ├── base.rs       # Provider trait
│   │   ├── anthropic.rs
│   │   ├── openai.rs
│   │   └── ...
│   ├── tools/            # 내장 도구
│   └── lib.rs
├── tests/                # 통합 테스트
└── Cargo.toml
```

**주요 기능:**
- 에이전트 실행 엔진
- LLM 제공자 추상화
- 도구 실행 시스템
- 메시지 처리

### 2. goose-cli

**위치:** `crates/goose-cli/`

**역할:** 커맨드 라인 인터페이스

```
crates/goose-cli/
├── src/
│   ├── main.rs           # CLI 진입점
│   ├── commands/         # 명령어 구현
│   │   ├── session.rs
│   │   ├── configure.rs
│   │   └── web.rs
│   └── ui/               # TUI 컴포넌트
├── static/               # 웹 UI 정적 파일
└── Cargo.toml
```

**주요 기능:**
- 명령어 파싱 (clap)
- 세션 관리
- 설정 인터페이스
- 웹 서버 (CLI web 모드)

### 3. goose-server

**위치:** `crates/goose-server/`

**역할:** Desktop 앱용 백엔드 API

```
crates/goose-server/
├── src/
│   ├── main.rs           # 서버 진입점
│   ├── routes/           # API 엔드포인트
│   ├── models/           # 데이터 모델
│   └── middleware/       # 미들웨어
├── ui/                   # OpenAPI 정의
└── Cargo.toml
```

**주요 기능:**
- REST API (Axum)
- WebSocket 통신
- 세션 상태 관리
- OpenAPI 스펙 생성

### 4. goose-mcp

**위치:** `crates/goose-mcp/`

**역할:** Model Context Protocol 구현

```
crates/goose-mcp/
├── src/
│   ├── developer/        # 개발자 도구
│   │   └── tools/
│   │       ├── shell.rs
│   │       ├── read.rs
│   │       └── write.rs
│   ├── computer/         # 컴퓨터 제어
│   └── lib.rs
├── examples/
│   └── mcp.rs            # MCP 서버 예제
└── Cargo.toml
```

**주요 기능:**
- MCP 서버 구현
- 도구 레지스트리
- 확장 시스템

### 5. goose-acp

**위치:** `crates/goose-acp/`

**역할:** Agent Communication Protocol

```
crates/goose-acp/
├── src/
│   ├── protocol.rs       # 프로토콜 정의
│   └── client.rs         # 클라이언트
└── Cargo.toml
```

**주요 기능:**
- 에이전트 간 통신
- 메시지 프로토콜
- 상태 동기화

### 6. goose-test & goose-test-support

**위치:** `crates/goose-test/`, `crates/goose-test-support/`

**역할:** 테스트 유틸리티

**주요 기능:**
- 테스트 헬퍼
- Mock 제공자
- 통합 테스트 지원

---

## Desktop UI

**위치:** `ui/desktop/`

**기술 스택:**
- **Electron**: 데스크톱 앱 프레임워크
- **React**: UI 라이브러리
- **TypeScript**: 타입 안전성
- **Vite**: 빌드 도구
- **Shadcn UI**: 컴포넌트 라이브러리

```
ui/desktop/
├── src/
│   ├── main.ts           # Electron 메인
│   ├── preload.ts        # Preload 스크립트
│   ├── renderer/         # React 앱
│   │   ├── App.tsx
│   │   ├── components/
│   │   └── hooks/
│   └── lib/
├── openapi.json          # API 스펙 (자동 생성)
├── package.json
└── forge.config.ts       # Electron Forge 설정
```

---

## 핵심 의존성

### Workspace 공통 의존성

```toml
[workspace.dependencies]
# MCP 프로토콜
rmcp = { version = "0.14.0", features = ["schemars", "auth"] }

# 비동기 런타임
tokio = { version = "1.49", features = ["full"] }

# 에러 핸들링
anyhow = "1.0"

# 직렬화
serde_json = "1.0"

# HTTP 클라이언트
reqwest = { version = "0.12.28", features = ["multipart"] }

# 서비스 추상화
tower = "0.5.2"
tower-http = "0.6.8"
```

---

## 데이터 흐름

### 1. 사용자 요청 → 응답

```
┌─────────────┐
│   User      │
│   Input     │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│ CLI/Desktop │────▶│ Goose Server │
│  Interface  │     │   (HTTP)     │
└─────────────┘     └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Goose Core  │
                    │   (Agent)    │
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
     ┌──────────┐   ┌──────────┐  ┌──────────┐
     │   LLM    │   │   MCP    │  │  Tools   │
     │ Provider │   │ Servers  │  │  System  │
     └──────────┘   └──────────┘  └──────────┘
```

### 2. 에이전트 실행 루프

```rust
// 의사 코드
loop {
    // 1. 사용자 메시지 수신
    let user_message = receive_message();

    // 2. LLM에 전송
    let llm_response = provider.complete(user_message);

    // 3. 도구 호출 확인
    if llm_response.has_tool_calls() {
        // 4. 도구 실행
        let tool_results = execute_tools(llm_response.tool_calls);

        // 5. 결과를 LLM에 다시 전송
        llm_response = provider.complete_with_tools(tool_results);
    }

    // 6. 최종 응답 반환
    send_response(llm_response);
}
```

---

## Provider 시스템

### Provider Trait

```rust
// crates/goose/src/providers/base.rs (개념적 구조)
pub trait Provider {
    async fn complete(
        &self,
        messages: Vec<Message>,
    ) -> anyhow::Result<Response>;

    async fn stream(
        &self,
        messages: Vec<Message>,
    ) -> anyhow::Result<Stream<Response>>;

    fn model(&self) -> &str;
}
```

### 지원 Provider

```
providers/
├── anthropic.rs      # Anthropic Claude
├── openai.rs         # OpenAI GPT
├── azure.rs          # Azure OpenAI
├── bedrock.rs        # Amazon Bedrock
├── gemini.rs         # Google Gemini
├── tetrate.rs        # Tetrate Router
└── github.rs         # GitHub Copilot
```

---

## MCP 아키텍처

### MCP 서버 구조

```
┌────────────────────────────────────────────┐
│           MCP Server                        │
├────────────────────────────────────────────┤
│                                             │
│  ┌─────────────────────────────────────┐  │
│  │       Tool Registry                  │  │
│  │  - shell                            │  │
│  │  - read_file                        │  │
│  │  - write_file                       │  │
│  │  - list_directory                   │  │
│  └─────────────────────────────────────┘  │
│                                             │
│  ┌─────────────────────────────────────┐  │
│  │     Resource Provider               │  │
│  │  - file://                          │  │
│  │  - http://                          │  │
│  └─────────────────────────────────────┘  │
│                                             │
│  ┌─────────────────────────────────────┐  │
│  │     Prompt Templates                │  │
│  │  - coding_assistant                 │  │
│  │  - debugger                         │  │
│  └─────────────────────────────────────┘  │
│                                             │
└────────────────────────────────────────────┘
```

---

## 설정 시스템

### 설정 파일 구조

```yaml
# config.yaml
version: "1.0"
default_provider: "anthropic"
log_level: "info"

# providers.yaml
providers:
  anthropic:
    api_key: "sk-ant-..."
    model: "claude-sonnet-4-5"

  openai:
    api_key: "sk-..."
    model: "gpt-5"

# extensions.yaml
extensions:
  developer:
    enabled: true
    allow_shell: true
    allow_file_ops: true

  computer_controller:
    enabled: true
    timeout: 300
```

---

## 빌드 시스템

### 개발 빌드

```bash
# Hermit 환경 활성화
source bin/activate-hermit

# 디버그 빌드
cargo build

# 특정 crate만 빌드
cargo build -p goose-cli
```

### 릴리스 빌드

```bash
# 릴리스 빌드
cargo build --release

# OpenAPI 포함 빌드
just release-binary
```

### Just 명령어

```bash
# OpenAPI 생성
just generate-openapi

# UI 실행
just run-ui

# MCP 테스트 기록
just record-mcp-tests
```

---

## 테스트 전략

### 단위 테스트

```bash
# 전체 테스트
cargo test

# 특정 crate 테스트
cargo test -p goose

# 특정 테스트
cargo test --package goose --test mcp_integration_test
```

### 통합 테스트

```
crates/goose/tests/
├── agent_test.rs
├── mcp_integration_test.rs
└── provider_test.rs
```

### Self-Test

```bash
# goose-self-test.yaml 실행
goose run --recipe goose-self-test.yaml
```

---

## 개발 워크플로우

### 1. 환경 설정

```bash
# Hermit 활성화
source bin/activate-hermit

# 의존성 설치
cargo build
```

### 2. 코드 작성

```bash
# 파일 수정 후
cargo fmt              # 포맷팅
cargo clippy           # Lint
cargo test -p <crate>  # 테스트
```

### 3. 서버 변경 시

```bash
# OpenAPI 재생성
just generate-openapi

# 빌드 및 테스트
cargo build
cargo test
```

---

## 다음 단계

아키텍처를 이해했다면, 다음 장에서는 핵심 에이전트 시스템을 상세히 살펴봅니다.

*다음 글에서는 Goose의 코어 에이전트 시스템과 실행 엔진을 분석합니다.*
