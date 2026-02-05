---
layout: post
title: "UI-TARS 완벽 가이드 (2) - 전체 아키텍처"
date: 2025-02-04
permalink: /ui-tars-guide-02-architecture/
author: ByteDance
categories: [AI]
tags: [UI-TARS, 아키텍처, 시스템 설계, MCP, 이벤트 스트림]
original_url: "https://github.com/bytedance/UI-TARS-desktop"
excerpt: "UI-TARS의 전체 시스템 아키텍처를 분석합니다. 모듈 간 관계, 데이터 흐름, 핵심 설계 패턴을 살펴봅니다."
---

## 전체 아키텍처 개요

UI-TARS는 **계층화된 모듈식 아키텍처**로 설계되어 있습니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │ UI-TARS Desktop │  │ Agent TARS CLI / Web UI         │  │
│  │   (Electron)    │  │                                 │  │
│  └────────┬────────┘  └────────────────┬────────────────┘  │
└───────────┼────────────────────────────┼────────────────────┘
            │                            │
┌───────────┼────────────────────────────┼────────────────────┐
│           │      Agent Layer           │                    │
│  ┌────────▼────────┐  ┌───────────────▼───────────────┐   │
│  │   GUI Agent     │  │        Agent TARS             │   │
│  │  (Action-based) │  │    (MCP Agent Extension)      │   │
│  └────────┬────────┘  └───────────────┬───────────────┘   │
└───────────┼────────────────────────────┼────────────────────┘
            │                            │
┌───────────┼────────────────────────────┼────────────────────┐
│           │     Framework Layer        │                    │
│  ┌────────▼────────────────────────────▼───────────────┐   │
│  │                    Tarko                             │   │
│  │  (Event Stream-based Meta Agent Framework)          │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │  Agent  │ │   LLM   │ │   MCP   │ │ Context │   │   │
│  │  │         │ │ Client  │ │  Agent  │ │Engineer │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
            │
┌───────────┼─────────────────────────────────────────────────┐
│           │     Infrastructure Layer                        │
│  ┌────────▼────────────────────────────────────────────┐   │
│  │              MCP Servers & Clients                   │   │
│  │  ┌────────┐ ┌────────┐ ┌──────────┐ ┌────────────┐  │   │
│  │  │Browser │ │  File  │ │ Commands │ │   Search   │  │   │
│  │  │ Server │ │ System │ │  Server  │ │   Server   │  │   │
│  │  └────────┘ └────────┘ └──────────┘ └────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
            │
┌───────────┼─────────────────────────────────────────────────┐
│           │       Execution Layer                           │
│  ┌────────▼────────────────────────────────────────────┐   │
│  │                   Operators                          │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────┐   │   │
│  │  │Browser │ │ NutJS  │ │  ADB   │ │    AIO     │   │   │
│  │  │Operator│ │Operator│ │Operator│ │  Sandbox   │   │   │
│  │  └────────┘ └────────┘ └────────┘ └────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 핵심 모듈 관계

### 1. Application Layer

사용자와 직접 상호작용하는 최상위 계층입니다.

```typescript
// UI-TARS Desktop (Electron)
apps/ui-tars/
├── src/main/          # 메인 프로세스
├── src/renderer/      # React UI
└── src/preload/       # IPC 브릿지

// Agent TARS CLI
multimodal/agent-tars/cli/
└── src/index.ts       # CLI 진입점
```

### 2. Agent Layer

실제 에이전트 로직을 구현하는 계층입니다.

```typescript
// GUI Agent - 시각 기반 UI 자동화
multimodal/gui-agent/
├── action-parser/     # LLM 출력 → 구조화된 액션
├── agent-sdk/         # GUIAgent 클래스
└── operators/         # 플랫폼별 실행 엔진

// Agent TARS - MCP 기반 범용 에이전트
multimodal/agent-tars/
├── interface/         # 타입 정의
├── core/              # AgentTARS 클래스
└── cli/               # CLI 구현
```

### 3. Framework Layer (Tarko)

모든 에이전트의 기반이 되는 메타 프레임워크입니다.

```typescript
multimodal/tarko/
├── agent/             # 기본 Agent 클래스
├── llm-client/        # 멀티 LLM 클라이언트
├── mcp-agent/         # MCP 통합 에이전트
├── context-engineer/  # 컨텍스트 처리
└── agent-ui/          # Web UI 컴포넌트
```

### 4. Infrastructure Layer

MCP 프로토콜 기반의 도구와 서비스입니다.

```typescript
packages/agent-infra/
├── mcp-servers/       # MCP 서버 구현
│   ├── browser/       # 브라우저 자동화
│   ├── filesystem/    # 파일 시스템
│   ├── commands/      # 명령어 실행
│   └── search/        # 웹 검색
├── mcp-client/        # MCP 클라이언트
└── browser/           # 브라우저 추상화
```

---

## 데이터 흐름

### 사용자 요청 처리 흐름

```
사용자 입력 (자연어)
    │
    ▼
┌─────────────────────────────────────┐
│         Application Layer           │
│  - 입력 검증                        │
│  - 세션 관리                        │
│  - UI 업데이트                      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│           Agent Layer               │
│  - 시스템 프롬프트 구성             │
│  - LLM 호출                         │
│  - 응답 파싱                        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│        Framework Layer (Tarko)      │
│  - 이벤트 스트림 관리               │
│  - 도구 호출 엔진                   │
│  - 컨텍스트 엔지니어링              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│       Infrastructure Layer          │
│  - MCP 서버 호출                    │
│  - 도구 실행                        │
│  - 결과 반환                        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│         Execution Layer             │
│  - 실제 동작 수행                   │
│  - 스크린샷 캡처                    │
│  - 상태 업데이트                    │
└─────────────────────────────────────┘
```

---

## 이벤트 스트림 시스템

Tarko의 핵심은 **이벤트 스트림 기반 아키텍처**입니다.

```typescript
// 이벤트 유형
enum AgentEventType {
  AGENT_START = 'agent_start',
  AGENT_END = 'agent_end',
  LLM_REQUEST = 'llm_request',
  LLM_RESPONSE = 'llm_response',
  TOOL_CALL_START = 'tool_call_start',
  TOOL_CALL_END = 'tool_call_end',
  ERROR = 'error',
  STREAM_CHUNK = 'stream_chunk'
}

// 이벤트 발행
eventStream.emit({
  type: AgentEventType.TOOL_CALL_START,
  toolName: 'browser_click',
  arguments: { x: 100, y: 200 }
});
```

### 이벤트 흐름 예시

```
┌──────────────────────────────────────────────────────────┐
│                    Agent Loop                             │
│                                                          │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐ │
│  │  START  │──▶│   LLM   │──▶│  TOOL   │──▶│   END   │ │
│  │         │   │ REQUEST │   │  CALL   │   │         │ │
│  └─────────┘   └────┬────┘   └────┬────┘   └─────────┘ │
│                     │             │                     │
│                     ▼             ▼                     │
│               ┌─────────┐   ┌─────────┐                │
│               │   LLM   │   │  TOOL   │                │
│               │RESPONSE │   │ RESULT  │                │
│               └─────────┘   └─────────┘                │
└──────────────────────────────────────────────────────────┘
```

---

## MCP (Model Context Protocol) 통합

### MCP 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Client                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │  - 서버 연결 관리                               │   │
│  │  - 도구 호출 라우팅                             │   │
│  │  - 리소스 접근                                  │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼────┐    ┌─────▼─────┐    ┌────▼────┐
   │  Stdio  │    │    SSE    │    │  HTTP   │
   │Transport│    │ Transport │    │Streaming│
   └────┬────┘    └─────┬─────┘    └────┬────┘
        │               │               │
   ┌────▼───────────────▼───────────────▼────┐
   │            MCP Servers                   │
   │  ┌──────┐ ┌────────┐ ┌────┐ ┌──────┐  │
   │  │Browse│ │Filesys │ │Cmd │ │Search│  │
   │  └──────┘ └────────┘ └────┘ └──────┘  │
   └──────────────────────────────────────────┘
```

### 지원 통신 방식

| 방식 | 설명 | 사용 사례 |
|------|------|----------|
| **Stdio** | 표준 입출력 | 로컬 프로세스 |
| **SSE** | Server-Sent Events | 실시간 스트리밍 |
| **HTTP Streaming** | HTTP 청크 전송 | 원격 서버 |
| **InMemory** | 프로세스 내 통신 | 같은 런타임 |

---

## 도구 호출 엔진

Tarko는 3가지 도구 호출 방식을 지원합니다.

### 1. Native Tool Call

```typescript
// 모델의 기본 함수 호출 기능 사용
const response = await llm.chat.completions.create({
  model: 'gpt-4o',
  messages: [...],
  tools: [{
    type: 'function',
    function: {
      name: 'browser_click',
      parameters: { /* JSON Schema */ }
    }
  }]
});
```

### 2. Prompt Engineering

```typescript
// 프롬프트 기반 도구 호출
const systemPrompt = `
Available tools:
- browser_click(x, y): Click at coordinates
- browser_type(text): Type text

Respond with: Action: tool_name(args)
`;
```

### 3. Structured Outputs

```typescript
// JSON Schema 기반 구조화된 출력
const response = await llm.chat.completions.create({
  model: 'gpt-4o',
  response_format: {
    type: 'json_schema',
    json_schema: toolCallSchema
  }
});
```

---

## 브라우저 제어 전략

Agent TARS는 3가지 브라우저 제어 전략을 지원합니다.

| 전략 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **DOM** | DOM 요소 기반 | 빠름, 정확함 | 복잡한 UI 어려움 |
| **Visual Grounding** | 스크린샷 기반 | 모든 UI 지원 | 느림, 토큰 소비 |
| **Hybrid** | DOM + Vision 조합 | 균형잡힌 성능 | 복잡한 로직 |

```typescript
// 전략 선택
const agent = new AgentTARS({
  browser: {
    control: 'hybrid'  // 'dom' | 'visual-grounding' | 'hybrid'
  }
});
```

---

## 핵심 설계 패턴

### 1. Strategy Pattern (전략 패턴)

브라우저 제어 전략에 적용되어 런타임에 전략을 교체할 수 있습니다.

### 2. Observer Pattern (관찰자 패턴)

이벤트 스트림 시스템에서 에이전트 상태 변화를 구독합니다.

### 3. Factory Pattern (팩토리 패턴)

Operator 생성 시 설정에 따라 적절한 구현체를 반환합니다.

### 4. Dependency Injection (의존성 주입)

커스텀 파서, 컨텍스트 처리기 등을 주입할 수 있습니다.

### 5. Template Method (템플릿 메서드)

Operator의 `initialize()`, `execute()` 등 생명주기 메서드를 정의합니다.

---

*다음 글에서는 UI-TARS Desktop Electron 앱을 상세히 분석합니다.*
