---
layout: post
title: "OpenClaw 완벽 가이드 (2) - Gateway 아키텍처"
date: 2025-02-04
permalink: /openclaw-guide-02-architecture/
author: Peter Steinberger
category: AI
tags: [OpenClaw, Gateway, WebSocket, Architecture, Protocol]
series: openclaw-guide
part: 2
original_url: "https://github.com/openclaw/openclaw"
excerpt: "OpenClaw Gateway의 WebSocket 아키텍처와 프로토콜을 분석합니다."
---

## Gateway 개요

**Gateway**는 OpenClaw의 핵심입니다. 모든 메시징 채널, 클라이언트, 에이전트를 연결하는 **단일 컨트롤 플레인**입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Gateway Architecture                          │
│                                                                  │
│   WhatsApp ──┐                          ┌── macOS App           │
│   Telegram ──┤                          │                       │
│   Discord ───┼── Gateway ───────────────┼── CLI                 │
│   Slack ─────┤   ws://127.0.0.1:18789   │                       │
│   Signal ────┘                          ├── WebChat UI          │
│                     │                   │                       │
│                     │                   └── iOS/Android Nodes   │
│                     ▼                                           │
│              Pi Agent (RPC)                                     │
│                     │                                           │
│                     ▼                                           │
│              Claude / GPT / Gemini                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 컴포넌트 구조

### 1. Gateway (데몬)

Gateway는 다음 역할을 수행합니다:

| 역할 | 설명 |
|------|------|
| **Provider 연결** | WhatsApp, Telegram 등 채널 관리 |
| **WS API** | 타입화된 WebSocket API 노출 |
| **이벤트 발행** | agent, chat, presence, health 이벤트 |
| **세션 관리** | 대화 세션 상태 유지 |

```typescript
// Gateway 이벤트 타입
type GatewayEvent =
  | "agent"      // 에이전트 응답
  | "chat"       // 채팅 메시지
  | "presence"   // 온라인 상태
  | "health"     // 헬스 체크
  | "heartbeat"  // 연결 유지
  | "cron"       // 예약 작업
  | "shutdown";  // 종료 신호
```

### 2. 클라이언트

클라이언트는 Gateway에 WebSocket으로 연결합니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Client Types                                │
│                                                                  │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│   │  macOS App  │   │     CLI     │   │   WebChat   │          │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘          │
│          │                 │                 │                  │
│          └─────────────────┼─────────────────┘                  │
│                            │                                    │
│                            ▼                                    │
│                    WebSocket 연결                               │
│                    (하나의 WS per 클라이언트)                   │
│                            │                                    │
│                            ▼                                    │
│   Requests: health, status, send, agent, system-presence       │
│   Events: tick, agent, presence, shutdown                       │
└─────────────────────────────────────────────────────────────────┘
```

### 3. 노드 (Nodes)

노드는 디바이스 기능을 Gateway에 노출합니다:

```typescript
// 노드 연결 시 role: "node" 지정
{
  type: "req",
  method: "connect",
  params: {
    role: "node",
    deviceId: "macbook-pro-m3",
    caps: ["canvas", "camera", "screen", "location"],
    commands: ["system.run", "system.notify", "camera.snap"]
  }
}
```

**노드가 제공하는 기능:**

| 명령어 | 설명 |
|--------|------|
| `canvas.*` | 캔버스 제어 |
| `camera.snap` | 카메라 스냅샷 |
| `camera.clip` | 카메라 클립 녹화 |
| `screen.record` | 화면 녹화 |
| `location.get` | 위치 정보 |
| `system.run` | 시스템 명령 실행 |
| `system.notify` | 알림 전송 |

---

## WebSocket 프로토콜

### 연결 수명주기

```
Client                    Gateway
  │                          │
  │──── req:connect ────────▶│
  │◀────── res (ok) ─────────│   (hello-ok: presence + health 포함)
  │                          │
  │◀───── event:presence ────│
  │◀───── event:tick ────────│
  │                          │
  │──── req:agent ──────────▶│
  │◀────── res:agent ────────│   (ack: {runId, status:"accepted"})
  │◀───── event:agent ───────│   (streaming)
  │◀────── res:agent ────────│   (final: {runId, status, summary})
  │                          │
```

### 프레임 형식

**요청 (Request)**
```json
{
  "type": "req",
  "id": "unique-request-id",
  "method": "agent",
  "params": {
    "message": "안녕하세요",
    "sessionKey": "main"
  }
}
```

**응답 (Response)**
```json
{
  "type": "res",
  "id": "unique-request-id",
  "ok": true,
  "payload": {
    "runId": "run-123",
    "status": "completed",
    "summary": "인사에 응답했습니다."
  }
}
```

**이벤트 (Event)**
```json
{
  "type": "event",
  "event": "agent",
  "payload": {
    "runId": "run-123",
    "chunk": "안녕하세요! 무엇을 도와드릴까요?"
  },
  "seq": 42,
  "stateVersion": 7
}
```

---

## 인증 및 페어링

### 디바이스 페어링

모든 WS 클라이언트는 **디바이스 ID**를 제공해야 합니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pairing Flow                                  │
│                                                                  │
│   1. 클라이언트 → connect (deviceId 포함)                       │
│   2. Gateway: 새 디바이스 ID? → 페어링 코드 발급                │
│   3. 사용자: openclaw pairing approve <channel> <code>          │
│   4. Gateway: 디바이스 토큰 발급                                │
│   5. 이후 연결 시 토큰으로 자동 인증                            │
└─────────────────────────────────────────────────────────────────┘
```

### 로컬 vs 원격 연결

| 연결 타입 | 인증 방식 |
|-----------|-----------|
| **로컬 (loopback)** | 자동 승인 가능 |
| **원격** | challenge 서명 + 명시적 승인 필요 |

### Gateway 인증

```json5
// ~/.openclaw/openclaw.json
{
  gateway: {
    auth: {
      mode: "password",  // "none" | "password" | "token"
      password: "your-secure-password",
      allowTailscale: true,  // Tailscale ID 헤더 신뢰
    },
  },
}
```

---

## 세션 모델

### 세션 타입

```
┌─────────────────────────────────────────────────────────────────┐
│                    Session Types                                 │
│                                                                  │
│   main ──────────▶ 직접 채팅 (DM)                              │
│                                                                  │
│   group:abc ─────▶ 그룹 채팅 (격리됨)                           │
│                                                                  │
│   channel:tel ───▶ 채널별 세션                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 세션 설정

```json5
{
  sessions: {
    defaults: {
      thinkingLevel: "medium",  // off|minimal|low|medium|high|xhigh
      verboseLevel: false,
      model: "anthropic/claude-opus-4-5",
      sendPolicy: "auto",  // auto|manual|off
    },
  },
}
```

### Agent-to-Agent 통신

에이전트 간 통신을 위한 세션 도구:

```typescript
// 활성 세션 목록
sessions_list()

// 세션 히스토리 조회
sessions_history({ sessionKey: "main" })

// 다른 세션에 메시지 전송
sessions_send({
  sessionKey: "group:dev-team",
  message: "빌드 완료되었습니다.",
  replyBack: true,  // 응답 받기
})
```

---

## Canvas Host

Canvas Host는 에이전트가 제어하는 HTML 렌더링 서버입니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Canvas Architecture                           │
│                                                                  │
│   Agent ──▶ canvas.push ──▶ Canvas Host ──▶ Browser/App        │
│                             (port 18793)                        │
│                                                                  │
│   Features:                                                     │
│   • A2UI (Agent-to-UI) 프로토콜                                 │
│   • 실시간 HTML/JS 렌더링                                       │
│   • eval 명령으로 JS 실행                                       │
│   • 스냅샷 캡처                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### A2UI 명령

```typescript
// HTML 푸시
canvas.push({ html: "<h1>Hello World</h1>" })

// JavaScript 실행
canvas.eval({ code: "document.title = 'Updated'" })

// 스냅샷 캡처
canvas.snapshot()

// 리셋
canvas.reset()
```

---

## 포트 구성

Gateway는 여러 포트를 사용합니다:

| 포트 | 용도 | 기본값 |
|------|------|--------|
| **Gateway** | WebSocket API | 18789 |
| **Browser Control** | 브라우저 서비스 | 18791 |
| **Browser Relay** | CDP 릴레이 | 18792 |
| **Canvas Host** | A2UI 서버 | 18793 |
| **CDP (openclaw)** | 관리 브라우저 | 18800+ |

```bash
# 커스텀 포트로 실행
openclaw gateway --port 19000

# 파생 포트도 자동 조정됨
# Browser: 19002, Relay: 19003, Canvas: 19004
```

---

## TypeBox 스키마

OpenClaw은 **TypeBox**로 프로토콜을 정의합니다:

```typescript
import { Type } from "@sinclair/typebox"

// 요청 스키마
const AgentRequest = Type.Object({
  message: Type.String(),
  sessionKey: Type.Optional(Type.String()),
  thinkingLevel: Type.Optional(ThinkingLevel),
})

// JSON Schema 자동 생성
// Swift 모델 자동 생성 (macOS/iOS 앱용)
```

---

## 원격 접근

### Tailscale 통합

```json5
{
  gateway: {
    tailscale: {
      mode: "serve",  // "off" | "serve" | "funnel"
      resetOnExit: true,
    },
    bind: "loopback",  // Serve/Funnel 시 필수
  },
}
```

| 모드 | 설명 |
|------|------|
| **off** | Tailscale 비활성화 |
| **serve** | tailnet 내부만 접근 (HTTPS) |
| **funnel** | 공개 인터넷 접근 (password 필수) |

### SSH 터널

```bash
# 로컬에서 원격 Gateway 접근
ssh -N -L 18789:127.0.0.1:18789 user@gateway-host
```

---

## 운영 명령

```bash
# Gateway 시작
openclaw gateway --port 18789 --verbose

# 상태 확인
openclaw status

# 헬스 체크
openclaw doctor

# 채널 상태
openclaw channels status --probe

# 로그 확인 (macOS)
./scripts/clawlog.sh
```

---

## 핵심 불변 규칙

1. **하나의 Gateway**: 호스트당 하나의 Gateway만 WhatsApp 세션 유지
2. **필수 핸드셰이크**: 첫 프레임은 반드시 `connect`
3. **이벤트 재전송 없음**: 클라이언트는 갭 발생 시 새로고침 필요
4. **멱등성 키**: side-effect 메서드(`send`, `agent`)는 중복 방지 키 필요

---

*다음 글에서는 설치 및 설정을 살펴봅니다.*
