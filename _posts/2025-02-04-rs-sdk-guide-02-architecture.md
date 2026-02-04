---
layout: post
title: "RS-SDK 가이드 - 아키텍처"
date: 2025-02-04
category: AI
tags: [rs-sdk, architecture, gateway, websocket, game-server]
series: rs-sdk-guide
part: 2
author: MaxBittker
original_url: https://github.com/MaxBittker/rs-sdk
---

## 아키텍처 개요

RS-SDK는 여러 구성 요소가 WebSocket으로 연결된 분산 아키텍처를 사용합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      RS-SDK Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐         ┌─────────────┐                      │
│   │   BotSDK    │ ◀─────▶ │   Gateway   │                      │
│   │  (Script)   │    WS   │  (Router)   │                      │
│   └─────────────┘         └──────┬──────┘                      │
│                                  │                              │
│                                  │ WS                           │
│                                  ▼                              │
│                          ┌─────────────┐                       │
│                          │  BotClient  │                       │
│                          │  (Browser)  │                       │
│                          └──────┬──────┘                       │
│                                 │                               │
│                                 │ WS                            │
│                                 ▼                               │
│                          ┌─────────────┐                       │
│                          │   Engine    │                       │
│                          │  (Server)   │                       │
│                          └─────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 핵심 원리

SDK는 게임 서버와 **직접 통신하지 않습니다**. 대신:

1. **BotClient** (브라우저)가 게임 서버에 연결
2. **Gateway**가 BotClient와 SDK 사이를 중계
3. BotClient가 게임 상태를 SDK로 전달
4. SDK가 저수준 액션(예: `walkTo(x,y)`)을 BotClient로 전송
5. BotClient가 게임 서버에서 액션 실행

## 디렉토리 구조

```
rs-sdk/
├── engine/           # 게임 서버
│   ├── src/          # 서버 소스 코드
│   ├── public/       # 정적 파일
│   │   ├── client/   # 표준 웹 클라이언트
│   │   └── bot/      # 봇 클라이언트
│   ├── view/         # EJS 템플릿
│   ├── data/         # 런타임 게임 데이터
│   └── prisma/       # DB 스키마/마이그레이션
│
├── webclient/        # TypeScript 웹 클라이언트
│   ├── src/
│   │   ├── client/   # 표준 클라이언트 코드
│   │   └── bot/      # 봇 전용 모듈 (BotSDK)
│   └── out/          # 빌드된 클라이언트 번들
│
├── gateway/          # WebSocket 라우터
│   ├── gateway.ts    # 메인 게이트웨이 서비스
│   ├── types.ts      # 프로토콜 타입
│   └── agent-state/  # 봇별 상태 파일
│
├── sdk/              # BotSDK 라이브러리
│   ├── index.ts      # BotSDK (저수준)
│   ├── actions.ts    # BotActions (고수준)
│   ├── runner.ts     # 스크립트 실행기
│   └── types.ts      # 타입 정의
│
├── mcp/              # MCP 서버
│   ├── server.ts     # MCP 서버 (stdio)
│   └── api/          # API 문서
│
├── bots/             # 봇 디렉토리
│   └── {username}/
│       ├── bot.env   # 자격 증명
│       ├── script.ts # 자동화 스크립트
│       └── lab_log.md # 세션 노트
│
├── learnings/        # 스킬별 가이드
│   ├── combat.md
│   ├── woodcutting.md
│   └── ...
│
└── scripts/          # 유틸리티 스크립트
    ├── create-bot.ts
    └── ...
```

## Engine (게임 서버)

LostCity 기반의 게임 서버입니다.

### 주요 역할

- 월드 시뮬레이션
- 플레이어/NPC 로직
- 네트워크 프로토콜
- 게임 콘텐츠 관리

### 실행 명령

```bash
cd engine

# 일반 시작
bun start

# 개발 모드 (핫 리로드)
bun run dev

# 콘텐츠 빌드
bun run build
```

### 데이터베이스

```bash
# SQLite 마이그레이션
bun run sqlite:migrate

# MySQL 마이그레이션
bun run db:migrate
```

## WebClient (웹 클라이언트)

브라우저 기반 게임 클라이언트입니다.

### 빌드 종류

| 타입 | 경로 | 설명 |
|------|------|------|
| Standard | `out/standard/` | 일반 웹 클라이언트 |
| Bot | `out/bot/` | SDK 연동 봇 클라이언트 |

### 빌드 명령

```bash
cd webclient

# 프로덕션 빌드
bun run build

# 개발 모드
bun run build:dev

# 엔진에 복사
cp out/standard/client.js ../engine/public/client/
cp out/bot/client.js ../engine/public/bot/
```

## Gateway (WebSocket 라우터)

봇 클라이언트와 SDK 간의 통신을 중계합니다.

### 메시지 흐름

```
SDK → Gateway → BotClient
         │
         ▼
    sdk_connect (인증)
    sdk_state (상태 수신)
    sdk_action_result (결과 수신)

BotClient → Gateway → SDK
               │
               ▼
          bot_state (상태 전송)
          bot_action_result (결과 전송)
```

### 실행

```bash
cd gateway
bun run gateway

# 개발 모드
bun run gateway:dev
```

### 인증

로그인 서버가 활성화되면 SDK 연결에 봇별 인증이 필요합니다:

```typescript
// 환경 변수
LOGIN_SERVER=true
LOGIN_HOST=localhost
LOGIN_PORT=43500
```

## SDK (Bot SDK)

봇 자동화 스크립트를 작성하기 위한 TypeScript 라이브러리입니다.

### 계층 구조

```
┌─────────────────────────────────────────┐
│           Your Script                   │
├─────────────────────────────────────────┤
│  BotActions (Porcelain)                 │
│  - 고수준, 도메인 인식                   │
│  - 효과 완료까지 대기                    │
├─────────────────────────────────────────┤
│  BotSDK (Plumbing)                      │
│  - 저수준, WebSocket API                │
│  - 서버 확인(ACK)까지 대기               │
├─────────────────────────────────────────┤
│  WebSocket                              │
└─────────────────────────────────────────┘
```

### 스크립트 실행기

```typescript
import { runScript } from '../../sdk/runner';

const result = await runScript(async (ctx) => {
  const { bot, sdk, log } = ctx;
  // 자동화 로직
}, {
  timeout: 60_000
});
```

## MCP (Model Context Protocol)

Claude Code와의 인터랙티브 통합을 위한 MCP 서버입니다.

### 자동 검색

프로젝트 루트의 `.mcp.json`으로 Claude Code가 자동 검색합니다:

```json
{
  "mcpServers": {
    "rs-agent": {
      "command": "bun",
      "args": ["run", "mcp/server.ts"]
    }
  }
}
```

## 통신 프로토콜

### SDK → Gateway 메시지

```typescript
interface SDKMessage {
  type: 'sdk_connect' | 'sdk_action';
  username?: string;
  password?: string;
  action?: BotAction;
  actionId?: string;
}
```

### Gateway → SDK 메시지

```typescript
interface SyncToSDKMessage {
  type: 'sdk_connected' | 'sdk_state' | 'sdk_action_result';
  success?: boolean;
  state?: BotWorldState;
  result?: ActionResult;
}
```

## 데이터 흐름 예시

나무 베기 액션의 흐름:

```
1. [SDK] bot.chopTree() 호출
2. [SDK → Gateway] 액션 전송: sendInteractLoc(tree.x, tree.z, ...)
3. [Gateway → BotClient] 액션 전달
4. [BotClient] 게임 서버에서 액션 실행
5. [Engine] 애니메이션 시작, XP 부여
6. [BotClient → Gateway] 새 상태 전송
7. [Gateway → SDK] 상태 업데이트
8. [SDK] chopTree() Promise 해결
```

## 다음 단계

다음 챕터에서는 실제로 봇을 생성하고 첫 스크립트를 작성합니다.

---

**이전 글**: [소개](/rs-sdk-guide-01-intro/)

**다음 글**: [시작하기](/rs-sdk-guide-03-getting-started/)
