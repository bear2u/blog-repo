---
layout: post
title: "RS-SDK 가이드 - MCP 통합"
date: 2025-02-04
category: AI
tags: [rs-sdk, mcp, claude-code, ai-agent, interactive]
series: rs-sdk-guide
part: 5
author: MaxBittker
original_url: https://github.com/MaxBittker/rs-sdk
---

## MCP (Model Context Protocol)

MCP는 Claude Code와 RS-SDK를 연결하는 프로토콜입니다. 이를 통해 Claude가 직접 봇을 제어할 수 있습니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Integration                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐        ┌──────────────┐                 │
│   │  Claude Code │ ◀────▶ │  MCP Server  │                 │
│   │              │  stdio │  (rs-agent)  │                 │
│   └──────────────┘        └──────┬───────┘                 │
│                                  │                          │
│                                  │ SDK API                  │
│                                  ▼                          │
│                          ┌──────────────┐                  │
│                          │   BotSDK     │                  │
│                          │   Gateway    │                  │
│                          └──────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 자동 검색 설정

프로젝트 루트에 `.mcp.json` 파일이 있으면 Claude Code가 자동으로 MCP 서버를 검색합니다:

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

## MCP 도구

### execute_code

TypeScript 코드를 봇 컨텍스트에서 실행합니다:

```typescript
// MCP를 통해 실행되는 코드
const state = sdk.getState();
log(`현재 위치: ${state?.player?.worldX}, ${state?.player?.worldZ}`);

const tree = sdk.findNearbyLoc(/^tree$/i);
if (tree) {
  await bot.chopTree(tree);
}

return { success: true };
```

### list_bots

사용 가능한 봇 목록을 조회합니다:

```
┌─────────────┬────────────┬──────────────┐
│ Username    │ Status     │ Last Active  │
├─────────────┼────────────┼──────────────┤
│ mybot       │ connected  │ 2 min ago    │
│ testbot     │ offline    │ 1 hour ago   │
│ grinder     │ connected  │ now          │
└─────────────┴────────────┴──────────────┘
```

### disconnect_bot

봇 연결을 종료합니다:

```typescript
// 특정 봇 연결 해제
await disconnectBot('mybot');
```

## Claude Code 사용 예시

### 봇 생성 및 시작

```bash
# Claude Code 터미널에서
claude "create a new bot named mybot and start woodcutting"
```

Claude가 자동으로:
1. `create-bot.ts` 스크립트로 봇 생성
2. MCP를 통해 봇 연결
3. 나무 베기 스크립트 실행

### 상태 확인

```bash
claude "what is my bot doing right now?"
```

응답 예:
```
mybot is currently at position (3222, 3218).
- Woodcutting level: 15
- Inventory: 12/28 (8 logs, 1 bronze axe)
- Status: Chopping tree
- Nearby: 2 trees, 1 man NPC
```

### 인터랙티브 제어

```bash
claude "make my bot walk to the bank and deposit all logs"
```

Claude가 생성하는 코드:
```typescript
// 은행으로 이동
await bot.walkTo(3185, 3436);

// 은행 열기
await bot.openBank();

// 로그 모두 입금
await bot.depositItem(/logs/i, -1);

// 은행 닫기
await bot.closeBank();

log('완료: 모든 로그 입금됨');
```

## 세션 워크플로우

### 1. 환경 파악

```typescript
// 현재 상태 확인
const state = sdk.getState();
const inv = sdk.getInventory();
const skills = state?.skills;

log(`위치: ${state?.player?.worldX}, ${state?.player?.worldZ}`);
log(`인벤토리: ${inv.length}/28`);
log(`스킬 레벨: ${skills?.map(s => `${s.name}: ${s.baseLevel}`).join(', ')}`);
```

### 2. 목표 설정

```typescript
// 목표: Woodcutting 레벨 20 달성
const targetLevel = 20;
const currentLevel = skills?.find(s => s.name === 'Woodcutting')?.baseLevel ?? 1;

if (currentLevel >= targetLevel) {
  log('목표 달성!');
  return { success: true };
}
```

### 3. 루프 실행

```typescript
// 목표 달성까지 반복
while (true) {
  await bot.dismissBlockingUI();

  const tree = sdk.findNearbyLoc(/^tree$/i);
  if (!tree) {
    log('나무 없음, 대기 중...');
    await sdk.waitForTicks(5);
    continue;
  }

  await bot.chopTree(tree);

  // 인벤토리 가득 차면 버리기
  if (sdk.getInventory().length >= 28) {
    for (const item of sdk.getInventory()) {
      if (item.name.includes('Logs')) {
        await sdk.sendDropItem(item.slot);
      }
    }
  }

  // 레벨 확인
  const newLevel = sdk.getState()?.skills?.find(s => s.name === 'Woodcutting')?.baseLevel;
  if (newLevel && newLevel >= targetLevel) {
    log(`목표 달성! Woodcutting ${newLevel}`);
    break;
  }
}
```

## MCP 서버 구조

```
mcp/
├── server.ts      # 메인 MCP 서버 (stdio 통신)
├── tools/         # MCP 도구 구현
│   ├── execute.ts
│   ├── list.ts
│   └── disconnect.ts
└── api/           # API 문서
    └── schema.json
```

### server.ts 핵심 구조

```typescript
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const server = new Server({
  name: 'rs-agent',
  version: '1.0.0',
}, {
  capabilities: {
    tools: {},
  },
});

// 도구 등록
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: 'execute_code',
      description: 'Execute TypeScript code in bot context',
      inputSchema: {
        type: 'object',
        properties: {
          code: { type: 'string' },
          botUsername: { type: 'string' },
          timeout: { type: 'number' },
        },
        required: ['code', 'botUsername'],
      },
    },
    // ... 다른 도구들
  ],
}));

// stdio 전송
const transport = new StdioServerTransport();
await server.connect(transport);
```

## 디버깅 팁

### MCP 로그 확인

```bash
# MCP 서버 로그
tail -f mcp/logs/server.log

# 봇 상태 로그
tail -f gateway/agent-state/mybot.json
```

### 연결 테스트

```bash
# MCP 서버 직접 테스트
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | bun mcp/server.ts
```

### 일반적인 문제

| 문제 | 해결책 |
|------|--------|
| MCP 서버 시작 안 됨 | `bun install` 재실행 |
| 봇 연결 실패 | Gateway 실행 확인 |
| 타임아웃 | timeout 값 증가 |
| 상태 동기화 안 됨 | 브라우저 새로고침 |

## 보안 고려사항

### 채팅 비활성화

기본적으로 채팅은 비활성화되어 있습니다:

```bash
# bot.env
SHOW_CHAT=false  # 스캠/프롬프트 인젝션 방지
```

### 코드 샌드박싱

MCP를 통해 실행되는 코드는 봇 컨텍스트 내에서만 실행됩니다. 시스템 접근이 제한됩니다.

## 다음 단계

다음 챕터에서는 다양한 스킬 자동화 패턴을 알아봅니다.

---

**이전 글**: [SDK API](/rs-sdk-guide-04-sdk-api/)

**다음 글**: [스킬 자동화](/rs-sdk-guide-06-skills/)
