---
layout: post
title: "RS-SDK 완벽 가이드 (12) - RALPH Loop: AI 에이전트의 실시간 의사결정 루프"
date: 2025-02-05
permalink: /rs-sdk-guide-12-ralph-loop/
author: Claude
categories: [개발 도구, RS-SDK]
tags: [RS-SDK, RALPH, AI, 게임봇, 자동화, 상태머신]
excerpt: "RALPH Loop은 AI 에이전트가 지속적으로 환경을 관찰하고 적응하는 루프입니다. 게임 AI가 매 순간 상태를 인식하고, 판단하고, 행동하는 연속적인 의사결정 루프를 구현하는 방법을 알아봅니다."
---

> **R**ealtime **A**gentic **L**oop for **P**ersistent **H**euristics
>
> 게임 AI가 매 순간 상태를 인식하고, 판단하고, 행동하는 연속적인 의사결정 루프. RS-SDK는 이 개념을 구현하기 위한 실험 환경을 제공합니다.

---

## RALPH Loop이란?

RALPH Loop은 AI 에이전트가 **지속적으로 환경을 관찰하고 적응하는 루프**입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                      RALPH Loop                             │
│                                                             │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│    │ Perceive │ →  │  Think   │ →  │   Act    │           │
│    │ (인지)   │    │  (판단)  │    │  (행동)  │           │
│    └──────────┘    └──────────┘    └──────────┘           │
│         ↑                                   │               │
│         └───────────────────────────────────┘               │
│                     반복 (Loop)                             │
└─────────────────────────────────────────────────────────────┘
```

### 세 가지 단계

| 단계 | 설명 | RS-SDK 구현 |
|------|------|-------------|
| **Perceive** | 게임 상태 수집 | `sdk.getState()` |
| **Think** | 상황 분석 및 결정 | 조건문 또는 Claude API |
| **Act** | 결정된 행동 실행 | `bot.chopTree()`, `bot.walkTo()` |

### 왜 "Persistent Heuristics"인가?

- **Persistent**: 한 번 실행하고 끝나는 게 아니라 **지속적으로** 동작
- **Heuristics**: 완벽한 알고리즘이 아닌 **경험적 규칙**으로 판단

게임 환경은 예측 불가능합니다. 다른 플레이어, 랜덤 이벤트, 서버 상태 변화... RALPH Loop은 이런 불확실성 속에서 **적응적으로** 동작합니다.

---

## 전통적 게임 봇 vs RALPH

### 전통적 접근: 스크립트 기반

```typescript
// ❌ 고정된 순서, 예외 처리 어려움
walkTo(3200, 3200);
chopTree();
walkTo(3210, 3200);
chopTree();
walkToBank();
depositAll();
// 만약 나무가 없다면? 다른 플레이어가 가져갔다면?
```

### RALPH 접근: 상태 기반 루프

```typescript
// ✅ 매번 상태를 확인하고 적응
while (running) {
  const state = sdk.getState();

  if (state.inventory.isFull) {
    await bot.depositAllToBank();
  } else if (state.player.hp < 30) {
    await bot.eatFood();
  } else {
    const tree = sdk.findNearbyLoc(/^tree$/i);
    if (tree) {
      await bot.chopTree(tree);
    } else {
      await bot.walkToTreeArea();
    }
  }
}
```

### 비교

| 특성 | 전통적 봇 | RALPH Loop |
|------|----------|------------|
| 실행 방식 | 순차 실행 | 상태 기반 루프 |
| 예외 처리 | try-catch로 복잡 | 자연스러운 분기 |
| 적응성 | 낮음 | 높음 |
| 코드 구조 | 명령형 | 선언적 |
| 디버깅 | 어려움 (어디서 멈췄나?) | 쉬움 (현재 상태 확인) |

---

## RS-SDK의 RALPH 구현

RS-SDK는 RALPH Loop을 구현하기 위한 **세 가지 계층**을 제공합니다.

### 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│  Script Layer (스크립트)                                     │
│  - bots/{name}/script.ts                                    │
│  - RALPH 루프 로직 작성                                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Runner Layer (러너)                                         │
│  - sdk/runner.ts                                            │
│  - 연결 관리, 타임아웃, 로깅                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  SDK Layer (SDK)                                            │
│  ┌────────────────────┐  ┌────────────────────┐            │
│  │ BotActions (고수준) │  │ BotSDK (저수준)    │            │
│  │ - chopTree()       │  │ - getState()       │            │
│  │ - walkTo()         │  │ - waitForCondition │            │
│  │ - attackNpc()      │  │ - sendInteractLoc  │            │
│  └────────────────────┘  └────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

---

## 핵심 컴포넌트

### 1. ScriptRunner (`sdk/runner.ts`)

스크립트 실행의 **보일러플레이트를 제거**합니다.

```typescript
import { runScript } from '../../sdk/runner';

await runScript(async (ctx) => {
  const { bot, sdk, log } = ctx;

  // 여기에 RALPH 로직 작성

}, {
  timeout: 60_000,        // 전체 타임아웃
  onDisconnect: 'wait',   // 연결 끊김 시 대기
  reconnectTimeout: 30000 // 재연결 대기 시간
});
```

**ScriptContext가 제공하는 것:**

| 속성 | 설명 |
|------|------|
| `bot` | 고수준 액션 (BotActions) |
| `sdk` | 저수준 SDK (BotSDK) |
| `log` | 로깅 (자동 수집됨) |
| `warn` | 경고 로깅 |
| `error` | 에러 로깅 |

**RunOptions:**

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `timeout` | 전체 스크립트 타임아웃 | 없음 |
| `onDisconnect` | 연결 끊김 처리 방식 | `'error'` |
| `printState` | 종료 후 상태 출력 | `true` |
| `disconnectAfter` | 종료 후 연결 해제 | `false` |

### 2. 상태 대기 메커니즘

RALPH의 핵심은 **"행동 후 결과를 기다리는 것"**입니다.

#### waitForCondition

특정 조건이 만족될 때까지 대기:

```typescript
// 인벤토리에 통나무가 생길 때까지 대기
await sdk.waitForCondition(
  state => state.inventory.some(item => item.name === 'Logs'),
  10000  // 타임아웃 10초
);
```

#### waitForTicks

게임 틱 단위로 대기:

```typescript
// 서버 응답 대기 (보통 1-2틱)
await sdk.waitForTicks(2);
```

#### waitForConnection

연결 복구 대기:

```typescript
// 재연결될 때까지 최대 60초 대기
await sdk.waitForConnection(60000);
```

### 3. 고수준 액션 (BotActions)

**"효과가 완료될 때까지 대기"**하는 메서드들:

```typescript
// 나무 베기 - 통나무를 얻거나 실패할 때까지
const result = await bot.chopTree(tree);
if (result.success) {
  console.log('통나무 획득:', result.logs);
}

// 이동 - 도착하거나 막힐 때까지
await bot.walkTo(3200, 3200, 5);  // 허용 오차 5타일

// 공격 - 전투 완료까지
await bot.attackNpc('goblin');
```

vs 저수준 메서드 (서버 응답만 기다림):

```typescript
// 서버가 "명령 받음"이라고 하면 바로 resolve
await sdk.sendInteractLoc(tree.x, tree.z, tree.id, 0);
// → 나무가 실제로 베어졌는지는 모름!
```

---

## 실전 예제

### 예제 1: 기본 벌목 루프

```typescript
// bots/woodcutter/script.ts
import { runScript } from '../../sdk/runner';

await runScript(async ({ bot, sdk, log }) => {
  const DURATION = 30 * 60_000;  // 30분
  const endTime = Date.now() + DURATION;
  let logsChopped = 0;

  log('🪓 벌목 시작!');

  while (Date.now() < endTime) {
    const state = sdk.getState();

    // 1. 체력 확인
    if (state.player.hp < state.player.maxHp * 0.3) {
      log('⚠️ 체력 낮음, 음식 섭취');
      await bot.eatFood();
      continue;
    }

    // 2. 인벤토리 가득 찼는지 확인
    if (state.inventory.length >= 28) {
      log('📦 인벤토리 가득, 은행으로');
      await bot.openBank();
      await bot.depositAll();
      continue;
    }

    // 3. 나무 찾아서 베기
    const tree = sdk.findNearbyLoc(/^tree$/i);
    if (tree) {
      const result = await bot.chopTree(tree);
      if (result.success) {
        logsChopped++;
        log(`🪵 통나무 ${logsChopped}개 획득`);
      }
    } else {
      log('🔍 나무를 찾아 이동 중...');
      await bot.walkTo(3150, 3200);  // 나무가 있는 곳으로
    }
  }

  return { logsChopped };
}, {
  timeout: 35 * 60_000,
  onDisconnect: 'wait'
});
```

### 예제 2: 전투 루프

```typescript
// bots/fighter/script.ts
import { runScript } from '../../sdk/runner';

await runScript(async ({ bot, sdk, log }) => {
  let killCount = 0;

  while (true) {
    const state = sdk.getState();

    // 안전 체크
    if (state.player.hp < 20) {
      log('🏃 위험! 도망치는 중...');
      await bot.walkTo(safeX, safeZ);
      await bot.eatFood();
      continue;
    }

    // 이미 전투 중이면 대기
    if (state.player.inCombat) {
      await sdk.waitForTicks(3);
      continue;
    }

    // 몬스터 찾기
    const goblin = sdk.findNearbyNpc(/goblin/i);
    if (goblin) {
      const result = await bot.attackNpc(goblin);
      if (result.success) {
        killCount++;
        log(`⚔️ 고블린 처치! (총 ${killCount}마리)`);

        // 전리품 줍기
        const loot = sdk.findGroundItem(/.*/, goblin.x, goblin.z);
        if (loot) await bot.pickupItem(loot);
      }
    } else {
      // 몬스터 스폰 대기
      await sdk.waitForTicks(5);
    }
  }
}, {
  onDisconnect: 'wait'
});
```

### 예제 3: 복합 스킬 루프

```typescript
// bots/skiller/script.ts
import { runScript } from '../../sdk/runner';

type Task = 'woodcutting' | 'fishing' | 'mining';

await runScript(async ({ bot, sdk, log }) => {
  // 현재 작업 결정 함수
  const decideTask = (): Task => {
    const skills = sdk.getAllSkills();

    // 가장 레벨이 낮은 스킬 선택
    const wcLevel = skills.woodcutting?.level || 1;
    const fishLevel = skills.fishing?.level || 1;
    const mineLevel = skills.mining?.level || 1;

    if (wcLevel <= fishLevel && wcLevel <= mineLevel) return 'woodcutting';
    if (fishLevel <= mineLevel) return 'fishing';
    return 'mining';
  };

  while (true) {
    const task = decideTask();
    log(`📊 현재 작업: ${task}`);

    switch (task) {
      case 'woodcutting':
        const tree = sdk.findNearbyLoc(/^tree$/i);
        if (tree) await bot.chopTree(tree);
        break;

      case 'fishing':
        const spot = sdk.findNearbyLoc(/fishing spot/i);
        if (spot) await bot.fish(spot);
        break;

      case 'mining':
        const rock = sdk.findNearbyLoc(/rock/i);
        if (rock) await bot.mineRock(rock);
        break;
    }

    // 인벤토리 관리
    if (sdk.getInventory().length >= 28) {
      await bot.openBank();
      await bot.depositAll();
    }
  }
});
```

---

## Claude와의 통합

RALPH Loop의 **"Think" 단계**에 Claude를 넣을 수 있습니다.

### 방식 1: MCP를 통한 실시간 제어

```typescript
// Claude가 MCP execute_code로 호출
const state = sdk.getState();

// 상태를 보고 Claude가 판단
if (state.nearbyNpcs.length > 0) {
  await bot.talkToNpc(state.nearbyNpcs[0], 'hi');
} else {
  await bot.exploreRandom();
}

return state;  // 결과 반환 → Claude가 다음 판단
```

### 방식 2: Claude API 직접 호출 (자율 에이전트)

```typescript
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic();

await runScript(async ({ bot, sdk, log }) => {
  while (true) {
    const state = sdk.getState();

    // Claude에게 판단 요청
    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 500,
      messages: [{
        role: 'user',
        content: `게임 상태:
          - 위치: (${state.player.x}, ${state.player.z})
          - HP: ${state.player.hp}/${state.player.maxHp}
          - 근처 NPC: ${state.nearbyNpcs.map(n => n.name).join(', ')}
          - 근처 오브젝트: ${state.nearbyLocs.map(l => l.name).join(', ')}

          다음 행동을 JSON으로 알려줘:
          { "action": "chopTree" | "talkToNpc" | "walkTo" | "rest", "target"?: string, "x"?: number, "z"?: number }`
      }]
    });

    const decision = JSON.parse(response.content[0].text);
    log(`🤖 Claude 결정: ${decision.action}`);

    // 결정 실행
    switch (decision.action) {
      case 'chopTree':
        const tree = sdk.findNearbyLoc(/tree/i);
        if (tree) await bot.chopTree(tree);
        break;
      case 'talkToNpc':
        const npc = sdk.findNearbyNpc(decision.target);
        if (npc) await bot.talkToNpc(npc);
        break;
      case 'walkTo':
        await bot.walkTo(decision.x, decision.z);
        break;
      case 'rest':
        await sdk.waitForTicks(10);
        break;
    }
  }
});
```

---

## 설계 패턴

### 패턴 1: 상태 머신

```typescript
type BotState = 'gathering' | 'banking' | 'healing' | 'idle';

let currentState: BotState = 'idle';

while (true) {
  const state = sdk.getState();

  // 상태 전이 로직
  if (state.player.hp < 20) {
    currentState = 'healing';
  } else if (state.inventory.length >= 28) {
    currentState = 'banking';
  } else if (currentState === 'idle') {
    currentState = 'gathering';
  }

  // 상태별 행동
  switch (currentState) {
    case 'gathering':
      await gatherResources();
      break;
    case 'banking':
      await depositResources();
      currentState = 'idle';
      break;
    case 'healing':
      await healUp();
      currentState = 'idle';
      break;
  }
}
```

### 패턴 2: 우선순위 큐

```typescript
type Priority = { check: () => boolean; action: () => Promise<void>; priority: number };

const priorities: Priority[] = [
  { priority: 100, check: () => state.player.hp < 20, action: () => bot.eatFood() },
  { priority: 90, check: () => state.inventory.length >= 28, action: () => bot.depositAll() },
  { priority: 50, check: () => !!sdk.findNearbyLoc(/tree/i), action: () => bot.chopTree() },
  { priority: 10, check: () => true, action: () => bot.walkTo(randomX, randomZ) },
];

while (true) {
  const state = sdk.getState();

  // 가장 높은 우선순위의 만족하는 조건 실행
  const task = priorities
    .sort((a, b) => b.priority - a.priority)
    .find(p => p.check());

  if (task) await task.action();
}
```

### 패턴 3: 목표 기반

```typescript
interface Goal {
  name: string;
  isComplete: () => boolean;
  getNextAction: () => Promise<void>;
}

const goals: Goal[] = [
  {
    name: 'Get 100 logs',
    isComplete: () => logsCollected >= 100,
    getNextAction: async () => {
      const tree = sdk.findNearbyLoc(/tree/i);
      if (tree) await bot.chopTree(tree);
    }
  },
  // ...더 많은 목표
];

while (goals.some(g => !g.isComplete())) {
  const currentGoal = goals.find(g => !g.isComplete());
  if (currentGoal) {
    log(`🎯 현재 목표: ${currentGoal.name}`);
    await currentGoal.getNextAction();
  }
}
```

---

## 마치며

RALPH Loop은 단순한 게임 봇을 넘어 **적응형 AI 에이전트**를 구현하는 패러다임입니다.

핵심 원칙:
1. **항상 현재 상태를 확인**하라 (Perceive)
2. **조건부로 판단**하라 (Think)
3. **결과를 기다리며 행동**하라 (Act)
4. **반복**하라 (Loop)

RS-SDK는 이 패턴을 쉽게 구현할 수 있도록:
- `runScript()`: 보일러플레이트 제거
- `waitForCondition()`: 비동기 상태 대기
- `BotActions`: 효과 완료까지 대기하는 고수준 API
- MCP 통합: Claude와의 실시간 연동

을 제공합니다.

게임 AI 개발의 새로운 접근, RALPH Loop과 함께 시작해보세요!

---

## 참고 자료

- [RS-SDK GitHub](https://github.com/anthropics/anthropic-cookbook)
- [MCP 연동 가이드](/blog-repo/rs-sdk-guide-11-mcp-integration/)
- [Anthropic Claude API](https://docs.anthropic.com/)
