---
layout: post
title: "RS-SDK 가이드 - 스킬 자동화"
date: 2025-02-04
categories: [개발 도구, RS-SDK]
tags: [rs-sdk, skills, woodcutting, mining, fishing, combat]
author: MaxBittker
original_url: https://github.com/MaxBittker/rs-sdk
---

## 스킬 자동화 개요

RS-SDK는 다양한 스킬 자동화를 지원합니다. 각 스킬은 고유한 패턴과 주의사항이 있습니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Skill Categories                         │
├─────────────────────────────────────────────────────────────┤
│  Gathering          │  Combat           │  Production       │
│  ─────────────────  │  ───────────────  │  ───────────────  │
│  Woodcutting        │  Attack           │  Cooking          │
│  Mining             │  Strength         │  Smithing         │
│  Fishing            │  Defence          │  Fletching        │
│                     │  Ranged           │  Crafting         │
│                     │  Magic            │                   │
└─────────────────────────────────────────────────────────────┘
```

## Woodcutting (벌목)

### 기본 패턴

```typescript
import { runScript } from '../../sdk/runner';

await runScript(async (ctx) => {
  const { bot, sdk, log } = ctx;

  while (true) {
    await bot.dismissBlockingUI();

    // 나무 찾기 (정확한 매칭)
    const tree = sdk.findNearbyLoc(/^tree$/i);

    if (!tree) {
      log('나무를 찾을 수 없음');
      await sdk.waitForTicks(5);
      continue;
    }

    // 나무 베기
    const result = await bot.chopTree(tree);

    if (result.success) {
      log(`통나무 획득! XP: ${result.xpGained}`);
    }

    // 인벤토리 관리
    if (sdk.getInventory().length >= 28) {
      await dropAllLogs(sdk);
    }
  }
}, { timeout: 300_000 });

async function dropAllLogs(sdk: BotSDK) {
  for (const item of sdk.getInventory()) {
    if (item.name.toLowerCase().includes('logs')) {
      await sdk.sendDropItem(item.slot);
    }
  }
}
```

### 나무 유형별 요구사항

| 나무 | 레벨 | 정규식 |
|------|------|--------|
| Tree | 1 | `/^tree$/i` |
| Oak | 15 | `/^oak$/i` |
| Willow | 30 | `/^willow$/i` |
| Maple | 45 | `/^maple$/i` |
| Yew | 60 | `/^yew$/i` |

### 주의사항

```typescript
// 잘못된 예 - "tree stump"도 매칭됨
const tree = sdk.findNearbyLoc(/tree/i);

// 올바른 예 - 정확히 "tree"만 매칭
const tree = sdk.findNearbyLoc(/^tree$/i);
```

## Mining (채굴)

### 기본 패턴

```typescript
await runScript(async (ctx) => {
  const { bot, sdk, log } = ctx;

  while (true) {
    await bot.dismissBlockingUI();

    // 광석 찾기
    const rock = sdk.findNearbyLoc(/copper rock|tin rock/i);

    if (!rock) {
      log('광석을 찾을 수 없음');
      await sdk.waitForTicks(5);
      continue;
    }

    // 채굴
    const result = await bot.mineRock(rock);

    if (result.success) {
      log(`광석 획득! XP: ${result.xpGained}`);
    }

    // 인벤토리 관리
    if (sdk.getInventory().length >= 28) {
      await dropOres(sdk);
    }
  }
}, { timeout: 300_000 });
```

### 광석 유형별 요구사항

| 광석 | 레벨 | 정규식 |
|------|------|--------|
| Copper/Tin | 1 | `/copper rock|tin rock/i` |
| Iron | 15 | `/iron rock/i` |
| Coal | 30 | `/coal rock/i` |
| Mithril | 55 | `/mithril rock/i` |
| Adamantite | 70 | `/adamantite rock/i` |

### 광석 고갈 처리

```typescript
// 광석이 고갈되면 다른 광석으로 이동
const rock = sdk.findNearbyLoc(/copper rock/i);

if (!rock || rock.name.includes('empty')) {
  // 다른 광석 찾기 또는 대기
  await sdk.waitForTicks(10);
}
```

## Fishing (낚시)

### 기본 패턴

```typescript
await runScript(async (ctx) => {
  const { bot, sdk, log } = ctx;

  while (true) {
    await bot.dismissBlockingUI();

    // 낚시 포인트 찾기 (NPC로 취급됨)
    const spot = sdk.findNearbyNpc(/fishing spot/i);

    if (!spot) {
      log('낚시 포인트를 찾을 수 없음');
      await sdk.waitForTicks(5);
      continue;
    }

    // 낚시
    const result = await bot.fish(spot);

    if (result.success) {
      log(`물고기 획득!`);
    }

    // 인벤토리 관리
    if (sdk.getInventory().length >= 28) {
      await dropFish(sdk);
    }
  }
}, { timeout: 300_000 });
```

### 낚시 유형

| 방법 | 도구 | 잡히는 물고기 |
|------|------|---------------|
| Net | Small fishing net | Shrimps, Anchovies |
| Bait | Fishing rod + bait | Sardine, Herring |
| Fly | Fly fishing rod + feathers | Trout, Salmon |
| Harpoon | Harpoon | Tuna, Swordfish |

### 도구 확인

```typescript
// 낚시 전 도구 확인
const hasNet = sdk.findInventoryItem(/small fishing net/i);
const hasRod = sdk.findInventoryItem(/fishing rod/i);
const hasBait = sdk.findInventoryItem(/fishing bait/i);

if (!hasNet && !hasRod) {
  log('낚시 도구가 없습니다');
  return { success: false };
}
```

## Combat (전투)

### 기본 전투 루프

```typescript
await runScript(async (ctx) => {
  const { bot, sdk, log } = ctx;

  const targetPattern = /cow|chicken/i;

  while (true) {
    await bot.dismissBlockingUI();

    const state = sdk.getState();

    // 전투 중인지 확인
    if (state?.combat?.inCombat) {
      // 체력 확인
      const hp = state.skills?.find(s => s.name === 'Hitpoints');
      if (hp && hp.boostedLevel < 10) {
        await bot.eatFood(/bread|meat/i);
      }

      await sdk.waitForTicks(2);
      continue;
    }

    // 새 대상 찾기
    const target = sdk.findNearbyNpc(targetPattern);

    if (!target) {
      log('대상을 찾을 수 없음');
      await sdk.waitForTicks(5);
      continue;
    }

    // 공격
    const result = await bot.attackNpc(targetPattern);

    if (result.success) {
      log(`${target.name} 공격 중...`);
    }
  }
}, { timeout: 600_000 });
```

### 전투 스타일

```typescript
// 전투 스타일에 따른 스킬 훈련
// Accurate: Attack 경험치
// Aggressive: Strength 경험치
// Defensive: Defence 경험치
// Controlled: 모든 스킬 경험치

// 전투 스타일 변경은 인터페이스를 통해 수행
```

### 자동 회복

```typescript
async function autoHeal(bot: BotActions, sdk: BotSDK) {
  const state = sdk.getState();
  const hp = state?.skills?.find(s => s.name === 'Hitpoints');

  if (!hp) return;

  // 체력이 50% 이하면 음식 먹기
  if (hp.boostedLevel < hp.baseLevel * 0.5) {
    const food = sdk.findInventoryItem(/bread|meat|fish/i);
    if (food) {
      await bot.eatFood(food.name);
    }
  }
}
```

### 루팅

```typescript
async function lootDrops(sdk: BotSDK) {
  // 지면 아이템 스캔
  const groundItems = await sdk.scanGroundItems();

  for (const item of groundItems) {
    if (item.name.match(/bones|hides|feathers/i)) {
      // 아이템 줍기
      await sdk.sendPickupItem(item.x, item.z, item.id);
      await sdk.waitForTicks(1);
    }
  }
}
```

## Cooking (요리)

### 기본 패턴

```typescript
await runScript(async (ctx) => {
  const { bot, sdk, log } = ctx;

  while (true) {
    await bot.dismissBlockingUI();

    // 날 음식 찾기
    const rawFood = sdk.findInventoryItem(/raw/i);

    if (!rawFood) {
      log('요리할 음식이 없습니다');
      break;
    }

    // 불/레인지 찾기
    const fire = sdk.findNearbyLoc(/fire|range|stove/i);

    if (!fire) {
      log('불을 찾을 수 없습니다');
      break;
    }

    // 요리
    const result = await bot.cookFood(rawFood);

    if (result.success) {
      log(`요리 완료!`);
    }
  }
}, { timeout: 300_000 });
```

## Smithing (대장간)

### 제련

```typescript
// 광석을 바로 변환
const ore = sdk.findInventoryItem(/copper ore|tin ore/i);
const furnace = sdk.findNearbyLoc(/furnace/i);

if (ore && furnace) {
  await sdk.sendInteractLoc(furnace.x, furnace.z, furnace.id, 0);
  await sdk.waitForCondition(state => state.modalOpen, 5000);
  // 제련 메뉴에서 선택
}
```

### 단조

```typescript
// 바를 장비로 변환
await bot.smithAtAnvil(itemId);
```

## 효율적인 스킬 훈련 팁

### 1. 인벤토리 관리

```typescript
// 인벤토리 공간 확인
function hasInventorySpace(sdk: BotSDK): boolean {
  return sdk.getInventory().length < 28;
}

// 불필요한 아이템 드롭
async function cleanInventory(sdk: BotSDK, keepPattern: RegExp) {
  for (const item of sdk.getInventory()) {
    if (!item.name.match(keepPattern)) {
      await sdk.sendDropItem(item.slot);
    }
  }
}
```

### 2. 위치 최적화

```typescript
// 가장 가까운 자원 선택
function findClosestResource(sdk: BotSDK, pattern: RegExp) {
  const resources = sdk.getState()?.nearbyLocs
    .filter(loc => loc.name.match(pattern))
    .sort((a, b) => a.distance - b.distance);

  return resources?.[0];
}
```

### 3. XP 추적

```typescript
// XP 변화 추적
let startXp = 0;

function trackXpGain(sdk: BotSDK, skillName: string) {
  const skill = sdk.getState()?.skills?.find(s => s.name === skillName);
  if (!skill) return 0;

  if (startXp === 0) {
    startXp = skill.xp;
    return 0;
  }

  return skill.xp - startXp;
}
```

## 다음 단계

다음 챕터에서는 뱅킹과 쇼핑 등 경제 시스템을 다룹니다.

---

**이전 글**: [MCP 통합](/rs-sdk-guide-05-mcp/)

**다음 글**: [경제 시스템](/rs-sdk-guide-07-economy/)
