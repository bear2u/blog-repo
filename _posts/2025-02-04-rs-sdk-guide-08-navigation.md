---
layout: post
title: "RS-SDK 가이드 - 이동 & 경로"
date: 2025-02-04
categories: [개발 도구, RS-SDK]
tags: [rs-sdk, navigation, pathfinding, walking, doors, gates]
author: MaxBittker
original_url: https://github.com/MaxBittker/rs-sdk
---

## 이동 시스템 개요

RS-SDK의 이동 시스템은 좌표 기반 걷기, 경로 찾기, 장애물 처리를 포함합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                   Navigation System                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│   │  Player  │───▶│Pathfinder│───▶│   Walk   │            │
│   │ Position │    │          │    │  Action  │            │
│   └──────────┘    └──────────┘    └──────────┘            │
│                         │                                   │
│                         ▼                                   │
│                   ┌──────────┐                             │
│                   │ Obstacle │                             │
│                   │ Handling │                             │
│                   │(Door/Gate)│                             │
│                   └──────────┘                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 기본 이동

### 좌표로 걷기

```typescript
// 특정 좌표로 이동
await bot.walkTo(3222, 3218);

// 저수준 API
await sdk.sendWalk(3222, 3218);
```

### 현재 위치 확인

```typescript
const state = sdk.getState();
const x = state?.player?.worldX;
const z = state?.player?.worldZ;

log(`현재 위치: (${x}, ${z})`);
```

### 거리 계산

```typescript
function getDistance(x1: number, z1: number, x2: number, z2: number): number {
  return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(z2 - z1, 2));
}

// 대상까지의 거리
const state = sdk.getState();
const playerX = state?.player?.worldX ?? 0;
const playerZ = state?.player?.worldZ ?? 0;
const targetX = 3185;
const targetZ = 3436;

const distance = getDistance(playerX, playerZ, targetX, targetZ);
log(`목표까지 거리: ${distance.toFixed(1)} 타일`);
```

## 경로 찾기

### 경로 계산

```typescript
// SDK 내장 경로 찾기
const path = sdk.findPath(targetX, targetZ);

if (path && path.length > 0) {
  log(`경로 발견: ${path.length} 스텝`);

  // 경로 따라 이동
  for (const step of path) {
    await sdk.sendWalk(step.x, step.z);
    await sdk.waitForTicks(1);
  }
}
```

### 도달 가능 여부 확인

```typescript
async function canReach(sdk: BotSDK, x: number, z: number): Promise<boolean> {
  const path = sdk.findPath(x, z);
  return path !== null && path.length > 0;
}

// 사용 예
if (await canReach(sdk, bankX, bankZ)) {
  await bot.walkTo(bankX, bankZ);
} else {
  log('목표에 도달할 수 없습니다');
}
```

## 장애물 처리

### 문 열기

```typescript
// 문 찾기
const door = sdk.findNearbyLoc(/door/i);

if (door) {
  // 문이 닫혀있는지 확인 (옵션에 "Open" 있으면 닫힌 상태)
  const canOpen = door.optionsWithIndex.some(opt =>
    opt.text.toLowerCase() === 'open'
  );

  if (canOpen) {
    await bot.openDoor(door);
  }
}
```

### 게이트 열기

```typescript
// 게이트 찾기
const gate = sdk.findNearbyLoc(/gate/i);

if (gate) {
  const canOpen = gate.optionsWithIndex.some(opt =>
    opt.text.toLowerCase() === 'open'
  );

  if (canOpen) {
    await bot.openDoor(gate); // openDoor로 게이트도 처리 가능
  }
}
```

### 장애물 감지 및 처리 패턴

```typescript
async function navigateWithObstacles(
  bot: BotActions,
  sdk: BotSDK,
  targetX: number,
  targetZ: number
) {
  const maxAttempts = 5;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    // 이동 시도
    await bot.walkTo(targetX, targetZ);
    await sdk.waitForTicks(3);

    // 도착 확인
    const state = sdk.getState();
    const currentX = state?.player?.worldX ?? 0;
    const currentZ = state?.player?.worldZ ?? 0;

    const distance = getDistance(currentX, currentZ, targetX, targetZ);

    if (distance < 3) {
      return true; // 도착
    }

    // 장애물 확인
    const door = sdk.findNearbyLoc(/door/i);
    const gate = sdk.findNearbyLoc(/gate/i);

    if (door) {
      await bot.openDoor(door);
    } else if (gate) {
      await bot.openDoor(gate);
    }

    await sdk.waitForTicks(2);
  }

  return false; // 도달 실패
}
```

## 주요 위치

### 위치 상수 정의

```typescript
const LOCATIONS = {
  // 은행
  VARROCK_BANK: { x: 3185, z: 3436 },
  LUMBRIDGE_BANK: { x: 3208, z: 3220 },
  FALADOR_BANK: { x: 2946, z: 3368 },

  // 스킬 위치
  LUMBRIDGE_TREES: { x: 3222, z: 3218 },
  VARROCK_MINE: { x: 3285, z: 3365 },
  BARBARIAN_FISHING: { x: 3104, z: 3433 },

  // 상점
  VARROCK_GENERAL_STORE: { x: 3212, z: 3247 },
  LUMBRIDGE_GENERAL_STORE: { x: 3211, z: 3247 },
};
```

### 위치 기반 이동

```typescript
async function goToLocation(bot: BotActions, locationName: string) {
  const location = LOCATIONS[locationName];

  if (!location) {
    throw new Error(`Unknown location: ${locationName}`);
  }

  await bot.walkTo(location.x, location.z);
}

// 사용 예
await goToLocation(bot, 'VARROCK_BANK');
```

## 런 에너지

RS-SDK 데모 서버에서는 런 에너지가 무제한입니다:

```typescript
// 런 에너지 걱정 없이 이동 가능
// 원본 게임과 달리 에너지가 소모되지 않음
await bot.walkTo(3185, 3436); // 장거리도 OK
```

## 이동 대기

### 도착까지 대기

```typescript
async function walkAndWait(
  bot: BotActions,
  sdk: BotSDK,
  x: number,
  z: number,
  timeout: number = 30000
): Promise<boolean> {
  await bot.walkTo(x, z);

  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    const state = sdk.getState();
    const currentX = state?.player?.worldX ?? 0;
    const currentZ = state?.player?.worldZ ?? 0;

    if (Math.abs(currentX - x) <= 1 && Math.abs(currentZ - z) <= 1) {
      return true;
    }

    await sdk.waitForTicks(1);
  }

  return false;
}
```

### 애니메이션 기반 대기

```typescript
// 걷기 애니메이션이 끝날 때까지 대기
async function waitUntilIdle(sdk: BotSDK, timeout: number = 10000) {
  const walkingAnimIds = [819, 820, 821, 822, 823, 824];

  await sdk.waitForCondition(
    state => !walkingAnimIds.includes(state.player?.animId ?? 0),
    timeout
  );
}
```

## 복잡한 경로 예시

### 은행 루프

```typescript
await runScript(async (ctx) => {
  const { bot, sdk, log } = ctx;

  const treeSpot = LOCATIONS.LUMBRIDGE_TREES;
  const bankSpot = LOCATIONS.LUMBRIDGE_BANK;

  while (true) {
    // 나무 위치로 이동
    log('나무 위치로 이동 중...');
    await navigateWithObstacles(bot, sdk, treeSpot.x, treeSpot.z);

    // 인벤토리 가득 찰 때까지 벌목
    while (sdk.getInventory().length < 28) {
      await bot.dismissBlockingUI();

      const tree = sdk.findNearbyLoc(/^tree$/i);
      if (tree) {
        await bot.chopTree(tree);
      }
    }

    // 은행으로 이동
    log('은행으로 이동 중...');
    await navigateWithObstacles(bot, sdk, bankSpot.x, bankSpot.z);

    // 입금
    await bot.openBank();
    await bot.depositItem(/logs/i, -1);
    await bot.closeBank();

    log('사이클 완료');
  }
}, { timeout: 1800_000 });
```

### 여러 위치 순회

```typescript
const route = [
  LOCATIONS.LUMBRIDGE_TREES,
  LOCATIONS.VARROCK_BANK,
  LOCATIONS.VARROCK_MINE,
  LOCATIONS.VARROCK_BANK,
];

for (const location of route) {
  await navigateWithObstacles(bot, sdk, location.x, location.z);
  log(`도착: (${location.x}, ${location.z})`);

  // 각 위치에서 작업 수행
  await sdk.waitForTicks(10);
}
```

## 문제 해결

### "Can't reach" 에러

```typescript
// 더 가까이 이동 시도
const state = sdk.getState();
const currentX = state?.player?.worldX ?? 0;
const currentZ = state?.player?.worldZ ?? 0;

// 목표 방향으로 중간 지점 계산
const midX = Math.floor((currentX + targetX) / 2);
const midZ = Math.floor((currentZ + targetZ) / 2);

await bot.walkTo(midX, midZ);
await sdk.waitForTicks(3);
await bot.walkTo(targetX, targetZ);
```

### 장애물에 막힘

```typescript
// 주변 문/게이트 확인
const obstacles = sdk.getState()?.nearbyLocs.filter(loc =>
  loc.name.match(/door|gate/i)
);

for (const obstacle of obstacles ?? []) {
  const canOpen = obstacle.optionsWithIndex.some(opt =>
    opt.text.toLowerCase() === 'open'
  );

  if (canOpen && obstacle.distance < 5) {
    await bot.openDoor(obstacle);
  }
}
```

## 다음 단계

다음 챕터에서는 에러 처리와 베스트 프랙티스를 다룹니다.

---

**이전 글**: [경제 시스템](/rs-sdk-guide-07-economy/)

**다음 글**: [베스트 프랙티스](/rs-sdk-guide-09-best-practices/)
