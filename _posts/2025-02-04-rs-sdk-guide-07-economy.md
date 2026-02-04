---
layout: post
title: "RS-SDK 가이드 - 경제 시스템"
date: 2025-02-04
category: AI
tags: [rs-sdk, economy, banking, shopping, items]
series: rs-sdk-guide
part: 7
author: MaxBittker
original_url: https://github.com/MaxBittker/rs-sdk
---

## 경제 시스템 개요

RS-SDK에서 경제 활동은 뱅킹, 쇼핑, 아이템 관리로 구성됩니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Economy System                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│   │ Inventory│◀──▶│   Bank   │◀──▶│   Shop   │            │
│   │  (28)    │    │ (Large)  │    │ (NPC)    │            │
│   └──────────┘    └──────────┘    └──────────┘            │
│        │                                                    │
│        ▼                                                    │
│   ┌──────────┐                                             │
│   │  Ground  │                                             │
│   │  Items   │                                             │
│   └──────────┘                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 뱅킹

### 은행 열기

```typescript
// 은행 부스 또는 뱅커 NPC로 은행 열기
await bot.openBank();

// 은행이 열렸는지 확인
const state = sdk.getState();
if (state?.modalOpen) {
  log('은행이 열렸습니다');
}
```

### 아이템 입금

```typescript
// 단일 아이템 입금
await bot.depositItem(/logs/i);

// 특정 수량 입금
await bot.depositItem(/logs/i, 5);

// 전부 입금
await bot.depositItem(/logs/i, -1);

// 모든 아이템 입금
for (const item of sdk.getInventory()) {
  await bot.depositItem(item.name, -1);
}
```

### 아이템 출금

```typescript
// 슬롯 번호로 출금 (1개)
await bot.withdrawItem(0);

// 특정 수량 출금
await bot.withdrawItem(0, 5);

// 전부 출금
await bot.withdrawItem(0, -1);
```

### 은행 닫기

```typescript
await bot.closeBank();
```

### 완전한 뱅킹 루프

```typescript
import { runScript } from '../../sdk/runner';

await runScript(async (ctx) => {
  const { bot, sdk, log } = ctx;

  // 나무 베기 위치
  const woodcuttingSpot = { x: 3222, z: 3218 };
  // 은행 위치
  const bankSpot = { x: 3185, z: 3436 };

  while (true) {
    // 나무 베기
    await bot.walkTo(woodcuttingSpot.x, woodcuttingSpot.z);

    while (sdk.getInventory().length < 28) {
      await bot.dismissBlockingUI();

      const tree = sdk.findNearbyLoc(/^tree$/i);
      if (tree) {
        await bot.chopTree(tree);
      }
    }

    log('인벤토리 가득 참, 은행으로 이동');

    // 은행으로 이동
    await bot.walkTo(bankSpot.x, bankSpot.z);

    // 뱅킹
    await bot.openBank();
    await bot.depositItem(/logs/i, -1);
    await bot.closeBank();

    log('입금 완료, 다시 나무 베기');
  }
}, { timeout: 1800_000 }); // 30분
```

## 쇼핑

### 상점 열기

```typescript
// 상점 주인 NPC 찾기
const shopkeeper = sdk.findNearbyNpc(/shopkeeper|store owner/i);

if (shopkeeper) {
  await bot.openShop(shopkeeper);
}
```

### 구매

```typescript
// 아이템 인덱스와 수량으로 구매
await bot.buyFromShop(0, 1);  // 첫 번째 아이템 1개
await bot.buyFromShop(2, 10); // 세 번째 아이템 10개
```

### 판매

```typescript
// 인벤토리 아이템 판매
await bot.sellToShop(/logs/i, 1);   // 1개 판매
await bot.sellToShop(/fish/i, -1);  // 전부 판매
```

### 상점 닫기

```typescript
await sdk.sendCloseModal();
```

### 구매 루프 예시

```typescript
await runScript(async (ctx) => {
  const { bot, sdk, log } = ctx;

  // 상점으로 이동
  await bot.walkTo(3212, 3247);

  // 상점 열기
  const shopkeeper = sdk.findNearbyNpc(/shopkeeper/i);
  if (!shopkeeper) {
    return { success: false, message: '상점 주인을 찾을 수 없음' };
  }

  await bot.openShop(shopkeeper);

  // 도끼 구매
  await bot.buyFromShop(0, 1); // 첫 번째 아이템

  // 상점 닫기
  await sdk.sendCloseModal();

  log('구매 완료');
  return { success: true };
}, { timeout: 60_000 });
```

## 아이템 관리

### 인벤토리 조회

```typescript
// 전체 인벤토리
const inventory = sdk.getInventory();

// 아이템 검색
const axe = sdk.findInventoryItem(/axe/i);
const logs = inventory.filter(item => item.name.includes('Logs'));

// 인벤토리 상태
log(`인벤토리: ${inventory.length}/28`);
```

### 아이템 사용

```typescript
// 슬롯 번호로 사용
await sdk.sendUseItem(0);

// 아이템 찾아서 사용
const food = sdk.findInventoryItem(/bread/i);
if (food) {
  await sdk.sendUseItem(food.slot);
}
```

### 아이템 버리기

```typescript
// 단일 아이템 버리기
await sdk.sendDropItem(0);

// 특정 아이템 모두 버리기
for (const item of sdk.getInventory()) {
  if (item.name.includes('Logs')) {
    await sdk.sendDropItem(item.slot);
  }
}
```

### 지면 아이템 줍기

```typescript
// 지면 아이템 스캔
const groundItems = await sdk.scanGroundItems();

for (const item of groundItems) {
  if (item.name.match(/coins|runes/i)) {
    await sdk.sendPickupItem(item.x, item.z, item.id);
    await sdk.waitForTicks(1);
  }
}
```

## 골드 관리

### 골드 확인

```typescript
const coins = sdk.findInventoryItem(/coins/i);
const goldAmount = coins?.count ?? 0;
log(`보유 골드: ${goldAmount}`);
```

### 골드 입금

```typescript
await bot.openBank();
await bot.depositItem(/coins/i, -1); // 전액 입금
await bot.closeBank();
```

## 거래 패턴

### 자원 수집 → 판매 루프

```typescript
await runScript(async (ctx) => {
  const { bot, sdk, log } = ctx;

  const resourceSpot = { x: 3222, z: 3218 };
  const shopSpot = { x: 3212, z: 3247 };

  while (true) {
    // 자원 수집
    await bot.walkTo(resourceSpot.x, resourceSpot.z);

    while (sdk.getInventory().length < 28) {
      const tree = sdk.findNearbyLoc(/^tree$/i);
      if (tree) await bot.chopTree(tree);
    }

    // 상점으로 이동
    await bot.walkTo(shopSpot.x, shopSpot.z);

    // 판매
    const shopkeeper = sdk.findNearbyNpc(/shopkeeper/i);
    if (shopkeeper) {
      await bot.openShop(shopkeeper);
      await bot.sellToShop(/logs/i, -1);
      await sdk.sendCloseModal();
    }

    log('판매 완료, 다시 수집');
  }
}, { timeout: 1800_000 });
```

### 물품 구매 → 가공 → 판매

```typescript
await runScript(async (ctx) => {
  const { bot, sdk, log } = ctx;

  // 1. 원재료 구매
  const supplier = sdk.findNearbyNpc(/general store/i);
  if (supplier) {
    await bot.openShop(supplier);
    await bot.buyFromShop(0, 28); // 원재료 구매
    await sdk.sendCloseModal();
  }

  // 2. 가공
  const furnace = sdk.findNearbyLoc(/furnace/i);
  if (furnace) {
    // 제련 등 가공 작업
  }

  // 3. 완제품 판매
  const buyer = sdk.findNearbyNpc(/specialty shop/i);
  if (buyer) {
    await bot.openShop(buyer);
    await bot.sellToShop(/bar/i, -1);
    await sdk.sendCloseModal();
  }

  return { success: true };
}, { timeout: 300_000 });
```

## 인벤토리 최적화

### 필수 아이템 슬롯 예약

```typescript
// 도구는 항상 첫 번째 슬롯에
const toolSlot = 0;

// 나머지 27슬롯 자원용
const maxResources = 27;

function hasSpace(sdk: BotSDK): boolean {
  return sdk.getInventory().length < 28;
}
```

### 아이템 정리

```typescript
async function organizeInventory(sdk: BotSDK, bot: BotActions) {
  const inventory = sdk.getInventory();

  // 도구 확인
  const hasTool = inventory.some(item =>
    item.name.match(/axe|pickaxe|rod/i)
  );

  if (!hasTool) {
    log('도구가 없습니다!');
    // 은행에서 도구 출금
    await bot.openBank();
    await bot.withdrawItem(0); // 첫 번째 슬롯에서 도구 출금
    await bot.closeBank();
  }
}
```

## 다음 단계

다음 챕터에서는 이동과 경로 찾기를 다룹니다.

---

**이전 글**: [스킬 자동화](/rs-sdk-guide-06-skills/)

**다음 글**: [이동 & 경로](/rs-sdk-guide-08-navigation/)
