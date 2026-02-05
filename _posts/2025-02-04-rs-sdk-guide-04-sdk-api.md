---
layout: post
title: "RS-SDK 가이드 - SDK API"
date: 2025-02-04
categories: [개발 도구, RS-SDK]
tags: [rs-sdk, api, botsdk, botactions, typescript]
author: MaxBittker
original_url: https://github.com/MaxBittker/rs-sdk
---

## API 계층 구조

RS-SDK는 두 가지 API 계층을 제공합니다:

```
┌─────────────────────────────────────────────────────┐
│                    Your Script                      │
├─────────────────────────────────────────────────────┤
│  BotActions (Porcelain - 고수준)                   │
│  - 도메인 인식 메서드                               │
│  - 효과 완료까지 대기                               │
│  - chopTree(), attackNpc(), openBank()             │
├─────────────────────────────────────────────────────┤
│  BotSDK (Plumbing - 저수준)                        │
│  - WebSocket 액션 프로토콜 1:1 매핑                │
│  - 서버 ACK까지 대기                               │
│  - sendInteractLoc(), getState(), findNearbyNpc() │
└─────────────────────────────────────────────────────┘
```

## BotSDK (저수준)

WebSocket API에 직접 매핑되는 저수준 메서드입니다.

### 연결

```typescript
import { BotSDK } from '../../sdk';

const sdk = new BotSDK({
  botUsername: 'mybot',
  password: 'password',
  gatewayUrl: 'wss://rs-sdk-demo.fly.dev/gateway',
  autoLaunchBrowser: 'auto',
  actionTimeout: 60000
});

await sdk.connect();
```

### 상태 조회

```typescript
// 현재 월드 상태
const state = sdk.getState();

// 상태 나이 (ms)
const age = sdk.getStateAge();

// 인벤토리
const inventory = sdk.getInventory();

// 스킬
const skills = state?.skills;
const woodcutting = skills?.find(s => s.name === 'Woodcutting');
```

### 상태 객체 (BotWorldState)

```typescript
interface BotWorldState {
  inGame: boolean;
  player: {
    worldX: number;
    worldZ: number;
    animId: number;
    combatLevel: number;
  };
  inventory: InventoryItem[];
  skills: SkillState[];
  nearbyNpcs: NearbyNpc[];
  nearbyLocs: NearbyLoc[];
  dialog: DialogState;
  modalOpen: boolean;
  combat?: {
    inCombat: boolean;
    target?: NearbyNpc;
  };
}
```

### 검색 메서드

```typescript
// 패턴으로 NPC 찾기
const cow = sdk.findNearbyNpc(/cow/i);

// 패턴으로 위치(오브젝트) 찾기
const tree = sdk.findNearbyLoc(/^tree$/i);

// 인벤토리 아이템 찾기
const axe = sdk.findInventoryItem(/axe/i);

// 지면 아이템 스캔
const groundItems = await sdk.scanGroundItems();
```

### 이동 액션

```typescript
// 특정 좌표로 걷기
await sdk.sendWalk(3222, 3218);

// 경로 찾기
const path = sdk.findPath(targetX, targetZ);
```

### 상호작용 액션

```typescript
// 위치(오브젝트)와 상호작용
await sdk.sendInteractLoc(loc.x, loc.z, loc.id, opIndex);

// NPC와 상호작용
await sdk.sendInteractNpc(npc.index, opIndex);

// 아이템 사용
await sdk.sendUseItem(slot);

// 아이템 버리기
await sdk.sendDropItem(slot);
```

### 다이얼로그 액션

```typescript
// 다이얼로그 옵션 클릭
await sdk.sendClickDialog(optionIndex);

// 모달 닫기
await sdk.sendCloseModal();
```

### 대기 유틸리티

```typescript
// 틱 대기
await sdk.waitForTicks(3);

// 조건 대기
await sdk.waitForCondition(
  state => state.dialog.isOpen,
  5000  // 타임아웃
);
```

## BotActions (고수준)

도메인 지식이 포함된 고수준 래퍼입니다.

### 초기화

```typescript
import { BotActions } from '../../sdk/actions';

const bot = new BotActions(sdk);
```

### UI 헬퍼

```typescript
// 레벨업 다이얼로그 등 차단 UI 닫기
await bot.dismissBlockingUI();

// 튜토리얼 스킵
await bot.skipTutorial();
```

### 이동

```typescript
// 좌표로 걷기
await bot.walkTo(3222, 3218);
```

### 스킬 액션

```typescript
// 나무 베기
const tree = sdk.findNearbyLoc(/^tree$/i);
const result = await bot.chopTree(tree);
// result: { success, message, xpGained }

// 광물 채굴
const rock = sdk.findNearbyLoc(/copper rock/i);
await bot.mineRock(rock);

// 낚시
const spot = sdk.findNearbyNpc(/fishing spot/i);
await bot.fish(spot);

// 요리
const fish = sdk.findInventoryItem(/raw/i);
await bot.cookFood(fish);
```

### 전투 액션

```typescript
// NPC 공격
await bot.attackNpc(/cow/i);

// 음식 먹기
await bot.eatFood(/bread/i);
```

### 뱅킹 액션

```typescript
// 은행 열기
await bot.openBank();

// 아이템 입금
await bot.depositItem(/logs/i);      // 1개 입금
await bot.depositItem(/coins/i, -1); // 전부 입금

// 아이템 출금
await bot.withdrawItem(0);      // 슬롯 0에서 1개
await bot.withdrawItem(0, -1);  // 슬롯 0에서 전부

// 은행 닫기
await bot.closeBank();
```

### 쇼핑 액션

```typescript
// 상점 열기
await bot.openShop(shopkeeper);

// 상점에서 구매
await bot.buyFromShop(itemIndex, quantity);

// 상점에 판매
await bot.sellToShop(/item/i, quantity);
```

### 제작 액션

```typescript
// 대장간 (모루)
await bot.smithAtAnvil(itemId);

// 활 깎기
await bot.fletchLogs();

// 가죽 제작
await bot.craftLeather(itemId);
```

### 문/게이트

```typescript
// 문 열기
const door = sdk.findNearbyLoc(/door/i);
await bot.openDoor(door);

// 게이트 열기
const gate = sdk.findNearbyLoc(/gate/i);
await bot.openDoor(gate);
```

## 액션 결과

모든 고수준 액션은 결과 객체를 반환합니다:

```typescript
interface ActionResult {
  success: boolean;
  message?: string;
  xpGained?: number;
}

// 사용 예
const result = await bot.chopTree(tree);
if (!result.success) {
  console.log(`실패: ${result.message}`);
}
```

## 타입 정의

### InventoryItem

```typescript
interface InventoryItem {
  slot: number;
  id: number;
  name: string;
  count: number;
}
```

### NearbyNpc

```typescript
interface NearbyNpc {
  index: number;
  id: number;
  name: string;
  distance: number;
  optionsWithIndex: Array<{
    text: string;
    opIndex: number;
  }>;
}
```

### NearbyLoc

```typescript
interface NearbyLoc {
  x: number;
  z: number;
  id: number;
  name: string;
  distance: number;
  optionsWithIndex: Array<{
    text: string;
    opIndex: number;
  }>;
}
```

### SkillState

```typescript
interface SkillState {
  name: string;
  baseLevel: number;
  boostedLevel: number;
  xp: number;
}
```

## API 선택 가이드

| 상황 | 사용 API |
|------|----------|
| 일반 자동화 | `bot.*` (BotActions) |
| 세밀한 제어 필요 | `sdk.*` (BotSDK) |
| 새 기능 구현 | `sdk.*` 조합 |
| 디버깅 | `sdk.getState()` |

## 다음 단계

다음 챕터에서는 MCP 통합을 통한 Claude Code 연동을 알아봅니다.

---

**이전 글**: [시작하기](/rs-sdk-guide-03-getting-started/)

**다음 글**: [MCP 통합](/rs-sdk-guide-05-mcp/)
