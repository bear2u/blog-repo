---
layout: post
title: "RS-SDK 가이드 - 베스트 프랙티스"
date: 2025-02-04
categories: [AI]
tags: [rs-sdk, best-practices, error-handling, patterns, tips]
author: MaxBittker
original_url: https://github.com/MaxBittker/rs-sdk
---

## 베스트 프랙티스 개요

안정적이고 효율적인 봇을 개발하기 위한 검증된 패턴과 팁입니다.

## 에러 처리

### 기본 에러 처리

```typescript
await runScript(async (ctx) => {
  const { bot, sdk, log, error } = ctx;

  try {
    const tree = sdk.findNearbyLoc(/^tree$/i);

    if (!tree) {
      log('나무를 찾을 수 없음, 대기 중...');
      await sdk.waitForTicks(5);
      return { success: false, reason: 'no_tree' };
    }

    const result = await bot.chopTree(tree);

    if (!result.success) {
      error(`벌목 실패: ${result.message}`);
      return { success: false, reason: result.message };
    }

    return { success: true, xpGained: result.xpGained };

  } catch (err) {
    error(`예외 발생: ${err.message}`);
    return { success: false, error: err.message };
  }
}, { timeout: 60_000 });
```

### 재시도 패턴

```typescript
async function withRetry<T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<T> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await operation();
    } catch (err) {
      lastError = err as Error;
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError;
}

// 사용 예
const result = await withRetry(
  () => bot.chopTree(tree),
  3,
  2000
);
```

### 타임아웃 처리

```typescript
async function withTimeout<T>(
  operation: Promise<T>,
  timeout: number,
  timeoutMessage: string = 'Operation timed out'
): Promise<T> {
  return Promise.race([
    operation,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(timeoutMessage)), timeout)
    )
  ]);
}

// 사용 예
try {
  const result = await withTimeout(
    bot.openBank(),
    10000,
    '은행 열기 시간 초과'
  );
} catch (err) {
  log(`타임아웃: ${err.message}`);
}
```

## 상태 검증

### 액션 전 검증

```typescript
function validateState(sdk: BotSDK): { valid: boolean; issues: string[] } {
  const state = sdk.getState();
  const issues: string[] = [];

  if (!state) {
    issues.push('상태를 가져올 수 없음');
    return { valid: false, issues };
  }

  if (!state.inGame) {
    issues.push('게임에 로그인되지 않음');
  }

  if (state.dialog?.isOpen) {
    issues.push('다이얼로그가 열려 있음');
  }

  if (state.modalOpen) {
    issues.push('모달이 열려 있음');
  }

  return { valid: issues.length === 0, issues };
}

// 사용 예
const { valid, issues } = validateState(sdk);
if (!valid) {
  log(`상태 문제: ${issues.join(', ')}`);
  await bot.dismissBlockingUI();
}
```

### 액션 후 검증

```typescript
async function verifyAction(
  sdk: BotSDK,
  condition: (state: BotWorldState) => boolean,
  timeout: number = 5000
): Promise<boolean> {
  try {
    await sdk.waitForCondition(condition, timeout);
    return true;
  } catch {
    return false;
  }
}

// 사용 예
await bot.chopTree(tree);

const gotLog = await verifyAction(
  sdk,
  state => state.inventory.some(item => item.name.includes('Logs')),
  10000
);

if (!gotLog) {
  log('통나무를 얻지 못함');
}
```

## 스크립트 구조

### 모듈화

```typescript
// skills/woodcutting.ts
export async function chopTrees(
  ctx: ScriptContext,
  options: { duration?: number; dropLogs?: boolean } = {}
) {
  const { bot, sdk, log } = ctx;
  const { duration = 300_000, dropLogs = true } = options;

  const endTime = Date.now() + duration;
  let logsChopped = 0;

  while (Date.now() < endTime) {
    await bot.dismissBlockingUI();

    const tree = sdk.findNearbyLoc(/^tree$/i);
    if (tree) {
      const result = await bot.chopTree(tree);
      if (result.success) logsChopped++;
    }

    if (dropLogs && sdk.getInventory().length >= 28) {
      await dropAllLogs(sdk);
    }
  }

  return { logsChopped };
}

// main script
import { chopTrees } from './skills/woodcutting';

await runScript(async (ctx) => {
  return await chopTrees(ctx, { duration: 600_000, dropLogs: true });
}, { timeout: 660_000 });
```

### 설정 분리

```typescript
// config.ts
export const CONFIG = {
  // 타이밍
  TICK_WAIT: 1,
  ACTION_TIMEOUT: 30000,
  SCRIPT_TIMEOUT: 300000,

  // 위치
  LOCATIONS: {
    LUMBRIDGE_TREES: { x: 3222, z: 3218 },
    VARROCK_BANK: { x: 3185, z: 3436 },
  },

  // 패턴
  PATTERNS: {
    TREE: /^tree$/i,
    LOGS: /logs/i,
    FOOD: /bread|meat|fish/i,
  },
};

// 사용
import { CONFIG } from './config';

const tree = sdk.findNearbyLoc(CONFIG.PATTERNS.TREE);
await bot.walkTo(CONFIG.LOCATIONS.LUMBRIDGE_TREES.x, CONFIG.LOCATIONS.LUMBRIDGE_TREES.z);
```

## 로깅

### 구조화된 로깅

```typescript
function createLogger(ctx: ScriptContext) {
  const { log, warn, error } = ctx;

  return {
    info: (msg: string, data?: object) => {
      const formatted = data ? `${msg} ${JSON.stringify(data)}` : msg;
      log(`[INFO] ${formatted}`);
    },

    warn: (msg: string, data?: object) => {
      const formatted = data ? `${msg} ${JSON.stringify(data)}` : msg;
      warn(`[WARN] ${formatted}`);
    },

    error: (msg: string, data?: object) => {
      const formatted = data ? `${msg} ${JSON.stringify(data)}` : msg;
      error(`[ERROR] ${formatted}`);
    },

    debug: (msg: string, data?: object) => {
      if (process.env.DEBUG) {
        const formatted = data ? `${msg} ${JSON.stringify(data)}` : msg;
        log(`[DEBUG] ${formatted}`);
      }
    },
  };
}

// 사용
const logger = createLogger(ctx);
logger.info('나무 베기 시작', { target: tree.name });
logger.error('실패', { reason: result.message });
```

### 진행 상황 추적

```typescript
class ProgressTracker {
  private startTime: number;
  private actions: number = 0;
  private xpGained: number = 0;

  constructor() {
    this.startTime = Date.now();
  }

  recordAction(xp: number = 0) {
    this.actions++;
    this.xpGained += xp;
  }

  getStats() {
    const elapsed = (Date.now() - this.startTime) / 1000 / 60; // 분
    return {
      actions: this.actions,
      xpGained: this.xpGained,
      elapsedMinutes: elapsed.toFixed(1),
      actionsPerMinute: (this.actions / elapsed).toFixed(1),
      xpPerMinute: (this.xpGained / elapsed).toFixed(1),
    };
  }
}

// 사용
const tracker = new ProgressTracker();

while (condition) {
  const result = await bot.chopTree(tree);
  if (result.success) {
    tracker.recordAction(result.xpGained ?? 0);
  }
}

log(`통계: ${JSON.stringify(tracker.getStats())}`);
```

## 안티 패턴 피하기

### 하드코딩된 대기 시간

```typescript
// 나쁨 ❌
await new Promise(r => setTimeout(r, 5000));

// 좋음 ✅
await sdk.waitForTicks(3);
await sdk.waitForCondition(state => !state.player.animId, 10000);
```

### 무한 루프 without 탈출 조건

```typescript
// 나쁨 ❌
while (true) {
  await bot.chopTree(tree);
}

// 좋음 ✅
const endTime = Date.now() + 300_000;
while (Date.now() < endTime) {
  await bot.dismissBlockingUI(); // 블로킹 UI 처리

  const tree = sdk.findNearbyLoc(/^tree$/i);
  if (!tree) {
    await sdk.waitForTicks(5);
    continue;
  }

  await bot.chopTree(tree);
}
```

### 상태 확인 없는 액션

```typescript
// 나쁨 ❌
await bot.chopTree(tree);
await bot.chopTree(tree);
await bot.chopTree(tree);

// 좋음 ✅
while (hasTreeNearby(sdk)) {
  await bot.dismissBlockingUI();

  const tree = sdk.findNearbyLoc(/^tree$/i);
  if (tree) {
    await bot.chopTree(tree);
  }

  // 인벤토리 확인
  if (sdk.getInventory().length >= 28) {
    break;
  }
}
```

## 스크립트 길이 가이드

| 지속 시간 | 사용 시점 |
|-----------|----------|
| **10-30초** | 새 스크립트 테스트, 디버깅, 단일 액션 |
| **2-5분** | 검증된 접근법, 신뢰도 구축 |
| **10분+** | 검증된 전략, 장시간 그라인딩 |

> **핵심 원칙**: 5분짜리 실패 실행보다 30초짜리 진단 실행 5번이 낫습니다.

## 디버깅 팁

### CLI로 상태 확인

```bash
# 봇 상태 확인
bun sdk/cli.ts mybot
```

### 상태 스냅샷

```typescript
function debugState(sdk: BotSDK, log: Function) {
  const state = sdk.getState();

  log('=== 상태 스냅샷 ===');
  log(`위치: (${state?.player?.worldX}, ${state?.player?.worldZ})`);
  log(`애니메이션: ${state?.player?.animId}`);
  log(`인벤토리: ${sdk.getInventory().length}/28`);
  log(`다이얼로그: ${state?.dialog?.isOpen ? '열림' : '닫힘'}`);
  log(`모달: ${state?.modalOpen ? '열림' : '닫힘'}`);
  log(`전투: ${state?.combat?.inCombat ? '전투 중' : '평화'}`);

  // 주변 객체
  const npcs = state?.nearbyNpcs.slice(0, 3);
  const locs = state?.nearbyLocs.slice(0, 3);

  log(`근처 NPC: ${npcs?.map(n => n.name).join(', ')}`);
  log(`근처 오브젝트: ${locs?.map(l => l.name).join(', ')}`);
  log('==================');
}
```

### 문제 격리

```typescript
// 문제가 있는 기능을 격리해서 테스트
await runScript(async (ctx) => {
  const { bot, sdk, log } = ctx;

  // 1. 상태 확인만
  debugState(sdk, log);

  // 2. 단일 액션만
  const tree = sdk.findNearbyLoc(/^tree$/i);
  log(`찾은 나무: ${tree?.name} at (${tree?.x}, ${tree?.z})`);

  // 3. 결과 확인
  if (tree) {
    const result = await bot.chopTree(tree);
    log(`결과: ${JSON.stringify(result)}`);
  }

  return { debug: true };
}, { timeout: 30_000 });
```

## 다음 단계

다음 챕터에서는 자체 서버 호스팅 방법을 다룹니다.

---

**이전 글**: [이동 & 경로](/rs-sdk-guide-08-navigation/)

**다음 글**: [서버 호스팅](/rs-sdk-guide-10-hosting/)
