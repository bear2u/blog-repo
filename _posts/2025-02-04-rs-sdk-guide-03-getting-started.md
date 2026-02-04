---
layout: post
title: "RS-SDK 가이드 - 시작하기"
date: 2025-02-04
category: AI
tags: [rs-sdk, tutorial, bot-creation, script, automation]
series: rs-sdk-guide
part: 3
author: MaxBittker
original_url: https://github.com/MaxBittker/rs-sdk
---

## 설치

### 저장소 클론

```bash
git clone https://github.com/MaxBittker/rs-sdk.git
cd rs-sdk
```

### 의존성 설치

```bash
bun install
```

## 봇 생성

### 자동 생성 스크립트

```bash
# 사용자명 지정
bun scripts/create-bot.ts mybot

# 랜덤 이름 자동 생성
bun scripts/create-bot.ts
```

### 생성되는 파일

```
bots/mybot/
├── bot.env        # 자격 증명 (자동 생성된 비밀번호)
├── lab_log.md     # 세션 노트 템플릿
└── script.ts      # 시작 스크립트
```

### bot.env 형식

```bash
BOT_USERNAME=mybot
PASSWORD=자동생성된비밀번호
SERVER=wss://rs-sdk-demo.fly.dev/gateway
```

### 사용자명 규칙

- 최대 12자
- 영숫자만 허용
- 이미 사용 중인 이름은 불가

## 첫 스크립트 실행

### 데모 서버 연결

기본적으로 데모 서버에 연결됩니다:

```bash
bun bots/mybot/script.ts
```

### 기본 스크립트 구조

```typescript
// bots/mybot/script.ts
import { runScript } from '../../sdk/runner';

const result = await runScript(async (ctx) => {
  const { bot, sdk, log } = ctx;

  // 현재 상태 확인
  const state = sdk.getState();
  log(`위치: ${state?.player?.worldX}, ${state?.player?.worldZ}`);

  // 튜토리얼 스킵 (새 캐릭터인 경우)
  await bot.skipTutorial();

  // 5분 동안 나무 베기
  const endTime = Date.now() + 5 * 60_000;
  let logsChopped = 0;

  while (Date.now() < endTime) {
    await bot.dismissBlockingUI();

    const tree = sdk.findNearbyLoc(/^tree$/i);
    if (tree) {
      const r = await bot.chopTree(tree);
      if (r.success) logsChopped++;
    }
  }

  log(`총 ${logsChopped}개 통나무 획득`);
  return { logsChopped };
}, {
  timeout: 6 * 60_000
});

console.log(`성공: ${result.success}`);
```

## 상태 확인 (CLI)

스크립트 작성 전 현재 상태를 확인하세요:

```bash
bun sdk/cli.ts mybot
```

출력 예:

```
┌─────────────────────────────────────────┐
│            Bot State: mybot             │
├─────────────────────────────────────────┤
│ Position: (3222, 3218)                  │
│ Combat Level: 3                         │
│ Total Level: 32                         │
├─────────────────────────────────────────┤
│ Inventory (5/28):                       │
│   - Bronze axe (1)                      │
│   - Logs (4)                            │
├─────────────────────────────────────────┤
│ Skills:                                 │
│   Attack: 1    Strength: 1    Defence: 1│
│   Woodcutting: 5   Mining: 1            │
├─────────────────────────────────────────┤
│ Nearby NPCs:                            │
│   - Man (dist: 3)                       │
│   - Chicken (dist: 7)                   │
├─────────────────────────────────────────┤
│ Nearby Locations:                       │
│   - Tree (dist: 2)                      │
│   - Tree (dist: 4)                      │
└─────────────────────────────────────────┘
```

## 튜토리얼 처리

새 캐릭터는 튜토리얼 영역에서 시작합니다:

```typescript
// 튜토리얼이 게임플레이를 차단함
// 다른 스크립트 실행 전 반드시 스킵

await bot.skipTutorial();

// 또는 sendSkipTutorial 직접 호출
await bot.sendSkipTutorial();
```

## 스크립트 컨텍스트

스크립트는 컨텍스트 객체를 받습니다:

| 속성 | 설명 |
|------|------|
| `bot` | BotActions 인스턴스 (고수준 액션) |
| `sdk` | BotSDK 인스턴스 (저수준 SDK) |
| `log` | 캡처된 로깅 (console.log와 유사) |
| `warn` | 캡처된 경고 |
| `error` | 캡처된 에러 |

## 실행 옵션

```typescript
await runScript(async (ctx) => {
  // ...
}, {
  timeout: 300_000,        // 전체 타임아웃 (ms)
  autoConnect: true,       // 자동 연결
  disconnectAfter: false   // 완료 후 연결 해제
});
```

## 실행 결과

```typescript
interface RunResult {
  success: boolean;
  result?: any;           // 스크립트 반환값
  error?: Error;          // 실패 시 에러
  duration: number;       // 총 시간 (ms)
  logs: LogEntry[];       // 캡처된 로그
  finalState: BotWorldState;
}
```

## 스크립트 지속 시간 가이드

| 지속 시간 | 사용 시점 |
|-----------|----------|
| **10-30초** | 새 스크립트, 단일 액션, 테스트 안 된 로직, 디버깅 |
| **2-5분** | 검증된 접근법, 신뢰도 구축 중 |
| **10분+** | 검증된 전략, 그라인딩 |

> **핵심**: 5분짜리 실패한 실행보다 30초짜리 진단 실행 5번이 낫습니다. **빠르게 실패하고 간단하게 시작하세요.**

## Claude Code와 함께 사용

Claude Code에서 직접 봇을 제어할 수 있습니다:

```bash
# Claude Code에서
claude "start a new bot with name: testbot"
```

MCP 자동 검색을 통해 `.mcp.json`이 활성화됩니다.

## 채팅 표시

기본적으로 채팅은 비활성화되어 있습니다 (스캠 및 프롬프트 인젝션 방지):

```bash
# bot.env에서 활성화
SHOW_CHAT=true
```

## 여러 스크립트 관리

다양한 작업을 위한 스크립트를 분리하세요:

```
bots/mybot/
├── script.ts           # 메인 스크립트
├── woodcutting.ts      # 나무 베기 전용
├── fishing.ts          # 낚시 전용
├── combat.ts           # 전투 훈련
└── banking.ts          # 뱅킹 루프
```

## 문제 해결

### "No state received"

봇이 게임에 연결되지 않음:
- 브라우저를 먼저 열거나 `autoLaunchBrowser: true` 사용

### 스크립트 멈춤

다이얼로그 확인:
```typescript
if (state.dialog.isOpen) {
  await sdk.sendClickDialog(0);
}
```

### "Can't reach"

경로가 차단됨:
- 더 가까이 걸어가기
- 다른 대상 찾기
- 게이트/문 확인

### 잘못된 대상

더 구체적인 정규식 사용:
```typescript
// 잘못됨 - "tree stump"도 매칭
/tree/i

// 올바름 - 정확히 "tree"만
/^tree$/i
```

## 다음 단계

다음 챕터에서는 SDK API를 자세히 살펴봅니다.

---

**이전 글**: [아키텍처](/rs-sdk-guide-02-architecture/)

**다음 글**: [SDK API](/rs-sdk-guide-04-sdk-api/)
