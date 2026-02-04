---
layout: post
title: "OpenClaw 완벽 가이드 (4) - 메시징 채널"
date: 2025-02-04
permalink: /openclaw-guide-04-channels/
author: Peter Steinberger
category: AI
tags: [OpenClaw, Channels, WhatsApp, Telegram, Discord, Slack, Signal]
series: openclaw-guide
part: 4
original_url: "https://github.com/openclaw/openclaw"
excerpt: "OpenClaw이 지원하는 다양한 메시징 채널의 설정과 사용법을 알아봅니다."
---

## 채널 개요

OpenClaw는 **15개 이상의 메시징 플랫폼**을 지원합니다. 모든 채널은 Gateway를 통해 연결되며, 하나의 AI가 모든 채널에 응답합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Supported Channels                            │
│                                                                  │
│   Built-in (Core):                                              │
│   ├── WhatsApp (Baileys)                                        │
│   ├── Telegram (grammY)                                         │
│   ├── Discord (discord.js)                                      │
│   ├── Slack (Bolt SDK)                                          │
│   ├── Signal (signal-cli)                                       │
│   ├── BlueBubbles (iMessage) ⭐ 권장                            │
│   ├── iMessage (legacy)                                         │
│   ├── Feishu/Lark                                               │
│   ├── LINE                                                      │
│   └── WebChat                                                   │
│                                                                  │
│   Extensions (Plugins):                                         │
│   ├── Microsoft Teams                                           │
│   ├── Google Chat                                               │
│   ├── Matrix                                                    │
│   ├── Mattermost                                                │
│   ├── Nextcloud Talk                                            │
│   ├── Nostr                                                     │
│   ├── Twitch                                                    │
│   ├── Zalo / Zalo Personal                                      │
│   └── Tlon (Urbit)                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## WhatsApp

**Baileys** 라이브러리를 사용한 WhatsApp Web 연결입니다.

### 설정

```json5
// ~/.openclaw/openclaw.json
{
  channels: {
    whatsapp: {
      // DM 허용 목록
      allowFrom: ["+1234567890"],

      // 그룹 허용 (설정 시 allowlist)
      groups: ["group_jid", "*"],  // "*"는 모든 그룹

      // DM 정책
      dmPolicy: "pairing",  // "pairing" | "open"
    },
  },
}
```

### QR 페어링

```bash
# QR 코드 표시
openclaw channels login

# 결과:
# ┌──────────────────────────────────────┐
# │  ▄▄▄▄▄▄▄  ▄  ▄▄▄  ▄  ▄▄▄▄▄▄▄         │
# │  █ ▄▄▄ █ ▀█ ▄▄█▄█▀  █ ▄▄▄ █         │
# │  ...                                 │
# └──────────────────────────────────────┘
# WhatsApp 앱 → 연결된 기기 → QR 스캔
```

### 주요 특징

| 기능 | 지원 |
|------|------|
| 텍스트 | ✅ |
| 이미지/비디오 | ✅ |
| 음성 메시지 | ✅ (전사) |
| 리액션 | ✅ |
| 그룹 | ✅ |
| 상태 표시 | ✅ |

---

## Telegram

**grammY** 라이브러리를 사용한 Telegram Bot API입니다.

### 봇 생성

1. [@BotFather](https://t.me/BotFather)에게 `/newbot` 전송
2. 봇 이름과 사용자명 설정
3. 토큰 복사

### 설정

```json5
{
  channels: {
    telegram: {
      botToken: "123456789:ABCDEF...",

      // 그룹 설정
      groups: {
        "*": {
          requireMention: true,  // @봇 멘션 필요
        },
        "-100123456789": {
          requireMention: false,  // 모든 메시지 응답
        },
      },

      // DM 허용
      allowFrom: ["user_id_1", "user_id_2"],

      // 웹훅 모드 (선택)
      webhookUrl: "https://your-domain.com/telegram",
      webhookSecret: "random-secret",
    },
  },
}
```

### 명령어

```bash
# 환경 변수로 설정
export TELEGRAM_BOT_TOKEN="123456:ABC..."

# 채널 상태 확인
openclaw channels status --probe
```

### 그룹 활성화 모드

| 모드 | 설명 |
|------|------|
| **mention** | `@bot` 멘션 시에만 응답 |
| **always** | 모든 메시지에 응답 |

채팅 명령: `/activation mention|always`

---

## Discord

**discord.js**를 사용한 Discord 봇입니다.

### 봇 생성

1. [Discord Developer Portal](https://discord.com/developers/applications) 접속
2. New Application → 봇 생성
3. Bot 탭 → TOKEN 복사
4. OAuth2 → bot + applications.commands 스코프
5. 서버에 봇 초대

### 설정

```json5
{
  channels: {
    discord: {
      token: "YOUR_DISCORD_TOKEN",

      // 슬래시 커맨드
      commands: {
        native: true,   // 네이티브 슬래시 커맨드
        text: true,     // 텍스트 커맨드 (!help)
      },

      // DM 설정
      dm: {
        policy: "pairing",  // "pairing" | "open"
        allowFrom: ["user_id"],
      },

      // 서버별 설정
      guilds: {
        "1234567890": {
          channels: ["9876543210"],  // 허용 채널
          allowFrom: ["*"],          // 모든 사용자
        },
      },

      // 미디어 크기 제한 (MB)
      mediaMaxMb: 25,
    },
  },
}
```

### 슬래시 커맨드

Discord에서 사용 가능한 명령:

- `/ask <message>` - AI에게 질문
- `/status` - 세션 상태
- `/new` - 세션 리셋
- `/think <level>` - 사고 레벨 설정

---

## Slack

**Bolt SDK**를 사용한 Slack 앱입니다.

### 앱 생성

1. [Slack API](https://api.slack.com/apps) 접속
2. Create New App → From scratch
3. OAuth & Permissions → Bot Token Scopes 추가:
   - `chat:write`
   - `channels:history`
   - `groups:history`
   - `im:history`
   - `users:read`
4. Socket Mode 활성화
5. Event Subscriptions → Subscribe to bot events:
   - `message.channels`
   - `message.groups`
   - `message.im`

### 설정

```json5
{
  channels: {
    slack: {
      botToken: "xoxb-...",   // Bot User OAuth Token
      appToken: "xapp-...",   // App-Level Token

      // DM 설정
      dm: {
        policy: "pairing",
        allowFrom: ["U12345"],
      },
    },
  },
}
```

### 환경 변수

```bash
export SLACK_BOT_TOKEN="xoxb-..."
export SLACK_APP_TOKEN="xapp-..."
```

---

## Signal

**signal-cli**를 사용한 Signal 연결입니다.

### 사전 요구사항

```bash
# signal-cli 설치
# macOS
brew install signal-cli

# Linux
# https://github.com/AsamK/signal-cli 참조
```

### 설정

```json5
{
  channels: {
    signal: {
      phoneNumber: "+1234567890",

      // 허용 목록
      allowFrom: ["+0987654321"],

      // 그룹 허용
      groups: ["group_id"],
    },
  },
}
```

### 등록

```bash
# 전화번호 등록
signal-cli -u +1234567890 register

# 인증 코드 입력
signal-cli -u +1234567890 verify 123456
```

---

## iMessage (BlueBubbles)

**BlueBubbles**는 iMessage 연동의 **권장 방법**입니다.

### BlueBubbles 서버 설정

1. macOS에 [BlueBubbles](https://bluebubbles.app/) 설치
2. 서버 시작 및 비밀번호 설정
3. Private API 활성화 (선택)

### 설정

```json5
{
  channels: {
    bluebubbles: {
      serverUrl: "http://localhost:1234",
      password: "your-bluebubbles-password",

      // 웹훅 경로
      webhookPath: "/webhook/bluebubbles",

      // 허용 목록
      allowFrom: ["email@icloud.com", "+1234567890"],
    },
  },
}
```

### 지원 기능

| 기능 | BlueBubbles | iMessage (legacy) |
|------|-------------|-------------------|
| 텍스트 | ✅ | ✅ |
| 미디어 | ✅ | ✅ |
| 리액션 | ✅ | ❌ |
| 편집/삭제 | ✅ | ❌ |
| 그룹 관리 | ✅ | ❌ |
| 효과 | ✅ | ❌ |

---

## Microsoft Teams (Extension)

### 설치

```bash
# 플러그인 설치
openclaw plugins install msteams
```

### 설정

```json5
{
  channels: {
    msteams: {
      appId: "your-app-id",
      appPassword: "your-app-password",

      // 허용 목록
      allowFrom: ["user@company.com"],

      // 그룹 정책
      groupPolicy: "pairing",  // "pairing" | "open"
      groupAllowFrom: ["*"],
    },
  },
}
```

---

## Google Chat (Extension)

### 설치

```bash
openclaw plugins install googlechat
```

### 설정

```json5
{
  channels: {
    googlechat: {
      // 서비스 계정 키 파일
      credentialsPath: "~/.openclaw/googlechat-credentials.json",

      // 허용 도메인
      allowedDomains: ["yourcompany.com"],
    },
  },
}
```

---

## WebChat

WebChat은 Gateway에 내장된 웹 채팅 UI입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    WebChat Architecture                          │
│                                                                  │
│   Browser ──▶ Gateway WS ──▶ Pi Agent ──▶ Model                │
│              (ws://127.0.0.1:18789)                             │
│                                                                  │
│   Features:                                                     │
│   • 실시간 스트리밍 응답                                        │
│   • 마크다운 렌더링                                              │
│   • 코드 하이라이팅                                              │
│   • 파일 업로드                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 접근

```bash
# Gateway 시작 후
open http://localhost:18789
```

---

## 채널 공통 기능

### DM 페어링

알 수 없는 발신자에게는 페어링 코드가 발급됩니다:

```bash
# 페어링 승인
openclaw pairing approve telegram ABC123

# 페어링 목록
openclaw pairing list
```

### 그룹 규칙

| 설정 | 설명 |
|------|------|
| `requireMention` | @멘션 필요 여부 |
| `replyTag` | 답장 시 멘션 태그 |
| `chunking` | 긴 메시지 분할 |

### 미디어 처리

```
┌─────────────────────────────────────────────────────────────────┐
│                    Media Pipeline                                │
│                                                                  │
│   이미지 ──▶ 리사이즈/압축 ──▶ 모델 전송                       │
│   오디오 ──▶ 전사 (Whisper) ──▶ 텍스트 변환                    │
│   비디오 ──▶ 프레임 추출 ──▶ 분석                              │
│   PDF ────▶ 텍스트 추출 ──▶ 컨텍스트 포함                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 채널 상태 확인

```bash
# 모든 채널 상태
openclaw channels status

# 프로브 포함 (실제 연결 테스트)
openclaw channels status --probe

# 특정 채널
openclaw channels status telegram
```

---

## 채팅 명령어

모든 채널에서 사용 가능한 명령:

| 명령 | 설명 |
|------|------|
| `/status` | 세션 상태 |
| `/new` | 세션 리셋 |
| `/reset` | 세션 리셋 (별칭) |
| `/compact` | 컨텍스트 압축 |
| `/think <level>` | 사고 레벨 (off~xhigh) |
| `/verbose on\|off` | 상세 모드 |
| `/usage off\|tokens\|full` | 사용량 표시 |
| `/restart` | Gateway 재시작 (그룹: owner만) |
| `/activation mention\|always` | 그룹 활성화 모드 |

---

*다음 글에서는 스킬 시스템을 살펴봅니다.*
