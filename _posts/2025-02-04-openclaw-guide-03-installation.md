---
layout: post
title: "OpenClaw 완벽 가이드 (3) - 설치 및 설정"
date: 2025-02-04
permalink: /openclaw-guide-03-installation/
author: Peter Steinberger
categories: [AI 에이전트, OpenClaw]
tags: [OpenClaw, Installation, Setup, Configuration, Onboarding]
original_url: "https://github.com/openclaw/openclaw"
excerpt: "OpenClaw의 설치 방법과 설정 파일 구성을 상세히 알아봅니다."
---

## 설치 방법 개요

OpenClaw는 다양한 설치 방법을 지원합니다:

| 방법 | 난이도 | 권장 대상 |
|------|--------|-----------|
| **npm (권장)** | ⭐ | 일반 사용자 |
| **pnpm** | ⭐ | 개발자 |
| **소스 빌드** | ⭐⭐ | 기여자 |
| **Docker** | ⭐⭐ | 서버 배포 |
| **Nix** | ⭐⭐⭐ | 선언적 구성 |

---

## 방법 1: npm 설치 (권장)

### 사전 요구사항

```bash
# Node.js 22+ 확인
node --version  # v22.12.0 이상

# npm 또는 pnpm
npm --version
```

### 설치

```bash
# 전역 설치
npm install -g openclaw@latest

# 또는 pnpm
pnpm add -g openclaw@latest

# 설치 확인
openclaw --version
```

### 온보딩 마법사

```bash
# 대화형 설정
openclaw onboard --install-daemon
```

온보딩 마법사가 안내하는 단계:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Onboarding Wizard                             │
│                                                                  │
│   1. 🔑 모델 선택 및 인증                                       │
│      - Anthropic OAuth 또는 API 키                              │
│      - OpenAI OAuth 또는 API 키                                 │
│                                                                  │
│   2. 📱 채널 설정                                                │
│      - WhatsApp QR 페어링                                       │
│      - Telegram 봇 토큰                                         │
│      - Discord/Slack 설정                                       │
│                                                                  │
│   3. 🛠️ 스킬 활성화                                              │
│      - 번들 스킬 선택                                           │
│      - 필수 바이너리 설치                                       │
│                                                                  │
│   4. 🔄 훅 설정                                                  │
│      - session-memory                                           │
│      - command-logger                                           │
│                                                                  │
│   5. 🚀 데몬 설치                                                │
│      - launchd (macOS) / systemd (Linux)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 방법 2: 소스에서 빌드

### 클론 및 설치

```bash
# 저장소 클론
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# 의존성 설치
pnpm install

# UI 빌드
pnpm ui:build

# 빌드
pnpm build

# 온보딩
pnpm openclaw onboard --install-daemon
```

### 개발 모드

```bash
# Gateway 개발 모드 (자동 리로드)
pnpm gateway:watch

# CLI 직접 실행
pnpm openclaw agent --message "테스트"

# 테스트 실행
pnpm test
```

---

## 방법 3: Docker

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  openclaw:
    image: openclaw/openclaw:latest
    container_name: openclaw
    restart: unless-stopped
    ports:
      - "18789:18789"
    volumes:
      - ./config:/root/.openclaw
      - ./workspace:/root/.openclaw/workspace
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
```

### 직접 실행

```bash
# 이미지 풀
docker pull openclaw/openclaw:latest

# 실행
docker run -d \
  --name openclaw \
  -p 18789:18789 \
  -v ~/.openclaw:/root/.openclaw \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  openclaw/openclaw:latest
```

### Dockerfile 커스텀

```dockerfile
FROM openclaw/openclaw:latest

# 추가 스킬 설치
COPY skills/ /root/.openclaw/workspace/skills/

# 설정 복사
COPY openclaw.json /root/.openclaw/openclaw.json

CMD ["openclaw", "gateway", "--port", "18789"]
```

---

## 설정 파일

### 파일 위치

```
~/.openclaw/
├── openclaw.json          # 메인 설정
├── credentials/           # 채널 인증 정보
│   ├── whatsapp/
│   └── ...
├── workspace/             # 에이전트 작업 공간
│   ├── AGENTS.md
│   ├── SOUL.md
│   ├── skills/
│   └── memory/
├── sessions/              # 세션 데이터
├── skills/                # 관리형 스킬
├── hooks/                 # 관리형 훅
└── logs/                  # 로그 파일
```

### 기본 설정 (openclaw.json)

```json5
// ~/.openclaw/openclaw.json
{
  // 에이전트 설정
  agent: {
    model: "anthropic/claude-opus-4-5",
  },

  // Gateway 설정
  gateway: {
    port: 18789,
    bind: "loopback",  // "loopback" | "all" | 특정 IP
  },

  // 채널 설정
  channels: {
    telegram: {
      botToken: "YOUR_BOT_TOKEN",
    },
    whatsapp: {
      allowFrom: ["+1234567890"],
    },
    discord: {
      token: "YOUR_DISCORD_TOKEN",
    },
  },

  // 브라우저 설정
  browser: {
    enabled: true,
    defaultProfile: "openclaw",
    color: "#FF4500",
  },

  // 스킬 설정
  skills: {
    entries: {
      github: { enabled: true },
      notion: { enabled: true, apiKey: "YOUR_NOTION_KEY" },
    },
  },
}
```

---

## 환경 변수

### 인증 관련

| 변수 | 설명 |
|------|------|
| `ANTHROPIC_API_KEY` | Anthropic API 키 |
| `OPENAI_API_KEY` | OpenAI API 키 |
| `ELEVENLABS_API_KEY` | ElevenLabs TTS API 키 |

### 채널 관련

| 변수 | 설명 |
|------|------|
| `TELEGRAM_BOT_TOKEN` | Telegram 봇 토큰 |
| `DISCORD_BOT_TOKEN` | Discord 봇 토큰 |
| `SLACK_BOT_TOKEN` | Slack 봇 토큰 |
| `SLACK_APP_TOKEN` | Slack 앱 토큰 |

### Gateway 관련

| 변수 | 설명 |
|------|------|
| `OPENCLAW_GATEWAY_PORT` | Gateway 포트 |
| `OPENCLAW_GATEWAY_TOKEN` | Gateway 인증 토큰 |
| `OPENCLAW_PROFILE` | 프로필 이름 |

### 설정 예시 (.env)

```bash
# ~/.profile 또는 ~/.zshrc
export ANTHROPIC_API_KEY="sk-ant-..."
export TELEGRAM_BOT_TOKEN="123456:ABC..."
export ELEVENLABS_API_KEY="..."
```

---

## 채널별 설정

### WhatsApp

```json5
{
  channels: {
    whatsapp: {
      // DM 허용 목록
      allowFrom: ["+1234567890", "+0987654321"],

      // 그룹 허용 (설정 시 allowlist로 동작)
      groups: ["*"],  // 모든 그룹 또는 특정 그룹 ID

      // DM 정책
      dmPolicy: "pairing",  // "pairing" | "open"
    },
  },
}
```

```bash
# QR 코드로 로그인
openclaw channels login
```

### Telegram

```json5
{
  channels: {
    telegram: {
      botToken: "123456:ABCDEF...",

      // 그룹 설정
      groups: {
        "*": {
          requireMention: true,  // @봇 멘션 필요
        },
      },

      // DM 허용
      allowFrom: ["user_id_1", "user_id_2"],

      // 웹훅 (선택)
      webhookUrl: "https://your-domain.com/webhook/telegram",
      webhookSecret: "your-secret",
    },
  },
}
```

### Discord

```json5
{
  channels: {
    discord: {
      token: "YOUR_DISCORD_TOKEN",

      // 네이티브 슬래시 커맨드
      commands: {
        native: true,
      },

      // DM 허용
      dm: {
        policy: "pairing",
        allowFrom: ["user_id"],
      },

      // 서버 설정
      guilds: {
        "guild_id": {
          channels: ["channel_id"],
        },
      },
    },
  },
}
```

### Slack

```json5
{
  channels: {
    slack: {
      botToken: "xoxb-...",
      appToken: "xapp-...",

      // DM 허용
      dm: {
        policy: "pairing",
        allowFrom: ["U12345"],
      },
    },
  },
}
```

---

## 데몬 설정

### macOS (launchd)

온보딩에서 `--install-daemon` 옵션으로 자동 설치됩니다.

```bash
# 수동 설치
openclaw daemon install

# 상태 확인
openclaw daemon status

# 재시작
openclaw daemon restart

# 제거
openclaw daemon uninstall
```

### Linux (systemd)

```bash
# 사용자 서비스 설치
openclaw daemon install --systemd

# 서비스 관리
systemctl --user status openclaw
systemctl --user restart openclaw
```

### 수동 실행

```bash
# 포그라운드
openclaw gateway --port 18789 --verbose

# 백그라운드
nohup openclaw gateway --port 18789 > /tmp/openclaw.log 2>&1 &
```

---

## Doctor 진단

```bash
# 전체 진단
openclaw doctor

# 출력 예시:
# ✓ Node.js version: 22.12.0
# ✓ Gateway config valid
# ✓ Anthropic API key configured
# ✓ Telegram bot token configured
# ⚠ WhatsApp not connected
# ✓ Browser enabled
# ✓ Skills loaded: 12
```

---

## 업데이트

```bash
# npm 업데이트
npm update -g openclaw

# 채널 변경
openclaw update --channel stable|beta|dev

# 업데이트 후 진단
openclaw doctor
```

---

## 문제 해결

### 일반적인 문제

| 문제 | 해결책 |
|------|--------|
| "Gateway already running" | `pkill -f openclaw-gateway` |
| "WhatsApp logged out" | `openclaw channels login` |
| "Permission denied" | Node 권한 확인, sudo 사용 자제 |
| "Port in use" | 포트 변경 또는 기존 프로세스 종료 |

### 로그 확인

```bash
# Gateway 로그
tail -f ~/.openclaw/logs/gateway.log

# macOS 통합 로그
./scripts/clawlog.sh

# 디버그 모드
openclaw gateway --verbose --debug
```

---

*다음 글에서는 메시징 채널을 상세히 살펴봅니다.*
