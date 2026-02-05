---
layout: post
title: "OpenClaw 완벽 가이드 (10) - 보안 & 배포"
date: 2025-02-04
permalink: /openclaw-guide-10-security/
author: Peter Steinberger
categories: [AI 에이전트, OpenClaw]
tags: [OpenClaw, Security, Docker, Sandboxing, Deployment, Tailscale]
original_url: "https://github.com/openclaw/openclaw"
excerpt: "OpenClaw의 보안 모델, 샌드박싱, Docker 배포, 원격 접근을 알아봅니다."
---

## 보안 모델 개요

OpenClaw는 **신뢰할 수 없는 입력**을 처리합니다. 메시징 채널의 DM은 누구나 보낼 수 있으므로, 보안이 중요합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Security Architecture                         │
│                                                                  │
│   Untrusted Input                                               │
│   (DM, 그룹 메시지)                                              │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────────┐                                           │
│   │  DM Pairing     │ ──▶ 허용 목록 확인                        │
│   │  Allowlist      │                                           │
│   └────────┬────────┘                                           │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                           │
│   │  Sandboxing     │ ──▶ non-main 세션 격리                    │
│   │  (Docker)       │                                           │
│   └────────┬────────┘                                           │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                           │
│   │  Tool Allow/    │ ──▶ 위험 도구 차단                        │
│   │  Deny Lists     │                                           │
│   └─────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## DM 정책

### 기본 동작

모든 채널에서 알 수 없는 발신자는 **페어링**이 필요합니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DM Pairing Flow                               │
│                                                                  │
│   1. 알 수 없는 사용자가 DM 전송                                │
│   2. OpenClaw: "페어링 코드: ABC123"                            │
│   3. 관리자: openclaw pairing approve telegram ABC123           │
│   4. 사용자가 허용 목록에 추가됨                                │
│   5. 이후 메시지 정상 처리                                       │
└─────────────────────────────────────────────────────────────────┘
```

### DM 정책 옵션

| 정책 | 설명 |
|------|------|
| `pairing` | 페어링 코드 필요 (기본) |
| `open` | 모든 DM 허용 (위험) |

```json5
{
  channels: {
    telegram: {
      dmPolicy: "pairing",
      allowFrom: ["user_id_1", "user_id_2"],
    },
    discord: {
      dm: {
        policy: "pairing",
        allowFrom: ["*"],  // open과 유사하지만 명시적
      },
    },
  },
}
```

### Doctor 경고

```bash
openclaw doctor

# 위험한 설정 경고:
# ⚠ Telegram DM policy is "open" - this is risky!
# ⚠ WhatsApp allowFrom includes "*" - anyone can message
```

---

## 샌드박싱

### 샌드박스 모드

```json5
{
  agents: {
    defaults: {
      sandbox: {
        mode: "non-main",  // "off" | "non-main" | "all"
      },
    },
  },
}
```

| 모드 | 설명 |
|------|------|
| `off` | 샌드박싱 비활성화 |
| `non-main` | main 세션 외 모든 세션 샌드박싱 (권장) |
| `all` | 모든 세션 샌드박싱 |

### Docker 샌드박스

non-main 세션은 **격리된 Docker 컨테이너**에서 실행됩니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Sandbox                                │
│                                                                  │
│   Host                          Container (per session)         │
│   ┌─────────────┐               ┌─────────────────────┐         │
│   │  Gateway    │──────────────▶│  bash               │         │
│   │             │               │  read/write (제한)  │         │
│   │             │               │  process            │         │
│   └─────────────┘               └─────────────────────┘         │
│                                                                  │
│   차단됨:                                                       │
│   • browser                                                     │
│   • canvas                                                      │
│   • nodes                                                       │
│   • cron                                                        │
│   • discord/slack 액션                                          │
│   • gateway 제어                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 샌드박스 설정

```json5
{
  agents: {
    defaults: {
      sandbox: {
        mode: "non-main",

        // 허용 도구
        allow: [
          "bash",
          "process",
          "read",
          "write",
          "edit",
          "sessions_list",
          "sessions_history",
          "sessions_send",
          "sessions_spawn",
        ],

        // 차단 도구
        deny: [
          "browser",
          "canvas",
          "nodes",
          "cron",
          "discord",
          "gateway",
        ],

        // Docker 설정
        docker: {
          image: "node:22-slim",
          setupCommand: "apt-get update && apt-get install -y git",
          network: "none",  // 네트워크 격리
          memory: "512m",
          cpus: "1",
        },
      },
    },
  },
}
```

---

## Gateway 인증

### 인증 모드

```json5
{
  gateway: {
    auth: {
      mode: "password",  // "none" | "password" | "token"
      password: "secure-password",
      allowTailscale: true,  // Tailscale ID 헤더 신뢰
    },
  },
}
```

| 모드 | 설명 |
|------|------|
| `none` | 인증 없음 (로컬 전용) |
| `password` | 비밀번호 인증 |
| `token` | 토큰 인증 |

### 토큰 인증

```bash
# 환경 변수로 토큰 설정
export OPENCLAW_GATEWAY_TOKEN="your-secure-token"

# 또는 설정 파일
{
  gateway: {
    auth: {
      mode: "token",
      token: "your-secure-token",
    },
  },
}
```

---

## Tailscale 통합

### Serve (tailnet 전용)

```json5
{
  gateway: {
    tailscale: {
      mode: "serve",  // tailnet 내부만 접근
    },
    bind: "loopback",  // 필수
  },
}
```

### Funnel (공개 접근)

```json5
{
  gateway: {
    tailscale: {
      mode: "funnel",  // 공개 인터넷
      resetOnExit: true,
    },
    bind: "loopback",
    auth: {
      mode: "password",  // funnel 시 필수
      password: "secure-password",
    },
  },
}
```

### Tailscale 설정 요약

| 설정 | Serve | Funnel |
|------|-------|--------|
| 접근 범위 | tailnet만 | 공개 |
| 인증 필요 | 선택 | 필수 (password) |
| bind | loopback | loopback |
| HTTPS | 자동 | 자동 |

---

## Docker 배포

### docker-compose.yml

```yaml
version: '3.8'

services:
  openclaw:
    image: openclaw/openclaw:latest
    container_name: openclaw
    restart: unless-stopped
    ports:
      - "18789:18789"
    volumes:
      - openclaw-config:/root/.openclaw
      - openclaw-workspace:/root/.openclaw/workspace
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY}
    healthcheck:
      test: ["CMD", "openclaw", "status"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  openclaw-config:
  openclaw-workspace:
```

### 환경 변수 파일

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
TELEGRAM_BOT_TOKEN=123456:ABC...
ELEVENLABS_API_KEY=...
OPENCLAW_GATEWAY_TOKEN=secure-gateway-token
```

### 실행

```bash
# 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 상태 확인
docker exec openclaw openclaw status
```

---

## 운영 배포

### systemd 서비스 (Linux)

```ini
# /etc/systemd/system/openclaw.service
[Unit]
Description=OpenClaw Gateway
After=network.target

[Service]
Type=simple
User=openclaw
WorkingDirectory=/home/openclaw
ExecStart=/usr/local/bin/openclaw gateway --port 18789
Restart=always
RestartSec=10
Environment=ANTHROPIC_API_KEY=sk-ant-...
Environment=TELEGRAM_BOT_TOKEN=123456:ABC...

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable openclaw
sudo systemctl start openclaw
```

### launchd (macOS)

온보딩에서 자동 설치:

```bash
openclaw onboard --install-daemon
```

수동 관리:

```bash
openclaw daemon status
openclaw daemon restart
openclaw daemon uninstall
```

---

## 로깅

### 로그 위치

```
~/.openclaw/logs/
├── gateway.log      # Gateway 로그
├── agent.log        # 에이전트 로그
├── commands.log     # 명령 로그 (훅)
├── cron.log         # Cron 작업 로그
└── hooks.log        # 훅 실행 로그
```

### 로그 레벨

```json5
{
  logging: {
    level: "info",  // "debug" | "info" | "warn" | "error"
    format: "json", // "json" | "pretty"
    maxSize: "10m", // 최대 파일 크기
    maxFiles: 5,    // 보관 파일 수
  },
}
```

### macOS 통합 로그

```bash
# OpenClaw 로그 조회
./scripts/clawlog.sh

# 실시간 추적
./scripts/clawlog.sh -f

# 카테고리 필터
./scripts/clawlog.sh --category gateway
```

---

## 보안 체크리스트

### 필수 설정

- [ ] `dmPolicy: "pairing"` 설정
- [ ] 허용 목록(`allowFrom`) 명시적 설정
- [ ] 샌드박싱 활성화 (`sandbox.mode: "non-main"`)
- [ ] Gateway 인증 활성화 (원격 접근 시)

### 권장 설정

- [ ] Tailscale Serve/Funnel 사용 (원격)
- [ ] 브라우저 도구 제한 (필요한 경우만)
- [ ] 정기적인 `openclaw doctor` 실행
- [ ] 로그 모니터링

### 금지 사항

- [ ] `dmPolicy: "open"` 사용 금지
- [ ] `allowFrom: ["*"]` 주의
- [ ] Gateway를 공개 인터넷에 직접 노출 금지
- [ ] 비밀번호/토큰을 코드에 하드코딩 금지

---

## 문제 해결

### 보안 관련 오류

| 오류 | 해결책 |
|------|--------|
| "Pairing required" | `openclaw pairing approve` 실행 |
| "Unauthorized" | Gateway 토큰 확인 |
| "Sandbox error" | Docker 설치/실행 확인 |
| "Permission denied" | 파일/디렉토리 권한 확인 |

### Doctor 진단

```bash
# 전체 진단
openclaw doctor

# 보안 집중 검사
openclaw doctor --security
```

---

## 마무리

OpenClaw는 강력한 개인 AI 어시스턴트입니다. 핵심 가치:

- **Local-first** - 직접 운영하는 Gateway
- **Multi-channel** - 모든 메시징 앱에서 AI와 대화
- **Extensible** - 스킬, 훅, 플러그인으로 확장
- **Secure** - 샌드박싱, 페어링, 인증
- **Always-on** - 항상 켜진 음성, 예약 작업

---

## 리소스

- **GitHub**: [github.com/openclaw/openclaw](https://github.com/openclaw/openclaw)
- **Docs**: [docs.openclaw.ai](https://docs.openclaw.ai)
- **Discord**: [discord.gg/clawd](https://discord.gg/clawd)
- **ClawHub**: [clawhub.com](https://clawhub.com)
- **라이선스**: MIT

---

*이 가이드 시리즈가 OpenClaw를 이해하고 활용하는 데 도움이 되길 바랍니다. 🦞*
