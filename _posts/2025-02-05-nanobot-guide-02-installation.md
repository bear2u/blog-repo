---
layout: post
title: "Nanobot 완벽 가이드 (2) - 설치 및 시작"
date: 2025-02-05
permalink: /nanobot-guide-02-installation/
author: HKUDS
categories: [AI 에이전트, Nanobot]
tags: [Nanobot, Installation, Configuration, Docker, CLI]
original_url: "https://github.com/HKUDS/nanobot"
excerpt: "Nanobot을 설치하고 설정하는 다양한 방법을 알아봅니다."
---

## 시스템 요구사항

| 요구사항 | 버전 |
|----------|------|
| **Python** | 3.11 이상 |
| **Node.js** | 18 이상 (WhatsApp 사용 시) |

---

## 설치 방법

### 방법 1: PyPI에서 설치 (권장)

```bash
pip install nanobot-ai
```

### 방법 2: uv로 설치 (빠름)

```bash
# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# nanobot 설치
uv tool install nanobot-ai
```

### 방법 3: 소스에서 설치 (개발용)

```bash
git clone https://github.com/HKUDS/nanobot.git
cd nanobot
pip install -e .
```

### 방법 4: Feishu 지원 추가

```bash
pip install nanobot-ai[feishu]
```

---

## 초기화

설치 후 초기화를 실행합니다.

```bash
nanobot onboard
```

이 명령은:
- `~/.nanobot/` 디렉토리 생성
- 기본 설정 파일 생성
- 워크스페이스 초기화

---

## 설정 파일

설정 파일 위치: `~/.nanobot/config.json`

### 기본 설정 예시

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
    }
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5"
    }
  },
  "tools": {
    "web": {
      "search": {
        "apiKey": "BSA-xxx"
      }
    }
  }
}
```

### 전체 설정 예시

```json
{
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5"
    }
  },
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
    },
    "groq": {
      "apiKey": "gsk_xxx"
    }
  },
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "123456:ABC...",
      "allowFrom": ["123456789"]
    },
    "whatsapp": {
      "enabled": false
    },
    "feishu": {
      "enabled": false,
      "appId": "cli_xxx",
      "appSecret": "xxx",
      "encryptKey": "",
      "verificationToken": "",
      "allowFrom": []
    }
  },
  "tools": {
    "web": {
      "search": {
        "apiKey": "BSA..."
      }
    }
  }
}
```

---

## API 키 발급

### LLM 프로바이더

| 프로바이더 | API 키 발급 |
|----------|-------------|
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) |
| Anthropic | [console.anthropic.com](https://console.anthropic.com) |
| OpenAI | [platform.openai.com](https://platform.openai.com) |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com) |
| Groq | [console.groq.com](https://console.groq.com) |
| Gemini | [aistudio.google.com](https://aistudio.google.com) |

### 웹 검색

| 서비스 | API 키 발급 |
|--------|-------------|
| Brave Search | [brave.com/search/api](https://brave.com/search/api/) |

> **Tip**: OpenRouter를 사용하면 하나의 API 키로 모든 LLM 모델에 접근할 수 있습니다.

---

## CLI 사용법

### 기본 명령어

```bash
# 단일 메시지로 대화
nanobot agent -m "What is 2+2?"

# 대화형 모드
nanobot agent

# 상태 확인
nanobot status
```

### 게이트웨이 (채널 연결)

```bash
# Telegram/WhatsApp/Feishu 게이트웨이 시작
nanobot gateway
```

### 채널 관리

```bash
# WhatsApp 로그인 (QR 스캔)
nanobot channels login

# 채널 상태 확인
nanobot channels status
```

### 스케줄된 작업 (Cron)

```bash
# 작업 추가
nanobot cron add --name "daily" --message "Good morning!" --cron "0 9 * * *"
nanobot cron add --name "hourly" --message "Check status" --every 3600
nanobot cron add --name "once" --message "Reminder!" --at "2025-02-05T15:00:00"

# 작업 목록
nanobot cron list

# 작업 제거
nanobot cron remove <job_id>
```

---

## 채널 설정

### Telegram (권장)

**1. 봇 생성**
- Telegram에서 `@BotFather` 검색
- `/newbot` 전송, 안내에 따라 설정
- 토큰 복사

**2. 설정**

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"]
    }
  }
}
```

> User ID는 `@userinfobot`에서 확인 가능

**3. 실행**

```bash
nanobot gateway
```

### WhatsApp

**요구사항**: Node.js 18 이상

**1. 디바이스 연결**

```bash
nanobot channels login
# QR 코드를 WhatsApp → 설정 → 연결된 기기에서 스캔
```

**2. 설정**

```json
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "allowFrom": ["+1234567890"]
    }
  }
}
```

**3. 실행** (터미널 2개)

```bash
# 터미널 1
nanobot channels login

# 터미널 2
nanobot gateway
```

### Feishu (飞书)

WebSocket 롱 커넥션 사용 - 퍼블릭 IP 불필요

**1. Feishu 봇 생성**
- [Feishu Open Platform](https://open.feishu.cn/app) 방문
- 새 앱 생성 → **Bot** 기능 활성화
- **권한**: `im:message` 추가
- **이벤트**: `im.message.receive_v1` 추가 (Long Connection 모드)
- **App ID**와 **App Secret** 확인

**2. 설정**

```json
{
  "channels": {
    "feishu": {
      "enabled": true,
      "appId": "cli_xxx",
      "appSecret": "xxx",
      "encryptKey": "",
      "verificationToken": "",
      "allowFrom": []
    }
  }
}
```

**3. 실행**

```bash
nanobot gateway
```

---

## 로컬 모델 (vLLM)

자체 로컬 모델을 vLLM이나 OpenAI 호환 서버로 실행할 수 있습니다.

**1. vLLM 서버 시작**

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

**2. 설정**

```json
{
  "providers": {
    "vllm": {
      "apiKey": "dummy",
      "apiBase": "http://localhost:8000/v1"
    }
  },
  "agents": {
    "defaults": {
      "model": "meta-llama/Llama-3.1-8B-Instruct"
    }
  }
}
```

**3. 사용**

```bash
nanobot agent -m "Hello from my local LLM!"
```

> **Tip**: 인증이 필요 없는 로컬 서버의 경우 `apiKey`는 빈 문자열이 아닌 아무 값이나 넣으면 됩니다.

---

## Docker

### 이미지 빌드

```bash
docker build -t nanobot .
```

### 초기화 (최초 1회)

```bash
docker run -v ~/.nanobot:/root/.nanobot --rm nanobot onboard
```

### 설정 편집

```bash
vim ~/.nanobot/config.json
```

### 게이트웨이 실행

```bash
docker run -v ~/.nanobot:/root/.nanobot -p 18790:18790 nanobot gateway
```

### 단일 명령 실행

```bash
# 메시지 보내기
docker run -v ~/.nanobot:/root/.nanobot --rm nanobot agent -m "Hello!"

# 상태 확인
docker run -v ~/.nanobot:/root/.nanobot --rm nanobot status
```

> **Tip**: `-v ~/.nanobot:/root/.nanobot` 플래그는 로컬 설정 디렉토리를 컨테이너에 마운트하여 설정과 워크스페이스가 컨테이너 재시작 후에도 유지됩니다.

---

## 워크스페이스 구조

```
~/.nanobot/
├── config.json        # 설정 파일
└── workspace/
    ├── AGENTS.md      # 에이전트 지침
    ├── TOOLS.md       # 도구 문서
    ├── SOUL.md        # 성격 정의
    ├── USER.md        # 사용자 정보
    ├── HEARTBEAT.md   # 주기적 작업
    ├── MEMORY.md      # 장기 메모리
    └── memory/        # 일별 메모리
```

---

## 문제 해결

### Python 버전 오류

```bash
# Python 버전 확인
python --version

# pyenv로 Python 3.11+ 설치
pyenv install 3.11
pyenv global 3.11
```

### 의존성 충돌

```bash
# 가상 환경 사용 권장
python -m venv .venv
source .venv/bin/activate
pip install nanobot-ai
```

### WhatsApp 연결 실패

```bash
# Node.js 버전 확인
node --version  # 18 이상 필요

# 세션 초기화
rm -rf ~/.nanobot/whatsapp-session
nanobot channels login
```

### Telegram 봇 응답 없음

1. `allowFrom`에 본인 User ID가 있는지 확인
2. 봇 토큰이 올바른지 확인
3. `nanobot gateway` 로그 확인

---

*다음 글에서는 Nanobot의 아키텍처를 분석합니다.*
