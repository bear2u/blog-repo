---
layout: post
title: "Nanobot 완벽 가이드 (1) - 소개 및 개요"
date: 2025-02-05
permalink: /nanobot-guide-01-intro/
author: HKUDS
categories: [AI 에이전트, Nanobot]
tags: [Nanobot, AI Assistant, Personal Agent, LLM, Telegram, WhatsApp]
original_url: "https://github.com/HKUDS/nanobot"
excerpt: "4,000줄의 코드로 구현된 초경량 개인 AI 어시스턴트 Nanobot을 소개합니다."
---

## Nanobot이란?

**Nanobot**은 **초경량 개인 AI 어시스턴트** 프레임워크입니다. [OpenClaw](https://github.com/openclaw/openclaw)에서 영감을 받아 만들어졌으며, **~4,000줄의 코드**로 핵심 에이전트 기능을 구현합니다 - OpenClaw의 43만 줄 대비 **99% 작습니다**.

```
┌─────────────────────────────────────────────┐
│            Nanobot의 핵심 가치              │
├─────────────────────────────────────────────┤
│  • ~4,000줄 코드 (99% 경량화)               │
│  • 멀티 채널 지원 (Telegram, WhatsApp 등)   │
│  • 다양한 LLM 프로바이더 지원               │
│  • 스킬 시스템으로 확장 가능                │
│  • Cron 기반 스케줄링                       │
│  • 서브에이전트로 백그라운드 작업           │
└─────────────────────────────────────────────┘
```

---

## 주요 특징

| 특징 | 설명 |
|------|------|
| **Ultra-Lightweight** | ~4,000줄 코드로 핵심 기능 구현 |
| **Research-Ready** | 읽기 쉽고 수정하기 쉬운 깔끔한 코드 |
| **Lightning Fast** | 최소한의 풋프린트로 빠른 시작 및 낮은 리소스 사용 |
| **Easy-to-Use** | 원클릭 배포 및 즉시 사용 가능 |
| **Multi-Channel** | Telegram, WhatsApp, Feishu 지원 |
| **Multi-Provider** | OpenRouter, Anthropic, OpenAI, DeepSeek, Groq 등 |

---

## 왜 Nanobot인가?

### 기존 AI 어시스턴트의 문제점

```
┌─────────────────────────────────────────────┐
│       기존 AI 어시스턴트의 한계              │
├─────────────────────────────────────────────┤
│  ❌ 수십만 줄의 복잡한 코드베이스            │
│  ❌ 연구/수정이 어려운 구조                  │
│  ❌ 무거운 의존성과 느린 시작 시간           │
│  ❌ 설정이 복잡하고 배포가 어려움            │
└─────────────────────────────────────────────┘
```

### Nanobot의 해결책

```
┌─────────────────────────────────────────────┐
│          Nanobot이 제공하는 가치            │
├─────────────────────────────────────────────┤
│  ✅ 4,000줄로 핵심 기능 모두 구현            │
│  ✅ 읽기 쉬운 코드로 연구/수정 용이          │
│  ✅ 빠른 시작과 낮은 리소스 사용             │
│  ✅ 원클릭 배포 (pip install + 설정)         │
└─────────────────────────────────────────────┘
```

---

## 사용 사례

| 사용 사례 | 설명 |
|----------|------|
| **24/7 실시간 시장 분석** | Discovery, Insights, Trends |
| **풀스택 소프트웨어 엔지니어** | Develop, Deploy, Scale |
| **스마트 일정 관리자** | Schedule, Automate, Organize |
| **개인 지식 어시스턴트** | Learn, Memory, Reasoning |

---

## 지원 채널

| 채널 | 난이도 | 설명 |
|------|--------|------|
| **Telegram** | Easy | 토큰만 있으면 됨 (권장) |
| **WhatsApp** | Medium | QR 코드 스캔 필요 |
| **Feishu** | Medium | 앱 자격 증명 필요 |

---

## 지원 LLM 프로바이더

| 프로바이더 | 용도 | API 키 발급 |
|----------|------|-------------|
| `openrouter` | LLM (권장, 모든 모델 접근) | [openrouter.ai](https://openrouter.ai) |
| `anthropic` | LLM (Claude 직접) | [console.anthropic.com](https://console.anthropic.com) |
| `openai` | LLM (GPT 직접) | [platform.openai.com](https://platform.openai.com) |
| `deepseek` | LLM (DeepSeek 직접) | [platform.deepseek.com](https://platform.deepseek.com) |
| `groq` | LLM + 음성 전사 (Whisper) | [console.groq.com](https://console.groq.com) |
| `gemini` | LLM (Gemini 직접) | [aistudio.google.com](https://aistudio.google.com) |
| `vllm` | 로컬 모델 | 자체 서버 |

---

## 빠른 시작

### 설치

```bash
# PyPI에서 설치 (안정)
pip install nanobot-ai

# 또는 uv로 설치 (빠름)
uv tool install nanobot-ai

# 또는 소스에서 설치 (개발용)
git clone https://github.com/HKUDS/nanobot.git
cd nanobot
pip install -e .
```

### 초기화 및 설정

```bash
# 초기화
nanobot onboard

# 설정 파일 편집 (~/.nanobot/config.json)
```

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
  }
}
```

### 사용

```bash
# 단일 메시지
nanobot agent -m "What is 2+2?"

# 대화형 모드
nanobot agent

# 게이트웨이 실행 (Telegram/WhatsApp 연결)
nanobot gateway
```

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                       Nanobot                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                    Channels                          │   │
│   │     Telegram   WhatsApp   Feishu   CLI              │   │
│   └───────────────────────┬─────────────────────────────┘   │
│                           │                                  │
│   ┌───────────────────────▼─────────────────────────────┐   │
│   │                   Message Bus                        │   │
│   │              (Inbound / Outbound)                    │   │
│   └───────────────────────┬─────────────────────────────┘   │
│                           │                                  │
│   ┌───────────────────────▼─────────────────────────────┐   │
│   │                   Agent Loop                         │   │
│   │                                                      │   │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │   │
│   │  │ Context │  │  Tools  │  │ Memory  │             │   │
│   │  │ Builder │  │Registry │  │         │             │   │
│   │  └─────────┘  └─────────┘  └─────────┘             │   │
│   │                                                      │   │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │   │
│   │  │ Skills  │  │Subagent │  │ Session │             │   │
│   │  │ Loader  │  │ Manager │  │ Manager │             │   │
│   │  └─────────┘  └─────────┘  └─────────┘             │   │
│   └───────────────────────┬─────────────────────────────┘   │
│                           │                                  │
│   ┌───────────────────────▼─────────────────────────────┐   │
│   │                  LLM Providers                       │   │
│   │    OpenRouter  Anthropic  OpenAI  Groq  vLLM       │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 프로젝트 구조

```
nanobot/
├── agent/          # 🧠 핵심 에이전트 로직
│   ├── loop.py     #    에이전트 루프 (LLM ↔ 도구 실행)
│   ├── context.py  #    프롬프트 빌더
│   ├── memory.py   #    영구 메모리
│   ├── skills.py   #    스킬 로더
│   ├── subagent.py #    백그라운드 작업 실행
│   └── tools/      #    내장 도구 (spawn 포함)
├── skills/         # 🎯 번들 스킬 (github, weather, tmux...)
├── channels/       # 📱 메시징 채널 통합
├── bus/            # 🚌 메시지 라우팅
├── cron/           # ⏰ 스케줄된 작업
├── heartbeat/      # 💓 주기적 웨이크업
├── providers/      # 🤖 LLM 프로바이더
├── session/        # 💬 대화 세션
├── config/         # ⚙️ 설정
└── cli/            # 🖥️ 명령어
```

---

## 이 가이드에서 다루는 내용

| # | 제목 | 내용 |
|---|------|------|
| 01 | **소개 및 개요** (현재 글) | Nanobot이란? 주요 특징 |
| 02 | **설치 및 시작** | 설치 방법, 설정, 빠른 시작 |
| 03 | **아키텍처 분석** | 모듈 구조, 설계 원칙 |
| 04 | **Agent Loop** | 핵심 처리 엔진 분석 |
| 05 | **Tools 시스템** | 내장 도구 및 확장 |
| 06 | **Channels** | Telegram, WhatsApp, Feishu 통합 |
| 07 | **Skills** | 스킬 시스템 및 커스텀 스킬 |
| 08 | **Cron & Heartbeat** | 스케줄링 및 주기적 작업 |
| 09 | **Providers** | LLM 프로바이더 및 로컬 모델 |
| 10 | **확장 및 커스터마이징** | 도구 추가, 스킬 개발, 워크스페이스 |

---

## 라이선스

Nanobot은 **MIT 라이선스**로 배포됩니다.

---

## 커뮤니티

- **[GitHub](https://github.com/HKUDS/nanobot)** - 소스 코드 및 이슈
- **[Discord](https://discord.gg/MnCvHqpUGB)** - 커뮤니티
- **[WeChat/Feishu](./COMMUNICATION.md)** - 중국 커뮤니티

---

*다음 글에서는 Nanobot의 설치 및 시작 방법을 살펴봅니다.*
