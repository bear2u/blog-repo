---
layout: page
title: Nanobot 가이드
permalink: /nanobot-guide/
icon: fas fa-robot
---

# Nanobot 완벽 가이드

> **~4,000줄의 코드로 구현된 초경량 개인 AI 어시스턴트**

Nanobot은 [HKUDS](https://github.com/HKUDS)에서 개발한 초경량 개인 AI 어시스턴트 프레임워크입니다. OpenClaw에서 영감을 받아 만들어졌으며, 핵심 에이전트 기능을 **~4,000줄의 코드**로 구현합니다 - OpenClaw의 43만 줄 대비 **99% 작습니다**.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/nanobot-guide-01-intro/) | Nanobot이란? 주요 특징, 아키텍처 개요 |
| 02 | [설치 및 시작](/nanobot-guide-02-installation/) | 설치 방법, 설정, 채널 연결 |
| 03 | [아키텍처 분석](/nanobot-guide-03-architecture/) | 모듈 구조, 데이터 흐름, 설계 원칙 |
| 04 | [Agent Loop](/nanobot-guide-04-agent-loop/) | 핵심 처리 엔진, 컨텍스트 빌더, 세션 관리 |
| 05 | [Tools 시스템](/nanobot-guide-05-tools/) | 내장 도구, 파일/셸/웹 도구, 커스텀 도구 |
| 06 | [Channels](/nanobot-guide-06-channels/) | Telegram, WhatsApp, Feishu 연동 |
| 07 | [Skills](/nanobot-guide-07-skills/) | 스킬 시스템, 내장 스킬, 커스텀 스킬 개발 |
| 08 | [Cron & Heartbeat](/nanobot-guide-08-cron/) | 스케줄링, 주기적 작업, 자동화 |
| 09 | [Providers](/nanobot-guide-09-providers/) | LLM 프로바이더, OpenRouter, vLLM |
| 10 | [확장 및 커스터마이징](/nanobot-guide-10-customization/) | 워크스페이스, 도구 추가, 베스트 프랙티스 |

---

## 주요 특징

- **Ultra-Lightweight** - ~4,000줄 코드로 핵심 기능 구현
- **Research-Ready** - 읽기 쉽고 수정하기 쉬운 깔끔한 코드
- **Lightning Fast** - 최소한의 풋프린트로 빠른 시작
- **Easy-to-Use** - 원클릭 배포 및 즉시 사용 가능
- **Multi-Channel** - Telegram, WhatsApp, Feishu 지원
- **Multi-Provider** - OpenRouter, Anthropic, OpenAI, Groq 등 지원

---

## 빠른 시작

```bash
# 설치
pip install nanobot-ai

# 초기화
nanobot onboard

# 설정 파일 편집 (~/.nanobot/config.json)
# API 키 입력

# 대화 시작
nanobot agent -m "Hello!"

# 또는 대화형 모드
nanobot agent

# 게이트웨이 실행 (Telegram/WhatsApp)
nanobot gateway
```

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                       Nanobot                                │
├─────────────────────────────────────────────────────────────┤
│   ┌─────────────────────────────────────────────────────┐   │
│   │                    Channels                          │   │
│   │     Telegram   WhatsApp   Feishu   CLI              │   │
│   └───────────────────────┬─────────────────────────────┘   │
│                           │                                  │
│   ┌───────────────────────▼─────────────────────────────┐   │
│   │                   Message Bus                        │   │
│   └───────────────────────┬─────────────────────────────┘   │
│                           │                                  │
│   ┌───────────────────────▼─────────────────────────────┐   │
│   │                   Agent Loop                         │   │
│   │  Context  │  Tools  │  Memory  │  Skills  │ Session │   │
│   └───────────────────────┬─────────────────────────────┘   │
│                           │                                  │
│   ┌───────────────────────▼─────────────────────────────┐   │
│   │                  LLM Providers                       │   │
│   │    OpenRouter  Anthropic  OpenAI  Groq  vLLM        │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 지원 채널

| 채널 | 난이도 | 설명 |
|------|--------|------|
| **Telegram** | Easy | 토큰만 있으면 됨 (권장) |
| **WhatsApp** | Medium | QR 코드 스캔 필요 |
| **Feishu** | Medium | 앱 자격 증명 필요 |

---

## 지원 LLM 프로바이더

| 프로바이더 | 용도 |
|----------|------|
| `openrouter` | 모든 모델 통합 접근 (권장) |
| `anthropic` | Claude 직접 연동 |
| `openai` | GPT 직접 연동 |
| `groq` | 초고속 추론 + 음성 전사 |
| `deepseek` | 비용 효율적 |
| `gemini` | Google 모델 |
| `vllm` | 로컬 모델 실행 |

---

## 관련 링크

- [GitHub 저장소](https://github.com/HKUDS/nanobot)
- [Discord 커뮤니티](https://discord.gg/MnCvHqpUGB)
