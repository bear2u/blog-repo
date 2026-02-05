---
layout: post
title: "TrendRadar 완벽 가이드 (1) - 소개 및 개요"
date: 2025-02-04
permalink: /trendradar-guide-01-intro/
author: sansan0
categories: [개발 도구, TrendRadar]
tags: [TrendRadar, News, Trend, RSS, MCP, AI]
original_url: "https://github.com/sansan0/TrendRadar"
excerpt: "30초 만에 배포 가능한 AI 기반 트렌드 모니터링 도구 TrendRadar를 소개합니다."
---

## TrendRadar란?

**TrendRadar**는 **30초 만에 배포 가능한 AI 기반 트렌드 모니터링 도구**입니다. 무의미한 스크롤을 멈추고, 관심 있는 뉴스와 트렌드만 확인할 수 있게 해줍니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    TrendRadar 핵심 가치                          │
├─────────────────────────────────────────────────────────────────┤
│  • 30초 배포 - GitHub Actions로 원클릭 설정                     │
│  • AI 분석 - GPT/Claude로 뉴스 요약 및 분석                     │
│  • 다중 알림 - WeChat, Telegram, Slack, Email 등                │
│  • MCP 지원 - AI 에이전트와 통합 가능                           │
│  • 경량화 - 최소한의 리소스로 최대 효율                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 주요 기능

### 1. 다중 플랫폼 지원

| 플랫폼 | 설명 |
|--------|------|
| **WeChat (기업/개인)** | 중국 최대 메신저 알림 |
| **Telegram** | 봇을 통한 실시간 알림 |
| **DingTalk** | 중국 비즈니스 메신저 |
| **Feishu (Lark)** | ByteDance의 협업 도구 |
| **Slack** | 글로벌 협업 플랫폼 |
| **Email** | 이메일 알림 |
| **ntfy** | 오픈소스 푸시 알림 |
| **Bark** | iOS 푸시 알림 |
| **Webhook** | 커스텀 연동 |

### 2. AI 분석 기능

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI Analysis Pipeline                         │
│                                                                  │
│   Raw News ──▶ Crawler ──▶ AI Summary ──▶ Push Notification    │
│                              │                                   │
│                              ▼                                   │
│                    ┌─────────────────┐                          │
│                    │  GPT / Claude   │                          │
│                    │  DeepSeek       │                          │
│                    │  Local LLM      │                          │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

- 뉴스 요약 및 핵심 포인트 추출
- 다국어 번역 지원
- 트렌드 분석 및 인사이트

### 3. MCP (Model Context Protocol) 지원

AI 에이전트(Claude Desktop 등)와 직접 통합:

```json
{
  "mcpServers": {
    "trendradar": {
      "command": "python",
      "args": ["-m", "mcp_server"]
    }
  }
}
```

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                    TrendRadar Architecture                       │
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │  Crawlers   │───▶│   Storage   │───▶│  Notifiers  │        │
│   │  (Sources)  │    │  (Cache)    │    │  (Channels) │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│          │                  │                   │                │
│          │                  │                   │                │
│          ▼                  ▼                   ▼                │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │  NewsNow    │    │  SQLite     │    │  Telegram   │        │
│   │  RSS Feeds  │    │  JSON       │    │  WeChat     │        │
│   │  Custom API │    │             │    │  Slack...   │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                  │
│   ┌─────────────────────────────────────────────────┐           │
│   │                 AI Analysis Layer                │           │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │           │
│   │  │ Summary │  │ Translate│  │ Analyze │         │           │
│   │  └─────────┘  └─────────┘  └─────────┘         │           │
│   └─────────────────────────────────────────────────┘           │
│                                                                  │
│   ┌─────────────────────────────────────────────────┐           │
│   │                   MCP Server                     │           │
│   │         (AI Agent Integration Layer)             │           │
│   └─────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 빠른 시작

### 방법 1: GitHub Actions (권장)

```bash
# 1. 레포지토리 Fork
# 2. Settings > Secrets에 환경 변수 설정
# 3. Actions 활성화 - 자동 실행!
```

### 방법 2: Docker

```bash
docker run -d \
  -e TELEGRAM_BOT_TOKEN=your_token \
  -e TELEGRAM_CHAT_ID=your_chat_id \
  wantcat/trendradar
```

### 방법 3: 로컬 설치

```bash
# 클론
git clone https://github.com/sansan0/TrendRadar.git
cd TrendRadar

# 의존성 설치
pip install -r requirements.txt

# 설정
cp config/config.example.yaml config/config.yaml
# config.yaml 편집

# 실행
python -m trendradar
```

---

## 설정 파일 구조

```yaml
# config/config.yaml

# 데이터 소스
sources:
  - type: newsnow
    enabled: true
    categories:
      - tech
      - finance
      - world

# AI 분석
ai:
  enabled: true
  provider: openai  # openai, anthropic, deepseek
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-mini

# 알림 채널
notifications:
  telegram:
    enabled: true
    bot_token: ${TELEGRAM_BOT_TOKEN}
    chat_id: ${TELEGRAM_CHAT_ID}

  wechat:
    enabled: false
    # ...

# 스케줄
schedule:
  interval: 3600  # 초 단위 (1시간)
  timezone: Asia/Seoul
```

---

## 프로젝트 구조

```
TrendRadar/
├── trendradar/           # 메인 패키지
│   ├── __main__.py       # 진입점
│   ├── context.py        # 컨텍스트 관리
│   ├── core/             # 핵심 로직
│   ├── crawler/          # 크롤러
│   ├── notification/     # 알림 채널
│   ├── ai/               # AI 분석
│   ├── report/           # 리포트 생성
│   ├── storage/          # 데이터 저장
│   └── utils/            # 유틸리티
├── mcp_server/           # MCP 서버
│   ├── server.py         # MCP 서버 구현
│   ├── tools/            # MCP 도구
│   ├── services/         # 서비스 레이어
│   └── utils/            # 유틸리티
├── config/               # 설정 파일
├── docker/               # Docker 관련
└── docs/                 # 문서
```

---

## 이 가이드에서 다루는 내용

1. **소개 및 개요** (현재 글)
2. **아키텍처** - 전체 시스템 구조
3. **크롤러 & 데이터 소스** - 뉴스 수집
4. **알림 시스템** - 다중 채널 알림
5. **AI 분석 & MCP** - AI 통합
6. **배포 및 활용** - 실전 사용법

---

## 라이선스

GPL-3.0 License

---

*다음 글에서는 TrendRadar의 아키텍처를 상세히 살펴봅니다.*
