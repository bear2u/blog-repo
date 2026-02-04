---
layout: post
title: "UI-TARS 완벽 가이드 (1) - 소개 및 개요"
date: 2025-02-04
permalink: /ui-tars-guide-01-intro/
author: ByteDance
category: AI
tags: [UI-TARS, Agent TARS, GUI Agent, Multimodal AI, ByteDance]
series: ui-tars-guide
part: 1
original_url: "https://github.com/bytedance/UI-TARS-desktop"
excerpt: "ByteDance의 멀티모달 AI 에이전트 스택 UI-TARS를 소개합니다. Agent TARS와 UI-TARS Desktop의 핵심 기능과 아키텍처를 살펴봅니다."
---

## UI-TARS란?

**TARS**는 ByteDance에서 개발한 **멀티모달 AI 에이전트 스택**으로, 현재 두 가지 프로젝트를 제공합니다:

| 프로젝트 | 설명 |
|---------|------|
| **Agent TARS** | 범용 멀티모달 AI 에이전트 - CLI와 Web UI 제공 |
| **UI-TARS Desktop** | 네이티브 GUI 에이전트 데스크톱 앱 |

---

## Agent TARS

Agent TARS는 **GUI Agent와 Vision의 힘을 터미널, 컴퓨터, 브라우저, 제품에 가져오는** 범용 멀티모달 AI 에이전트입니다.

### 핵심 특징

- **원클릭 CLI** - 헤드풀(Web UI) 및 헤드리스(서버) 실행 모두 지원
- **하이브리드 브라우저 에이전트** - GUI Agent, DOM, 또는 하이브리드 전략으로 브라우저 제어
- **이벤트 스트림** - 프로토콜 기반 Event Stream이 컨텍스트 엔지니어링과 Agent UI를 구동
- **MCP 통합** - MCP 서버를 마운트하여 실제 도구들과 연결

### 빠른 시작

```bash
# npx로 실행
npx @agent-tars/cli@latest

# 전역 설치 (Node.js >= 22 필요)
npm install @agent-tars/cli@latest -g

# 선호하는 모델 프로바이더로 실행
agent-tars --provider anthropic --model claude-3-7-sonnet-latest --apiKey your-api-key
```

---

## UI-TARS Desktop

UI-TARS Desktop은 **UI-TARS 모델 기반의 네이티브 GUI 에이전트**를 제공하는 데스크톱 애플리케이션입니다.

### 핵심 특징

- 🤖 **Vision-Language Model 기반** 자연어 제어
- 🖥️ **스크린샷 및 시각 인식** 지원
- 🎯 **정밀한 마우스/키보드 제어**
- 💻 **크로스 플랫폼** 지원 (Windows/macOS/Browser)
- 🔄 **실시간 피드백** 및 상태 표시
- 🔐 **완전 로컬 처리**로 프라이버시 보장

### 지원 Operator

| Operator | 설명 |
|----------|------|
| **Local Computer** | NutJS로 로컬 컴퓨터 직접 제어 |
| **Local Browser** | 로컬 Chrome 브라우저 자동화 |
| **Remote Computer** | 프록시를 통한 원격 컴퓨터 제어 |
| **Remote Browser** | 클라우드 브라우저 자동화 |

---

## 프로젝트 구조

```
UI-TARS-desktop/
├── apps/
│   └── ui-tars/              # Electron 데스크톱 앱
├── multimodal/
│   ├── agent-tars/           # Agent TARS (CLI, Core, Interface)
│   ├── gui-agent/            # GUI Agent SDK
│   ├── tarko/                # 핵심 에이전트 프레임워크
│   └── omni-tars/            # Omni Agent
├── packages/
│   ├── agent-infra/          # MCP 서버/클라이언트
│   ├── ui-tars/              # UI-TARS SDK
│   └── common/               # 공통 유틸리티
└── examples/                 # 예제 코드
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **런타임** | Electron 34.x, Node.js 22+ |
| **프론트엔드** | React 18, Tailwind CSS |
| **상태관리** | Zustand, Jotai |
| **번들러** | Vite, RSLib |
| **언어** | TypeScript 5.5+ |
| **브라우저 자동화** | Puppeteer |
| **데스크톱 자동화** | NutJS |
| **모바일 자동화** | Appium ADB |

---

## 지원 LLM 프로바이더

| 프로바이더 | 모델 예시 |
|-----------|----------|
| OpenAI | GPT-4o, GPT-4 Turbo, o1, o3-mini |
| Anthropic | Claude 3.x 시리즈 |
| Google | Gemini |
| VolcEngine | Doubao-1.5-UI-TARS |
| DeepSeek | DeepSeek |
| Ollama | 로컬 모델 |
| LM Studio | 로컬 모델 |

---

## 이 가이드에서 다루는 내용

1. **소개 및 개요** (현재 글)
2. **전체 아키텍처** - 시스템 구조와 데이터 흐름
3. **UI-TARS Desktop 앱** - Electron 앱 상세 분석
4. **Agent TARS Core** - CLI와 핵심 에이전트 로직
5. **GUI Agent SDK** - 액션 파서와 에이전트 SDK
6. **Operators** - Browser, NutJS, ADB 연산자
7. **Tarko 프레임워크** - 핵심 에이전트 프레임워크
8. **MCP 인프라** - 서버와 클라이언트 구현
9. **컨텍스트 엔지니어링** - 컨텍스트 처리와 최적화
10. **활용 가이드** - 실제 사용 예제와 확장 방법

---

## 라이선스

이 프로젝트는 **Apache License 2.0**으로 배포됩니다.

---

*다음 글에서는 UI-TARS의 전체 아키텍처를 살펴봅니다.*
