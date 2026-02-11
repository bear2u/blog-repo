---
layout: page
title: Goose 가이드
permalink: /goose-guide/
icon: fas fa-robot
---

# Goose 완벽 가이드

> **로컬 머신에서 동작하는 확장 가능한 오픈소스 AI 에이전트**

**Goose**는 Block에서 개발한 강력한 AI 에이전트 프레임워크로, 복잡한 개발 작업을 처음부터 끝까지 자동화합니다. Rust로 작성되어 빠르고 안정적이며, Desktop 앱과 CLI 두 가지 인터페이스를 제공합니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/blog-repo/goose-guide-01-intro/) | Goose란? 주요 특징, 기술 스택, 아키텍처 개요 |
| 02 | [설치 및 시작](/blog-repo/goose-guide-02-installation/) | 설치 방법, LLM 설정, 첫 세션, 명령어 |
| 03 | [아키텍처 분석](/blog-repo/goose-guide-03-architecture/) | Workspace 구조, Crate 분석, 의존성 |
| 04 | [코어 에이전트 시스템](/blog-repo/goose-guide-04-core-agent/) | 에이전트 실행 루프, Provider, Tool System |
| 05 | [CLI 인터페이스](/blog-repo/goose-guide-05-cli/) | CLI 명령어, 세션 관리, Recipe 실행 |
| 06 | [Desktop 앱](/blog-repo/goose-guide-06-desktop/) | Electron 아키텍처, React UI, API 통신 |
| 07 | [MCP 통합](/blog-repo/goose-guide-07-mcp/) | Model Context Protocol, 확장 시스템 |
| 08 | [서버 및 API](/blog-repo/goose-guide-08-server-api/) | Backend API, REST 엔드포인트, WebSocket |
| 09 | [확장 및 커스터마이징](/blog-repo/goose-guide-09-customization/) | .goosehints, 커스텀 도구, Recipe |
| 10 | [개발 및 기여 가이드](/blog-repo/goose-guide-10-contributing/) | 개발 환경 구축, 코드 품질, PR 가이드 |

---

## 주요 특징

### 🤖 완전 자율 에이전트
- 프로젝트 전체를 처음부터 생성
- 코드 작성 및 실행
- 자동 디버깅
- 워크플로우 오케스트레이션

### 🔌 최대 유연성
- 모든 LLM 제공자 지원 (OpenAI, Anthropic, Gemini 등)
- 멀티 모델 설정 가능
- MCP를 통한 확장
- Desktop + CLI 듀얼 인터페이스

### 🚀 개발자 친화적
- 로컬 실행 (데이터 안전)
- 오픈소스 (Apache 2.0)
- Rust 기반 (빠르고 안정적)
- 확장 가능한 플러그인 시스템

---

## 빠른 시작

### 설치

```bash
# macOS/Linux
curl -fsSL https://github.com/block/goose/releases/latest/download/install.sh | bash

# Windows (Git Bash/PowerShell)
curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash
```

### 설정

```bash
goose configure
> Configure Providers
> Tetrate Agent Router (추천)
```

### 세션 시작

```bash
goose session
```

### 간단한 작업

```
G❯ create an interactive tic-tac-toe game in JavaScript
```

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                      Goose Ecosystem                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────────┐              ┌─────────────────┐     │
│   │  Desktop UI     │              │   CLI           │     │
│   │  (Electron)     │              │   (Terminal)    │     │
│   └────────┬────────┘              └────────┬────────┘     │
│            │                                │              │
│            └───────────┬────────────────────┘              │
│                        │                                   │
│             ┌──────────▼──────────┐                        │
│             │  Goose Server       │                        │
│             │  (Backend API)      │                        │
│             └──────────┬──────────┘                        │
│                        │                                   │
│             ┌──────────▼──────────┐                        │
│             │  Goose Core         │                        │
│             │  (Agent Engine)     │                        │
│             └──────────┬──────────┘                        │
│                        │                                   │
│      ┌─────────────────┼─────────────────┐                │
│      │                 │                 │                │
│  ┌───▼────┐      ┌────▼─────┐     ┌────▼─────┐          │
│  │ LLM    │      │ MCP      │     │ Tool     │          │
│  │Provider│      │ Servers  │     │ System   │          │
│  └────────┘      └──────────┘     └──────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 기술 | 용도 |
|------|------|
| **Rust** | 코어 에이전트 엔진 |
| **Tokio** | 비동기 런타임 |
| **Axum** | 웹 서버 프레임워크 |
| **Electron** | Desktop 앱 |
| **React** | UI 라이브러리 |
| **rmcp** | Model Context Protocol |

---

## Workspace 구조

```
crates/
├── goose                # 코어 에이전트 로직
├── goose-cli            # CLI 인터페이스
├── goose-server         # 백엔드 API
├── goose-mcp            # MCP 확장
├── goose-acp            # Agent Communication Protocol
├── goose-test           # 테스트 유틸리티
└── goose-test-support   # 테스트 지원

ui/desktop/              # Electron 앱
```

---

## 지원 LLM 제공자

- **OpenAI** (GPT-4, GPT-5)
- **Anthropic** (Claude Sonnet 4.5, Opus)
- **Amazon Bedrock**
- **Azure OpenAI**
- **Google Gemini**
- **Tetrate Agent Router** (추천)
- **GitHub Copilot**
- **로컬 모델** (Ollama 등)

---

## 주요 기능

### 프로젝트 생성

```
User: create a REST API with authentication
Goose: ✓ 프로젝트 구조 생성
      ✓ 의존성 추가
      ✓ 인증 시스템 구현
      ✓ 테스트 작성
      완료!
```

### 코드 리팩토링

```
User: refactor this to use async/await
Goose: [코드 분석]
      [리팩토링 계획]
      ✓ 변경 완료
```

### 버그 디버깅

```
User: fix the failing test
Goose: [테스트 실행]
      [에러 분석]
      [수정 적용]
      ✓ 테스트 통과
```

---

## MCP 확장

Goose는 Model Context Protocol을 통해 확장할 수 있습니다:

### 내장 확장

- **Developer**: 셸 명령, 파일 작업, Git
- **Computer Controller**: 브라우저 자동화, 스크린샷

### 커스텀 확장

```rust
pub struct MyCustomTool;

#[async_trait]
impl Tool for MyCustomTool {
    fn name(&self) -> &str { "my_tool" }

    async fn execute(&self, args: Value) -> Result<String> {
        // 구현
    }
}
```

---

## 프로젝트별 커스터마이징

### .goosehints

```markdown
# Project: My Web App

## Tech Stack
- React 18
- TypeScript
- Vite

## Coding Standards
- Use functional components
- Write type-safe code

## Commands
```bash
npm run dev
npm test
```
```

### .gooseignore

```gitignore
node_modules/
.env
*.key
```

---

## 커뮤니티

| 리소스 | 링크 |
|--------|------|
| **GitHub** | [github.com/block/goose](https://github.com/block/goose) |
| **문서** | [block.github.io/goose](https://block.github.io/goose) |
| **Discord** | [discord.gg/goose-oss](https://discord.gg/goose-oss) |
| **YouTube** | [@goose-oss](https://www.youtube.com/@goose-oss) |
| **Twitter** | [@goose_oss](https://x.com/goose_oss) |

---

## 라이선스

Apache 2.0 - Block (ai-oss-tools@block.xyz)

---

## 다음 단계

1. [소개 및 개요](/blog-repo/goose-guide-01-intro/)부터 시작하기
2. [설치 및 시작](/blog-repo/goose-guide-02-installation/)으로 직접 설치
3. [GitHub 저장소](https://github.com/block/goose) 방문

**Happy Coding with Goose! 🦢**
