---
layout: post
title: "OpenCode 가이드 - 소개 및 주요 특징"
date: 2025-02-04
categories: [AI]
tags: [opencode, ai-agent, cli, coding-assistant, open-source]
author: anomalyco
original_url: https://github.com/anomalyco/opencode
---

## OpenCode란?

**OpenCode**는 Anthropic의 Claude Code와 유사한 기능을 제공하는 **100% 오픈소스 AI 코딩 에이전트**입니다. 터미널에서 실행되며, AI의 도움을 받아 코드 작성, 디버깅, 리팩토링, 코드 분석 등 다양한 개발 작업을 수행할 수 있습니다.

```
   ____                   ______          __
  / __ \____  ___  ____  / ____/___  ____/ /__
 / / / / __ \/ _ \/ __ \/ /   / __ \/ __  / _ \
/ /_/ / /_/ /  __/ / / / /___/ /_/ / /_/ /  __/
\____/ .___/\___/_/ /_/\____/\____/\__,_/\___/
    /_/

The open source AI coding agent.
```

## Claude Code와의 차이점

OpenCode는 Claude Code의 대안으로 설계되었지만, 몇 가지 핵심적인 차별점이 있습니다:

| 특성 | Claude Code | OpenCode |
|------|-------------|----------|
| **소스 코드** | 비공개 | 100% 오픈소스 (MIT) |
| **AI 프로바이더** | Anthropic Claude 전용 | 멀티 프로바이더 지원 |
| **LSP 지원** | 제한적 | 기본 내장 |
| **인터페이스** | CLI | TUI 중심 설계 |
| **아키텍처** | 모놀리식 | 클라이언트-서버 분리 |
| **커뮤니티** | Anthropic 운영 | 오픈 커뮤니티 |

### 1. 100% 오픈소스

OpenCode는 MIT 라이선스로 GitHub에 완전히 공개되어 있습니다. 누구나 소스 코드를 확인하고, 수정하고, 기여할 수 있습니다.

```bash
# 소스 코드 확인
git clone https://github.com/anomalyco/opencode
```

### 2. 프로바이더 독립성

특정 AI 프로바이더에 종속되지 않습니다. Claude, OpenAI, Google, Azure, 심지어 로컬 모델까지 자유롭게 선택할 수 있습니다.

```json
{
  "provider": {
    "anthropic": { "apiKey": "..." },
    "openai": { "apiKey": "..." },
    "google": { "apiKey": "..." }
  }
}
```

### 3. TUI 중심 설계

Neovim 사용자와 terminal.shop 개발자들이 만든 프로젝트답게, 터미널에서 최상의 경험을 제공합니다.

### 4. 클라이언트-서버 아키텍처

TUI 프론트엔드는 하나의 클라이언트일 뿐입니다. 서버는 로컬에서 실행하면서 모바일 앱이나 웹에서 원격으로 제어할 수 있습니다.

## 주요 기능

### 에이전트 시스템

OpenCode는 두 가지 기본 내장 에이전트를 제공합니다:

```
┌─────────────────────────────────────────┐
│  Tab 키로 에이전트 전환                  │
├─────────────────────────────────────────┤
│  build   - 기본 에이전트, 모든 도구 사용 │
│  plan    - 읽기 전용, 분석/계획 모드     │
├─────────────────────────────────────────┤
│  @general - 복잡한 검색/멀티스텝 작업    │
└─────────────────────────────────────────┘
```

- **build**: 파일 편집, 명령어 실행 등 모든 작업 가능
- **plan**: 파일 수정 불가, bash 명령 전 확인 요청
- **general**: `@general`로 호출하는 서브에이전트

### 내장 도구

```typescript
// 사용 가능한 도구 목록
const tools = {
  edit: "파일 편집",
  bash: "쉘 명령 실행",
  read: "파일 읽기",
  write: "파일 작성",
  glob: "파일 패턴 검색",
  grep: "텍스트 검색",
  lsp: "Language Server 연동",
  webfetch: "웹 페이지 가져오기",
  websearch: "웹 검색",
  task: "서브에이전트 작업"
}
```

### LSP 통합

기본으로 Language Server Protocol을 지원하여 코드 분석 정확도가 높습니다:

```yaml
지원 기능:
  - 코드 정의 이동
  - 참조 찾기
  - 호버 정보
  - 심볼 검색
  - 진단 정보
```

### MCP (Model Context Protocol) 지원

외부 도구를 MCP 서버로 연동할 수 있습니다:

```json
{
  "mcp": {
    "filesystem": {
      "type": "local",
      "command": ["npx", "@modelcontextprotocol/server-filesystem", "/path"]
    }
  }
}
```

### 스킬 시스템

Claude Code 스타일의 SKILL.md 파일을 지원합니다:

```markdown
# .opencode/skills/deploy/SKILL.md

---
name: deploy
description: 프로덕션 배포 자동화
---

배포 시 다음 단계를 따르세요:
1. 테스트 실행
2. 빌드 생성
3. 배포 명령 실행
```

## 지원 플랫폼

| 플랫폼 | 설치 방법 |
|--------|-----------|
| macOS (Apple Silicon) | `brew install anomalyco/tap/opencode` |
| macOS (Intel) | `brew install anomalyco/tap/opencode` |
| Linux | `brew install anomalyco/tap/opencode` |
| Windows | `scoop install opencode` 또는 `choco install opencode` |
| Arch Linux | `paru -S opencode-bin` |
| 범용 | `npm i -g opencode-ai@latest` |

## 데스크톱 앱 (베타)

터미널 외에도 네이티브 데스크톱 앱을 제공합니다:

```bash
# macOS (Homebrew)
brew install --cask opencode-desktop

# Windows (Scoop)
scoop bucket add extras
scoop install extras/opencode-desktop
```

## 시스템 요구사항

- **Node.js**: 18.0 이상 (npm 설치 시)
- **Bun**: 1.0 이상 (권장)
- **운영체제**: macOS, Linux, Windows
- **터미널**: 트루컬러 지원 권장

## 빠른 시작

```bash
# 1. 설치
npm i -g opencode-ai@latest

# 2. API 키 설정 (예: Anthropic)
export ANTHROPIC_API_KEY="your-api-key"

# 3. 프로젝트에서 실행
cd your-project
opencode
```

## 커뮤니티 & 지원

- **GitHub**: [github.com/anomalyco/opencode](https://github.com/anomalyco/opencode)
- **Discord**: [discord.gg/opencode](https://discord.gg/opencode)
- **X (Twitter)**: [@opencode](https://x.com/opencode)

## OpenCode Zen

유료 구독 서비스인 **OpenCode Zen**을 통해 추가 모델과 기능을 사용할 수 있습니다:

```bash
# Zen 인증
opencode auth
```

## 다음 단계

다음 챕터에서는 OpenCode의 다양한 설치 방법을 상세히 알아봅니다.

---

**다음 글**: [설치 가이드](/opencode-guide-02-installation/)
