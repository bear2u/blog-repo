---
layout: post
title: "OpenCode 가이드 - 설정 및 권한"
date: 2025-02-04
categories: [AI 코딩 에이전트, OpenCode]
tags: [opencode, configuration, permissions, settings, security]
author: anomalyco
original_url: https://github.com/anomalyco/opencode
---

## 설정 시스템 개요

OpenCode는 다층 설정 시스템을 사용합니다. 여러 위치의 설정이 병합되며, 우선순위에 따라 덮어씁니다.

## 설정 파일 위치

### 설정 우선순위 (낮음 → 높음)

1. **원격 .well-known/opencode** - 조직 기본값
2. **전역 설정** - `~/.config/opencode/opencode.json`
3. **커스텀 설정** - `OPENCODE_CONFIG` 환경 변수
4. **프로젝트 설정** - `opencode.json` 또는 `opencode.jsonc`
5. **.opencode 디렉토리** - `.opencode/opencode.json`
6. **인라인 설정** - `OPENCODE_CONFIG_CONTENT` 환경 변수
7. **관리형 설정** - 엔터프라이즈 전용

### 플랫폼별 관리형 설정 경로

| 플랫폼 | 경로 |
|--------|------|
| macOS | `/Library/Application Support/opencode` |
| Windows | `C:\ProgramData\opencode` |
| Linux | `/etc/opencode` |

## 설정 파일 형식

### opencode.json / opencode.jsonc

```json
{
  "$schema": "https://opencode.ai/config.json",

  // 기본 모델
  "model": "anthropic/claude-sonnet-4-20250514",

  // 기본 에이전트
  "default_agent": "build",

  // AI 프로바이더 설정
  "provider": {
    "anthropic": {
      "apiKey": "sk-ant-..."
    }
  },

  // 에이전트 설정
  "agent": {
    "custom": {
      "name": "custom",
      "description": "커스텀 에이전트"
    }
  },

  // 권한 설정
  "permission": {
    "edit": { "*": "allow" },
    "bash": "ask"
  },

  // MCP 서버
  "mcp": {
    "server-name": {
      "type": "local",
      "command": ["node", "server.js"]
    }
  },

  // LSP 설정
  "lsp": {
    "typescript": {
      "command": ["typescript-language-server", "--stdio"]
    }
  },

  // 스킬 경로
  "skills": {
    "paths": ["./custom-skills"]
  },

  // 플러그인
  "plugin": ["./plugins/custom-plugin.js"],

  // 추가 지시사항
  "instructions": ["항상 한국어로 응답하세요"]
}
```

## 핵심 설정 옵션

### 모델 설정

```json
{
  // 기본 모델
  "model": "anthropic/claude-sonnet-4-20250514",

  // 에이전트별 모델
  "agent": {
    "fast": {
      "model": "anthropic/claude-haiku-3-5-20241022"
    }
  }
}
```

### 온도 및 생성 파라미터

```json
{
  "agent": {
    "creative": {
      "temperature": 0.9,
      "topP": 0.95
    }
  }
}
```

### 지시사항 (Instructions)

```json
{
  "instructions": [
    "항상 TypeScript를 사용하세요",
    "함수형 프로그래밍 스타일을 선호합니다",
    "테스트 코드를 함께 작성하세요"
  ]
}
```

## 권한 시스템

### 권한 수준

| 수준 | 설명 |
|------|------|
| `allow` | 자동 허용 |
| `ask` | 사용자 확인 요청 |
| `deny` | 자동 거부 |

### 도구별 권한

```json
{
  "permission": {
    // 전체 기본값
    "*": "allow",

    // Bash 명령
    "bash": "ask",

    // 파일 편집
    "edit": {
      "*": "allow",
      "*.env": "deny",
      "*.env.*": "deny",
      "node_modules/**": "deny"
    },

    // 파일 읽기
    "read": {
      "*": "allow",
      ".env*": "ask"
    },

    // 외부 디렉토리
    "external_directory": {
      "*": "ask",
      "/tmp/**": "allow"
    },

    // 특수 권한
    "doom_loop": "ask",      // 무한 루프 감지
    "question": "allow",      // 사용자 질문
    "plan_enter": "allow",    // Plan 모드 진입
    "plan_exit": "deny"       // Plan 모드 종료
  }
}
```

### 패턴 매칭

글로브 패턴을 사용하여 세밀한 권한 제어가 가능합니다:

```json
{
  "permission": {
    "edit": {
      "*": "allow",

      // 특정 확장자
      "*.env": "deny",
      "*.key": "deny",
      "*.pem": "deny",

      // 특정 디렉토리
      "node_modules/**": "deny",
      ".git/**": "deny",

      // 예외 허용
      "docs/**": "allow",
      "*.md": "allow"
    }
  }
}
```

## 환경 변수

### 주요 환경 변수

```bash
# 설정 파일 경로
export OPENCODE_CONFIG="/path/to/opencode.json"

# 인라인 설정
export OPENCODE_CONFIG_CONTENT='{"model":"anthropic/claude-haiku"}'

# .opencode 디렉토리 경로
export OPENCODE_CONFIG_DIR="/path/to/.opencode"

# 프로젝트 설정 비활성화
export OPENCODE_DISABLE_PROJECT_CONFIG=1

# 외부 스킬 비활성화
export OPENCODE_DISABLE_EXTERNAL_SKILLS=1

# 권한 오버라이드
export OPENCODE_PERMISSION='{"bash":"allow"}'

# 실험적 LSP
export OPENCODE_EXPERIMENTAL_LSP_TY=1
```

### 테스트용 환경 변수

```bash
# 테스트용 관리형 설정 디렉토리
export OPENCODE_TEST_MANAGED_CONFIG_DIR="/tmp/test-config"
```

## .opencode 디렉토리 구조

프로젝트 내 `.opencode/` 디렉토리 구조:

```
.opencode/
├── opencode.json       # 프로젝트 설정
├── agents/             # 커스텀 에이전트
│   └── reviewer.json
├── commands/           # 커스텀 명령
│   └── deploy.json
├── skills/             # 커스텀 스킬
│   └── testing/
│       └── SKILL.md
├── plugins/            # 플러그인
│   └── custom.js
└── plans/              # Plan 모드 저장소
    └── feature-plan.md
```

## LSP 설정

### 기본 제공 LSP

OpenCode는 여러 LSP 서버를 기본 지원합니다:

- TypeScript (typescript-language-server)
- Python (pyright)
- Go (gopls)
- Rust (rust-analyzer)

### LSP 비활성화

```json
{
  "lsp": false
}
```

### 특정 LSP 비활성화

```json
{
  "lsp": {
    "pyright": {
      "disabled": true
    }
  }
}
```

### 커스텀 LSP 추가

```json
{
  "lsp": {
    "my-lsp": {
      "command": ["my-lsp-server", "--stdio"],
      "extensions": [".custom"],
      "env": {
        "MY_LSP_CONFIG": "value"
      }
    }
  }
}
```

## 실험적 기능

```json
{
  "experimental": {
    // OpenTelemetry 활성화
    "openTelemetry": true,

    // MCP 타임아웃 (ms)
    "mcp_timeout": 60000
  }
}
```

## Prettier 설정

OpenCode 프로젝트 자체의 포맷팅 설정:

```json
{
  "prettier": {
    "semi": false,
    "printWidth": 120
  }
}
```

## 설정 마이그레이션

### mode → agent 마이그레이션

이전 `mode` 필드는 `agent`로 자동 마이그레이션됩니다:

```json
// 이전 방식 (deprecated)
{
  "mode": {
    "custom": { "prompt": "..." }
  }
}

// 새 방식
{
  "agent": {
    "custom": {
      "mode": "primary",
      "prompt": "..."
    }
  }
}
```

### tools → permission 마이그레이션

이전 `tools` 필드는 `permission`으로 마이그레이션됩니다.

## 설정 검증

설정 스키마 URL:

```json
{
  "$schema": "https://opencode.ai/config.json"
}
```

## 설정 디버깅

```bash
# 현재 설정 확인
opencode config show

# 설정 소스 확인
opencode config sources
```

## 엔터프라이즈 배포

관리형 설정 디렉토리를 사용하면 조직 전체에 일관된 설정을 배포할 수 있습니다:

```bash
# Linux/macOS
sudo mkdir -p /etc/opencode
sudo cp opencode.json /etc/opencode/

# Windows (PowerShell, 관리자)
New-Item -Path "C:\ProgramData\opencode" -ItemType Directory
Copy-Item opencode.json -Destination "C:\ProgramData\opencode\"
```

## 다음 단계

다음 챕터에서는 MCP(Model Context Protocol) 통합을 알아봅니다.

---

**이전 글**: [AI 프로바이더](/opencode-guide-06-providers/)

**다음 글**: [MCP 통합](/opencode-guide-08-mcp/)
