---
layout: post
title: "OpenCode 가이드 - MCP 통합"
date: 2025-02-04
categories: [AI 코딩 에이전트, OpenCode]
tags: [opencode, mcp, model-context-protocol, tools, integration]
author: anomalyco
original_url: https://github.com/anomalyco/opencode
---

## MCP란?

**Model Context Protocol (MCP)**은 AI 모델과 외부 도구/데이터 소스를 연결하는 표준 프로토콜입니다. OpenCode는 MCP 클라이언트를 내장하여 다양한 MCP 서버와 통합할 수 있습니다.

```
┌─────────────────────────────────────────────────────────┐
│                    OpenCode                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌───────────┐     ┌───────────┐     ┌───────────┐   │
│   │  Agent    │ ──▶ │    MCP    │ ──▶ │   Tools   │   │
│   │  System   │     │  Client   │     │ (외부)    │   │
│   └───────────┘     └─────┬─────┘     └───────────┘   │
│                           │                            │
│                           ▼                            │
│   ┌─────────────────────────────────────────────────┐ │
│   │              MCP Servers                         │ │
│   │  ┌────────┐  ┌────────┐  ┌────────┐            │ │
│   │  │ Local  │  │ Remote │  │ OAuth  │            │ │
│   │  │ Server │  │ Server │  │ Server │            │ │
│   │  └────────┘  └────────┘  └────────┘            │ │
│   └─────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## MCP 서버 유형

### Local (로컬 서버)

로컬 프로세스로 실행되는 MCP 서버:

```json
{
  "mcp": {
    "filesystem": {
      "type": "local",
      "command": ["npx", "@modelcontextprotocol/server-filesystem", "/path"],
      "environment": {
        "DEBUG": "true"
      },
      "timeout": 30000
    }
  }
}
```

### Remote (원격 서버)

HTTP/HTTPS를 통해 접속하는 원격 MCP 서버:

```json
{
  "mcp": {
    "remote-api": {
      "type": "remote",
      "url": "https://mcp.example.com/api",
      "headers": {
        "Authorization": "Bearer token"
      }
    }
  }
}
```

## MCP 서버 설정

### 기본 설정

```json
{
  "mcp": {
    "server-name": {
      "type": "local",           // "local" 또는 "remote"
      "command": ["cmd", "args"], // local 전용
      "url": "https://...",       // remote 전용
      "enabled": true,            // 활성화 여부
      "timeout": 30000            // 타임아웃 (ms)
    }
  }
}
```

### 환경 변수 전달

```json
{
  "mcp": {
    "database": {
      "type": "local",
      "command": ["npx", "@modelcontextprotocol/server-postgres"],
      "environment": {
        "DATABASE_URL": "postgresql://localhost/mydb",
        "PGPASSWORD": "${PGPASSWORD}"
      }
    }
  }
}
```

### 서버 비활성화

```json
{
  "mcp": {
    "server-name": {
      "enabled": false
    }
  }
}
```

## OAuth 인증

원격 MCP 서버가 OAuth를 요구하는 경우:

### 자동 OAuth (기본)

```json
{
  "mcp": {
    "oauth-server": {
      "type": "remote",
      "url": "https://api.example.com/mcp"
      // oauth 기본 활성화
    }
  }
}
```

### OAuth 명시적 설정

```json
{
  "mcp": {
    "oauth-server": {
      "type": "remote",
      "url": "https://api.example.com/mcp",
      "oauth": {
        "clientId": "your-client-id",
        "clientSecret": "your-client-secret",
        "scope": "read write"
      }
    }
  }
}
```

### OAuth 비활성화

```json
{
  "mcp": {
    "no-oauth-server": {
      "type": "remote",
      "url": "https://api.example.com/mcp",
      "oauth": false
    }
  }
}
```

### OAuth 인증 흐름

```bash
# CLI에서 인증
opencode mcp auth server-name
```

## 인기 있는 MCP 서버

### Filesystem Server

파일 시스템 접근:

```json
{
  "mcp": {
    "filesystem": {
      "type": "local",
      "command": [
        "npx",
        "@modelcontextprotocol/server-filesystem",
        "/allowed/path"
      ]
    }
  }
}
```

### PostgreSQL Server

데이터베이스 접근:

```json
{
  "mcp": {
    "postgres": {
      "type": "local",
      "command": ["npx", "@modelcontextprotocol/server-postgres"],
      "environment": {
        "DATABASE_URL": "postgresql://user:pass@localhost/db"
      }
    }
  }
}
```

### GitHub Server

GitHub API 접근:

```json
{
  "mcp": {
    "github": {
      "type": "local",
      "command": ["npx", "@modelcontextprotocol/server-github"],
      "environment": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### Slack Server

Slack 통합:

```json
{
  "mcp": {
    "slack": {
      "type": "local",
      "command": ["npx", "@modelcontextprotocol/server-slack"],
      "environment": {
        "SLACK_BOT_TOKEN": "${SLACK_BOT_TOKEN}"
      }
    }
  }
}
```

### Brave Search Server

웹 검색:

```json
{
  "mcp": {
    "brave-search": {
      "type": "local",
      "command": ["npx", "@modelcontextprotocol/server-brave-search"],
      "environment": {
        "BRAVE_API_KEY": "${BRAVE_API_KEY}"
      }
    }
  }
}
```

## MCP 관리 명령어

### 상태 확인

```bash
opencode mcp status
```

출력 예:

```
┌─────────────────┬────────────┐
│ Server          │ Status     │
├─────────────────┼────────────┤
│ filesystem      │ connected  │
│ postgres        │ connected  │
│ github          │ needs_auth │
│ disabled-server │ disabled   │
└─────────────────┴────────────┘
```

### OAuth 인증

```bash
# 특정 서버 인증
opencode mcp auth github

# 인증 제거
opencode mcp auth remove github
```

### 연결/연결 해제

```bash
# 수동 연결
opencode mcp connect server-name

# 연결 해제
opencode mcp disconnect server-name
```

## MCP 도구 사용

MCP 서버가 제공하는 도구는 에이전트가 자동으로 사용할 수 있습니다:

```
"GitHub에서 이 저장소의 최근 이슈를 가져와줘"

→ MCP github 서버의 list_issues 도구 호출
```

### 도구 이름 규칙

MCP 도구는 `{서버명}_{도구명}` 형식으로 등록됩니다:

```
filesystem_read_file
filesystem_write_file
postgres_query
github_list_issues
```

## MCP 리소스

MCP 서버가 제공하는 리소스에 접근:

```typescript
// 리소스 목록 조회
const resources = await MCP.resources()

// 리소스 읽기
const content = await MCP.readResource("filesystem", "file:///path/to/file")
```

## MCP 프롬프트

MCP 서버가 제공하는 프롬프트 템플릿:

```typescript
// 프롬프트 목록 조회
const prompts = await MCP.prompts()

// 프롬프트 실행
const result = await MCP.getPrompt("server", "prompt-name", { arg: "value" })
```

## 고급 설정

### 타임아웃 전역 설정

```json
{
  "experimental": {
    "mcp_timeout": 60000
  }
}
```

### 커스텀 헤더 (원격 서버)

```json
{
  "mcp": {
    "custom-server": {
      "type": "remote",
      "url": "https://api.example.com/mcp",
      "headers": {
        "X-Custom-Header": "value",
        "Authorization": "Bearer ${API_TOKEN}"
      }
    }
  }
}
```

### 작업 디렉토리 (로컬 서버)

로컬 MCP 서버는 기본적으로 프로젝트 디렉토리에서 실행됩니다.

## 디버깅

### 로그 확인

MCP 서버의 stderr 출력이 로그에 기록됩니다:

```typescript
transport.stderr?.on("data", (chunk: Buffer) => {
  log.info(`mcp stderr: ${chunk.toString()}`, { key })
})
```

### 연결 실패 시

```
┌───────────────────────────────────────────┐
│ MCP Authentication Required               │
│                                           │
│ Server "github" requires authentication.  │
│ Run: opencode mcp auth github             │
└───────────────────────────────────────────┘
```

## MCP 서버 개발

직접 MCP 서버를 개발할 수도 있습니다:

```typescript
// my-mcp-server.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js"
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js"

const server = new Server({
  name: "my-server",
  version: "1.0.0"
})

server.setRequestHandler("tools/list", async () => ({
  tools: [{
    name: "my_tool",
    description: "My custom tool",
    inputSchema: { type: "object", properties: {} }
  }]
}))

server.setRequestHandler("tools/call", async (request) => {
  // 도구 실행 로직
  return { content: [{ type: "text", text: "Result" }] }
})

const transport = new StdioServerTransport()
await server.connect(transport)
```

## 다음 단계

다음 챕터에서는 TUI와 데스크톱 앱을 살펴봅니다.

---

**이전 글**: [설정 및 권한](/opencode-guide-07-configuration/)

**다음 글**: [TUI & 데스크톱 앱](/opencode-guide-09-tui-desktop/)
