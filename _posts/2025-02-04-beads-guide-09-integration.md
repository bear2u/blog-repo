---
layout: post
title: "Beads 완벽 가이드 (9) - 확장 및 통합"
date: 2025-02-04
permalink: /beads-guide-09-integration/
author: Steve Yegge
category: AI
tags: [Beads, MCP, Integration, API, Extension]
series: beads-guide
part: 9
original_url: "https://github.com/steveyegge/beads"
excerpt: "Beads의 MCP 서버, 외부 통합, API 확장 방법을 분석합니다."
---

## MCP 통합

Beads는 **Model Context Protocol (MCP)** 서버를 내장하여 AI 에이전트와의 긴밀한 통합을 지원합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      MCP Architecture                            │
│                                                                  │
│   AI Agent (Claude, GPT, etc.)                                  │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────┐                                           │
│   │   MCP Server    │    Beads MCP 서버                         │
│   │   (beads-mcp)   │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐    ┌─────────────────┐                   │
│   │  Per-Workspace  │───▶│    SQLite DB    │                   │
│   │     Daemon      │    │   (.beads/)     │                   │
│   └─────────────────┘    └─────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## MCP 서버 설정

### Claude Desktop 설정

```json
// ~/.config/claude-desktop/config.json (Linux)
// ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)

{
  "mcpServers": {
    "beads": {
      "command": "beads-mcp",
      "args": ["--workspace", "/path/to/project"]
    }
  }
}
```

### 다중 워크스페이스

```json
{
  "mcpServers": {
    "beads-webapp": {
      "command": "beads-mcp",
      "args": ["--workspace", "/Users/alice/projects/webapp"]
    },
    "beads-backend": {
      "command": "beads-mcp",
      "args": ["--workspace", "/Users/alice/projects/backend"]
    }
  }
}
```

---

## MCP 도구 목록

### beads_ready

```typescript
// AI 에이전트가 호출 가능한 Ready 작업 조회
{
  name: "beads_ready",
  description: "Get list of ready tasks (no open blockers)",
  inputSchema: {
    type: "object",
    properties: {
      priority: {
        type: "array",
        items: { type: "integer" },
        description: "Filter by priority levels (0-4)"
      },
      label: {
        type: "string",
        description: "Filter by label"
      }
    }
  }
}

// 응답 예시
{
  "issues": [
    {
      "id": "bd-a1b2",
      "title": "Add OAuth login",
      "priority": 0,
      "type": "feature",
      "status": "open"
    }
  ]
}
```

### beads_create

```typescript
{
  name: "beads_create",
  description: "Create a new issue",
  inputSchema: {
    type: "object",
    required: ["title"],
    properties: {
      title: { type: "string" },
      description: { type: "string" },
      priority: { type: "integer", default: 2 },
      type: {
        type: "string",
        enum: ["task", "bug", "feature", "epic"]
      },
      parent: { type: "string" },
      labels: {
        type: "array",
        items: { type: "string" }
      }
    }
  }
}
```

### beads_update

```typescript
{
  name: "beads_update",
  description: "Update an existing issue",
  inputSchema: {
    type: "object",
    required: ["id"],
    properties: {
      id: { type: "string" },
      title: { type: "string" },
      status: {
        type: "string",
        enum: ["open", "in_progress", "blocked", "closed"]
      },
      priority: { type: "integer" },
      assignee: { type: "string" }
    }
  }
}
```

### beads_close

```typescript
{
  name: "beads_close",
  description: "Close an issue",
  inputSchema: {
    type: "object",
    required: ["id"],
    properties: {
      id: { type: "string" },
      reason: { type: "string" }
    }
  }
}
```

### beads_dep_add

```typescript
{
  name: "beads_dep_add",
  description: "Add a dependency between issues",
  inputSchema: {
    type: "object",
    required: ["from_id", "to_id"],
    properties: {
      from_id: { type: "string" },
      to_id: { type: "string" },
      type: {
        type: "string",
        enum: ["blocks", "parent-child", "related", "discovered-from"],
        default: "blocks"
      }
    }
  }
}
```

---

## MCP 서버 구현

```go
// cmd/beads-mcp/main.go

func main() {
    server := mcp.NewServer("beads", "0.21.0")

    // 도구 등록
    server.AddTool(mcp.Tool{
        Name:        "beads_ready",
        Description: "Get list of ready tasks",
        Handler:     handleReady,
    })

    server.AddTool(mcp.Tool{
        Name:        "beads_create",
        Description: "Create a new issue",
        Handler:     handleCreate,
    })

    // 리소스 등록 (선택적)
    server.AddResource(mcp.Resource{
        URI:         "beads://issues",
        Description: "All issues in the workspace",
        Handler:     handleListResource,
    })

    // 서버 시작
    server.Serve()
}

func handleReady(params json.RawMessage) (interface{}, error) {
    var req ReadyRequest
    json.Unmarshal(params, &req)

    store := getStore()
    issues, err := store.GetReadyIssues()
    if err != nil {
        return nil, err
    }

    // 필터 적용
    if len(req.Priority) > 0 {
        issues = filterByPriority(issues, req.Priority)
    }

    return map[string]interface{}{"issues": issues}, nil
}
```

---

## GitHub 통합

### bd gh 명령어

```bash
# GitHub Issue에서 임포트
bd gh import --repo owner/repo --issue 123

# GitHub Issue로 익스포트
bd gh export bd-a1b2 --repo owner/repo

# GitHub Issues 동기화
bd gh sync --repo owner/repo --labels beads

# PR 연결
bd gh link bd-a1b2 --pr 456
```

### GitHub Actions

{% raw %}
```yaml
# .github/workflows/beads-sync.yml

name: Beads Sync

on:
  push:
    branches: [main]
    paths:
      - '.beads/issues.jsonl'

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Beads
        run: |
          curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash

      - name: Sync to GitHub Issues
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          bd gh sync --repo ${{ github.repository }}
```
{% endraw %}

---

## Jira 통합

```bash
# Jira에서 임포트
bd jira import --project PROJ --filter "status = Open"

# Jira로 익스포트
bd jira export bd-a1b2 --project PROJ

# 양방향 동기화
bd jira sync --project PROJ
```

```yaml
# .beads/config.yaml

integrations:
  jira:
    url: https://company.atlassian.net
    project: PROJ
    sync_labels: true
    priority_map:
      0: Highest
      1: High
      2: Medium
      3: Low
      4: Lowest
```

---

## Linear 통합

```bash
# Linear에서 임포트
bd linear import --team TEAM

# Linear로 익스포트
bd linear export bd-a1b2
```

---

## Webhook 지원

```yaml
# .beads/config.yaml

webhooks:
  - url: https://hooks.slack.com/services/xxx/yyy/zzz
    events:
      - issue.created
      - issue.closed
    filter:
      priority: [0, 1]  # P0, P1만

  - url: https://api.example.com/beads-webhook
    events:
      - "*"  # 모든 이벤트
    headers:
      Authorization: "Bearer ${WEBHOOK_TOKEN}"
```

```go
// internal/webhooks/dispatcher.go

type WebhookEvent struct {
    Type      string          `json:"type"`
    Timestamp time.Time       `json:"timestamp"`
    Issue     *types.Issue    `json:"issue,omitempty"`
    Actor     string          `json:"actor,omitempty"`
}

func (d *Dispatcher) Notify(event WebhookEvent) {
    for _, hook := range d.webhooks {
        if hook.Matches(event) {
            go d.send(hook, event)
        }
    }
}
```

---

## 플러그인 시스템

### 커스텀 명령어

```go
// plugins/my-plugin/main.go

package main

import "github.com/steveyegge/beads/plugin"

func init() {
    plugin.RegisterCommand("my-cmd", &MyCommand{})
}

type MyCommand struct{}

func (c *MyCommand) Run(args []string) error {
    // 커스텀 로직
    return nil
}
```

### 플러그인 설치

```bash
# 플러그인 설치
bd plugin install github.com/user/beads-plugin-xxx

# 플러그인 목록
bd plugin list

# 플러그인 제거
bd plugin remove xxx
```

---

## API 직접 사용

### Go 라이브러리

```go
import "github.com/steveyegge/beads/pkg/beads"

func main() {
    client, err := beads.NewClient("/path/to/workspace")
    if err != nil {
        log.Fatal(err)
    }

    // Ready 작업 조회
    issues, err := client.Ready()

    // 이슈 생성
    issue, err := client.Create(&beads.CreateOptions{
        Title:    "New feature",
        Priority: 1,
        Type:     "feature",
    })

    // 이슈 종료
    err = client.Close(issue.ID, "Completed")
}
```

### REST API (데몬)

```bash
# 데몬이 실행 중일 때 HTTP API 사용 가능
curl -X GET http://localhost:8080/api/ready
curl -X POST http://localhost:8080/api/issues \
  -H "Content-Type: application/json" \
  -d '{"title": "New issue", "priority": 2}'
```

---

## 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `BEADS_WORKSPACE` | 워크스페이스 경로 | 현재 디렉토리 |
| `BEADS_NO_DAEMON` | 데몬 비활성화 | false |
| `BEADS_SYNC_DEBOUNCE` | 동기화 디바운스 (ms) | 5000 |
| `BEADS_AUTO_PUSH` | 자동 Git 푸시 | false |
| `BEADS_LOG_LEVEL` | 로그 레벨 | info |

---

*다음 글에서는 활용 가이드와 베스트 프랙티스를 살펴봅니다.*
