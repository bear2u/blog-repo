---
layout: post
title: "Goose 완벽 가이드 (07) - MCP 통합"
date: 2026-02-11
permalink: /goose-guide-07-mcp/
author: Block
categories: [AI 에이전트, 개발 도구]
tags: [Goose, MCP, Model Context Protocol, Extensions, Tools]
original_url: "https://github.com/block/goose"
excerpt: "Model Context Protocol을 통한 Goose 확장 시스템"
---

## MCP 개요

**Model Context Protocol (MCP)**는 AI 에이전트가 외부 도구 및 데이터 소스에 접근할 수 있게 해주는 표준 프로토콜입니다.

Goose는 MCP를 통해 다양한 기능을 확장할 수 있습니다.

---

## MCP 아키텍처

```
┌──────────────────────────────────────────────────────┐
│              Goose + MCP Architecture                 │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────┐                                     │
│  │ Goose Core  │                                     │
│  └──────┬──────┘                                     │
│         │                                             │
│         ▼                                             │
│  ┌─────────────────┐                                 │
│  │  MCP Client     │                                 │
│  │  (rmcp crate)   │                                 │
│  └────────┬────────┘                                 │
│           │                                           │
│    ┌──────┴──────┬──────────┬─────────┐             │
│    │             │          │         │             │
│    ▼             ▼          ▼         ▼             │
│  ┌────┐      ┌────┐    ┌────┐    ┌────┐           │
│  │MCP │      │MCP │    │MCP │    │MCP │           │
│  │Srv1│      │Srv2│    │Srv3│    │Srv4│           │
│  └────┘      └────┘    └────┘    └────┘           │
│  Developer  Computer  Custom   External           │
│                                                       │
└──────────────────────────────────────────────────────┘
```

---

## MCP 구성 요소

### 1. Tools (도구)

에이전트가 실행할 수 있는 기능

```rust
// 도구 정의 예시
pub struct ShellTool {
    allowed_commands: Vec<String>,
}

impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute shell commands"
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let command = args["command"].as_str().unwrap();

        // 실행 로직
        let output = Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .await?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}
```

### 2. Resources (리소스)

에이전트가 접근할 수 있는 데이터 소스

```rust
pub struct FileResource {
    base_path: PathBuf,
}

impl Resource for FileResource {
    fn uri_scheme(&self) -> &str {
        "file"
    }

    async fn read(&self, uri: &str) -> Result<Vec<u8>> {
        let path = self.resolve_path(uri)?;
        fs::read(path).await
    }

    async fn list(&self, uri: &str) -> Result<Vec<ResourceInfo>> {
        // 디렉토리 목록 반환
        todo!()
    }
}
```

### 3. Prompts (프롬프트 템플릿)

재사용 가능한 프롬프트

```rust
pub struct CodingAssistantPrompt;

impl Prompt for CodingAssistantPrompt {
    fn name(&self) -> &str {
        "coding_assistant"
    }

    fn description(&self) -> &str {
        "Helper for coding tasks"
    }

    fn template(&self) -> &str {
        r#"
You are a coding assistant. Help the user with:
- Writing clean, maintainable code
- Following best practices
- Adding proper documentation
- Writing tests

Current task: {{task}}
        "#
    }
}
```

---

## 내장 MCP 서버

### 1. Developer MCP

**위치:** `crates/goose-mcp/src/developer/`

**제공 도구:**

#### shell - 셸 명령 실행

```json
{
  "name": "shell",
  "description": "Execute shell commands",
  "parameters": {
    "command": "string"
  }
}
```

**사용 예시:**
```
User: Run the tests
Goose: [Uses shell tool]
      shell(command="cargo test")
```

#### read_file - 파일 읽기

```json
{
  "name": "read_file",
  "description": "Read file contents",
  "parameters": {
    "path": "string"
  }
}
```

#### write_file - 파일 쓰기

```json
{
  "name": "write_file",
  "description": "Write content to file",
  "parameters": {
    "path": "string",
    "content": "string"
  }
}
```

#### list_directory - 디렉토리 목록

```json
{
  "name": "list_directory",
  "description": "List directory contents",
  "parameters": {
    "path": "string"
  }
}
```

#### git_* - Git 작업

```
git_status
git_diff
git_commit
git_push
```

**액세스 제어:**

```yaml
# extensions.yaml
developer:
  enabled: true
  allow_shell: true
  allow_file_read: true
  allow_file_write: true
  allow_git: true
  allowed_paths:
    - "/home/user/projects"
  blocked_commands:
    - "rm -rf /"
    - "format"
```

### 2. Computer Controller MCP

**제공 기능:**

- **브라우저 자동화**
  ```
  open_browser(url)
  click_element(selector)
  fill_form(data)
  ```

- **스크린샷**
  ```
  screenshot(path)
  ```

- **웹 스크래핑**
  ```
  scrape_page(url, selectors)
  ```

**설정:**

```yaml
computer_controller:
  enabled: true
  timeout: 300
  headless: true
```

---

## 커스텀 MCP 서버 만들기

### 1. MCP 서버 구조

```rust
// custom_mcp/src/main.rs
use goose_mcp::{Server, Tool, Resource};

#[tokio::main]
async fn main() -> Result<()> {
    let mut server = Server::new("my-custom-server");

    // 도구 등록
    server.register_tool(Box::new(MyCustomTool::new()));

    // 리소스 등록
    server.register_resource(Box::new(MyCustomResource::new()));

    // 서버 시작
    server.serve().await
}
```

### 2. 커스텀 도구 구현

```rust
// custom_mcp/src/tools/database.rs
use async_trait::async_trait;

pub struct DatabaseTool {
    connection: sqlx::Pool<Postgres>,
}

#[async_trait]
impl Tool for DatabaseTool {
    fn name(&self) -> &str {
        "query_database"
    }

    fn description(&self) -> &str {
        "Execute SQL queries"
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query to execute"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let query = args["query"].as_str().unwrap();

        let rows = sqlx::query(query)
            .fetch_all(&self.connection)
            .await?;

        Ok(serde_json::to_string(&rows)?)
    }
}
```

### 3. MCP 서버 등록

```bash
# Goose에 커스텀 MCP 서버 추가
goose configure
> Add Extension
> Custom MCP Server

# 서버 정보 입력
Name: my-custom-server
Command: /path/to/custom_mcp
Arguments: --port 9000
```

### 4. 설정 파일

```yaml
# ~/.config/goose/extensions.yaml
custom_servers:
  - name: my-custom-server
    command: /path/to/custom_mcp
    args:
      - --port
      - "9000"
    env:
      DATABASE_URL: postgres://...
    enabled: true
```

---

## MCP Inspector 사용

MCP 서버를 개발할 때 유용한 도구:

```bash
# MCP Inspector 설치
npm install -g @modelcontextprotocol/inspector

# Goose MCP 서버 테스트
npx @modelcontextprotocol/inspector \
  cargo run -p goose-mcp --example mcp

# 브라우저에서 열림
# http://localhost:5173
```

**Inspector 기능:**
- 도구 목록 확인
- 도구 실행 테스트
- 리소스 탐색
- 프롬프트 테스트

---

## 실전 예제

### 예제 1: Slack 통합

```rust
// slack_mcp/src/main.rs
use slack_api::Slack;

pub struct SlackTool {
    client: Slack,
}

#[async_trait]
impl Tool for SlackTool {
    fn name(&self) -> &str {
        "send_slack_message"
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let channel = args["channel"].as_str().unwrap();
        let message = args["message"].as_str().unwrap();

        self.client
            .post_message(channel, message)
            .await?;

        Ok("Message sent".to_string())
    }
}
```

**사용:**
```
User: Send a message to #engineering saying "Deploy complete"
Goose: [Uses send_slack_message tool]
      ✓ Message sent to #engineering
```

### 예제 2: Jira 통합

```rust
pub struct JiraTool {
    client: jira::Client,
}

impl Tool for JiraTool {
    // create_issue
    // update_issue
    // search_issues
}
```

**사용:**
```
User: Create a bug ticket for the login issue
Goose: [Uses create_issue tool]
      ✓ Created PROJ-123: Login form validation error
```

### 예제 3: 데이터베이스 관리

```rust
pub struct DatabaseMigrationTool {
    // 마이그레이션 실행
    // 롤백
    // 스키마 확인
}
```

**사용:**
```
User: Run the latest database migration
Goose: [Uses run_migration tool]
      ✓ Applied migration 2024_01_15_add_users_table
```

---

## 보안 고려사항

### 1. 액세스 제어

```rust
pub struct AccessControl {
    allowed_paths: Vec<PathBuf>,
    allowed_commands: Vec<String>,
    blocked_patterns: Vec<Regex>,
}

impl AccessControl {
    pub fn check_path(&self, path: &Path) -> Result<()> {
        for allowed in &self.allowed_paths {
            if path.starts_with(allowed) {
                return Ok(());
            }
        }
        Err(anyhow!("Path not allowed"))
    }

    pub fn check_command(&self, cmd: &str) -> Result<()> {
        // 위험한 명령어 차단
        for pattern in &self.blocked_patterns {
            if pattern.is_match(cmd) {
                return Err(anyhow!("Command blocked"));
            }
        }
        Ok(())
    }
}
```

### 2. 승인 시스템

```rust
pub enum ApprovalMode {
    Always,        // 항상 승인 필요
    SmartApproval, // 위험한 작업만
    Autonomous,    // 자동 승인
}

impl Tool for ShellTool {
    async fn execute(&self, args: Value) -> Result<String> {
        let command = args["command"].as_str().unwrap();

        // 승인 확인
        if self.requires_approval(command) {
            if !self.request_approval(command).await? {
                return Err(anyhow!("User denied"));
            }
        }

        // 실행
        self.run_command(command).await
    }

    fn requires_approval(&self, command: &str) -> bool {
        // 위험한 명령어 체크
        command.contains("rm ")
            || command.contains("format")
            || command.contains("delete")
    }
}
```

### 3. 샌드박싱

```rust
// Docker 컨테이너에서 실행
pub struct SandboxedShell {
    container_id: String,
}

impl SandboxedShell {
    async fn execute_in_sandbox(&self, cmd: &str) -> Result<String> {
        let output = Command::new("docker")
            .args(&["exec", &self.container_id, "sh", "-c", cmd])
            .output()
            .await?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}
```

---

## MCP 프로토콜 상세

### 요청/응답 형식

```json
// Tool 호출 요청
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "shell",
    "arguments": {
      "command": "ls -la"
    }
  },
  "id": 1
}

// 응답
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "total 48\ndrwxr-xr-x ..."
      }
    ]
  },
  "id": 1
}
```

### 도구 목록 조회

```json
// 요청
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 2
}

// 응답
{
  "jsonrpc": "2.0",
  "result": {
    "tools": [
      {
        "name": "shell",
        "description": "Execute shell commands",
        "inputSchema": { ... }
      },
      // ...
    ]
  },
  "id": 2
}
```

---

## 다음 단계

MCP 확장을 이해했다면, 다음 장에서는 서버 및 API를 살펴봅니다.

*다음 글에서는 Goose 서버 아키텍처와 REST API를 분석합니다.*
