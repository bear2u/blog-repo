---
layout: post
title: "Goose 완벽 가이드 (04) - 코어 에이전트 시스템"
date: 2026-02-11
permalink: /goose-guide-04-core-agent/
author: Block
categories: [AI 에이전트, 개발 도구]
tags: [Goose, Agent, LLM, Execution Loop, Message Processing]
original_url: "https://github.com/block/goose"
excerpt: "Goose의 핵심 에이전트 엔진과 실행 메커니즘"
---

## 에이전트 개요

Goose의 핵심은 **자율적으로 작업을 수행하는 AI 에이전트**입니다. 이 에이전트는 사용자 요청을 받아 계획을 세우고, 필요한 도구를 사용하며, 작업을 완료할 때까지 반복적으로 실행됩니다.

---

## 에이전트 실행 루프

```
┌─────────────────────────────────────────────────────────────┐
│                  Agent Execution Loop                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐                                         │
│  │  User Request  │                                         │
│  └───────┬────────┘                                         │
│          │                                                   │
│          ▼                                                   │
│  ┌────────────────┐                                         │
│  │ Build Context  │  ← System prompt + Conversation history │
│  └───────┬────────┘                                         │
│          │                                                   │
│          ▼                                                   │
│  ┌────────────────┐                                         │
│  │  LLM Request   │  → Provider (OpenAI, Anthropic, etc)    │
│  └───────┬────────┘                                         │
│          │                                                   │
│          ▼                                                   │
│  ┌────────────────┐                                         │
│  │ Parse Response │                                         │
│  └───────┬────────┘                                         │
│          │                                                   │
│     ┌────┴─────┐                                           │
│     │          │                                           │
│     ▼          ▼                                           │
│  ┌──────┐  ┌────────────┐                                 │
│  │ Text │  │ Tool Calls │                                 │
│  └──┬───┘  └─────┬──────┘                                 │
│     │            │                                         │
│     │            ▼                                         │
│     │      ┌────────────┐                                 │
│     │      │Execute Tool│                                 │
│     │      └─────┬──────┘                                 │
│     │            │                                         │
│     │            ▼                                         │
│     │      ┌────────────┐                                 │
│     │      │Tool Results│                                 │
│     │      └─────┬──────┘                                 │
│     │            │                                         │
│     │            ▼                                         │
│     │      Loop back to LLM ────┐                        │
│     │                             │                        │
│     ▼                             │                        │
│  ┌────────────────┐              │                        │
│  │ Return to User │◄─────────────┘                        │
│  └────────────────┘                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 핵심 컴포넌트

### 1. Agent (에이전트 코어)

**파일:** `crates/goose/src/agents/agent.rs`

```rust
// 개념적 구조
pub struct Agent {
    provider: Box<dyn Provider>,
    tools: ToolRegistry,
    context: ConversationContext,
    config: AgentConfig,
}

impl Agent {
    pub async fn run(&mut self, user_message: String) -> Result<Response> {
        // 1. 컨텍스트 빌드
        let messages = self.context.build_messages(user_message);

        // 2. LLM 호출
        loop {
            let response = self.provider.complete(messages).await?;

            // 3. 도구 호출 확인
            if let Some(tool_calls) = response.tool_calls {
                // 4. 도구 실행
                let results = self.execute_tools(tool_calls).await?;

                // 5. 결과를 컨텍스트에 추가
                messages.extend(results);

                // 6. 다시 LLM 호출
                continue;
            }

            // 7. 최종 응답 반환
            return Ok(response);
        }
    }
}
```

### 2. Message 시스템

```rust
pub enum Message {
    System(String),      // 시스템 프롬프트
    User(String),        // 사용자 입력
    Assistant(String),   // 어시스턴트 응답
    Tool {               // 도구 호출 결과
        name: String,
        result: String,
    },
}
```

### 3. Tool Execution

```rust
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

impl Agent {
    async fn execute_tools(
        &self,
        tool_calls: Vec<ToolCall>
    ) -> Result<Vec<Message>> {
        let mut results = Vec::new();

        for call in tool_calls {
            let tool = self.tools.get(&call.name)?;
            let result = tool.execute(call.arguments).await?;

            results.push(Message::Tool {
                name: call.name,
                result,
            });
        }

        Ok(results)
    }
}
```

---

## 컨텍스트 관리

### Conversation Context

에이전트는 대화 히스토리를 관리하며, 매 요청마다 컨텍스트를 구성합니다.

```rust
pub struct ConversationContext {
    system_prompt: String,
    history: Vec<Message>,
    max_tokens: usize,
}

impl ConversationContext {
    pub fn build_messages(&self, user_input: String) -> Vec<Message> {
        let mut messages = vec![
            Message::System(self.system_prompt.clone())
        ];

        // 히스토리 추가 (토큰 제한 고려)
        messages.extend(self.truncate_history());

        // 현재 사용자 입력
        messages.push(Message::User(user_input));

        messages
    }

    fn truncate_history(&self) -> Vec<Message> {
        // 토큰 제한에 맞게 히스토리 자르기
        // 최신 메시지 우선 유지
        todo!()
    }
}
```

---

## System Prompt

Goose는 강력한 시스템 프롬프트를 사용하여 에이전트의 행동을 정의합니다.

```text
You are Goose, an AI agent that helps developers with software engineering tasks.

# Capabilities
- Write and execute code
- Read and modify files
- Run shell commands
- Search the web
- Debug issues
- Manage git repositories

# Guidelines
- Always explain what you're doing before taking action
- Ask for clarification when requirements are unclear
- Write clean, maintainable code
- Follow best practices
- Test your changes

# Tools Available
[Tool descriptions from MCP servers...]

# Working Directory
Current directory: /path/to/project

# Context from .goosehints
[Project-specific hints...]
```

---

## Provider 통합

### Provider Trait

```rust
#[async_trait]
pub trait Provider: Send + Sync {
    async fn complete(
        &self,
        messages: Vec<Message>,
    ) -> Result<ProviderResponse>;

    async fn stream(
        &self,
        messages: Vec<Message>,
    ) -> Result<Box<dyn Stream<Item = ProviderResponse>>>;

    fn supports_tools(&self) -> bool;
    fn model(&self) -> &str;
}
```

### Anthropic Provider 예시

```rust
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

#[async_trait]
impl Provider for AnthropicProvider {
    async fn complete(
        &self,
        messages: Vec<Message>,
    ) -> Result<ProviderResponse> {
        let request = self.build_request(messages)?;

        let response = self.client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .json(&request)
            .send()
            .await?;

        let body: AnthropicResponse = response.json().await?;

        Ok(self.parse_response(body)?)
    }
}
```

---

## Tool System

### Tool Registry

```rust
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn register(&mut self, name: String, tool: Box<dyn Tool>) {
        self.tools.insert(name, tool);
    }

    pub fn get(&self, name: &str) -> Option<&Box<dyn Tool>> {
        self.tools.get(name)
    }

    pub fn list(&self) -> Vec<ToolDescription> {
        self.tools.values()
            .map(|tool| tool.description())
            .collect()
    }
}
```

### Tool Trait

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> ToolDescription;

    async fn execute(
        &self,
        args: serde_json::Value
    ) -> Result<String>;
}
```

### 내장 도구 예시

```rust
pub struct ShellTool;

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> ToolDescription {
        ToolDescription {
            name: "shell".to_string(),
            description: "Execute shell commands".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    }
                },
                "required": ["command"]
            }),
        }
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let command = args["command"].as_str()
            .ok_or_else(|| anyhow!("Missing command"))?;

        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .await?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}
```

---

## 실행 모드

### 1. Interactive Mode (대화형)

사용자가 각 단계를 확인하고 승인합니다.

```
User: create a Python script
Goose: I'll create a Python script. Here's my plan:
      1. Create main.py
      2. Write the script content
      3. Make it executable

      Proceed? [y/n]
```

### 2. Smart Approval Mode

안전한 작업은 자동 실행, 위험한 작업은 승인 요청

```
Goose: Creating file main.py... ✓
Goose: Writing content... ✓
Goose: About to execute 'chmod +x main.py'
      This will modify file permissions. Approve? [y/n]
```

### 3. Autonomous Mode

모든 작업을 자동으로 실행

```
Goose: Creating file main.py... ✓
Goose: Writing content... ✓
Goose: Making executable... ✓
Goose: Task completed!
```

---

## 에러 핸들링

### Retry 전략

```rust
pub struct RetryConfig {
    max_retries: usize,
    backoff: Duration,
}

impl Agent {
    async fn execute_with_retry(
        &self,
        tool_call: ToolCall
    ) -> Result<String> {
        let mut attempts = 0;

        loop {
            match self.execute_tool(tool_call.clone()).await {
                Ok(result) => return Ok(result),
                Err(e) if attempts < self.config.retry.max_retries => {
                    attempts += 1;
                    tokio::time::sleep(
                        self.config.retry.backoff * attempts as u32
                    ).await;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
    }
}
```

### Error Recovery

```rust
impl Agent {
    async fn handle_error(&mut self, error: Error) -> Result<()> {
        // 1. 에러를 LLM에 전달
        let error_message = format!("Error: {}", error);

        self.context.history.push(Message::Assistant(
            error_message
        ));

        // 2. LLM이 에러 분석 및 해결 시도
        let response = self.provider.complete(
            self.context.build_messages(
                "The previous action failed. Please analyze the error and try again.".to_string()
            )
        ).await?;

        // 3. 새로운 해결 시도
        self.handle_response(response).await
    }
}
```

---

## Plan Mode

복잡한 작업을 위해 Goose는 `/plan` 명령을 지원합니다.

```
User: /plan create a full-stack web app
Goose: I'll create a detailed plan for the web app:

      Phase 1: Project Setup
      - Initialize project structure
      - Set up dependencies
      - Configure development environment

      Phase 2: Backend
      - Design database schema
      - Implement API endpoints
      - Add authentication

      Phase 3: Frontend
      - Create React components
      - Implement state management
      - Connect to backend

      Phase 4: Testing & Deployment
      - Write tests
      - Set up CI/CD
      - Deploy to production

      Approve plan? [y/n]
```

---

## 성능 최적화

### 1. Streaming Responses

```rust
impl Agent {
    pub async fn run_stream(
        &mut self,
        user_message: String
    ) -> Result<impl Stream<Item = String>> {
        let messages = self.context.build_messages(user_message);

        self.provider.stream(messages).await
    }
}
```

### 2. Caching

```rust
pub struct CacheConfig {
    enabled: bool,
    ttl: Duration,
}

// LLM 응답 캐싱
// 동일한 요청에 대해 재사용
```

### 3. Parallel Tool Execution

```rust
impl Agent {
    async fn execute_tools_parallel(
        &self,
        tool_calls: Vec<ToolCall>
    ) -> Result<Vec<Message>> {
        let futures: Vec<_> = tool_calls
            .into_iter()
            .map(|call| self.execute_tool(call))
            .collect();

        let results = futures::future::join_all(futures).await;

        // ...
    }
}
```

---

## 다음 단계

에이전트 코어를 이해했다면, 다음 장에서는 CLI 인터페이스를 상세히 살펴봅니다.

*다음 글에서는 Goose CLI의 명령어 시스템과 사용 방법을 분석합니다.*
