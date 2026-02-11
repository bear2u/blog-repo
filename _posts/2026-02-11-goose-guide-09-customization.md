---
layout: post
title: "Goose 완벽 가이드 (09) - 확장 및 커스터마이징"
date: 2026-02-11
permalink: /goose-guide-09-customization/
author: Block
categories: [AI 에이전트, 개발 도구]
tags: [Goose, Customization, Plugin, Extension, Configuration]
original_url: "https://github.com/block/goose"
excerpt: "Goose를 프로젝트에 맞게 커스터마이징하는 방법"
---

## 커스터마이징 개요

Goose는 다양한 방식으로 커스터마이징할 수 있습니다:

- **.goosehints** - 프로젝트 컨텍스트 제공
- **.gooseignore** - 파일/디렉토리 제외
- **커스텀 Provider** - 새로운 LLM 통합
- **커스텀 MCP 서버** - 새로운 도구 추가
- **설정 파일** - 동작 변경

---

## .goosehints - 프로젝트 힌트

### 개요

`.goosehints` 파일은 Goose에게 프로젝트별 컨텍스트를 제공합니다.

```bash
# 프로젝트 루트에 생성
touch .goosehints
```

### 기본 템플릿

```markdown
# Project: My Web App

## Overview
This is a full-stack web application built with React and FastAPI.

## Tech Stack
- **Frontend**: React 18, TypeScript, Vite, TailwindCSS
- **Backend**: Python 3.11, FastAPI, SQLAlchemy
- **Database**: PostgreSQL
- **Testing**: Vitest, Pytest

## Project Structure
```
src/
├── frontend/       # React app
├── backend/        # FastAPI server
├── shared/         # Shared types
└── tests/          # Test files
```

## Coding Standards

### Python
- Use type hints
- Follow PEP 8
- Add docstrings to all functions
- Prefer async/await for I/O operations

### TypeScript
- Use functional components
- Prefer hooks over classes
- Write prop types
- Use const for immutable values

### Testing
- Write unit tests for utilities
- Integration tests for API endpoints
- E2E tests for critical flows
- Maintain >80% coverage

## Git Workflow
- Branch naming: `feature/`, `bugfix/`, `hotfix/`
- Commit messages: Conventional Commits
- Always create PR before merging

## Important Notes
- Never commit .env files
- Always run tests before pushing
- Update documentation when changing APIs
```

### 고급 힌트

```markdown
# Project-Specific Commands

## Development
```bash
# Start dev server
npm run dev          # Frontend (port 5173)
python run.py        # Backend (port 8000)
```

## Testing
```bash
npm test             # Frontend tests
pytest               # Backend tests
npm run e2e          # E2E tests
```

## Database
```bash
# Migrations
alembic upgrade head  # Apply migrations
alembic revision      # Create new migration
```

## Common Tasks

### Adding a new API endpoint
1. Create route in `backend/routes/`
2. Add schema in `backend/schemas/`
3. Write tests in `tests/api/`
4. Update OpenAPI docs

### Adding a new component
1. Create component in `frontend/components/`
2. Add types in `frontend/types/`
3. Write tests in `tests/components/`
4. Update Storybook

## Architecture Decisions

### State Management
- Use Zustand for global state
- React Query for server state
- Local state with useState

### API Design
- REST for CRUD operations
- WebSocket for real-time features
- GraphQL for complex queries (future)

### Error Handling
- Use Result type pattern
- Log all errors to Sentry
- Show user-friendly messages
```

---

## .gooseignore - 파일 제외

### 개요

`.gooseignore` 파일은 Goose가 접근하지 않을 파일/디렉토리를 지정합니다.

```bash
# 프로젝트 루트에 생성
touch .gooseignore
```

### 기본 템플릿

```gitignore
# Dependencies
node_modules/
venv/
.venv/
__pycache__/

# Build outputs
dist/
build/
*.pyc
*.pyo

# Environment variables
.env
.env.local
.env.production

# Secrets
*.key
*.pem
secrets/
credentials.json

# Logs
*.log
logs/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Large files
*.mp4
*.zip
*.tar.gz

# Generated
coverage/
.next/
.nuxt/

# Git
.git/
.github/workflows/  # CI 파일은 읽지 않기
```

---

## 커스텀 Provider

### 새로운 LLM Provider 추가

```rust
// custom-provider/src/lib.rs
use goose::providers::{Provider, ProviderResponse};
use async_trait::async_trait;

pub struct CustomLLMProvider {
    api_key: String,
    endpoint: String,
}

#[async_trait]
impl Provider for CustomLLMProvider {
    async fn complete(
        &self,
        messages: Vec<Message>,
    ) -> Result<ProviderResponse> {
        // API 호출
        let response = reqwest::Client::new()
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&messages)
            .send()
            .await?;

        // 응답 파싱
        let body: ApiResponse = response.json().await?;

        Ok(ProviderResponse {
            content: body.choices[0].message.content.clone(),
            tool_calls: body.tool_calls,
            usage: body.usage,
        })
    }

    fn model(&self) -> &str {
        "custom-model-v1"
    }

    fn supports_tools(&self) -> bool {
        true
    }
}
```

### Provider 등록

```rust
// Goose에 Provider 추가
use goose::ProviderRegistry;

let mut registry = ProviderRegistry::new();
registry.register("custom-llm", Box::new(CustomLLMProvider::new()));
```

---

## 커스텀 도구

### 간단한 도구 예시

```rust
// custom-tools/src/calculator.rs
use goose::tools::Tool;

pub struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Perform mathematical calculations"
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate"
                }
            },
            "required": ["expression"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let expr = args["expression"].as_str().unwrap();

        // 계산 수행 (실제로는 안전한 파서 사용)
        let result = eval_expression(expr)?;

        Ok(result.to_string())
    }
}
```

### 복잡한 도구 예시

```rust
// custom-tools/src/ai_image_gen.rs
pub struct ImageGeneratorTool {
    api_key: String,
}

#[async_trait]
impl Tool for ImageGeneratorTool {
    fn name(&self) -> &str {
        "generate_image"
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let prompt = args["prompt"].as_str().unwrap();

        // DALL-E API 호출
        let response = self.call_dalle_api(prompt).await?;

        // 이미지 저장
        let image_path = self.save_image(response.image_url).await?;

        Ok(format!("Image saved to: {}", image_path))
    }

    async fn call_dalle_api(&self, prompt: &str) -> Result<ImageResponse> {
        // API 호출 로직
        todo!()
    }

    async fn save_image(&self, url: &str) -> Result<String> {
        // 이미지 다운로드 및 저장
        todo!()
    }
}
```

---

## 커스텀 Recipe

### Recipe 파일 작성

```yaml
# recipes/setup-project.yaml
name: "Full-Stack Project Setup"
description: "Initialize a complete full-stack project"

variables:
  project_name: "my-app"
  use_typescript: true
  use_docker: true

steps:
  - name: "Create project structure"
    prompt: |
      Create a full-stack project structure with:
      - Frontend directory (React + {{use_typescript ? 'TypeScript' : 'JavaScript'}})
      - Backend directory (FastAPI)
      - Shared types
      - Tests directory

  - name: "Initialize frontend"
    prompt: |
      In the frontend directory:
      - Run 'npm create vite@latest . -- --template {{use_typescript ? 'react-ts' : 'react'}}'
      - Install TailwindCSS
      - Set up ESLint and Prettier

  - name: "Initialize backend"
    prompt: |
      In the backend directory:
      - Create virtual environment
      - Install FastAPI, SQLAlchemy, Alembic
      - Set up project structure
      - Create initial migration

  - name: "Add Docker support"
    condition: "{{use_docker}}"
    prompt: |
      Create Docker files:
      - Dockerfile for frontend
      - Dockerfile for backend
      - docker-compose.yml

  - name: "Create README"
    prompt: |
      Create a comprehensive README.md with:
      - Project overview
      - Setup instructions
      - Development guide
      - Testing guide
```

### Recipe 실행

```bash
# Recipe 실행
goose run --recipe recipes/setup-project.yaml

# 변수 오버라이드
goose run --recipe recipes/setup-project.yaml \
  --var project_name=awesome-app \
  --var use_typescript=false
```

---

## 프로젝트별 설정

### 프로젝트 설정 파일

```yaml
# .goose/config.yaml
provider: anthropic
model: claude-sonnet-4-5

execution_mode: smart_approval

extensions:
  - developer
  - computer_controller

tools:
  allow:
    - shell
    - read_file
    - write_file
  deny:
    - delete_file

paths:
  allowed:
    - /home/user/projects
  blocked:
    - /home/user/.ssh
    - /etc

auto_save: true
save_interval: 300  # seconds
```

---

## 고급 커스터마이징

### 1. 커스텀 System Prompt

```rust
// custom-system-prompt/src/lib.rs
pub fn build_custom_prompt(project: &Project) -> String {
    format!(
        r#"
You are an expert {tech_stack} developer working on {project_name}.

# Project Context
{project_description}

# Your Responsibilities
- Write production-ready code
- Follow project conventions
- Add comprehensive tests
- Update documentation

# Tech Stack
{tech_stack_details}

# Current Task
{current_task}
        "#,
        tech_stack = project.tech_stack,
        project_name = project.name,
        project_description = project.description,
        tech_stack_details = project.tech_details(),
        current_task = project.current_task,
    )
}
```

### 2. 커스텀 승인 로직

```rust
pub struct CustomApprovalPolicy {
    rules: Vec<ApprovalRule>,
}

impl CustomApprovalPolicy {
    pub fn should_request_approval(&self, action: &Action) -> bool {
        for rule in &self.rules {
            if rule.matches(action) {
                return rule.requires_approval;
            }
        }
        false
    }
}

pub struct ApprovalRule {
    pattern: Regex,
    requires_approval: bool,
}
```

### 3. 커스텀 로깅

```rust
use tracing_subscriber::fmt;

pub fn setup_custom_logging() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_thread_ids(true)
        .with_level(true)
        .with_file(true)
        .with_line_number(true)
        .with_writer(std::io::stderr)
        .init();
}
```

---

## 팀 설정 공유

### 1. .goose/ 디렉토리 구조

```
.goose/
├── config.yaml          # 프로젝트 설정
├── hints.md             # 프로젝트 힌트
├── ignore              # 제외 파일
├── recipes/            # 커스텀 레시피
│   ├── setup.yaml
│   ├── deploy.yaml
│   └── test.yaml
└── tools/              # 커스텀 도구
    └── project_tool.rs
```

### 2. Git에 커밋

```gitignore
# .gitignore

# Goose 설정은 커밋
.goose/config.yaml
.goose/hints.md
.goose/recipes/

# 개인 설정은 제외
.goose/local.yaml
```

### 3. 팀원들이 사용

```bash
# 프로젝트 클론
git clone https://github.com/team/project.git
cd project

# Goose가 자동으로 .goose/ 설정 로드
goose session
```

---

## 실전 예제

### 예제 1: 모노레포 프로젝트

```markdown
# .goosehints

# Monorepo Structure
```
apps/
  web/          # Next.js app
  mobile/       # React Native
packages/
  ui/           # Shared components
  utils/        # Shared utilities
```

## Working with Monorepo
- Use pnpm workspaces
- Run commands from root with `-w`
- Shared packages in packages/

## Commands
```bash
pnpm install           # Install all deps
pnpm run dev          # Start all apps
pnpm run build        # Build all
pnpm run test         # Test all
```
```

### 예제 2: 마이크로서비스

```markdown
# .goosehints

# Microservices Architecture

## Services
- **auth-service** (port 8001): Authentication
- **user-service** (port 8002): User management
- **order-service** (port 8003): Order processing
- **payment-service** (port 8004): Payments

## Inter-Service Communication
- Use gRPC for sync calls
- Use RabbitMQ for async events

## Development
```bash
# Start all services
docker-compose up

# Start specific service
cd services/auth-service
cargo run
```

## Testing
- Unit tests: `cargo test` in each service
- Integration tests: `./scripts/integration-test.sh`
```

---

## 다음 단계

커스터마이징을 마스터했다면, 마지막 장에서는 개발 및 기여 방법을 알아봅니다.

*다음 글에서는 Goose 프로젝트에 기여하는 방법과 개발 가이드를 살펴봅니다.*
