---
layout: post
title: "Goose 완벽 가이드 (08) - 서버 및 API"
date: 2026-02-11
permalink: /goose-guide-08-server-api/
author: Block
categories: [AI 에이전트, 개발 도구]
tags: [Goose, Server, API, REST, WebSocket, Axum]
original_url: "https://github.com/block/goose"
excerpt: "Goose 백엔드 서버 아키텍처와 REST API 상세 분석"
---

## 서버 개요

**Goose Server** (`goosed`)는 Desktop 앱과 Web 인터페이스를 지원하는 백엔드 API 서버입니다.

**기술 스택:**
- **Axum**: 웹 프레임워크
- **Tower**: 미들웨어
- **Tokio**: 비동기 런타임
- **OpenAPI**: API 스펙

---

## 서버 아키텍처

```
┌────────────────────────────────────────────────────┐
│               Goose Server (goosed)                 │
├────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────────────────────────────────┐ │
│  │         HTTP Server (Axum)                   │ │
│  └───────────┬──────────────────────────────────┘ │
│              │                                     │
│    ┌─────────┼─────────┬──────────┐              │
│    │         │         │          │              │
│    ▼         ▼         ▼          ▼              │
│  ┌────┐  ┌────┐   ┌────┐    ┌────┐             │
│  │REST│  │ WS │   │CORS│    │Auth│             │
│  │API │  │    │   │    │    │    │             │
│  └─┬──┘  └─┬──┘   └────┘    └────┘             │
│    │       │                                     │
│    ▼       ▼                                     │
│  ┌──────────────────┐                            │
│  │   Router Layer   │                            │
│  └────────┬─────────┘                            │
│           │                                      │
│  ┌────────▼─────────────────────────────────┐   │
│  │         Service Layer                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌────────┐ │   │
│  │  │ Sessions │  │ Messages │  │ Config │ │   │
│  │  └──────────┘  └──────────┘  └────────┘ │   │
│  └────────┬─────────────────────────────────┘   │
│           │                                      │
│  ┌────────▼─────────┐                            │
│  │   Goose Core     │                            │
│  │   (Agent)        │                            │
│  └──────────────────┘                            │
│                                                     │
└────────────────────────────────────────────────────┘
```

---

## 서버 시작

### 바이너리 실행

```bash
# 기본 포트 (8080)
goosed

# 포트 지정
goosed --port 3000

# 호스트 지정
goosed --host 0.0.0.0 --port 8080

# 로그 레벨
goosed --log-level debug
```

### 소스에서 빌드

```bash
cd crates/goose-server

# 디버그 빌드
cargo build

# 실행
cargo run

# 릴리스 빌드
cargo build --release
./target/release/goosed
```

---

## REST API

### API 엔드포인트

#### 세션 관리

```http
# 세션 목록
GET /api/sessions

# 세션 생성
POST /api/sessions
Content-Type: application/json

{
  "name": "My Project",
  "working_directory": "/path/to/project"
}

# 세션 조회
GET /api/sessions/{sessionId}

# 세션 삭제
DELETE /api/sessions/{sessionId}
```

#### 메시지

```http
# 메시지 전송
POST /api/sessions/{sessionId}/messages
Content-Type: application/json

{
  "content": "Create a Python script",
  "role": "user"
}

# 응답
{
  "id": "msg_123",
  "content": "I'll create a Python script...",
  "role": "assistant",
  "created_at": "2026-02-11T10:30:00Z"
}

# 메시지 히스토리
GET /api/sessions/{sessionId}/messages

# 스트리밍 메시지
POST /api/sessions/{sessionId}/messages/stream
Content-Type: application/json

{
  "content": "Write a function..."
}
```

#### Provider 설정

```http
# Provider 목록
GET /api/providers

# Provider 설정
POST /api/providers
Content-Type: application/json

{
  "name": "anthropic",
  "api_key": "sk-ant-...",
  "model": "claude-sonnet-4-5"
}

# 현재 Provider
GET /api/providers/current
```

#### Extension 관리

```http
# Extension 목록
GET /api/extensions

# Extension 추가
POST /api/extensions
Content-Type: application/json

{
  "name": "developer",
  "config": {
    "allow_shell": true
  }
}

# Extension 토글
PATCH /api/extensions/{extensionId}/toggle

# Extension 삭제
DELETE /api/extensions/{extensionId}
```

#### 설정

```http
# 설정 조회
GET /api/config

# 설정 업데이트
PATCH /api/config
Content-Type: application/json

{
  "execution_mode": "smart_approval",
  "auto_save": true
}
```

---

## WebSocket API

### 연결

```javascript
const ws = new WebSocket('ws://localhost:8080/api/sessions/{sessionId}/stream');

ws.onopen = () => {
  console.log('Connected');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Message:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected');
};
```

### 메시지 형식

```json
// 클라이언트 → 서버
{
  "type": "message",
  "content": "Create a function..."
}

// 서버 → 클라이언트 (스트리밍)
{
  "type": "content_delta",
  "delta": "I'll create"
}

{
  "type": "content_delta",
  "delta": " a function"
}

{
  "type": "tool_use",
  "tool": "write_file",
  "arguments": {
    "path": "main.py",
    "content": "..."
  }
}

{
  "type": "message_complete"
}
```

---

## 서버 구현

### main.rs

```rust
// crates/goose-server/src/main.rs
use axum::{Router, routing::{get, post}};
use tower_http::cors::CorsLayer;

#[tokio::main]
async fn main() -> Result<()> {
    // 로거 초기화
    tracing_subscriber::fmt::init();

    // 상태 초기화
    let state = AppState::new();

    // 라우터 생성
    let app = Router::new()
        .route("/api/sessions", get(list_sessions).post(create_session))
        .route("/api/sessions/:id", get(get_session).delete(delete_session))
        .route("/api/sessions/:id/messages", post(send_message))
        .route("/api/sessions/:id/stream", get(stream_handler))
        .route("/api/providers", get(list_providers).post(add_provider))
        .route("/api/extensions", get(list_extensions).post(add_extension))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // 서버 시작
    let addr = "0.0.0.0:8080".parse()?;
    println!("Server listening on {}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}
```

### 상태 관리

```rust
// crates/goose-server/src/state.rs
#[derive(Clone)]
pub struct AppState {
    sessions: Arc<RwLock<HashMap<String, Session>>>,
    agent_core: Arc<GooseCore>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            agent_core: Arc::new(GooseCore::new()),
        }
    }

    pub async fn get_session(&self, id: &str) -> Option<Session> {
        self.sessions.read().await.get(id).cloned()
    }

    pub async fn add_session(&self, session: Session) {
        self.sessions.write().await.insert(session.id.clone(), session);
    }
}
```

### 라우트 핸들러

```rust
// crates/goose-server/src/routes/sessions.rs
pub async fn create_session(
    State(state): State<AppState>,
    Json(payload): Json<CreateSessionRequest>,
) -> Result<Json<Session>, ApiError> {
    let session = Session {
        id: Uuid::new_v4().to_string(),
        name: payload.name,
        working_directory: payload.working_directory,
        created_at: Utc::now(),
        messages: vec![],
    };

    state.add_session(session.clone()).await;

    Ok(Json(session))
}

pub async fn send_message(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Json(payload): Json<SendMessageRequest>,
) -> Result<Json<Message>, ApiError> {
    let session = state
        .get_session(&session_id)
        .await
        .ok_or(ApiError::NotFound)?;

    // 에이전트 실행
    let response = state.agent_core
        .run(&session, payload.content)
        .await?;

    Ok(Json(response))
}
```

### 스트리밍 핸들러

```rust
// crates/goose-server/src/routes/stream.rs
pub async fn stream_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Response {
    ws.on_upgrade(|socket| handle_socket(socket, state, session_id))
}

async fn handle_socket(
    mut socket: WebSocket,
    state: AppState,
    session_id: String,
) {
    while let Some(Ok(msg)) = socket.recv().await {
        if let Message::Text(text) = msg {
            let request: ClientMessage = serde_json::from_str(&text).unwrap();

            // 스트림 응답 생성
            let stream = state.agent_core
                .run_stream(&session_id, request.content)
                .await
                .unwrap();

            // 스트림 데이터 전송
            tokio::pin!(stream);
            while let Some(chunk) = stream.next().await {
                let response = ServerMessage::ContentDelta {
                    delta: chunk,
                };

                socket.send(Message::Text(
                    serde_json::to_string(&response).unwrap()
                )).await.ok();
            }

            // 완료 메시지
            socket.send(Message::Text(
                serde_json::to_string(&ServerMessage::Complete).unwrap()
            )).await.ok();
        }
    }
}
```

---

## OpenAPI 스펙

### 스펙 생성

```bash
# OpenAPI 스펙 생성
just generate-openapi

# 생성 파일
# ui/desktop/openapi.json
```

### 스펙 구조

```yaml
openapi: 3.0.0
info:
  title: Goose API
  version: 1.23.0

paths:
  /api/sessions:
    get:
      summary: List sessions
      responses:
        '200':
          description: OK
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Session'

    post:
      summary: Create session
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateSessionRequest'
      responses:
        '201':
          description: Created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Session'

components:
  schemas:
    Session:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        working_directory:
          type: string
        created_at:
          type: string
          format: date-time
```

---

## 미들웨어

### CORS

```rust
use tower_http::cors::{CorsLayer, Any};

let cors = CorsLayer::new()
    .allow_origin(Any)
    .allow_methods(Any)
    .allow_headers(Any);

let app = Router::new()
    // routes...
    .layer(cors);
```

### 로깅

```rust
use tower_http::trace::{TraceLayer, DefaultMakeSpan};

let app = Router::new()
    // routes...
    .layer(
        TraceLayer::new_for_http()
            .make_span_with(DefaultMakeSpan::new())
    );
```

### 인증 (향후)

```rust
#[derive(Clone)]
pub struct AuthLayer {
    secret: String,
}

impl<S> tower::Layer<S> for AuthLayer {
    type Service = AuthMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AuthMiddleware {
            inner,
            secret: self.secret.clone(),
        }
    }
}
```

---

## 에러 핸들링

```rust
// crates/goose-server/src/error.rs
#[derive(Debug)]
pub enum ApiError {
    NotFound,
    BadRequest(String),
    InternalError(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::NotFound => (
                StatusCode::NOT_FOUND,
                "Resource not found"
            ),
            ApiError::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                msg.as_str()
            ),
            ApiError::InternalError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                msg.as_str()
            ),
        };

        (status, Json(json!({
            "error": message
        }))).into_response()
    }
}
```

---

## 테스트

### 통합 테스트

```rust
// crates/goose-server/tests/api_test.rs
#[tokio::test]
async fn test_create_session() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/sessions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"name": "test", "working_directory": "/tmp"}"#
                ))
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);
}
```

---

## 다음 단계

서버 및 API를 이해했다면, 다음 장에서는 확장 및 커스터마이징을 살펴봅니다.

*다음 글에서는 Goose를 커스터마이징하고 확장하는 방법을 알아봅니다.*
