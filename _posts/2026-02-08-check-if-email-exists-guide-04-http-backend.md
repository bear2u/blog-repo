---
layout: post
title: "check-if-email-exists 완벽 가이드 (04) - HTTP 백엔드"
date: 2026-02-08
categories: [개발 도구, 백엔드]
tags: [Email Validation, Rust, SMTP, API, Docker]
permalink: /check-if-email-exists-guide-04-http-backend/
excerpt: "Docker 기반 HTTP API 서버 완벽 가이드"
original_url: "https://github.com/reacherhq/check-if-email-exists"
---

# check-if-email-exists 완벽 가이드 (04) - HTTP 백엔드

## 목차
1. [HTTP 백엔드 개요](#http-백엔드-개요)
2. [Docker 이미지 상세](#docker-이미지-상세)
3. [API 엔드포인트](#api-엔드포인트)
4. [요청/응답 형식](#요청응답-형식)
5. [프록시 설정](#프록시-설정)
6. [인증 및 보안](#인증-및-보안)
7. [환경변수 및 설정](#환경변수-및-설정)
8. [다음 챕터 예고](#다음-챕터-예고)

---

## HTTP 백엔드 개요

**check-if-email-exists** HTTP 백엔드는 Rust의 Warp 웹 프레임워크로 구축된 REST API 서버입니다.

### 아키텍처

```
┌─────────────────────────────────────────────────────┐
│              HTTP 백엔드 아키텍처                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐                                  │
│  │  클라이언트   │                                  │
│  │ (curl/Python)│                                  │
│  └──────┬───────┘                                  │
│         │ HTTP POST                                │
│         │                                          │
│  ┌──────▼──────────────────────────┐              │
│  │      Warp Web Server             │              │
│  │      (포트 8080)                 │              │
│  ├──────────────────────────────────┤              │
│  │  /v0/check_email (레거시)        │              │
│  │  /v1/check_email (최신)          │              │
│  │  /v1/bulk (대량 검증)            │              │
│  └──────┬──────────────────────────┘              │
│         │                                          │
│  ┌──────▼──────────────────────────┐              │
│  │   Check Email Core               │              │
│  │   (검증 엔진)                    │              │
│  └──────┬──────────────────────────┘              │
│         │                                          │
│  ┌──────▼──────────────────────────┐              │
│  │   SMTP 서버                      │              │
│  │   (포트 25)                      │              │
│  └─────────────────────────────────┘              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 주요 구성 요소

| 구성 요소 | 기술 스택 | 역할 |
|---------|---------|------|
| **웹 프레임워크** | Warp | HTTP 요청 처리 |
| **검증 엔진** | check-if-email-exists core | 이메일 검증 로직 |
| **비동기 런타임** | Tokio | 동시성 처리 |
| **WebDriver** | ChromeDriver | Headless 검증 (Gmail, Yahoo) |
| **설정 관리** | TOML / 환경변수 | 런타임 설정 |

---

## Docker 이미지 상세

### 공식 이미지

Docker Hub에서 제공하는 공식 이미지:

```bash
# 최신 버전
docker pull reacherhq/backend:latest

# 특정 버전
docker pull reacherhq/backend:v0.11.7

# Commercial License Trial 버전
docker pull reacherhq/commercial-license-trial:latest
```

### 이미지 크기 및 레이어

```
REPOSITORY                              SIZE
reacherhq/backend                       ~500MB
├─ Ubuntu base                          ~100MB
├─ Rust binary                          ~50MB
├─ ChromeDriver                         ~150MB
└─ Dependencies                         ~200MB
```

### 기본 실행

```bash
# 포트 8080에서 실행
docker run -p 8080:8080 reacherhq/backend:latest
```

**출력:**

```log
2026-02-08T10:00:00.123456Z  INFO reacher: Running Reacher version="0.11.7"
2026-02-08T10:00:00.234567Z  INFO reacher: Backend name backend="backend-dev"
Starting ChromeDriver 124.0.6367.78 on port 9515
ChromeDriver was started successfully.
2026-02-08T10:00:00.456789Z  INFO reacher: Server is listening host=0.0.0.0 port=8080
```

### 고급 실행 옵션

```bash
# 백그라운드에서 실행 + 자동 재시작
docker run -d \
  --name reacher-backend \
  --restart unless-stopped \
  -p 8080:8080 \
  -v /path/to/backend_config.toml:/app/backend_config.toml \
  reacherhq/backend:latest

# 메모리 제한
docker run -p 8080:8080 \
  --memory="2g" \
  --memory-swap="2g" \
  reacherhq/backend:latest

# CPU 제한
docker run -p 8080:8080 \
  --cpus="2.0" \
  reacherhq/backend:latest
```

---

## API 엔드포인트

HTTP 백엔드는 3가지 주요 엔드포인트를 제공합니다.

### 엔드포인트 비교

| 엔드포인트 | 버전 | 특징 | 권장 사용 |
|----------|------|------|----------|
| `/v0/check_email` | 레거시 | 즉시 실행, 간단한 요청 | 레거시 코드 |
| `/v1/check_email` | 최신 | 제한율(throttle) 지원 | 새 프로젝트 |
| `/v1/bulk` | 최신 | 대량 검증 | 배치 작업 |

### 1. /v0/check_email (레거시)

#### 요청

```bash
curl -X POST http://localhost:8080/v0/check_email \
  -H 'Content-Type: application/json' \
  -d '{
    "to_email": "user@example.com",
    "from_email": "sender@mycompany.com",
    "hello_name": "mycompany.com",
    "proxy": {
      "host": "my-proxy.io",
      "port": 1080,
      "username": "proxyuser",
      "password": "proxypass"
    },
    "smtp_port": 25
  }'
```

#### 요청 필드

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|-----|------|------|--------|------|
| `to_email` | string | ✅ | - | 검증할 이메일 주소 |
| `from_email` | string | ❌ | `reacher.email@gmail.com` | MAIL FROM 이메일 |
| `hello_name` | string | ❌ | `gmail.com` | EHLO 명령에 사용할 도메인 |
| `proxy` | object | ❌ | null | SOCKS5 프록시 설정 |
| `smtp_port` | number | ❌ | 25 | SMTP 포트 |

#### 응답

```json
{
  "input": "user@example.com",
  "is_reachable": "safe",
  "misc": {
    "is_disposable": false,
    "is_role_account": false,
    "gravatar_url": null
  },
  "mx": {
    "accepts_mail": true,
    "records": [
      "mx1.example.com.",
      "mx2.example.com."
    ]
  },
  "smtp": {
    "can_connect_smtp": true,
    "has_full_inbox": false,
    "is_catch_all": false,
    "is_deliverable": true,
    "is_disabled": false,
    "verif_method": {
      "type": "Smtp",
      "host": "mx1.example.com.",
      "port": 25,
      "used_proxy": false
    }
  },
  "syntax": {
    "address": "user@example.com",
    "domain": "example.com",
    "is_valid_syntax": true,
    "username": "user",
    "normalized_email": "user@example.com",
    "suggestion": null
  }
}
```

### 2. /v1/check_email (최신, 권장)

v1 엔드포인트는 제한율(throttle) 기능을 지원합니다.

#### 요청

```bash
curl -X POST http://localhost:8080/v1/check_email \
  -H 'Content-Type: application/json' \
  -d '{
    "to_email": "user@example.com"
  }'
```

#### 추가 설정 옵션

```json
{
  "to_email": "user@example.com",
  "verif_method": {
    "gmail": {
      "type": "smtp",
      "from_email": "noreply@mycompany.com",
      "hello_name": "mycompany.com",
      "proxy": "proxy1",
      "smtp_port": 25
    },
    "yahoo": {
      "type": "headless"
    },
    "proxies": {
      "proxy1": {
        "host": "my-proxy.io",
        "port": 1080,
        "username": "user",
        "password": "pass"
      }
    }
  }
}
```

#### 응답

v0와 동일한 형식이지만, 제한율 초과 시:

```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60
}
```

### 3. /v1/bulk (대량 검증)

여러 이메일을 한 번에 검증합니다.

#### 요청

```bash
curl -X POST http://localhost:8080/v1/bulk \
  -H 'Content-Type: application/json' \
  -d '{
    "emails": [
      "user1@example.com",
      "user2@gmail.com",
      "user3@yahoo.com"
    ]
  }'
```

#### 응답

```json
{
  "results": [
    {
      "input": "user1@example.com",
      "is_reachable": "safe",
      // ... 전체 검증 결과
    },
    {
      "input": "user2@gmail.com",
      "is_reachable": "invalid",
      // ... 전체 검증 결과
    },
    {
      "input": "user3@yahoo.com",
      "is_reachable": "risky",
      // ... 전체 검증 결과
    }
  ],
  "summary": {
    "total": 3,
    "safe": 1,
    "invalid": 1,
    "risky": 1,
    "unknown": 0
  }
}
```

---

## 요청/응답 형식

### 에러 응답

API는 HTTP 상태 코드와 함께 에러를 반환합니다.

#### 400 Bad Request

```json
{
  "error": "Invalid email format",
  "field": "to_email"
}
```

#### 401 Unauthorized

```json
{
  "error": "Missing or invalid x-reacher-secret header"
}
```

#### 429 Too Many Requests

```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60,
  "limit": "60 requests per minute"
}
```

#### 500 Internal Server Error

```json
{
  "error": "SMTP connection timeout",
  "details": "Failed to connect to mx1.example.com:25"
}
```

### Content-Type

모든 요청과 응답은 `application/json`을 사용합니다:

```http
POST /v1/check_email HTTP/1.1
Host: localhost:8080
Content-Type: application/json
Accept: application/json
```

---

## 프록시 설정

프록시는 SMTP 포트 25 차단을 우회하고 IP 평판을 관리하는 데 필수적입니다.

### 단일 프록시 설정

#### 환경변수로 설정

```bash
docker run -p 8080:8080 \
  -e RCH__PROXY__HOST=my-proxy.io \
  -e RCH__PROXY__PORT=1080 \
  -e RCH__PROXY__USERNAME=proxyuser \
  -e RCH__PROXY__PASSWORD=proxypass \
  reacherhq/backend:latest
```

#### 설정 파일로 설정

`backend_config.toml`:

```toml
[proxy]
host = "my-proxy.io"
port = 1080
username = "proxyuser"
password = "proxypass"
timeout_ms = 10000
```

### 다중 프록시 설정

도메인별로 다른 프록시를 사용할 수 있습니다.

#### 설정 파일

```toml
[overrides.proxies]
proxy1 = { host = "proxy1.io", port = 1080, username = "user1", password = "pass1" }
proxy2 = { host = "proxy2.io", port = 1081, username = "user2", password = "pass2" }
proxy3 = { host = "proxy3.io", port = 1082 }

# Gmail은 proxy1 사용
[overrides.gmail]
type = "smtp"
proxy = "proxy1"
hello_name = "mycompany.com"
from_email = "noreply@mycompany.com"

# Yahoo는 proxy2 사용
[overrides.yahoo]
type = "headless"
proxy = "proxy2"

# Hotmail은 proxy3 사용
[overrides.hotmailb2c]
type = "smtp"
proxy = "proxy3"
```

#### 환경변수로 설정

```bash
docker run -p 8080:8080 \
  -e RCH__OVERRIDES__PROXIES__PROXY1__HOST=proxy1.io \
  -e RCH__OVERRIDES__PROXIES__PROXY1__PORT=1080 \
  -e RCH__OVERRIDES__PROXIES__PROXY2__HOST=proxy2.io \
  -e RCH__OVERRIDES__PROXIES__PROXY2__PORT=1081 \
  -e RCH__OVERRIDES__GMAIL__TYPE=smtp \
  -e RCH__OVERRIDES__GMAIL__PROXY=proxy1 \
  reacherhq/backend:latest
```

### 프록시 검증

프록시가 제대로 작동하는지 확인:

```bash
# 프록시 사용 확인
curl -X POST http://localhost:8080/v0/check_email \
  -H 'Content-Type: application/json' \
  -d '{"to_email": "test@gmail.com"}' | jq '.smtp.verif_method'
```

**출력:**

```json
{
  "type": "Smtp",
  "host": "gmail-smtp-in.l.google.com.",
  "port": 25,
  "used_proxy": true
}
```

---

## 인증 및 보안

### Shared Secret 인증

HTTP 헤더를 통한 간단한 인증 방식:

#### 설정

```toml
header_secret = "my-super-secret-key"
```

또는 환경변수:

```bash
docker run -p 8080:8080 \
  -e RCH__HEADER_SECRET=my-super-secret-key \
  reacherhq/backend:latest
```

#### 요청

```bash
curl -X POST http://localhost:8080/v1/check_email \
  -H 'Content-Type: application/json' \
  -H 'x-reacher-secret: my-super-secret-key' \
  -d '{"to_email": "test@example.com"}'
```

#### 인증 실패

```json
{
  "error": "Unauthorized",
  "message": "Missing or invalid x-reacher-secret header"
}
```

### HTTPS 설정

프로덕션 환경에서는 리버스 프록시를 통해 HTTPS를 설정합니다.

#### Nginx 리버스 프록시

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### CORS 설정

기본적으로 CORS는 모든 오리진을 허용합니다. 프로덕션에서는 제한해야 합니다.

```rust
// backend/src/main.rs (참고용)
let cors = warp::cors()
    .allow_origin("https://myapp.com")
    .allow_methods(vec!["POST"])
    .allow_headers(vec!["Content-Type", "x-reacher-secret"]);
```

---

## 환경변수 및 설정

### 주요 환경변수

| 환경변수 | 기본값 | 설명 |
|---------|--------|------|
| `RCH__BACKEND_NAME` | `backend-dev` | 백엔드 식별자 |
| `RCH__HTTP_HOST` | `127.0.0.1` | 바인딩 호스트 |
| `RCH__HTTP_PORT` | `8080` | 바인딩 포트 |
| `RCH__HELLO_NAME` | `localhost` | EHLO 명령에 사용할 도메인 |
| `RCH__FROM_EMAIL` | `hello@localhost` | MAIL FROM 이메일 |
| `RCH__SMTP_TIMEOUT` | 없음 (무제한) | SMTP 타임아웃 (초) |
| `RCH__HEADER_SECRET` | 없음 | 인증 시크릿 |
| `RCH__SENTRY_DSN` | 없음 | Sentry 오류 추적 DSN |
| `RCH__WEBDRIVER_ADDR` | `http://localhost:9515` | ChromeDriver 주소 |

### Throttle 설정

요청 제한율을 설정하여 IP 평판을 보호합니다:

```toml
[throttle]
max_requests_per_second = 20
max_requests_per_minute = 60
max_requests_per_hour = 1000
max_requests_per_day = 10000
```

환경변수:

```bash
docker run -p 8080:8080 \
  -e RCH__THROTTLE__MAX_REQUESTS_PER_MINUTE=60 \
  -e RCH__THROTTLE__MAX_REQUESTS_PER_DAY=10000 \
  reacherhq/backend:latest
```

### 검증 방법 Override

도메인별 검증 방법을 지정합니다:

```toml
[overrides.gmail]
type = "smtp"  # 또는 "api"
proxy = "proxy1"
hello_name = "mycompany.com"
from_email = "noreply@mycompany.com"

[overrides.yahoo]
type = "headless"  # 또는 "smtp", "api"

[overrides.hotmailb2c]
type = "headless"  # 또는 "smtp"

[overrides.hotmailb2b]
type = "smtp"  # B2B는 SMTP만 지원
```

### 전체 설정 예시

```toml
backend_name = "production-backend-1"
http_host = "0.0.0.0"
http_port = 8080
hello_name = "mycompany.com"
from_email = "noreply@mycompany.com"
smtp_timeout = 45
header_secret = "production-secret-key"
webdriver_addr = "http://localhost:9515"

[proxy]
host = "main-proxy.io"
port = 1080
username = "proxyuser"
password = "proxypass"
timeout_ms = 10000

[throttle]
max_requests_per_minute = 60
max_requests_per_day = 10000

[overrides.proxies]
proxy1 = { host = "gmail-proxy.io", port = 1080 }
proxy2 = { host = "yahoo-proxy.io", port = 1081 }

[overrides.gmail]
type = "smtp"
proxy = "proxy1"

[overrides.yahoo]
type = "headless"
proxy = "proxy2"
```

### 설정 파일 마운트

```bash
docker run -p 8080:8080 \
  -v /path/to/backend_config.toml:/app/backend_config.toml:ro \
  reacherhq/backend:latest
```

---

## 다음 챕터 예고

### 챕터 05: 고급 활용

다음 챕터에서는 프로덕션 환경을 위한 고급 주제를 다룹니다:

1. Self-hosting 가이드 (VPS, 클라우드 배포)
2. RabbitMQ 통합 (대량 처리 아키텍처)
3. AWS SQS 통합
4. 프로덕션 배포 체크리스트
5. 성능 최적화 및 튜닝
6. 모니터링 및 로깅 (Prometheus, Grafana)

---

## 결론

이 챕터에서는 **check-if-email-exists** HTTP 백엔드의 모든 측면을 살펴보았습니다.

### 핵심 요약

**API 엔드포인트:**

- `/v0/check_email`: 레거시, 즉시 실행
- `/v1/check_email`: 최신, throttle 지원 (권장)
- `/v1/bulk`: 대량 검증

**프록시 설정:**

- 단일 프록시: 모든 요청에 동일 프록시
- 다중 프록시: 도메인별 다른 프록시
- 필수 이유: 포트 25 차단 우회, IP 평판 관리

**보안:**

- Shared Secret 인증 (`x-reacher-secret` 헤더)
- HTTPS (Nginx 리버스 프록시)
- Throttle (요청 제한율)

### 프로덕션 체크리스트

- [ ] `header_secret` 설정
- [ ] HTTPS 활성화
- [ ] 프록시 설정
- [ ] Throttle 설정
- [ ] 모니터링 구축 (다음 챕터)
- [ ] 백업 전략 수립

### 참고 자료

- Backend 소스 코드: https://github.com/reacherhq/check-if-email-exists/tree/main/backend
- OpenAPI 문서: https://docs.reacher.email/advanced/openapi
- Docker Hub: https://hub.docker.com/r/reacherhq/backend

다음 챕터에서는 대규모 트래픽을 처리하기 위한 고급 아키텍처를 알아보겠습니다.
