---
layout: post
title: "check-if-email-exists 완벽 가이드 (02) - 빠른 시작"
date: 2026-02-08
categories: [개발 도구, 백엔드]
tags: [Email Validation, Rust, SMTP, API, Docker]
permalink: /check-if-email-exists-guide-02-quick-start/
excerpt: "Docker, CLI, Rust 라이브러리로 이메일 검증 시작하기"
original_url: "https://github.com/reacherhq/check-if-email-exists"
---

# check-if-email-exists 완벽 가이드 (02) - 빠른 시작

## 목차
1. [사용 방법 개요](#사용-방법-개요)
2. [방법 1: Docker를 통한 HTTP 백엔드](#방법-1-docker를-통한-http-백엔드)
3. [방법 2: CLI 바이너리 사용](#방법-2-cli-바이너리-사용)
4. [방법 3: Rust 라이브러리 통합](#방법-3-rust-라이브러리-통합)
5. [첫 이메일 검증 실습](#첫-이메일-검증-실습)
6. [문제 해결](#문제-해결)
7. [다음 챕터 예고](#다음-챕터-예고)

---

## 사용 방법 개요

**check-if-email-exists**는 3가지 방법으로 사용할 수 있습니다:

```
┌─────────────────────────────────────────────────────┐
│            3가지 사용 방법 비교                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  방법 1: Docker (HTTP 백엔드) 🥇 인기                  │
│  ┌─────────────────────────────────────────┐       │
│  │  Docker → HTTP API → 어떤 언어에서나 호출  │       │
│  │  난이도: ⭐                               │       │
│  │  추천: 프로덕션 환경, 팀 협업              │       │
│  └─────────────────────────────────────────┘       │
│                                                     │
│  방법 2: CLI 바이너리                               │
│  ┌─────────────────────────────────────────┐       │
│  │  다운로드 → 터미널에서 즉시 실행           │       │
│  │  난이도: ⭐                               │       │
│  │  추천: 로컬 테스트, 개인 사용              │       │
│  └─────────────────────────────────────────┘       │
│                                                     │
│  방법 3: Rust 라이브러리                            │
│  ┌─────────────────────────────────────────┐       │
│  │  Cargo.toml → 직접 코드 통합              │       │
│  │  난이도: ⭐⭐                             │       │
│  │  추천: Rust 프로젝트, 커스터마이징         │       │
│  └─────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

### 선택 가이드

| 상황 | 추천 방법 |
|-----|---------|
| 프로덕션 배포 | 방법 1: Docker |
| Python/Node.js 프로젝트 | 방법 1: Docker |
| 빠른 테스트 | 방법 2: CLI |
| Rust 프로젝트 | 방법 3: 라이브러리 |
| 대량 검증 | 방법 1: Docker + RabbitMQ |

---

## 방법 1: Docker를 통한 HTTP 백엔드

가장 인기 있는 방법입니다. Docker만 설치되어 있으면 몇 초 만에 시작할 수 있습니다.

### 1.1 사전 요구사항

```bash
# Docker 설치 확인
docker --version
# Docker version 20.10.0 이상 권장
```

Docker가 없다면: https://docs.docker.com/get-docker/

### 1.2 Docker 컨테이너 실행

```bash
# 최신 이미지 다운로드 및 실행
docker run -p 8080:8080 reacherhq/backend:latest
```

**출력 예시:**

```log
2026-02-08T10:30:15.123456Z  INFO reacher: Running Reacher version="0.11.7"
Starting ChromeDriver 124.0.6367.78 on port 9515
ChromeDriver was started successfully.
2026-02-08T10:30:15.456789Z  INFO reacher: Server is listening host=0.0.0.0 port=8080
```

### 1.3 API 요청 보내기

**터미널에서 curl 사용:**

```bash
curl -X POST http://localhost:8080/v0/check_email \
  -H 'Content-Type: application/json' \
  -d '{"to_email": "test@gmail.com"}'
```

**Python에서 요청:**

```python
import requests

response = requests.post(
    'http://localhost:8080/v0/check_email',
    json={'to_email': 'test@gmail.com'}
)

result = response.json()
print(f"Is reachable: {result['is_reachable']}")
```

**Node.js에서 요청:**

```javascript
const axios = require('axios');

async function checkEmail(email) {
  const response = await axios.post('http://localhost:8080/v0/check_email', {
    to_email: email
  });

  console.log('Is reachable:', response.data.is_reachable);
  return response.data;
}

checkEmail('test@gmail.com');
```

### 1.4 프록시 설정 (옵션)

SOCKS5 프록시를 통해 검증하려면:

```bash
curl -X POST http://localhost:8080/v0/check_email \
  -H 'Content-Type: application/json' \
  -d '{
    "to_email": "test@gmail.com",
    "proxy": {
      "host": "my-proxy.io",
      "port": 1080,
      "username": "proxyuser",
      "password": "proxypass"
    }
  }'
```

### 1.5 백그라운드에서 실행

```bash
# 데몬 모드로 실행
docker run -d \
  --name reacher \
  --restart unless-stopped \
  -p 8080:8080 \
  reacherhq/backend:latest

# 로그 확인
docker logs -f reacher

# 중지
docker stop reacher

# 재시작
docker start reacher
```

### 1.6 환경 변수로 설정

```bash
docker run -p 8080:8080 \
  -e RCH__FROM_EMAIL=my-email@example.com \
  -e RCH__HELLO_NAME=example.com \
  -e RCH__PROXY__HOST=my-proxy.io \
  -e RCH__PROXY__PORT=1080 \
  reacherhq/backend:latest
```

---

## 방법 2: CLI 바이너리 사용

터미널에서 직접 이메일을 검증하는 가장 간단한 방법입니다.

### 2.1 바이너리 다운로드

**릴리스 페이지에서 다운로드:**
https://github.com/reacherhq/check-if-email-exists/releases

```bash
# Linux (x86_64)
wget https://github.com/reacherhq/check-if-email-exists/releases/download/v0.11.7/check_if_email_exists-linux-x86_64

# macOS (Apple Silicon)
wget https://github.com/reacherhq/check-if-email-exists/releases/download/v0.11.7/check_if_email_exists-macos-arm64

# macOS (Intel)
wget https://github.com/reacherhq/check-if-email-exists/releases/download/v0.11.7/check_if_email_exists-macos-x86_64

# Windows
# check_if_email_exists-windows-x86_64.exe 다운로드
```

### 2.2 실행 권한 부여 (Linux/macOS)

```bash
chmod +x check_if_email_exists-linux-x86_64
mv check_if_email_exists-linux-x86_64 /usr/local/bin/check_if_email_exists
```

### 2.3 기본 사용법

```bash
# 도움말 확인
check_if_email_exists --help
```

**출력:**

```
check-if-email-exists-cli
Check if an email address exists without sending any email.

USAGE:
    check_if_email_exists [OPTIONS] <TO_EMAIL>

ARGS:
    <TO_EMAIL>    The email to check

OPTIONS:
        --check-gravatar <CHECK_GRAVATAR>
            Whether to check if a gravatar image is existing [default: false]

        --from-email <FROM_EMAIL>
            The email to use in the `MAIL FROM:` SMTP command
            [default: reacher.email@gmail.com]

        --gmail-verif-method <GMAIL_VERIF_METHOD>
            Select how to verify Gmail: api or smtp [default: smtp]

    -h, --help
            Print help information

        --haveibeenpwned-api-key <HAVEIBEENPWNED_API_KEY>
            HaveIBeenPwned API key

        --hello-name <HELLO_NAME>
            The name to use in the `EHLO:` SMTP command [default: gmail.com]

        --proxy-host <PROXY_HOST>
            Use the specified SOCKS5 proxy host

        --proxy-port <PROXY_PORT>
            SOCKS5 proxy port [default: 1080]

        --proxy-username <PROXY_USERNAME>
            Proxy username

        --proxy-password <PROXY_PASSWORD>
            Proxy password

        --smtp-port <SMTP_PORT>
            The port to use for SMTP [default: 25]

    -V, --version
            Print version information
```

### 2.4 이메일 검증 실행

**간단한 검증:**

```bash
check_if_email_exists test@gmail.com
```

**출력 (JSON):**

```json
{
  "input": "test@gmail.com",
  "is_reachable": "invalid",
  "misc": {
    "is_disposable": false,
    "is_role_account": false
  },
  "mx": {
    "accepts_mail": true,
    "records": ["gmail-smtp-in.l.google.com."]
  },
  "smtp": {
    "can_connect_smtp": true,
    "is_deliverable": false
  },
  "syntax": {
    "is_valid_syntax": true,
    "domain": "gmail.com",
    "username": "test"
  }
}
```

### 2.5 고급 옵션 사용

**프록시를 통한 검증:**

```bash
check_if_email_exists test@gmail.com \
  --proxy-host my-proxy.io \
  --proxy-port 1080 \
  --proxy-username myuser \
  --proxy-password mypass
```

**커스텀 발신자 설정:**

```bash
check_if_email_exists test@gmail.com \
  --from-email noreply@example.com \
  --hello-name example.com
```

**Gravatar 및 HaveIBeenPwned 체크:**

```bash
check_if_email_exists test@gmail.com \
  --check-gravatar true \
  --haveibeenpwned-api-key YOUR_API_KEY
```

### 2.6 디버그 모드

상세한 로그를 보려면:

```bash
RUST_LOG=debug check_if_email_exists test@gmail.com
```

**디버그 출력 예시:**

```log
[DEBUG] Resolving MX records for gmail.com
[DEBUG] Found 5 MX records
[DEBUG] Connecting to SMTP server: gmail-smtp-in.l.google.com:25
[DEBUG] EHLO gmail.com
[DEBUG] MAIL FROM: <reacher.email@gmail.com>
[DEBUG] RCPT TO: <test@gmail.com>
[DEBUG] Response: 550 5.1.1 User unknown
[INFO] Result: invalid
```

---

## 방법 3: Rust 라이브러리 통합

Rust 프로젝트에 직접 통합하여 최대한의 유연성을 확보합니다.

### 3.1 Cargo.toml에 의존성 추가

```toml
[dependencies]
check-if-email-exists = "0.11"
tokio = { version = "1.0", features = ["full"] }
```

### 3.2 기본 사용 예제

```rust
use check_if_email_exists::{check_email, CheckEmailInput};

#[tokio::main]
async fn main() {
    // 검증할 이메일 주소 설정
    let mut input = CheckEmailInput::new(vec!["test@gmail.com".into()]);

    // 이메일 검증 실행
    let result = check_email(&input).await;

    // 결과 출력
    println!("{:#?}", result);
}
```

### 3.3 옵션 설정 예제

```rust
use check_if_email_exists::{
    check_email,
    CheckEmailInput,
    CheckEmailInputProxy,
};

#[tokio::main]
async fn main() {
    // 입력 설정 생성
    let mut input = CheckEmailInput::new(vec!["test@gmail.com".into()]);

    // 발신자 이메일 설정
    input
        .set_from_email("noreply@example.com".into())
        .set_hello_name("example.com".into());

    // 프록시 설정
    input.set_proxy(CheckEmailInputProxy {
        host: "my-proxy.io".into(),
        port: 1080,
        username: Some("proxyuser".into()),
        password: Some("proxypass".into()),
    });

    // Gravatar 체크 활성화
    input.set_check_gravatar(true);

    // 검증 실행
    let result = check_email(&input).await;

    // is_reachable 필드만 출력
    for email_result in result {
        println!("{}: {}",
            email_result.input,
            email_result.is_reachable
        );
    }
}
```

### 3.4 여러 이메일 동시 검증

```rust
use check_if_email_exists::{check_email, CheckEmailInput};

#[tokio::main]
async fn main() {
    // 여러 이메일 주소 설정
    let emails = vec![
        "user1@gmail.com".to_string(),
        "user2@yahoo.com".to_string(),
        "user3@outlook.com".to_string(),
    ];

    let input = CheckEmailInput::new(emails);

    // 병렬로 검증 (내부적으로 처리됨)
    let results = check_email(&input).await;

    // 결과 테이블 출력
    println!("{:<30} | {:<10}", "Email", "Status");
    println!("{:-<30}-+-{:-<10}", "", "");

    for result in results {
        println!("{:<30} | {:<10}",
            result.input,
            result.is_reachable
        );
    }
}
```

**출력 예시:**

```
Email                          | Status
------------------------------+----------
user1@gmail.com                | invalid
user2@yahoo.com                | safe
user3@outlook.com              | risky
```

### 3.5 에러 처리

```rust
use check_if_email_exists::{check_email, CheckEmailInput};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let input = CheckEmailInput::new(vec!["test@gmail.com".into()]);

    match check_email(&input).await.get(0) {
        Some(result) => {
            // 검증 성공
            match result.is_reachable.as_str() {
                "safe" => println!("✅ Safe to send"),
                "invalid" => println!("❌ Invalid email"),
                "risky" => println!("⚠️ Risky email"),
                "unknown" => println!("❓ Cannot determine"),
                _ => println!("Unexpected result"),
            }
        }
        None => {
            // 검증 실패
            eprintln!("Failed to verify email");
        }
    }

    Ok(())
}
```

### 3.6 커스텀 타임아웃 설정

```rust
use check_if_email_exists::{check_email, CheckEmailInput};
use std::time::Duration;

#[tokio::main]
async fn main() {
    let mut input = CheckEmailInput::new(vec!["test@gmail.com".into()]);

    // SMTP 타임아웃 설정 (10초)
    input.set_smtp_timeout(Duration::from_secs(10));

    let result = check_email(&input).await;
    println!("{:#?}", result);
}
```

---

## 첫 이메일 검증 실습

실제로 다양한 이메일 주소를 검증해 봅시다.

### 실습 1: 유효한 이메일

```bash
# Docker 방법
curl -X POST http://localhost:8080/v0/check_email \
  -H 'Content-Type: application/json' \
  -d '{"to_email": "amaury@reacher.email"}'

# CLI 방법
check_if_email_exists amaury@reacher.email
```

**예상 결과:**

```json
{
  "input": "amaury@reacher.email",
  "is_reachable": "safe",
  "smtp": {
    "is_deliverable": true
  }
}
```

### 실습 2: 무효한 이메일

```bash
curl -X POST http://localhost:8080/v0/check_email \
  -H 'Content-Type: application/json' \
  -d '{"to_email": "nonexistent@gmail.com"}'
```

**예상 결과:**

```json
{
  "input": "nonexistent@gmail.com",
  "is_reachable": "invalid",
  "smtp": {
    "is_deliverable": false
  }
}
```

### 실습 3: 일회용 이메일

```bash
curl -X POST http://localhost:8080/v0/check_email \
  -H 'Content-Type: application/json' \
  -d '{"to_email": "test@tempmail.com"}'
```

**예상 결과:**

```json
{
  "input": "test@tempmail.com",
  "is_reachable": "risky",
  "misc": {
    "is_disposable": true
  }
}
```

### 실습 4: Catch-all 도메인

```bash
curl -X POST http://localhost:8080/v0/check_email \
  -H 'Content-Type: application/json' \
  -d '{"to_email": "anything@catchall-domain.com"}'
```

**예상 결과:**

```json
{
  "input": "anything@catchall-domain.com",
  "is_reachable": "risky",
  "smtp": {
    "is_catch_all": true
  }
}
```

---

## 문제 해결

### 문제 1: Docker 포트 충돌

**증상:**

```
Error: Bind for 0.0.0.0:8080 failed: port is already allocated
```

**해결:**

```bash
# 다른 포트 사용
docker run -p 8081:8080 reacherhq/backend:latest

# 또는 기존 프로세스 종료
lsof -ti:8080 | xargs kill
```

### 문제 2: SMTP 포트 25 차단

**증상:**

```json
{
  "is_reachable": "unknown",
  "smtp": {
    "can_connect_smtp": false
  }
}
```

**원인:** 대부분의 클라우드 제공자(AWS, GCP, Azure)는 포트 25를 차단합니다.

**해결:**

```bash
# 프록시 사용
curl -X POST http://localhost:8080/v0/check_email \
  -H 'Content-Type: application/json' \
  -d '{
    "to_email": "test@gmail.com",
    "proxy": {
      "host": "proxy-with-port-25.com",
      "port": 1080
    }
  }'
```

### 문제 3: CLI 실행 권한 오류 (Linux/macOS)

**증상:**

```
Permission denied
```

**해결:**

```bash
chmod +x check_if_email_exists
```

### 문제 4: 느린 검증 속도

**원인:** Gmail, Yahoo 등은 SMTP 검증을 제한할 수 있습니다.

**해결:**

```bash
# 타임아웃 증가
docker run -p 8080:8080 \
  -e RCH__SMTP_TIMEOUT=30 \
  reacherhq/backend:latest
```

---

## 다음 챕터 예고

### 챕터 03: 검증 메커니즘

다음 챕터에서는 **check-if-email-exists**가 이메일을 검증하는 내부 메커니즘을 상세히 다룹니다:

1. Syntax 검증 (정규식, 형식 체크)
2. DNS MX 레코드 조회
3. SMTP 핸드셰이크 과정
4. Email Deliverability 판단 알고리즘
5. Disposable Email 감지 방법
6. Catch-all 주소 감지 메커니즘
7. Role Account 판별
8. Gravatar 및 HaveIBeenPwned 통합

---

## 결론

이 챕터에서는 **check-if-email-exists**를 시작하는 3가지 방법을 모두 살펴보았습니다:

### 핵심 요약

| 방법 | 장점 | 사용 사례 |
|-----|------|---------|
| **Docker** | 언어 독립적, 프로덕션 준비 | API 서버, 마이크로서비스 |
| **CLI** | 설치 없음, 즉시 사용 | 로컬 테스트, 배치 스크립트 |
| **Rust 라이브러리** | 최대 유연성, 성능 | Rust 앱, 커스터마이징 |

### 추천 시작 방법

1. **처음 시도**: CLI로 빠르게 테스트
2. **프로젝트 통합**: Docker로 HTTP API 구축
3. **고급 사용**: Rust 라이브러리로 커스터마이징

### 참고 자료

- CLI 문서: https://github.com/reacherhq/check-if-email-exists/tree/main/cli
- Backend 문서: https://github.com/reacherhq/check-if-email-exists/tree/main/backend
- Rust API 문서: https://docs.rs/check-if-email-exists

다음 챕터에서는 각 검증 단계가 어떻게 작동하는지 깊이 있게 알아보겠습니다.
