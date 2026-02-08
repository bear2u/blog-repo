---
layout: page
title: check-if-email-exists 가이드
permalink: /check-if-email-exists-guide/
icon: fas fa-envelope-open-text
---

# check-if-email-exists 완벽 가이드

> **이메일을 보내지 않고 이메일 주소 유효성 검증**

**check-if-email-exists**는 실제로 이메일을 보내지 않고 이메일 주소의 유효성을 검증하는 Rust 라이브러리 & HTTP 백엔드입니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/blog-repo/check-if-email-exists-guide-01-intro/) | 프로젝트 소개, 14가지 검증 항목, JSON 출력 |
| 02 | [빠른 시작](/blog-repo/check-if-email-exists-guide-02-quick-start/) | Docker/CLI/Rust 3가지 시작 방법 |
| 03 | [검증 메커니즘](/blog-repo/check-if-email-exists-guide-03-verification/) | Syntax/DNS/SMTP 검증 상세 분석 |
| 04 | [HTTP 백엔드](/blog-repo/check-if-email-exists-guide-04-http-backend/) | API 엔드포인트, 프록시, 보안 설정 |
| 05 | [고급 활용](/blog-repo/check-if-email-exists-guide-05-advanced/) | RabbitMQ, SQS, 프로덕션 배포 |
| 06 | [개발 및 기여](/blog-repo/check-if-email-exists-guide-06-development/) | Rust 개발 환경, 기여 가이드 |

---

## 주요 특징

### ✅ 14가지 검증 항목

| 검증 항목 | 설명 |
|----------|------|
| **Email Reachability** | 이메일 도달 가능성 (safe/risky/invalid/unknown) |
| **Syntax Validation** | 문법 검증 (형식, 정규식) |
| **DNS Records** | MX 레코드 유효성 검사 |
| **Disposable Email** | 일회용 이메일 주소 감지 |
| **SMTP Server** | 메일 서버 연결 테스트 |
| **Email Deliverability** | 실제 전송 가능 여부 |
| **Mailbox Disabled** | 비활성화된 메일박스 감지 |
| **Full Inbox** | 메일박스 가득참 감지 |
| **Catch-all Address** | 모든 이메일 수신 주소 감지 |
| **Role Account** | 역할 계정 (info@, support@) 감지 |
| **Gravatar URL** | Gravatar 프로필 사진 URL |
| **Have I Been Pwned** | 데이터 유출 이력 확인 |

### 🚀 3가지 사용 방법

```
┌─────────────────────────────────────────────────────────────┐
│              check-if-email-exists                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Docker HTTP Backend (⭐ 가장 인기)                       │
│     └─ docker run -p 8080:8080 reacherhq/backend           │
│                                                              │
│  2. CLI Binary                                               │
│     └─ check_if_email_exists user@example.com              │
│                                                              │
│  3. Rust Library                                             │
│     └─ use check_if_email_exists::check_email;             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 빠른 시작

### 1. Docker로 시작 (가장 쉬움)

```bash
# HTTP 백엔드 실행
docker run -p 8080:8080 reacherhq/backend:latest

# API 요청
curl -X POST http://localhost:8080/v0/check_email \
  -H "Content-Type: application/json" \
  -d '{"to_email": "user@example.com"}'
```

### 2. CLI 다운로드

```bash
# 릴리스 페이지에서 바이너리 다운로드
# https://github.com/reacherhq/check-if-email-exists/releases

# 실행
check_if_email_exists user@example.com
```

### 3. Rust 라이브러리

```toml
[dependencies]
check-if-email-exists = "0.9"
```

```rust
use check_if_email_exists::{check_email, CheckEmailInput};

async fn verify() {
    let mut input = CheckEmailInput::new(vec!["user@example.com".into()]);
    let result = check_email(&input).await;
    println!("{:?}", result);
}
```

---

## JSON 출력 예시

```json
{
  "input": "user@gmail.com",
  "is_reachable": "safe",
  "misc": {
    "is_disposable": false,
    "is_role_account": false,
    "is_b2c": true
  },
  "mx": {
    "accepts_mail": true,
    "records": ["gmail-smtp-in.l.google.com."]
  },
  "smtp": {
    "can_connect_smtp": true,
    "has_full_inbox": false,
    "is_catch_all": false,
    "is_deliverable": true,
    "is_disabled": false
  },
  "syntax": {
    "domain": "gmail.com",
    "is_valid_syntax": true,
    "username": "user"
  }
}
```

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                  Verification Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Email Input (user@example.com)                              │
│         ↓                                                    │
│  1. Syntax Validation                                        │
│         ↓                                                    │
│  2. DNS/MX Records Check                                     │
│         ↓                                                    │
│  3. SMTP Server Connection                                   │
│         ↓                                                    │
│  4. Additional Checks                                        │
│     • Disposable Email Detection                             │
│     • Catch-all Detection                                    │
│     • Gravatar Lookup                                        │
│     • HaveIBeenPwned Check                                   │
│         ↓                                                    │
│  Result: is_reachable (safe/risky/invalid/unknown)           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 기술 | 용도 |
|------|------|
| Rust | 핵심 라이브러리 (fast-smtp, tokio) |
| Actix-web | HTTP 백엔드 서버 |
| PostgreSQL | 결과 저장 (선택) |
| RabbitMQ | 대량 처리 큐 (선택) |
| AWS SQS | 클라우드 메시지 큐 (선택) |
| Docker | 컨테이너 배포 |

---

## 사용 사례

### 1. 회원가입 폼 검증

```javascript
// 프론트엔드에서 호출
async function validateEmail(email) {
  const response = await fetch('http://localhost:8080/v0/check_email', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ to_email: email })
  });

  const result = await response.json();

  if (result.is_reachable === 'invalid') {
    alert('유효하지 않은 이메일 주소입니다.');
    return false;
  }

  return true;
}
```

### 2. 이메일 목록 정리

```bash
# 대량 검증
cat email_list.txt | xargs -I {} \
  check_if_email_exists {}
```

### 3. 스팸 방지

```rust
// Disposable Email 차단
if result.misc.is_disposable {
    return Err("일회용 이메일은 사용할 수 없습니다.");
}
```

### 4. 이메일 마케팅 준비

```python
# Invalid/Risky 제거
valid_emails = []
for email in email_list:
    result = check_email(email)
    if result['is_reachable'] in ['safe', 'unknown']:
        valid_emails.append(email)
```

---

## 프로덕션 요구사항

### 포트 25 필수

SMTP 검증을 위해 **아웃바운드 포트 25**가 열려 있어야 합니다.

```bash
# 포트 25 테스트
telnet smtp.gmail.com 25
```

**클라우드 제한사항:**
- AWS EC2: 기본적으로 포트 25 차단 (요청 필요)
- GCP: 포트 25 차단 (우회 불가)
- Azure: 포트 25 차단 (우회 불가)
- DigitalOcean: 포트 25 열림 ✅
- Vultr: 포트 25 열림 ✅

---

## 성능

| 메트릭 | 값 |
|--------|-----|
| **평균 검증 시간** | 2-5초/이메일 |
| **동시 처리** | 100+ concurrent |
| **처리량** | ~1000 이메일/분 (단일 인스턴스) |
| **메모리 사용** | ~50MB (베이스) |
| **CPU 사용** | 낮음 (I/O bound) |

---

## 라이선스

| 라이선스 | 용도 |
|---------|------|
| **AGPL-3.0** | 오픈소스 (무료) |
| **Commercial** | 상업적 사용 (유료, 문의 필요) |

**AGPL-3.0 요구사항:**
- 소스 코드 공개 필수
- 네트워크 서비스로 제공 시에도 소스 공개
- 수정 사항 공개

**Commercial License:**
- 소스 코드 공개 불필요
- 자유로운 상업적 사용
- 문의: amaury@reacher.email

---

## 관련 링크

- **GitHub**: [https://github.com/reacherhq/check-if-email-exists](https://github.com/reacherhq/check-if-email-exists)
- **Docs**: [https://docs.rs/check-if-email-exists](https://docs.rs/check-if-email-exists)
- **Live Demo**: [https://reacher.email](https://reacher.email)
- **Docker Hub**: [https://hub.docker.com/r/reacherhq/backend](https://hub.docker.com/r/reacherhq/backend)

---

*이메일 검증으로 더 나은 사용자 경험과 데이터 품질을 제공하세요!* ✉️
