---
layout: post
title: "check-if-email-exists 완벽 가이드 (03) - 검증 메커니즘"
date: 2026-02-08
categories: [개발 도구, 백엔드]
tags: [Email Validation, Rust, SMTP, API, Docker]
permalink: /check-if-email-exists-guide-03-verification/
excerpt: "이메일 검증의 4단계 메커니즘과 알고리즘 상세 분석"
original_url: "https://github.com/reacherhq/check-if-email-exists"
---

# check-if-email-exists 완벽 가이드 (03) - 검증 메커니즘

## 목차
1. [검증 파이프라인 개요](#검증-파이프라인-개요)
2. [1단계: Syntax 검증](#1단계-syntax-검증)
3. [2단계: DNS/MX 레코드 검증](#2단계-dnsmx-레코드-검증)
4. [3단계: SMTP 서버 검증](#3단계-smtp-서버-검증)
5. [4단계: 추가 검증 (Misc)](#4단계-추가-검증-misc)
6. [최종 판정: is_reachable 계산](#최종-판정-is_reachable-계산)
7. [특수 케이스 처리](#특수-케이스-처리)
8. [다음 챕터 예고](#다음-챕터-예고)

---

## 검증 파이프라인 개요

**check-if-email-exists**는 4단계 검증 파이프라인을 사용합니다. 각 단계는 독립적으로 실행되며, 결과를 종합하여 최종 판정을 내립니다.

### 전체 검증 흐름

```
입력: user@example.com
  │
  ├─> [1] Syntax 검증 (즉시)
  │   ├─ 정규식 매칭
  │   ├─ 도메인/사용자명 파싱
  │   └─ 일반적인 오타 감지
  │
  ├─> [2] DNS/MX 검증 (병렬)
  │   ├─ MX 레코드 조회
  │   ├─ 우선순위 정렬
  │   └─ A/AAAA 레코드 폴백
  │
  ├─> [3] SMTP 검증 (순차)
  │   ├─ 서버 연결 (포트 25)
  │   ├─ EHLO 핸드셰이크
  │   ├─ MAIL FROM 명령
  │   ├─ RCPT TO 명령
  │   └─ 응답 코드 분석
  │
  ├─> [4] 추가 검증 (병렬)
  │   ├─ Disposable Email 체크
  │   ├─ Role Account 체크
  │   ├─ Gravatar 조회
  │   └─ HaveIBeenPwned 조회
  │
  └─> [5] 최종 판정
      └─ is_reachable: safe/invalid/risky/unknown
```

### 검증 단계별 소요 시간

| 단계 | 평균 소요 시간 | 실패 시 |
|-----|--------------|---------|
| Syntax | < 1ms | 즉시 중단 |
| DNS/MX | 50-200ms | 다음 단계 진행 |
| SMTP | 1-10초 | unknown 반환 |
| Misc | 100-500ms | 선택적 |
| **전체** | **1-12초** | - |

---

## 1단계: Syntax 검증

가장 빠르고 기본적인 검증 단계입니다. 네트워크 요청 없이 로컬에서 즉시 수행됩니다.

### 검증 항목

#### 1.1 RFC 5322 형식 검증

이메일 주소는 다음 형식을 따라야 합니다:

```
local-part @ domain

local-part:
  - 길이: 1-64자
  - 허용 문자: a-z, A-Z, 0-9, . _ % + -
  - 점(.)은 시작/끝 불가, 연속 불가

domain:
  - 길이: 1-255자
  - 허용 문자: a-z, A-Z, 0-9, . -
  - 최소 1개의 점(.) 필요
  - 최상위 도메인 2자 이상
```

#### 1.2 정규식 패턴

**check-if-email-exists**는 다음과 유사한 정규식을 사용합니다:

```regex
^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$
```

#### 1.3 코드 예시 (Rust)

```rust
// core/src/syntax/mod.rs 구조

pub fn check_syntax(email: &str) -> SyntaxDetails {
    let email = email.trim().to_lowercase();

    // 1. @ 기호 존재 확인
    let parts: Vec<&str> = email.split('@').collect();
    if parts.len() != 2 {
        return SyntaxDetails {
            is_valid_syntax: false,
            address: email,
            domain: String::new(),
            username: String::new(),
            suggestion: None,
        };
    }

    let username = parts[0];
    let domain = parts[1];

    // 2. local-part 검증
    if username.is_empty() || username.len() > 64 {
        return invalid_syntax();
    }

    // 3. domain 검증
    if domain.is_empty() || domain.len() > 255 {
        return invalid_syntax();
    }

    // 4. 정규식 검증
    if !EMAIL_REGEX.is_match(&email) {
        return invalid_syntax();
    }

    // 5. 오타 제안
    let suggestion = get_similar_mail_provider(domain);

    SyntaxDetails {
        is_valid_syntax: true,
        address: email,
        domain: domain.to_string(),
        username: username.to_string(),
        normalized_email: normalize_email(&email),
        suggestion,
    }
}
```

### 오타 감지 및 제안

일반적인 오타를 감지하고 올바른 도메인을 제안합니다:

```rust
// 오타 → 제안
gmai.com       → gmail.com
gmial.com      → gmail.com
yahooo.com     → yahoo.com
hotmial.com    → hotmail.com
outlok.com     → outlook.com
```

**Levenshtein 거리 알고리즘** 사용:

```rust
use levenshtein::levenshtein;

pub fn get_similar_mail_provider(domain: &str) -> Option<String> {
    let common_providers = vec![
        "gmail.com",
        "yahoo.com",
        "hotmail.com",
        "outlook.com",
        "icloud.com",
    ];

    let mut best_match = None;
    let mut min_distance = 3; // 최대 거리 3

    for provider in common_providers {
        let distance = levenshtein(domain, provider);
        if distance < min_distance {
            min_distance = distance;
            best_match = Some(provider.to_string());
        }
    }

    best_match
}
```

### JSON 출력

```json
{
  "syntax": {
    "address": "user@gmail.com",
    "domain": "gmail.com",
    "is_valid_syntax": true,
    "username": "user",
    "normalized_email": "user@gmail.com",
    "suggestion": null
  }
}
```

**오타가 있는 경우:**

```json
{
  "syntax": {
    "address": "user@gmai.com",
    "domain": "gmai.com",
    "is_valid_syntax": true,
    "username": "user",
    "normalized_email": "user@gmai.com",
    "suggestion": "gmail.com"
  }
}
```

---

## 2단계: DNS/MX 레코드 검증

도메인이 실제로 이메일을 받을 수 있는지 DNS 레코드를 조회합니다.

### MX 레코드란?

**MX (Mail Exchange)** 레코드는 도메인의 이메일 서버를 지정하는 DNS 레코드입니다.

```
예시: gmail.com의 MX 레코드

gmail.com.  MX  5   gmail-smtp-in.l.google.com.
gmail.com.  MX  10  alt1.gmail-smtp-in.l.google.com.
gmail.com.  MX  20  alt2.gmail-smtp-in.l.google.com.
gmail.com.  MX  30  alt3.gmail-smtp-in.l.google.com.
gmail.com.  MX  40  alt4.gmail-smtp-in.l.google.com.

숫자 = 우선순위 (낮을수록 우선)
```

### 검증 과정

```
1. DNS 쿼리 발송
   ├─> MX 레코드 조회
   └─> 응답 대기 (타임아웃: 5초)

2. 결과 파싱
   ├─> 레코드 존재 → accepts_mail: true
   ├─> 레코드 없음 → A/AAAA 레코드 확인
   └─> 모두 없음 → accepts_mail: false

3. 우선순위 정렬
   └─> 가장 낮은 우선순위부터 사용
```

### 코드 예시 (Rust)

```rust
// core/src/mx/mod.rs 구조

use hickory_resolver::TokioAsyncResolver;
use hickory_resolver::config::*;

pub async fn check_mx(domain: &str) -> Result<MxDetails, MxError> {
    // 1. DNS Resolver 생성
    let resolver = TokioAsyncResolver::tokio(
        ResolverConfig::default(),
        ResolverOpts::default(),
    )?;

    // 2. MX 레코드 조회
    match resolver.mx_lookup(domain).await {
        Ok(lookup) => {
            let mut records: Vec<String> = lookup
                .iter()
                .map(|mx| mx.exchange().to_string())
                .collect();

            // 3. 우선순위 정렬
            records.sort_by_key(|mx| {
                lookup
                    .iter()
                    .find(|r| r.exchange().to_string() == *mx)
                    .map(|r| r.preference())
                    .unwrap_or(u16::MAX)
            });

            Ok(MxDetails {
                accepts_mail: true,
                records,
            })
        }
        Err(_) => {
            // 4. A/AAAA 레코드 폴백
            let has_a_record = resolver.lookup_ip(domain).await.is_ok();

            Ok(MxDetails {
                accepts_mail: has_a_record,
                records: vec![],
            })
        }
    }
}
```

### JSON 출력

**MX 레코드가 있는 경우:**

```json
{
  "mx": {
    "accepts_mail": true,
    "records": [
      "gmail-smtp-in.l.google.com.",
      "alt1.gmail-smtp-in.l.google.com.",
      "alt2.gmail-smtp-in.l.google.com.",
      "alt3.gmail-smtp-in.l.google.com.",
      "alt4.gmail-smtp-in.l.google.com."
    ]
  }
}
```

**MX 레코드가 없는 경우:**

```json
{
  "mx": {
    "accepts_mail": false,
    "records": []
  }
}
```

---

## 3단계: SMTP 서버 검증

가장 중요하고 복잡한 단계입니다. 실제 SMTP 서버와 통신하여 메일박스의 존재를 확인합니다.

### SMTP 프로토콜 이해

SMTP는 이메일 전송을 위한 텍스트 기반 프로토콜입니다:

```
클라이언트 → 서버: EHLO gmail.com
서버 → 클라이언트: 250 mx.google.com Hello

클라이언트 → 서버: MAIL FROM:<sender@example.com>
서버 → 클라이언트: 250 OK

클라이언트 → 서버: RCPT TO:<recipient@gmail.com>
서버 → 클라이언트: 250 OK (존재함)
                   550 User not found (존재하지 않음)
                   552 Mailbox full (메일함 가득 찼음)

클라이언트 → 서버: QUIT
서버 → 클라이언트: 221 Bye
```

### SMTP 응답 코드

| 코드 | 의미 | is_deliverable |
|-----|------|----------------|
| 250 | Accepted | ✅ true |
| 451 | Temporary failure | ❓ unknown |
| 452 | Insufficient storage | ⚠️ false (full_inbox) |
| 550 | User not found | ❌ false |
| 551 | User not local | ❌ false |
| 552 | Mailbox full | ⚠️ false (full_inbox) |
| 553 | Invalid mailbox | ❌ false |
| 554 | Rejected | ❌ false |

### 검증 흐름

```
1. 서버 연결
   ├─> TCP 소켓 열기 (포트 25)
   ├─> 타임아웃: 10초
   └─> 프록시 사용 (선택적)

2. EHLO 핸드셰이크
   ├─> EHLO example.com
   └─> 250 응답 대기

3. MAIL FROM 명령
   ├─> MAIL FROM:<sender@example.com>
   └─> 250 응답 대기

4. RCPT TO 명령 (핵심)
   ├─> RCPT TO:<target@gmail.com>
   └─> 응답 코드 분석:
       ├─ 250 → 존재함
       ├─ 550 → 존재하지 않음
       ├─ 552 → 메일함 가득 참
       └─ 기타 → unknown

5. QUIT 또는 RSET
   └─> 연결 정리
```

### 코드 예시 (Rust)

```rust
// core/src/smtp/connect.rs 구조

use async_smtp::{
    smtp::commands::*,
    smtp::response::Response,
    SmtpClient, SmtpTransport,
};

pub async fn check_smtp(
    to_email: &str,
    host: &str,
    port: u16,
    from_email: &str,
    hello_name: &str,
) -> Result<SmtpDetails, SmtpError> {
    // 1. SMTP 서버 연결
    let mut client = SmtpClient::new(host, port)
        .timeout(Duration::from_secs(10))
        .await?;

    // 2. EHLO 핸드셰이크
    let ehlo_response = client.command(EhloCommand::new(hello_name)).await?;
    if !ehlo_response.is_positive() {
        return Err(SmtpError::EhloFailed);
    }

    // 3. MAIL FROM 명령
    let mail_response = client
        .command(MailCommand::new(from_email.into()))
        .await?;
    if !mail_response.is_positive() {
        return Err(SmtpError::MailFromFailed);
    }

    // 4. RCPT TO 명령 (핵심)
    let rcpt_response = client
        .command(RcptCommand::new(to_email.into()))
        .await?;

    // 5. 응답 분석
    let (is_deliverable, is_disabled, has_full_inbox) =
        parse_rcpt_response(&rcpt_response);

    // 6. QUIT
    client.command(QuitCommand).await.ok();

    Ok(SmtpDetails {
        can_connect_smtp: true,
        has_full_inbox,
        is_catch_all: false, // 별도 체크 필요
        is_deliverable,
        is_disabled,
    })
}

fn parse_rcpt_response(response: &Response) -> (bool, bool, bool) {
    let code = response.code();
    let message = response.message().join(" ").to_lowercase();

    match code {
        250 => (true, false, false),  // 성공
        550 | 551 | 553 => {
            let is_disabled = message.contains("disabled")
                || message.contains("inactive");
            (false, is_disabled, false)
        }
        452 | 552 => (false, false, true),  // 메일함 가득 참
        _ => (false, false, false),  // 기타 실패
    }
}
```

### Catch-all 주소 감지

Catch-all 도메인은 존재하지 않는 주소도 250 응답을 반환합니다:

```rust
pub async fn is_catch_all(
    domain: &str,
    mx_host: &str,
) -> bool {
    // 무작위 이메일 주소 생성
    let random_email = format!(
        "{}@{}",
        generate_random_string(32),
        domain
    );

    // RCPT TO 테스트
    match check_smtp(&random_email, mx_host, 25, ...).await {
        Ok(result) if result.is_deliverable => true,  // Catch-all
        _ => false,
    }
}
```

### JSON 출력

```json
{
  "smtp": {
    "can_connect_smtp": true,
    "has_full_inbox": false,
    "is_catch_all": false,
    "is_deliverable": true,
    "is_disabled": false,
    "verif_method": {
      "type": "Smtp",
      "host": "gmail-smtp-in.l.google.com.",
      "port": 25,
      "used_proxy": false
    }
  }
}
```

---

## 4단계: 추가 검증 (Misc)

이메일 주소에 대한 보조 정보를 수집합니다.

### 4.1 Disposable Email 감지

일회용 이메일 제공자를 감지합니다:

```rust
// mailchecker 라이브러리 사용

use mailchecker::is_valid;

pub fn is_disposable_email(domain: &str) -> bool {
    !is_valid(domain)  // mailchecker는 유효한 도메인만 true 반환
}

// 알려진 일회용 이메일 제공자:
// - tempmail.com
// - guerrillamail.com
// - 10minutemail.com
// - mailinator.com
// - ...수천 개의 도메인
```

### 4.2 Role Account 감지

역할 계정(일반적인 비즈니스 계정)을 감지합니다:

```rust
const ROLE_ACCOUNTS: &[&str] = &[
    "abuse", "admin", "administrator", "billing",
    "contact", "help", "hostmaster", "info",
    "noreply", "no-reply", "postmaster", "sales",
    "security", "support", "webmaster",
];

pub fn is_role_account(username: &str) -> bool {
    ROLE_ACCOUNTS.contains(&username.to_lowercase().as_str())
}
```

**예시:**

```
admin@example.com       → true
support@company.com     → true
john.doe@company.com    → false
```

### 4.3 Gravatar 조회

Gravatar는 이메일 주소와 연결된 프로필 이미지 서비스입니다:

```rust
use md5::{Md5, Digest};

pub async fn get_gravatar_url(email: &str) -> Option<String> {
    // 1. 이메일을 소문자로 변환
    let email = email.trim().to_lowercase();

    // 2. MD5 해시 생성
    let mut hasher = Md5::new();
    hasher.update(email.as_bytes());
    let hash = format!("{:x}", hasher.finalize());

    // 3. Gravatar URL 생성
    let url = format!("https://gravatar.com/avatar/{}", hash);

    // 4. URL 존재 확인
    let response = reqwest::get(&url).await.ok()?;
    if response.status().is_success() {
        Some(url)
    } else {
        None
    }
}
```

### 4.4 HaveIBeenPwned 조회

이메일 주소가 데이터 유출에 포함되었는지 확인합니다:

```rust
pub async fn check_haveibeenpwned(
    email: &str,
    api_key: &str,
) -> Option<Vec<String>> {
    let url = format!(
        "https://haveibeenpwned.com/api/v3/breachedaccount/{}",
        email
    );

    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header("hibp-api-key", api_key)
        .send()
        .await
        .ok()?;

    if response.status().is_success() {
        let breaches: Vec<Breach> = response.json().await.ok()?;
        Some(breaches.iter().map(|b| b.name.clone()).collect())
    } else {
        None
    }
}
```

### JSON 출력

```json
{
  "misc": {
    "is_disposable": false,
    "is_role_account": false,
    "gravatar_url": "https://gravatar.com/avatar/abc123...",
    "haveibeenpwned": ["Adobe", "LinkedIn", "Dropbox"]
  }
}
```

---

## 최종 판정: is_reachable 계산

모든 검증 결과를 종합하여 최종 신뢰도를 계산합니다.

### 판정 알고리즘

```rust
pub fn calculate_reachable(
    syntax: &SyntaxDetails,
    mx: &MxDetails,
    smtp: &SmtpDetails,
    misc: &MiscDetails,
) -> Reachable {
    // 1. Syntax 실패 → invalid
    if !syntax.is_valid_syntax {
        return Reachable::Invalid;
    }

    // 2. MX 레코드 없음 → invalid
    if !mx.accepts_mail {
        return Reachable::Invalid;
    }

    // 3. SMTP 연결 실패 → unknown
    if !smtp.can_connect_smtp {
        return Reachable::Unknown;
    }

    // 4. SMTP 검증 결과
    if !smtp.is_deliverable {
        return Reachable::Invalid;
    }

    // 5. 위험 요소 체크 → risky
    if misc.is_disposable
        || misc.is_role_account
        || smtp.is_catch_all
        || smtp.has_full_inbox
    {
        return Reachable::Risky;
    }

    // 6. 모든 검증 통과 → safe
    Reachable::Safe
}
```

### 결정 트리

```
Syntax 유효?
├─ No → invalid
└─ Yes
    │
    MX 레코드 존재?
    ├─ No → invalid
    └─ Yes
        │
        SMTP 연결 가능?
        ├─ No → unknown
        └─ Yes
            │
            SMTP Deliverable?
            ├─ No → invalid
            └─ Yes
                │
                위험 요소 존재?
                ├─ Yes (Disposable, Role, Catch-all, Full) → risky
                └─ No → safe
```

### 각 판정의 의미

| 판정 | 의미 | 바운스율 예상 | 권장 조치 |
|-----|------|--------------|----------|
| **safe** | 안전하게 발송 가능 | < 2% | 즉시 발송 |
| **invalid** | 발송 불가능 | ~100% | 발송 금지 |
| **risky** | 위험 요소 존재 | 2-20% | 주의해서 발송 |
| **unknown** | 판단 불가 | 알 수 없음 | 재시도 또는 보류 |

---

## 특수 케이스 처리

### Gmail 검증

Gmail은 SMTP 검증을 제한하므로 특별한 처리가 필요합니다:

```rust
// Gmail API 사용 (선택적)
pub enum GmailVerifMethod {
    Smtp,  // 기본 SMTP (제한적)
    Api,   // Gmail API (더 정확함)
}

// API 방식
async fn verify_gmail_api(email: &str) -> bool {
    // Gmail API를 사용하여 계정 존재 확인
    // 구현은 생략
}
```

### Yahoo 검증

Yahoo도 유사한 제한이 있습니다:

```rust
pub enum YahooVerifMethod {
    Smtp,      // 기본 SMTP
    Api,       // Yahoo API
    Headless,  // Selenium/Puppeteer (가장 정확함)
}
```

### Hotmail/Outlook 검증

```rust
pub enum HotmailVerifMethod {
    Smtp,      // 기본 SMTP
    Headless,  // Headless 브라우저
}

pub enum HotmailB2bVerifMethod {
    Smtp,  // B2B는 SMTP만 지원
}
```

---

## 다음 챕터 예고

### 챕터 04: HTTP 백엔드

다음 챕터에서는 Docker 기반 HTTP 백엔드를 상세히 다룹니다:

1. Docker 이미지 상세 설명
2. API 엔드포인트 완전 가이드 (/v0/check_email, /v1/check_email)
3. 요청/응답 형식 및 에러 처리
4. 프록시 설정 및 관리
5. 인증 및 보안 설정
6. 환경변수 및 설정 파일

---

## 결론

이 챕터에서는 **check-if-email-exists**의 핵심 검증 메커니즘을 깊이 있게 살펴보았습니다.

### 핵심 요약

**4단계 검증 파이프라인:**

1. **Syntax**: 정규식, 형식 체크, 오타 감지
2. **DNS/MX**: 도메인 이메일 서버 확인
3. **SMTP**: 실제 메일박스 존재 확인 (핵심)
4. **Misc**: 보조 정보 수집

**최종 판정:**

- `safe`: 모든 검증 통과
- `invalid`: 주요 검증 실패
- `risky`: 위험 요소 존재
- `unknown`: 판단 불가

### 검증 정확도 향상 팁

1. **프록시 사용**: 포트 25 차단 우회
2. **타임아웃 조정**: 느린 서버 대응
3. **재시도 로직**: 일시적 실패 처리
4. **특수 도메인 처리**: Gmail, Yahoo 등

### 참고 자료

- Core 소스 코드: https://github.com/reacherhq/check-if-email-exists/tree/main/core
- SMTP RFC: https://tools.ietf.org/html/rfc5321
- DNS RFC: https://tools.ietf.org/html/rfc1035

다음 챕터에서는 이 검증 엔진을 HTTP API로 노출하는 백엔드 시스템을 알아보겠습니다.
