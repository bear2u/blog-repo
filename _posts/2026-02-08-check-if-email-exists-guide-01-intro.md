---
layout: post
title: "check-if-email-exists 완벽 가이드 (01) - 소개 및 개요"
date: 2026-02-08
categories: [개발 도구, 백엔드]
tags: [Email Validation, Rust, SMTP, API, Docker]
permalink: /check-if-email-exists-guide-01-intro/
excerpt: "이메일을 보내지 않고 주소 유효성을 검증하는 오픈소스 도구"
original_url: "https://github.com/reacherhq/check-if-email-exists"
---

# check-if-email-exists 완벽 가이드 (01) - 소개 및 개요

## 목차
1. [프로젝트 소개](#프로젝트-소개)
2. [이메일 검증이 필요한 이유](#이메일-검증이-필요한-이유)
3. [주요 특징](#주요-특징)
4. [검증 항목 14가지](#검증-항목-14가지)
5. [JSON 출력 예시](#json-출력-예시)
6. [Live Demo 및 SaaS](#live-demo-및-saas)
7. [다음 챕터 예고](#다음-챕터-예고)

---

## 프로젝트 소개

**check-if-email-exists**는 이메일을 실제로 보내지 않고도 이메일 주소의 유효성을 검증할 수 있는 Rust 기반 오픈소스 라이브러리입니다. AGPL-3.0 라이선스로 공개되어 있으며, 상용 라이선스도 제공됩니다.

### 프로젝트 개요

```
┌─────────────────────────────────────────────────────┐
│         check-if-email-exists 아키텍처               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │  Rust 라이브러리  │  │   CLI 도구    │  │  HTTP    │ │
│  │  (Core)      │  │  (Binary)    │  │  Backend │ │
│  └──────────────┘  └──────────────┘  └──────────┘ │
│         │                  │                │      │
│         └──────────────────┴────────────────┘      │
│                       │                            │
│              ┌────────▼────────┐                   │
│              │  Email 검증 엔진  │                   │
│              ├─────────────────┤                   │
│              │ • Syntax Check  │                   │
│              │ • DNS/MX Check  │                   │
│              │ • SMTP Check    │                   │
│              │ • Misc Checks   │                   │
│              └─────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

### 제공 형태

1. **Rust 라이브러리**: Cargo를 통해 직접 코드에 통합
2. **CLI 도구**: 터미널에서 즉시 사용 가능한 실행 파일
3. **HTTP 백엔드**: Docker를 통한 REST API 서버

### 프로젝트 정보

| 항목 | 내용 |
|-----|------|
| **언어** | Rust (2018 Edition) |
| **버전** | v0.11.7 (2026년 2월 기준) |
| **라이선스** | AGPL-3.0 / Commercial License |
| **저장소** | https://github.com/reacherhq/check-if-email-exists |
| **문서** | https://docs.rs/check-if-email-exists |
| **Docker Hub** | reacherhq/backend |

---

## 이메일 검증이 필요한 이유

### 1. 이메일 마케팅의 문제

이메일 마케팅을 진행할 때 가장 큰 문제는 **바운스율(Bounce Rate)**입니다. 존재하지 않는 이메일 주소로 발송하면:

- 발신자 평판(Sender Reputation)이 하락합니다
- 스팸으로 분류될 확률이 높아집니다
- 이메일 서비스 제공자(ESP)로부터 제재를 받을 수 있습니다
- 불필요한 발송 비용이 발생합니다

### 2. 기존 솔루션의 한계

많은 유료 서비스들이 이메일 검증을 제공하지만:

```
Hunter.io        →  유료 (크레딧 기반)
Verify-email.org →  유료 (구독 기반)
Email-checker.net →  제한적 무료 + 유료
```

**check-if-email-exists**는 이러한 서비스의 오픈소스 대안입니다.

### 3. 실시간 검증의 중요성

회원가입 폼에서 실시간으로 이메일을 검증하면:

- 오타로 인한 가입 실패를 방지
- 일회용 이메일 차단
- 스팸 계정 생성 방지
- 사용자 경험 개선

---

## 주요 특징

### 1. 이메일 전송 없이 검증

기존 방식은 실제로 이메일을 보내고 반응을 확인했지만, **check-if-email-exists**는:

```
기존 방식:
User → Email 발송 → SMTP 서버 → 바운스 확인 (시간 소요)

check-if-email-exists:
User → SMTP 핸드셰이크 → 즉시 결과 반환 (빠름)
```

### 2. 포괄적인 검증

단순 형식 검증을 넘어 다음을 확인합니다:

- DNS MX 레코드 존재 여부
- SMTP 서버 연결 가능 여부
- 실제 메일박스 존재 여부
- 일회용 이메일 여부
- 역할 계정(role account) 여부

### 3. 높은 정확도

검증 결과를 4단계로 분류:

| 결과 | 의미 | 바운스율 |
|-----|------|---------|
| `safe` | 안전하게 발송 가능 | < 2% |
| `invalid` | 발송 불가능 | ~100% |
| `risky` | 위험 요소 존재 | 2-20% |
| `unknown` | 판단 불가 | 알 수 없음 |

### 4. 프로덕션 준비 완료

- Docker 지원으로 쉬운 배포
- RabbitMQ 통합으로 대량 처리 가능
- PostgreSQL 연동으로 결과 저장
- Kubernetes 배포 가능
- 프록시 지원으로 IP 평판 관리

---

## 검증 항목 14가지

**check-if-email-exists**는 총 14가지 항목을 검증합니다:

### ✅ 구현 완료 (12가지)

| # | 기능 | 설명 | JSON 필드 |
|---|------|------|-----------|
| 1 | **Email Reachability** | 전체 신뢰도 점수 | `is_reachable` |
| 2 | **Syntax Validation** | 이메일 형식 검증 | `syntax.is_valid_syntax` |
| 3 | **DNS Records** | MX 레코드 존재 확인 | `mx.accepts_mail` |
| 4 | **Disposable Email** | 일회용 이메일 감지 | `misc.is_disposable` |
| 5 | **SMTP Server** | SMTP 서버 연결 확인 | `smtp.can_connect_smtp` |
| 6 | **Email Deliverability** | 실제 전송 가능 여부 | `smtp.is_deliverable` |
| 7 | **Mailbox Disabled** | 비활성화된 계정 감지 | `smtp.is_disabled` |
| 8 | **Full Inbox** | 메일함 가득 참 확인 | `smtp.has_full_inbox` |
| 9 | **Catch-all Address** | Catch-all 주소 감지 | `smtp.is_catch_all` |
| 10 | **Role Account** | 역할 계정 확인 | `misc.is_role_account` |
| 11 | **Gravatar** | Gravatar 프로필 확인 | `misc.gravatar_url` |
| 12 | **Have I Been Pwned** | 데이터 유출 확인 | `misc.haveibeenpwned` |

### 🔜 계획 중 (2가지)

| # | 기능 | 설명 | 이슈 |
|---|------|------|-----|
| 13 | **Free Email Provider** | 무료 이메일 제공자 확인 | [#89](https://github.com/reacherhq/check-if-email-exists/issues/89) |
| 14 | **Honeypot Detection** | 스팸트랩 감지 | [#91](https://github.com/reacherhq/check-if-email-exists/issues/91) |

---

## JSON 출력 예시

### 예시 1: 비활성화된 Gmail 주소

`someone@gmail.com`을 검증한 결과:

```json
{
  "input": "someone@gmail.com",
  "is_reachable": "invalid",
  "misc": {
    "is_disposable": false,
    "is_role_account": false,
    "is_b2c": true
  },
  "mx": {
    "accepts_mail": true,
    "records": [
      "alt3.gmail-smtp-in.l.google.com.",
      "gmail-smtp-in.l.google.com.",
      "alt1.gmail-smtp-in.l.google.com.",
      "alt4.gmail-smtp-in.l.google.com.",
      "alt2.gmail-smtp-in.l.google.com."
    ]
  },
  "smtp": {
    "can_connect_smtp": true,
    "has_full_inbox": false,
    "is_catch_all": false,
    "is_deliverable": false,
    "is_disabled": true
  },
  "syntax": {
    "domain": "gmail.com",
    "is_valid_syntax": true,
    "username": "someone",
    "suggestion": null
  }
}
```

### 출력 필드 해석

```
is_reachable: "invalid"
└─> SMTP 검증 결과:
    ├─ can_connect_smtp: true (연결 성공)
    ├─ is_deliverable: false (전송 불가)
    └─ is_disabled: true (계정 비활성화)
```

### 예시 2: 유효한 이메일 주소

`amaury@reacher.email`을 검증한 결과:

```json
{
  "input": "amaury@reacher.email",
  "is_reachable": "safe",
  "misc": {
    "is_disposable": false,
    "is_role_account": false,
    "gravatar_url": "https://gravatar.com/avatar/..."
  },
  "mx": {
    "accepts_mail": true,
    "records": [
      "mx1.reacher.email.",
      "mx2.reacher.email."
    ]
  },
  "smtp": {
    "can_connect_smtp": true,
    "has_full_inbox": false,
    "is_catch_all": false,
    "is_deliverable": true,
    "is_disabled": false
  },
  "syntax": {
    "domain": "reacher.email",
    "is_valid_syntax": true,
    "username": "amaury",
    "normalized_email": "amaury@reacher.email",
    "suggestion": null
  }
}
```

### 검증 흐름

```
입력: someone@gmail.com
  │
  ├─> [1] Syntax 검증 ✅
  │   └─ 형식: username@domain 올바름
  │
  ├─> [2] DNS/MX 검증 ✅
  │   └─ MX 레코드 5개 발견
  │
  ├─> [3] SMTP 검증 ✅
  │   ├─ 연결: 성공
  │   ├─ EHLO: gmail.com
  │   ├─ MAIL FROM: reacher.email@gmail.com
  │   └─ RCPT TO: someone@gmail.com → 550 Disabled
  │
  └─> [4] 최종 판정: invalid
```

---

## Live Demo 및 SaaS

### Live Demo

공식 웹사이트에서 즉시 테스트할 수 있습니다:

**https://reacher.email**

### SaaS 버전

셀프 호스팅이 부담스러운 경우, 공식 SaaS를 이용할 수 있습니다:

| 기능 | 오픈소스 | SaaS (Reacher) |
|-----|---------|---------------|
| 이메일 검증 | ✅ | ✅ |
| 웹 대시보드 | ❌ | ✅ |
| API 토큰 관리 | ❌ | ✅ |
| 결과 히스토리 | ❌ | ✅ |
| 대량 검증 | 수동 설정 필요 | ✅ |
| 기술 지원 | 커뮤니티 | 전문 지원 |
| 프록시 관리 | 직접 관리 | 자동 관리 |

### 가격 정책

- **무료 티어**: 월 100회 검증
- **프로 플랜**: 종량제 (검증당 과금)
- **엔터프라이즈**: 맞춤형 계약

자세한 정보: https://reacher.email/pricing

---

## 다음 챕터 예고

### 챕터 02: 빠른 시작

다음 챕터에서는 실제로 **check-if-email-exists**를 설치하고 사용하는 방법을 다룹니다:

1. Docker를 통한 HTTP 백엔드 실행
2. CLI 바이너리 다운로드 및 사용
3. Rust 프로젝트에 라이브러리 통합
4. 첫 이메일 검증 실습

### 전체 가이드 시리즈

1. **소개 및 개요** (현재)
2. 빠른 시작
3. 검증 메커니즘
4. HTTP 백엔드
5. 고급 활용
6. 개발 및 기여

---

## 결론

**check-if-email-exists**는 이메일 검증을 위한 강력하고 유연한 오픈소스 솔루션입니다. Rust의 성능과 안정성을 바탕으로, 실제 이메일 발송 없이 높은 정확도로 이메일 주소를 검증할 수 있습니다.

### 핵심 요약

- 이메일 전송 없이 SMTP 검증
- 12가지 검증 항목 지원
- 4단계 신뢰도 분류 (safe/invalid/risky/unknown)
- Rust 라이브러리, CLI, HTTP API 제공
- Docker와 Kubernetes 배포 지원
- AGPL-3.0 오픈소스 라이선스

### 참고 자료

- 공식 저장소: https://github.com/reacherhq/check-if-email-exists
- API 문서: https://docs.rs/check-if-email-exists
- 공식 웹사이트: https://reacher.email
- Docker Hub: https://hub.docker.com/r/reacherhq/backend

다음 챕터에서는 실제로 도구를 설치하고 첫 이메일 검증을 수행해 보겠습니다.
