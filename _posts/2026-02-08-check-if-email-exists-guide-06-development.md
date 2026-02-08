---
layout: post
title: "check-if-email-exists 완벽 가이드 (06) - 개발 및 기여"
date: 2026-02-08
categories: [개발 도구, 백엔드]
tags: [Email Validation, Rust, SMTP, API, Docker]
permalink: /check-if-email-exists-guide-06-development/
excerpt: "Rust 개발 환경, 프로젝트 구조, 기여 가이드 완벽 정리"
original_url: "https://github.com/reacherhq/check-if-email-exists"
---

# check-if-email-exists 완벽 가이드 (06) - 개발 및 기여

## 목차
1. [개발 환경 설정](#개발-환경-설정)
2. [프로젝트 구조](#프로젝트-구조)
3. [빌드 및 실행](#빌드-및-실행)
4. [테스트](#테스트)
5. [기여 가이드](#기여-가이드)
6. [라이선스 이해](#라이선스-이해)
7. [시리즈 총정리](#시리즈-총정리)

---

## 개발 환경 설정

**check-if-email-exists**는 Rust로 작성되어 있습니다. 개발을 시작하려면 Rust 툴체인이 필요합니다.

### Rust 설치

#### Linux / macOS

```bash
# Rustup 설치 (공식 방법)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 환경변수 로드
source $HOME/.cargo/env

# 설치 확인
rustc --version
cargo --version
```

**출력 예시:**

```
rustc 1.75.0 (82e1608df 2023-12-21)
cargo 1.75.0 (1d8b05cdd 2023-11-20)
```

#### Windows

1. https://rustup.rs/ 에서 `rustup-init.exe` 다운로드
2. 실행하여 설치
3. Visual Studio C++ Build Tools 설치 (필요 시)

### 필수 의존성

#### Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    libsqlite3-dev
```

#### macOS

```bash
brew install openssl sqlite
```

### 추가 도구

```bash
# Rust 포맷터 (코드 스타일 통일)
rustup component add rustfmt

# Clippy (린터)
rustup component add clippy

# cargo-watch (자동 재빌드)
cargo install cargo-watch
```

---

## 프로젝트 구조

**check-if-email-exists**는 Cargo Workspace로 구성된 모노레포입니다.

### Workspace 구조

```
check-if-email-exists/
├── Cargo.toml              # Workspace 루트 설정
├── Cargo.lock              # 의존성 잠금 파일
├── Makefile                # 빌드/테스트 스크립트
├── README.md               # 프로젝트 소개
├── LICENSE.md              # 라이선스 정보
├── LICENSE.AGPL            # AGPL-3.0 전문
│
├── core/                   # 핵심 검증 로직
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs          # 라이브러리 엔트리
│       ├── syntax/         # Syntax 검증
│       ├── mx/             # DNS/MX 검증
│       ├── smtp/           # SMTP 검증
│       ├── misc/           # 추가 검증
│       └── util/           # 유틸리티
│
├── cli/                    # CLI 바이너리
│   ├── Cargo.toml
│   └── src/
│       └── main.rs         # CLI 엔트리
│
├── backend/                # HTTP 백엔드
│   ├── Cargo.toml
│   ├── backend_config.toml # 설정 파일
│   ├── openapi.json        # API 스펙
│   └── src/
│       ├── main.rs         # 서버 엔트리
│       ├── routes/         # HTTP 라우트
│       ├── worker/         # RabbitMQ Worker
│       └── storage/        # PostgreSQL 연동
│
├── sqs/                    # AWS SQS 통합 (베타)
│   ├── Cargo.toml
│   └── src/
│       └── main.rs
│
├── docs/                   # 문서
│   ├── README.md
│   ├── getting-started/
│   ├── self-hosting/
│   └── advanced/
│
└── .github/                # CI/CD
    └── workflows/
        └── pr.yml          # PR 자동 빌드/테스트
```

### 핵심 모듈 상세

#### 1. core 모듈

검증 로직의 핵심입니다:

```
core/src/
├── lib.rs                  # 공개 API 정의
│   └── pub fn check_email()
│
├── syntax/
│   └── mod.rs              # 이메일 형식 검증
│       ├── check_syntax()
│       └── get_similar_mail_provider()
│
├── mx/
│   └── mod.rs              # DNS MX 레코드 조회
│       └── check_mx()
│
├── smtp/
│   ├── mod.rs              # SMTP 검증 메인
│   ├── connect.rs          # SMTP 연결 (400줄)
│   ├── parser.rs           # 응답 파싱 (300줄)
│   ├── verif_method.rs     # 검증 방법 (530줄)
│   ├── gmail/              # Gmail 특화 검증
│   ├── yahoo/              # Yahoo 특화 검증
│   └── outlook/            # Outlook 특화 검증
│
├── misc/
│   ├── mod.rs              # 추가 검증
│   ├── roles.txt           # Role account 목록
│   └── b2c.txt             # 무료 이메일 제공자 목록
│
└── util/
    ├── input_output.rs     # 입출력 구조체 (350줄)
    └── sentry.rs           # 오류 추적
```

#### 2. backend 모듈

HTTP 서버 및 Worker:

```
backend/src/
├── main.rs                 # 서버 시작
│   └── async fn main()
│
├── routes/
│   ├── check_email/
│   │   ├── v0.rs           # /v0/check_email
│   │   └── v1.rs           # /v1/check_email
│   ├── bulk/
│   │   └── v1.rs           # /v1/bulk
│   └── health.rs           # /health
│
├── worker/
│   ├── mod.rs              # Worker 메인
│   ├── single_shot.rs      # 단일 검증
│   └── bulk.rs             # 대량 검증
│
└── storage/
    ├── mod.rs
    └── postgres/
        ├── mod.rs
        └── store.rs        # DB 저장 로직
```

### 주요 파일 크기 (라인 수)

| 파일 | 라인 수 | 역할 |
|-----|--------|------|
| `core/src/smtp/verif_method.rs` | 531 | 검증 방법 관리 |
| `core/src/smtp/connect.rs` | 396 | SMTP 연결 |
| `core/src/util/input_output.rs` | 353 | 입출력 타입 |
| `core/src/smtp/parser.rs` | 291 | SMTP 응답 파싱 |
| `core/src/lib.rs` | 281 | 라이브러리 엔트리 |
| `core/src/smtp/mod.rs` | 234 | SMTP 검증 메인 |
| `core/src/syntax/mod.rs` | 199 | Syntax 검증 |

---

## 빌드 및 실행

### 전체 프로젝트 빌드

```bash
# 레포지토리 클론
git clone https://github.com/reacherhq/check-if-email-exists.git
cd check-if-email-exists

# 전체 빌드 (debug 모드)
cargo build

# 전체 빌드 (release 모드)
cargo build --release
```

**빌드 시간:**

- Debug 모드: ~5분 (첫 빌드), ~30초 (증분 빌드)
- Release 모드: ~10분 (첫 빌드), ~2분 (증분 빌드)

### 개별 모듈 빌드

```bash
# Core 라이브러리만 빌드
cargo build -p check-if-email-exists

# CLI만 빌드
cargo build -p cli --bin check_if_email_exists

# Backend만 빌드
cargo build -p backend --bin reacher_backend
```

### 실행

#### CLI 실행

```bash
# Debug 빌드로 실행
cargo run -p cli -- test@gmail.com

# Release 빌드로 실행
cargo run -p cli --release -- test@gmail.com

# 또는 빌드된 바이너리 직접 실행
./target/release/check_if_email_exists test@gmail.com
```

#### Backend 실행

```bash
# 기본 실행
make run

# 또는 직접 실행
cd backend && cargo run --bin reacher_backend

# Worker 모드로 실행 (RabbitMQ + PostgreSQL 필요)
make run-with-worker
```

### 자동 재빌드 (개발 중)

```bash
# 파일 변경 시 자동 재빌드
cargo watch -x run

# 특정 패키지만 감시
cargo watch -p backend -x run
```

---

## 테스트

### 단위 테스트

```bash
# 전체 테스트 실행
cargo test

# 특정 모듈만 테스트
cargo test -p check-if-email-exists

# 테스트 이름 필터링
cargo test smtp

# 출력 표시 (println! 등)
cargo test -- --nocapture

# 병렬 실행 비활성화
cargo test -- --test-threads=1
```

### 통합 테스트

```bash
# Backend 통합 테스트
cargo test -p backend --test integration_tests
```

### 테스트 작성 예시

```rust
// core/src/syntax/mod.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_email() {
        let result = check_syntax("user@example.com");
        assert!(result.is_valid_syntax);
        assert_eq!(result.domain, "example.com");
        assert_eq!(result.username, "user");
    }

    #[test]
    fn test_invalid_email_no_at() {
        let result = check_syntax("userexample.com");
        assert!(!result.is_valid_syntax);
    }

    #[test]
    fn test_typo_suggestion() {
        let result = check_syntax("user@gmai.com");
        assert_eq!(result.suggestion, Some("gmail.com".to_string()));
    }
}
```

### 커버리지 측정

```bash
# tarpaulin 설치
cargo install cargo-tarpaulin

# 커버리지 실행
cargo tarpaulin --out Html
```

---

## 기여 가이드

**check-if-email-exists**는 오픈소스 프로젝트이며 기여를 환영합니다.

### 기여 프로세스

#### 1. Fork 및 Clone

```bash
# 1. GitHub에서 Fork 클릭
# 2. 자신의 레포지토리 클론
git clone https://github.com/YOUR_USERNAME/check-if-email-exists.git
cd check-if-email-exists

# 3. 원본 레포지토리를 upstream으로 추가
git remote add upstream https://github.com/reacherhq/check-if-email-exists.git
```

#### 2. 브랜치 생성

```bash
# 최신 코드로 업데이트
git checkout main
git pull upstream main

# Feature 브랜치 생성
git checkout -b feature/my-awesome-feature
```

**브랜치 네이밍 규칙:**

```
feature/   - 새 기능
fix/       - 버그 수정
docs/      - 문서 변경
refactor/  - 리팩토링
test/      - 테스트 추가
```

#### 3. 코드 작성

```bash
# 코드 변경
vim core/src/smtp/mod.rs

# 포맷팅 (필수)
cargo fmt

# Clippy 검사 (필수)
cargo clippy

# 테스트 실행 (필수)
cargo test
```

#### 4. 커밋

**Conventional Commits** 형식을 따릅니다:

```bash
# 형식
git commit -m "type(scope): subject"

# 예시
git commit -m "feat(smtp): add timeout configuration"
git commit -m "fix(core): handle empty MX records"
git commit -m "docs: update README with new examples"
git commit -m "test(cli): add integration tests"
```

**커밋 타입:**

- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 스타일 (포맷팅, 세미콜론 등)
- `refactor`: 리팩토링
- `perf`: 성능 개선
- `test`: 테스트 추가/수정
- `chore`: 빌드 프로세스, 도구 변경

#### 5. Push 및 PR 생성

```bash
# Push
git push origin feature/my-awesome-feature

# GitHub에서 Pull Request 생성
# - 제목: feat(smtp): add timeout configuration
# - 설명: 변경 사항 상세 설명
```

### PR 체크리스트

PR을 생성하기 전에 다음을 확인하세요:

- [ ] `cargo fmt` 실행됨
- [ ] `cargo clippy` 경고 없음
- [ ] `cargo test` 모두 통과
- [ ] 새로운 기능에 테스트 추가
- [ ] 문서 업데이트 (README, docs/)
- [ ] Conventional Commits 형식 준수
- [ ] CHANGELOG.md 업데이트 (중요 변경 시)

### 코드 스타일

#### Rust 스타일 가이드

```rust
// 좋은 예시
pub async fn check_email(input: &CheckEmailInput) -> CheckEmailOutput {
    // 로직
}

// 나쁜 예시 (타입 힌트 없음)
pub async fn check_email(input) {
    // 로직
}

// 좋은 예시 (에러 처리)
match smtp_result {
    Ok(result) => result,
    Err(e) => {
        tracing::error!("SMTP verification failed: {}", e);
        return default_smtp_details();
    }
}

// 나쁜 예시 (unwrap 사용)
let result = smtp_result.unwrap();
```

#### 네이밍 규칙

```rust
// 함수: snake_case
fn check_syntax() {}

// 구조체: PascalCase
struct CheckEmailInput {}

// 상수: SCREAMING_SNAKE_CASE
const MAX_TIMEOUT: u64 = 60;

// 변수: snake_case
let is_valid_syntax = true;
```

### 이슈 보고

버그를 발견했다면 이슈를 생성하세요:

**이슈 템플릿:**

```markdown
**버그 설명**
간단한 설명

**재현 방법**
1. '...'로 이동
2. '....' 클릭
3. '....' 스크롤
4. 에러 확인

**예상 동작**
무엇이 일어날 것으로 예상했는지

**실제 동작**
실제로 무엇이 일어났는지

**환경**
- OS: [e.g. Ubuntu 22.04]
- Rust 버전: [e.g. 1.75.0]
- 프로젝트 버전: [e.g. 0.11.7]

**추가 컨텍스트**
스크린샷, 로그 등
```

---

## 라이선스 이해

**check-if-email-exists**는 **듀얼 라이선스** 모델을 사용합니다.

### AGPL-3.0 (오픈소스)

#### 허용

- ✅ 상업적 사용
- ✅ 수정
- ✅ 배포
- ✅ 특허 사용

#### 조건

- 📋 라이선스 및 저작권 고지 포함
- 📋 상태 변경 명시
- 📋 소스 코드 공개 (네트워크 서비스도 포함)
- 📋 동일 라이선스로 배포

#### AGPL-3.0의 특징

AGPL-3.0은 GPL-3.0의 네트워크 버전입니다:

```
GPL-3.0:
프로그램을 배포하면 소스 공개 필요

AGPL-3.0:
프로그램을 네트워크 서비스로 제공해도 소스 공개 필요
```

**예시:**

```
시나리오 1: 오픈소스 프로젝트
✅ check-if-email-exists를 사용하여 오픈소스 프로젝트 개발
✅ 코드를 GitHub에 AGPL-3.0로 공개
✅ 라이선스 조건 충족

시나리오 2: 내부 도구
❌ check-if-email-exists를 사용하여 내부 이메일 검증 시스템 구축
❌ 외부 사용자도 접근 가능 (네트워크 서비스)
❌ 소스 코드를 공개하지 않음
❌ 라이선스 위반 → Commercial License 필요
```

### Commercial License

#### 언제 필요한가?

Commercial License가 필요한 경우:

1. **SaaS 서비스**: 이메일 검증을 서비스로 제공
2. **프로프라이어터리 제품**: 소스 코드를 공개하고 싶지 않음
3. **내부 도구**: 외부 사용자가 접근 가능한 내부 시스템

#### 구매 방법

https://reacher.email/pricing

#### 가격 (2026년 2월 기준)

- **Commercial License Trial**: 무료 (일 10,000 검증 제한)
- **Commercial License**: 문의 필요 (제한 없음)

### 라이선스 비교

| 항목 | AGPL-3.0 | Commercial License |
|-----|----------|-------------------|
| **소스 공개** | 필수 | 불필요 |
| **상업적 사용** | 가능 (소스 공개 시) | 가능 |
| **SaaS 제공** | 가능 (소스 공개 시) | 가능 |
| **프로프라이어터리** | 불가능 | 가능 |
| **비용** | 무료 | 유료 |
| **지원** | 커뮤니티 | 공식 지원 |

---

## 시리즈 총정리

6개 챕터에 걸쳐 **check-if-email-exists**의 모든 것을 살펴보았습니다.

### 챕터 요약

#### 챕터 01: 소개 및 개요

- 프로젝트 소개 및 동기
- 14가지 검증 항목
- JSON 출력 형식
- Live Demo 및 SaaS

#### 챕터 02: 빠른 시작

- Docker를 통한 HTTP 백엔드
- CLI 바이너리 사용
- Rust 라이브러리 통합
- 첫 이메일 검증 실습

#### 챕터 03: 검증 메커니즘

- 4단계 검증 파이프라인
- Syntax, DNS/MX, SMTP, Misc 검증
- is_reachable 계산 알고리즘
- Gmail, Yahoo, Outlook 특수 처리

#### 챕터 04: HTTP 백엔드

- Docker 이미지 및 API 엔드포인트
- /v0/check_email, /v1/check_email, /v1/bulk
- 프록시 설정 (단일/다중)
- 인증 및 보안 (Shared Secret, HTTPS)

#### 챕터 05: 고급 활용

- Self-hosting (VPS, AWS, GCP)
- RabbitMQ 통합 (큐 기반 아키텍처)
- 확장 전략 (전용 서버, 서버리스, Kubernetes)
- 성능 최적화 및 모니터링

#### 챕터 06: 개발 및 기여 (현재)

- Rust 개발 환경 설정
- 프로젝트 구조 (core, cli, backend, sqs)
- 빌드, 테스트, 기여 프로세스
- 라이선스 이해 (AGPL-3.0 vs Commercial)

### 핵심 개념

**이메일 검증 파이프라인:**

```
입력 → Syntax → DNS/MX → SMTP → Misc → is_reachable
```

**is_reachable 판정:**

- `safe`: 안전하게 발송 가능 (< 2% 바운스율)
- `invalid`: 발송 불가능 (~100% 바운스율)
- `risky`: 위험 요소 존재 (2-20% 바운스율)
- `unknown`: 판단 불가

**3가지 사용 방법:**

1. **Docker**: HTTP API (프로덕션 권장)
2. **CLI**: 터미널 실행 (테스트 용이)
3. **Rust 라이브러리**: 직접 통합 (최대 유연성)

### 프로덕션 체크리스트

- [ ] 포트 25 개방 확인 (또는 프록시 설정)
- [ ] `header_secret` 설정
- [ ] HTTPS 활성화
- [ ] Throttle 설정 (60/분, 10,000/일)
- [ ] 모니터링 구축 (Prometheus + Grafana)
- [ ] 백업 설정 (PostgreSQL)
- [ ] 문서화 (배포 절차, 장애 대응)

### 학습 로드맵

**초급 (1-2시간):**
1. 챕터 01 읽기
2. Docker로 빠른 시작 (챕터 02)
3. 첫 이메일 검증 실습

**중급 (1일):**
1. 검증 메커니즘 이해 (챕터 03)
2. HTTP 백엔드 배포 (챕터 04)
3. 프록시 설정 실습

**고급 (1주):**
1. RabbitMQ 통합 (챕터 05)
2. 프로덕션 배포 (챕터 05)
3. 소스 코드 분석 (챕터 06)

**전문가 (지속적):**
1. 기여 참여 (챕터 06)
2. 성능 최적화
3. 커뮤니티 활동

---

## 결론

**check-if-email-exists**는 이메일 검증을 위한 강력하고 유연한 오픈소스 도구입니다.

### 프로젝트의 강점

1. **오픈소스**: AGPL-3.0으로 자유롭게 사용 가능
2. **Rust**: 메모리 안전성과 고성능
3. **포괄적**: 14가지 검증 항목
4. **유연함**: CLI, 라이브러리, HTTP API 제공
5. **확장 가능**: RabbitMQ, Kubernetes 지원
6. **활발한 커뮤니티**: GitHub 스타 4,000+

### 다음 단계

**사용자:**

1. Docker로 시작하여 첫 검증 실행
2. 프로덕션 요구사항에 맞게 확장
3. 커뮤니티에서 도움 받기

**개발자:**

1. 소스 코드 탐색
2. 테스트 작성 및 실행
3. PR 제출하여 기여

### 참고 자료

- **공식 저장소**: https://github.com/reacherhq/check-if-email-exists
- **문서**: https://docs.reacher.email
- **API 문서**: https://docs.rs/check-if-email-exists
- **SaaS**: https://reacher.email
- **Discord**: https://discord.gg/reacher (커뮤니티)

### 감사의 말

이 가이드 시리즈가 **check-if-email-exists**를 이해하고 활용하는 데 도움이 되었기를 바랍니다.

질문이나 피드백은 GitHub Issues 또는 Discord에서 환영합니다!

---

**전체 시리즈:**

1. [소개 및 개요](/check-if-email-exists-guide-01-intro/)
2. [빠른 시작](/check-if-email-exists-guide-02-quick-start/)
3. [검증 메커니즘](/check-if-email-exists-guide-03-verification/)
4. [HTTP 백엔드](/check-if-email-exists-guide-04-http-backend/)
5. [고급 활용](/check-if-email-exists-guide-05-advanced/)
6. [개발 및 기여](/check-if-email-exists-guide-06-development/) (현재)

Happy Email Verification! 📧✅
