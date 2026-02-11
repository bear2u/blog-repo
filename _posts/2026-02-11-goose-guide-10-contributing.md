---
layout: post
title: "Goose 완벽 가이드 (10) - 개발 및 기여 가이드"
date: 2026-02-11
permalink: /goose-guide-10-contributing/
author: Block
categories: [AI 에이전트, 개발 도구]
tags: [Goose, Contributing, Development, Open Source, Rust]
original_url: "https://github.com/block/goose"
excerpt: "Goose 프로젝트 개발 환경 구축과 기여 방법"
---

## 개발 환경 구축

### 1. 저장소 클론

```bash
# 저장소 클론
git clone https://github.com/block/goose.git
cd goose
```

### 2. Hermit 활성화

Goose는 [Hermit](https://cashapp.github.io/hermit/)를 사용하여 개발 도구를 관리합니다.

```bash
# Hermit 환경 활성화
source bin/activate-hermit

# 도구가 자동으로 설치됨:
# - Rust toolchain
# - Just
# - Node.js
# - 기타 필요한 도구들
```

### 3. 빌드

```bash
# 디버그 빌드
cargo build

# 릴리스 빌드
cargo build --release

# 특정 crate만 빌드
cargo build -p goose-cli
```

### 4. 테스트

```bash
# 전체 테스트
cargo test

# 특정 crate 테스트
cargo test -p goose

# 특정 테스트 실행
cargo test --test mcp_integration_test
```

---

## 개발 워크플로우

### 기본 워크플로우

```bash
# 1. Hermit 활성화
source bin/activate-hermit

# 2. 코드 작성

# 3. 포맷팅
cargo fmt

# 4. 빌드
cargo build

# 5. 테스트
cargo test -p <crate>

# 6. Lint
cargo clippy --all-targets -- -D warnings

# 7. 서버 변경 시 OpenAPI 재생성
just generate-openapi
```

### Just 명령어

```bash
# 사용 가능한 명령어 보기
just --list

# 릴리스 바이너리 빌드
just release-binary

# OpenAPI 생성
just generate-openapi

# UI 실행
just run-ui

# MCP 테스트 기록
just record-mcp-tests
```

---

## 코드 품질 가이드

### 1. Rust 코드 스타일

#### 주석 규칙

```rust
// ❌ 나쁜 예: 코드가 하는 일을 단순 반복
// Initialize the user
let user = User::new();

// Return the result
return result;

// ✅ 좋은 예: "왜" 그렇게 하는지 설명
// Pre-allocate capacity to avoid reallocation during the loop
let mut results = Vec::with_capacity(items.len());

// Use binary search since the list is sorted
let index = items.binary_search(&target)?;
```

#### 에러 핸들링

```rust
// ❌ 나쁜 예: 불필요한 컨텍스트
read_file(path)
    .context("Failed to read file")?  // 이미 에러에 포함된 정보

// ✅ 좋은 예: 유용한 컨텍스트 추가
read_file(path)
    .with_context(|| format!("Reading config from {}", path.display()))?
```

#### 간결성

```rust
// ❌ 나쁜 예: 불필요하게 Optional
struct Config {
    enabled: Option<bool>,  // 단순히 true/false면 충분
}

// ✅ 좋은 예
struct Config {
    enabled: bool,  // 기본값 false
}

// ❌ 나쁜 예: 과도한 방어적 코드
if let Some(value) = option {
    if !value.is_empty() {
        if value.len() > 0 {  // 중복 체크
            process(value);
        }
    }
}

// ✅ 좋은 예: Rust 타입 시스템 신뢰
if let Some(value) = option.filter(|v| !v.is_empty()) {
    process(value);
}
```

### 2. 테스트 작성

#### 위치

```bash
# 단위 테스트: 같은 파일에
// src/lib.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_something() {
        // ...
    }
}

# 통합 테스트: tests/ 폴더에
// tests/integration_test.rs
```

#### 새 기능 추가 시

```bash
# 1. goose-self-test.yaml에 테스트 추가
vim goose-self-test.yaml

# 2. 빌드 및 실행
cargo build
goose run --recipe goose-self-test.yaml
```

### 3. Provider 구현

```rust
// crates/goose/src/providers/new_provider.rs

// Provider trait 구현
// providers/base.rs 참조
use crate::providers::base::Provider;

pub struct NewProvider {
    // ...
}

#[async_trait]
impl Provider for NewProvider {
    async fn complete(&self, messages: Vec<Message>) -> Result<Response> {
        // 구현
    }

    fn model(&self) -> &str {
        "model-name"
    }

    fn supports_tools(&self) -> bool {
        true
    }
}
```

### 4. MCP Extension 구현

```rust
// crates/goose-mcp/src/my_extension/

// 새 도구는 goose-mcp crate에
pub mod tools;

// tools/my_tool.rs
use goose::tools::Tool;

pub struct MyTool;

#[async_trait]
impl Tool for MyTool {
    // 구현
}
```

---

## 기여 가이드

### 1. 이슈 찾기

```bash
# GitHub에서 이슈 찾기
# https://github.com/block/goose/issues

# Good First Issue 라벨 찾기
# https://github.com/block/goose/labels/good%20first%20issue
```

### 2. 브랜치 생성

```bash
# 기능 추가
git checkout -b feature/my-feature

# 버그 수정
git checkout -b bugfix/fix-issue-123

# 문서 개선
git checkout -b docs/improve-readme
```

### 3. 커밋 메시지

```bash
# Conventional Commits 형식 사용
feat: add new provider for Gemini
fix: resolve MCP connection timeout
docs: update installation guide
test: add tests for agent loop
refactor: simplify provider trait
```

### 4. Pull Request 생성

```bash
# 변경사항 커밋
git add .
git commit -m "feat: add new feature"

# Push
git push origin feature/my-feature

# GitHub에서 PR 생성
# https://github.com/block/goose/compare
```

#### PR 체크리스트

- [ ] 코드가 `cargo fmt`로 포맷팅됨
- [ ] `cargo clippy`가 경고 없이 통과
- [ ] 모든 테스트 통과
- [ ] 새 기능에 테스트 추가
- [ ] 문서 업데이트 (필요시)
- [ ] CHANGELOG 업데이트 (필요시)
- [ ] Co-Authored-By 추가 (AI 도움 받은 경우)

```bash
# 커밋 메시지 예시
feat: add support for Gemini Pro

Implements Gemini Pro provider with streaming support.

- Add GeminiProvider struct
- Implement Provider trait
- Add tests for Gemini integration
- Update provider documentation

Co-Authored-By: goose <goose@block.xyz>
```

---

## AI 도움 받기

### HOWTOAI.md 가이드라인

Goose 프로젝트는 [HOWTOAI.md](https://github.com/block/goose/blob/main/HOWTOAI.md)에 AI 사용 가이드라인이 있습니다.

#### ✅ 추천 용도

- 보일러플레이트 코드 생성
- 테스트 작성
- 문서 작성
- 리팩토링
- 유틸리티 함수 생성

#### ❌ 피해야 할 용도

- 복잡한 비즈니스 로직 (철저한 리뷰 없이)
- 보안 중요 코드
- 이해하지 못하는 코드
- 대규모 아키텍처 변경
- 데이터베이스 마이그레이션

#### 워크플로우

```bash
# 1. Goose로 코드 생성
goose session
> Implement a new tool for ...

# 2. 생성된 코드 리뷰
# - 모든 줄 이해하기
# - 보안 이슈 확인
# - 패턴 확인

# 3. 테스트
cargo test -p <crate>

# 4. Lint
cargo clippy

# 5. PR에 AI 사용 명시
# "Generated with goose, reviewed and tested by human"
```

---

## 커뮤니티

### Discord

```
https://discord.gg/goose-oss
```

**채널:**
- `#general` - 일반 토론
- `#help` - 도움 요청
- `#development` - 개발 토론
- `#contributions` - 기여 관련

### GitHub Discussions

```
https://github.com/block/goose/discussions
```

**카테고리:**
- Ideas - 새로운 아이디어
- Q&A - 질문과 답변
- Show and Tell - 프로젝트 공유

---

## 릴리스 프로세스

### 버전 관리

Goose는 [Semantic Versioning](https://semver.org/)을 따릅니다:

```
MAJOR.MINOR.PATCH
1.23.0

MAJOR: 호환성 없는 API 변경
MINOR: 기능 추가 (호환성 유지)
PATCH: 버그 수정
```

### 릴리스 체크리스트

1. **버전 업데이트**
   ```bash
   # Cargo.toml에서 버전 업데이트
   vim Cargo.toml
   ```

2. **CHANGELOG 업데이트**
   ```bash
   vim CHANGELOG.md
   ```

3. **태그 생성**
   ```bash
   git tag -a v1.24.0 -m "Release v1.24.0"
   git push origin v1.24.0
   ```

4. **GitHub Release 생성**
   - CI가 자동으로 바이너리 빌드
   - Release notes 작성

---

## 문제 해결

### 빌드 실패

```bash
# 캐시 정리
cargo clean

# 의존성 업데이트
cargo update

# 재빌드
cargo build
```

### 테스트 실패

```bash
# 자세한 출력
cargo test -- --nocapture

# 특정 테스트만
cargo test test_name -- --nocapture

# 로그 레벨 증가
RUST_LOG=debug cargo test
```

### Clippy 경고

```bash
# 자동 수정 가능한 항목 수정
cargo clippy --fix

# 모든 경고 보기
cargo clippy --all-targets
```

---

## 고급 주제

### 성능 프로파일링

```bash
# 릴리스 빌드로 프로파일링
cargo build --release

# Flamegraph 생성
cargo flamegraph --bin goose
```

### 벤치마킹

```bash
# 벤치마크 실행
cargo bench -p goose-bench
```

### 크로스 컴파일

```bash
# ARM64용 빌드
cargo build --target aarch64-unknown-linux-gnu

# Windows용 빌드
cargo build --target x86_64-pc-windows-gnu
```

---

## 참고 자료

### 공식 문서

- **GitHub**: https://github.com/block/goose
- **문서**: https://block.github.io/goose
- **CONTRIBUTING.md**: https://github.com/block/goose/blob/main/CONTRIBUTING.md
- **HOWTOAI.md**: https://github.com/block/goose/blob/main/HOWTOAI.md

### Rust 리소스

- **The Rust Book**: https://doc.rust-lang.org/book/
- **Async Book**: https://rust-lang.github.io/async-book/
- **Rust by Example**: https://doc.rust-lang.org/rust-by-example/

### 관련 프로젝트

- **MCP Specification**: https://modelcontextprotocol.io/
- **Anthropic SDK**: https://github.com/anthropics/anthropic-sdk-rust
- **OpenAI SDK**: https://github.com/openai/openai-rust

---

## 마무리

이것으로 Goose 완벽 가이드를 마칩니다!

Goose는 강력하고 확장 가능한 AI 에이전트 프레임워크입니다. 이 가이드를 통해 Goose의 모든 측면을 이해하고, 프로젝트에 활용하거나 기여할 수 있기를 바랍니다.

**Happy Coding with Goose! 🦢**

---

## 전체 시리즈 목차

1. [소개 및 개요](/goose-guide-01-intro/)
2. [설치 및 시작](/goose-guide-02-installation/)
3. [아키텍처 분석](/goose-guide-03-architecture/)
4. [코어 에이전트 시스템](/goose-guide-04-core-agent/)
5. [CLI 인터페이스](/goose-guide-05-cli/)
6. [Desktop 앱](/goose-guide-06-desktop/)
7. [MCP 통합](/goose-guide-07-mcp/)
8. [서버 및 API](/goose-guide-08-server-api/)
9. [확장 및 커스터마이징](/goose-guide-09-customization/)
10. [개발 및 기여 가이드](/goose-guide-10-contributing/) ← 현재 페이지
