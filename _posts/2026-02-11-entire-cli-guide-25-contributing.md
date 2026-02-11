---
layout: post
title: "Entire CLI 완벽 가이드 (25) - Contributing"
date: 2026-02-11
permalink: /entire-cli-guide-25-contributing/
author: Entire Team
categories: [AI 코딩, 개발 도구, 오픈소스]
tags: [Entire, Contributing, Open Source, Pull Request]
original_url: "https://github.com/entireio/cli"
excerpt: "Entire CLI에 기여하기 - 테스트, 코드 스타일, PR 프로세스 완벽 가이드"
---

## 개요

Entire CLI는 **오픈소스 프로젝트**입니다. 모든 기여를 환영하며, 이 챕터에서는 프로젝트에 기여하는 방법을 상세히 안내합니다.

---

## 기여 방법

### 1. 이슈 제기

**Bug Report:**

```markdown
## Bug Description
Rewind fails with "checkpoint not found" error

## Steps to Reproduce
1. entire enable
2. claude "Add feature"
3. git commit -m "Add feature"
4. entire rewind
   → Error: checkpoint not found

## Expected Behavior
Should show available checkpoints

## Environment
- Entire CLI version: v0.3.0
- OS: macOS 14.0
- Go version: 1.25.6
```

**Feature Request:**

```markdown
## Feature Description
Add support for Aider agent

## Use Case
Many developers use Aider for AI coding

## Proposed Implementation
1. Create agent/aider/ package
2. Implement Agent interface
3. Add hook scripts

## Alternatives Considered
- Manual session tracking
- Third-party integration
```

### 2. 코드 기여

**Good First Issues:**

- 문서 개선 (README, CLAUDE.md)
- 테스트 추가
- 버그 수정 (good-first-issue 라벨)

**Advanced Contributions:**

- 새 Strategy 구현
- 새 Agent 추가
- 성능 최적화

---

## 개발 프로세스

### 1. Fork 및 Clone

```bash
# 1. GitHub에서 Fork
# https://github.com/entireio/cli → Fork 버튼

# 2. Clone
git clone https://github.com/<your-username>/cli.git
cd cli

# 3. Upstream 추가
git remote add upstream https://github.com/entireio/cli.git
```

### 2. 브랜치 생성

```bash
# Feature 브랜치
git checkout -b feature/add-aider-support

# Bugfix 브랜치
git checkout -b fix/rewind-checkpoint-not-found

# Documentation 브랜치
git checkout -b docs/update-contributing
```

**브랜치 이름 규칙:**

- `feature/<description>` - 새 기능
- `fix/<description>` - 버그 수정
- `docs/<description>` - 문서 개선
- `refactor/<description>` - 리팩토링
- `test/<description>` - 테스트 추가

### 3. 코드 작성

```bash
# 파일 생성/편집
vim cmd/entire/cli/agent/aider/agent.go

# 테스트 작성
vim cmd/entire/cli/agent/aider/agent_test.go
```

### 4. 테스트

```bash
# 단위 테스트
mise run test

# 통합 테스트
mise run test:integration

# 전체 테스트 (CI)
mise run test:ci
```

### 5. 커밋 전 체크리스트

**필수 (CI 실패 방지):**

```bash
# 1. 포맷팅
mise run fmt

# 2. 린팅
mise run lint

# 3. 전체 테스트
mise run test:ci

# 또는 한 번에
mise run fmt && mise run lint && mise run test:ci
```

**권장:**

```bash
# 중복 코드 검사
mise run dup:staged
```

### 6. 커밋

```bash
git add .
git commit -m "Add Aider agent support"
```

**커밋 메시지 규칙:**

```
<type>: <subject>

<body>

<footer>
```

**Types:**

- `feat`: 새 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `refactor`: 리팩토링
- `test`: 테스트 추가
- `chore`: 빌드/도구 변경

**예시:**

```
feat: Add Aider agent support

Implements Agent interface for Aider AI coding tool:
- Create agent/aider package
- Implement session management
- Add hook scripts
- Add tests

Closes #123
```

### 7. Push

```bash
git push origin feature/add-aider-support
```

---

## Pull Request 프로세스

### 1. PR 생성

GitHub에서 "Compare & pull request" 버튼 클릭

**PR 템플릿:**

```markdown
## Description
Brief description of changes

## Related Issue
Closes #123

## Type of Change
- [ ] Bug fix
- [x] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing Done
- [x] Unit tests pass
- [x] Integration tests pass
- [x] Manual testing completed

## Checklist
- [x] Code follows project style
- [x] Tests added/updated
- [x] Documentation updated
- [x] CI passes
```

### 2. Code Review

Maintainer가 리뷰를 진행합니다.

**일반적인 피드백:**

```go
// ❌ Bad
func SaveCheckpoint(id string) error {
    // 에러 처리 없음
    data := readFile(id)
    return nil
}

// ✓ Good
func SaveCheckpoint(id id.CheckpointID) error {
    data, err := readFile(id.String())
    if err != nil {
        return fmt.Errorf("failed to read file: %w", err)
    }
    return nil
}
```

### 3. 수정 사항 반영

```bash
# 피드백 반영
vim cmd/entire/cli/agent/aider/agent.go

# 테스트
mise run test

# 커밋
git add .
git commit -m "Address review feedback"

# Push (자동으로 PR 업데이트)
git push origin feature/add-aider-support
```

### 4. Merge

Maintainer가 승인하면 merge됩니다.

**Merge 조건:**

- ✅ CI 통과
- ✅ 1+ approvals
- ✅ No merge conflicts
- ✅ Code review completed

---

## 코드 스타일

### Go 코드 규칙

#### 1. 에러 처리

```go
// ❌ Bad - 에러 무시
_ = doSomething()

// ✓ Good - 명시적 처리
if err := doSomething(); err != nil {
    return fmt.Errorf("operation failed: %w", err)
}

// ✓ Good - nolint 주석
//nolint:errcheck // Cleanup operation, error is not critical
os.Remove(tempFile)
```

#### 2. 변수 이름

```go
// ❌ Bad
s := "session-id"
r := repo

// ✓ Good
sessionID := "session-id"
repository := repo

// ✓ Good (짧은 스코프에서는 OK)
for _, f := range files {
    processFile(f)
}
```

#### 3. 함수 길이

```go
// ❌ Bad - 200 줄 함수
func doEverything() error {
    // ...
}

// ✓ Good - 작은 함수로 분리
func doStep1() error { /* ... */ }
func doStep2() error { /* ... */ }
func doEverything() error {
    if err := doStep1(); err != nil {
        return err
    }
    return doStep2()
}
```

#### 4. 테스트

```go
func TestFeature_BasicCase(t *testing.T) {
    t.Parallel()  // 필수!

    // Given
    input := "test"

    // When
    result := ProcessInput(input)

    // Then
    assert.Equal(t, "expected", result)
}
```

### 문서화

#### 1. Package 주석

```go
// Package agent provides abstractions for AI coding tools.
//
// This package defines the Agent interface and implementations
// for Claude Code, Gemini CLI, and other AI agents.
package agent
```

#### 2. 함수 주석

```go
// SaveChanges saves checkpoint data to storage.
//
// This method is called after each AI response in manual-commit
// strategy or after each commit in auto-commit strategy.
//
// Returns error if git operations fail or storage is unavailable.
func (s *Strategy) SaveChanges(ctx SaveContext) error {
    // ...
}
```

#### 3. 복잡한 로직 주석

```go
// Shadow branch migration: if user does stash→pull→apply, HEAD changes
// but work isn't committed. The shadow branch would be orphaned at the
// old commit. We detect this and rename the branch to the new commit.
if baseChanged && oldShadowBranchExists {
    migrateShadowBranch(oldBranch, newBranch)
}
```

---

## 테스트 작성

### 단위 테스트

**파일 명명:**

```
feature.go → feature_test.go
```

**테스트 함수:**

```go
func TestFeature_SpecificCase(t *testing.T) {
    t.Parallel()

    // Setup
    input := "test"

    // Execute
    result := ProcessInput(input)

    // Assert
    if result != "expected" {
        t.Errorf("got %s, want %s", result, "expected")
    }
}
```

**Table-driven tests:**

```go
func TestParseSessionID(t *testing.T) {
    t.Parallel()

    tests := []struct {
        name  string
        input string
        want  string
    }{
        {"valid", "2026-02-11-abc123", "abc123"},
        {"invalid", "invalid", ""},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            t.Parallel()
            got := ParseSessionID(tt.input)
            if got != tt.want {
                t.Errorf("got %s, want %s", got, tt.want)
            }
        })
    }
}
```

### 통합 테스트

```go
//go:build integration

package integration_test

func TestWorkflow_EndToEnd(t *testing.T) {
    t.Parallel()

    RunForAllStrategies(t, func(t *testing.T, env *TestEnv, strategy string) {
        // 1. Setup
        setupRepo(env)

        // 2. Enable
        runCommand("entire", "enable", "--strategy", strategy)

        // 3. Create session
        createSession(env)

        // 4. Commit
        commitChanges(env)

        // 5. Verify checkpoint created
        checkpoints := listCheckpoints(env)
        assert.Len(t, checkpoints, 1)

        // 6. Rewind
        rewindToCheckpoint(env, checkpoints[0])

        // 7. Verify state restored
        verifyFilesRestored(env)
    })
}
```

---

## CI/CD

### GitHub Actions

**Workflow (.github/workflows/test.yml):**

```yaml
name: Test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install mise
        run: curl https://mise.run | sh

      - name: Install dependencies
        run: mise install

      - name: Run tests
        run: mise run test:ci
```

**CI가 확인하는 것:**

1. `gofmt` - 코드 포맷팅
2. `golangci-lint` - 린팅
3. Unit tests - 단위 테스트
4. Integration tests - 통합 테스트
5. Race detector - 동시성 버그

### Pre-commit Hook (선택)

```bash
# .git/hooks/pre-commit
#!/bin/bash

echo "Running pre-commit checks..."

# Format
mise run fmt

# Lint
mise run lint || exit 1

# Test
mise run test || exit 1

echo "Pre-commit checks passed!"
```

---

## 보안

### 취약점 보고

**Public 이슈로 보고하지 마세요!**

**올바른 방법:**

1. [SECURITY.md](https://github.com/entireio/cli/blob/main/SECURITY.md) 읽기
2. 보안팀에 이메일 전송
3. 비공개 채널로 논의

### 민감 정보 제거

```go
// ❌ Bad - API key 로깅
logging.Info(ctx, "api call", slog.String("key", apiKey))

// ✓ Good - 로깅 안함
// API key는 절대 로깅하지 않음

// ✓ Good - Redact 패키지 사용
import "github.com/entireio/cli/redact"

redacted := redact.Sensitive(apiKey)
```

---

## 커뮤니티

### Discord

[Join our Discord](https://discord.gg/4WXDu2Ph)

**채널:**

- `#general` - 일반 논의
- `#development` - 개발 질문
- `#contributions` - 기여 관련

### GitHub Discussions

[GitHub Discussions](https://github.com/entireio/cli/discussions)

**카테고리:**

- **Q&A** - 질문
- **Ideas** - 기능 제안
- **Show and tell** - 프로젝트 공유

---

## 릴리스 프로세스

### 버전 관리

Semantic Versioning 사용:

```
v<major>.<minor>.<patch>

v0.3.0  → v0.3.1 (patch)
v0.3.1  → v0.4.0 (minor)
v0.4.0  → v1.0.0 (major)
```

**변경 타입:**

- **Patch** - 버그 수정
- **Minor** - 새 기능 (하위 호환)
- **Major** - Breaking changes

### 릴리스 노트

```markdown
# v0.4.0 - 2026-02-15

## Features
- Add Aider agent support (#123)
- Implement auto-summarization (#145)

## Bug Fixes
- Fix rewind on main branch (#156)
- Resolve checkpoint not found error (#167)

## Breaking Changes
- None

## Contributors
- @user1
- @user2
```

---

## 체크리스트

### PR 체크리스트

- [ ] 브랜치 이름이 규칙을 따름
- [ ] 커밋 메시지가 명확함
- [ ] 테스트 추가/업데이트됨
- [ ] 문서 업데이트됨 (필요 시)
- [ ] `mise run fmt` 실행
- [ ] `mise run lint` 실행
- [ ] `mise run test:ci` 통과
- [ ] PR 템플릿 작성
- [ ] 관련 이슈 연결

### 리뷰어 체크리스트

- [ ] 코드가 프로젝트 스타일을 따름
- [ ] 에러 처리가 적절함
- [ ] 테스트가 충분함
- [ ] 문서가 정확함
- [ ] Breaking changes 확인
- [ ] 성능 영향 검토
- [ ] 보안 검토

---

## 감사의 말

Entire CLI에 기여해주셔서 감사합니다! 🎉

모든 기여자는 [CONTRIBUTORS.md](https://github.com/entireio/cli/blob/main/CONTRIBUTORS.md)에 기록됩니다.

---

## 추가 리소스

- [GitHub Repository](https://github.com/entireio/cli)
- [CLAUDE.md](https://github.com/entireio/cli/blob/main/CLAUDE.md) - Architecture reference
- [CONTRIBUTING.md](https://github.com/entireio/cli/blob/main/CONTRIBUTING.md) - Contributing guide
- [Code of Conduct](https://github.com/entireio/cli/blob/main/CODE_OF_CONDUCT.md)
- [Discord Community](https://discord.gg/4WXDu2Ph)

---

## 마치며

이것으로 **Entire CLI 완벽 가이드 시리즈**를 마칩니다!

**전체 25개 챕터:**

1. 소개 및 개요
2. 설치 및 시작하기
3. 핵심 개념 (Session, Checkpoint, Strategy)
4. 일반적인 워크플로우
5. 명령어 레퍼런스
6-15. [이전 챕터들]
16. Subagent Tracking
17. Logging 시스템
18. Rewind 메커니즘
19. Resume 기능
20. Auto-Summarization
21. Token Usage Tracking
22. 개발 환경 설정
23. 코드 구조
24. Agent 통합
25. Contributing

**Happy Coding!** 🚀

---

*Entire CLI와 함께 AI 코딩을 더욱 효율적으로!*
