---
layout: post
title: "Entire CLI 완벽 가이드 (22) - 개발 환경 설정"
date: 2026-02-11
permalink: /entire-cli-guide-22-development-setup/
author: Entire Team
categories: [AI 코딩, 개발 도구, 개발]
tags: [Entire, Development, Go, mise, Testing]
original_url: "https://github.com/entireio/cli"
excerpt: "Entire CLI 개발 환경 설정 - mise, Go, 테스트 실행 완벽 가이드"
---

## 개요

Entire CLI는 **Go 1.25.x**로 작성되었으며, **mise**를 빌드 도구로 사용합니다. 이 챕터에서는 개발 환경을 설정하고 첫 기여를 준비하는 방법을 알아봅니다.

---

## 요구사항

### 필수 도구

```bash
# 1. Git
git --version
# git version 2.40+

# 2. mise (버전 관리 및 태스크 러너)
curl https://mise.run | sh

# 3. Go (mise가 자동 설치)
# go 1.25.6 (mise.toml에 정의됨)
```

---

## 프로젝트 클론

### 1. Repository Clone

```bash
git clone https://github.com/entireio/cli.git
cd cli
```

### 2. mise 신뢰 설정

mise는 보안을 위해 프로젝트 설정을 신뢰해야 합니다.

```bash
mise trust

# 출력:
mise: trusted config file: /path/to/cli/mise.toml
```

### 3. 의존성 설치

```bash
# mise가 Go와 도구들을 자동 설치
mise install

# 출력:
mise: installing go@1.25.6
mise: installing golangci-lint@2.8.0
mise: installing shellcheck@latest
```

### 4. Go 모듈 다운로드

```bash
go mod download

# 출력:
go: downloading github.com/spf13/cobra v1.8.0
go: downloading github.com/charmbracelet/huh v0.3.0
...
```

---

## mise 설정 (mise.toml)

### Tools 섹션

```toml
[tools]
go = { version = '1.25.6', postinstall = "go install github.com/go-delve/delve/cmd/dlv@latest" }
golangci-lint = '2.8.0'
shellcheck = 'latest'
```

**설명:**

- **go** - Go 1.25.6 설치, dlv (debugger) 자동 설치
- **golangci-lint** - 린터
- **shellcheck** - 쉘 스크립트 린터

### Tasks 섹션

```toml
[tasks.fmt]
description = "Run gofmt"
run = "gofmt -s -w ."

[tasks.test]
description = "Run tests"
run = "go test ./..."

[tasks."test:integration"]
description = "Run integration tests"
run = "go test -tags=integration ./cmd/entire/cli/integration_test/..."

[tasks."test:ci"]
description = "Run all tests (unit + integration) with race detection"
run = "go test -tags=integration -race ./..."

[tasks.build]
description = "Build the CLI"
run = """
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
go build -ldflags "-X github.com/entireio/cli/cmd/entire/cli/buildinfo.Version=${VERSION} -X github.com/entireio/cli/cmd/entire/cli/buildinfo.Commit=${COMMIT}" -o entire ./cmd/entire
"""

[tasks.lint]
description = "Run golangci-lint"
run = "golangci-lint run"
```

---

## 빌드

### 개발용 빌드

```bash
mise run build

# 출력:
go build -ldflags "-X ...Version=dev ..." -o entire ./cmd/entire

# 바이너리 확인
./entire version
# Entire CLI dev (commit: abc123)
```

### 릴리스 빌드

```bash
# 모든 플랫폼용 빌드 (goreleaser 사용)
mise run build:all

# 출력:
dist/entire_linux_amd64/entire
dist/entire_darwin_amd64/entire
dist/entire_darwin_arm64/entire
```

### 바이너리 설치

```bash
# 로컬 설치
go install ./cmd/entire

# 또는
cp ./entire /usr/local/bin/

# 확인
which entire
entire version
```

---

## 테스트 실행

### 단위 테스트 (Unit Tests)

```bash
mise run test

# 출력:
?   	github.com/entireio/cli/cmd/entire	[no test files]
ok  	github.com/entireio/cli/cmd/entire/cli	0.234s
ok  	github.com/entireio/cli/cmd/entire/cli/checkpoint	0.156s
ok  	github.com/entireio/cli/cmd/entire/cli/strategy	1.023s
```

**특정 패키지만:**

```bash
go test ./cmd/entire/cli/strategy

# 또는 특정 테스트만
go test -run TestManualCommit ./cmd/entire/cli/strategy
```

### 통합 테스트 (Integration Tests)

```bash
mise run test:integration

# 출력:
ok  	github.com/entireio/cli/cmd/entire/cli/integration_test	5.678s
```

**통합 테스트 특징:**

- `//go:build integration` 태그 사용
- 실제 Git 저장소 생성
- 모든 전략 테스트 (`RunForAllStrategies`)
- `cmd/entire/cli/integration_test/` 디렉토리

### CI 테스트 (전체 + Race Detector)

```bash
mise run test:ci

# 출력:
go test -tags=integration -race ./...
ok  	github.com/entireio/cli/cmd/entire/cli	0.234s
ok  	github.com/entireio/cli/cmd/entire/cli/integration_test	5.678s
```

**Race detector:**

- 동시성 버그 감지
- CI에서 필수
- 느리지만 안전

### 병렬 테스트

모든 테스트는 `t.Parallel()`을 사용해야 합니다.

```go
func TestFeature(t *testing.T) {
    t.Parallel()  // 필수!

    // 테스트 코드...
}

// 예외: os.Chdir(), os.Setenv() 사용 시
func TestWithChdir(t *testing.T) {
    // t.Parallel() 호출 안함
    os.Chdir("/tmp")
}
```

---

## 린팅 및 포맷팅

### gofmt (코드 포맷팅)

```bash
mise run fmt

# 또는
gofmt -s -w .
```

**CI에서 자동 확인:**

```bash
# CI가 실행하는 명령
gofmt -l .
# 출력 있으면 실패
```

### golangci-lint (린팅)

```bash
mise run lint

# 출력:
cmd/entire/cli/setup.go:45:2: unused variable 'err' (unused)
cmd/entire/cli/strategy/common.go:123:1: line is too long (lll)
```

**설정 파일:**

```yaml
# .golangci.yaml
linters:
  enable:
    - unused
    - staticcheck
    - errcheck
    - gofmt
    - lll  # Line length
    - dupl # Code duplication (threshold 75)
```

### 중복 코드 검사

```bash
# Threshold 50 (권장)
mise run dup

# 출력:
=== Duplication Summary (by file) ===
  3 manual_commit_hooks.go
  2 auto_commit.go
  1 common.go

# Staged 파일만 (threshold 75)
mise run dup:staged
```

---

## 커밋 전 체크리스트

### 필수 단계 (CI 실패 방지)

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

**CI가 실패하는 이유:**

- `gofmt` 차이 → `mise run fmt` 실행
- Lint 에러 → `mise run lint` 실행 및 수정
- 테스트 실패 → `mise run test` 실행 및 수정

---

## 개발 워크플로우

### 1. Feature 브랜치 생성

```bash
git checkout -b feature/my-feature
```

### 2. 코드 작성

```bash
# 파일 편집
vim cmd/entire/cli/my_feature.go

# 테스트 작성
vim cmd/entire/cli/my_feature_test.go
```

### 3. 테스트 실행

```bash
# 단위 테스트
go test -run TestMyFeature ./cmd/entire/cli

# 통합 테스트
mise run test:integration
```

### 4. 커밋 전 체크

```bash
mise run fmt && mise run lint && mise run test:ci
```

### 5. 커밋

```bash
git add .
git commit -m "Add my feature"
```

### 6. Pull Request

```bash
git push origin feature/my-feature

# GitHub에서 PR 생성
```

---

## 디버깅

### dlv (Delve) 사용

```bash
# dlv 설치 (mise가 자동 설치함)
go install github.com/go-delve/delve/cmd/dlv@latest

# 디버깅 시작
dlv debug ./cmd/entire -- enable

# 또는 테스트 디버깅
dlv test ./cmd/entire/cli/strategy -- -test.run TestManualCommit
```

**Breakpoint 설정:**

```bash
(dlv) break cmd/entire/cli/setup.go:123
(dlv) continue
(dlv) print variable
(dlv) next
```

### 로그 디버깅

```bash
# DEBUG 로그 활성화
ENTIRE_LOG_LEVEL=debug ./entire enable

# 로그 확인
tail -f .entire/logs/entire.log | jq .
```

---

## IDE 설정

### VS Code

**.vscode/settings.json:**

```json
{
  "go.useLanguageServer": true,
  "go.lintTool": "golangci-lint",
  "go.lintOnSave": "workspace",
  "editor.formatOnSave": true,
  "go.formatTool": "gofmt",
  "go.testFlags": ["-v", "-race"],
  "go.testTimeout": "30s"
}
```

**.vscode/launch.json:**

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Entire",
      "type": "go",
      "request": "launch",
      "mode": "debug",
      "program": "${workspaceFolder}/cmd/entire",
      "args": ["enable"]
    }
  ]
}
```

### GoLand

**Settings → Go → Build Tags:**

```
integration
```

**Run Configuration:**

- Package path: `./cmd/entire`
- Arguments: `enable`

---

## 트러블슈팅

### 1. "mise: command not found"

```bash
# mise 재설치
curl https://mise.run | sh

# 쉘 reload
source ~/.zshrc  # or ~/.bashrc
```

### 2. "go: cannot find main module"

```bash
# go.mod 확인
ls go.mod

# 없으면 프로젝트 root에 있는지 확인
cd /path/to/cli
```

### 3. 테스트 실패

```bash
# 캐시 삭제
go clean -testcache

# 재실행
mise run test
```

### 4. Lint 에러 무시

```go
// nolint:errcheck // 이유 설명
os.Remove(file)
```

---

## 고급 개발 팁

### 1. 코드 중복 방지

```bash
# 커밋 전 중복 검사
mise run dup:staged

# helper 함수 확인
grep -r "func.*Helper" cmd/entire/cli/
```

### 2. 테스트 커버리지

```bash
# 커버리지 생성
go test -coverprofile=coverage.out ./...

# HTML 리포트
go tool cover -html=coverage.out
```

### 3. 벤치마크

```bash
# 벤치마크 실행
go test -bench=. ./cmd/entire/cli/strategy

# CPU 프로파일
go test -cpuprofile=cpu.prof -bench=. ./...
go tool pprof cpu.prof
```

---

## 다음 단계

개발 환경 설정을 완료했습니다! 다음 챕터에서는:

- **코드 구조** - 패키지 구성, 주요 파일
- **Agent 통합** - Gemini CLI, 새 Agent 추가
- **Contributing** - 기여 가이드, 테스트, PR 프로세스

---

*다음 글에서는 Entire CLI의 코드 구조를 살펴봅니다.*
