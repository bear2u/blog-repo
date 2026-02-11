---
layout: post
title: "Goose 완벽 가이드 (05) - CLI 인터페이스"
date: 2026-02-11
permalink: /goose-guide-05-cli/
author: Block
categories: [AI 에이전트, 개발 도구]
tags: [Goose, CLI, Commands, Terminal, Session Management]
original_url: "https://github.com/block/goose"
excerpt: "Goose CLI의 모든 명령어와 고급 사용법"
---

## CLI 개요

Goose CLI는 터미널에서 AI 에이전트를 사용할 수 있는 강력한 인터페이스입니다.

```bash
# 기본 명령어
goose session      # 세션 시작
goose configure    # 설정
goose web          # 웹 인터페이스
goose doctor       # 진단
```

---

## 명령어 상세

### 1. session - 세션 관리

#### 새 세션 시작

```bash
# 기본 세션
goose session

# 특정 디렉토리에서
cd /path/to/project
goose session
```

#### 세션 재개

```bash
# 마지막 세션 재개
goose session -r
goose session --resume

# 특정 세션 재개
goose session --resume <session-id>
```

#### 세션 목록

```bash
# 모든 세션 보기
goose session --list

# 출력 예시
# Sessions:
# 1. session_20260211_140532 (active)
# 2. session_20260210_093012
# 3. session_20260209_151423
```

#### 세션 삭제

```bash
# 특정 세션 삭제
goose session --delete <session-id>

# 모든 세션 삭제
goose session --delete-all
```

---

### 2. configure - 설정 관리

```bash
# 설정 메뉴 열기
goose configure
```

#### 메뉴 옵션

```
┌   goose-configure
│
◆  What would you like to configure?
│  ● Configure Providers
│  ○ Add Extension
│  ○ Toggle Extensions
│  ○ Remove Extension
│  ○ goose settings
└
```

#### Provider 설정

```bash
goose configure
> Configure Providers
> Anthropic

# API 키 입력
ANTHROPIC_API_KEY: sk-ant-...

# 모델 선택
> claude-sonnet-4-5
```

#### Extension 추가

```bash
goose configure
> Add Extension
> Built-in Extension
> Developer

# 설정
Allow shell commands: Yes
Allow file operations: Yes
```

#### Extension 토글

```bash
goose configure
> Toggle Extensions

# 활성화/비활성화 선택
[x] Developer
[x] Computer Controller
[ ] Custom Extension
```

---

### 3. web - 웹 인터페이스

CLI 사용자를 위한 웹 기반 인터페이스

```bash
# 웹 인터페이스 시작
goose web

# 자동으로 브라우저 열기
goose web --open

# 포트 지정
goose web --port 3000

# 호스트 지정
goose web --host 0.0.0.0
```

**접속:**
```
http://localhost:8080
```

**기능:**
- 채팅 인터페이스
- 파일 브라우저
- 세션 관리
- 설정 패널

---

### 4. doctor - 진단 도구

시스템 상태 및 설정 확인

```bash
# 진단 실행
goose doctor

# 출력 예시
✓ Goose version: 1.23.0
✓ Rust toolchain: 1.75.0
✓ Config directory: ~/.config/goose
✓ Provider configured: Anthropic (claude-sonnet-4-5)
✓ Extensions: 2 active
  - Developer
  - Computer Controller
⚠ Warning: API key expires in 7 days
```

---

### 5. logs - 로그 조회

```bash
# 최근 로그 보기
goose logs

# 실시간 로그 (tail)
goose logs --follow

# 로그 레벨 필터
goose logs --level error
goose logs --level warn
goose logs --level info

# 특정 세션 로그
goose logs --session <session-id>
```

---

### 6. run - Recipe 실행

사전 정의된 작업 레시피 실행

```bash
# Recipe 파일 실행
goose run --recipe recipe.yaml

# 예시: Self-test
goose run --recipe goose-self-test.yaml
```

#### Recipe 파일 형식

```yaml
# recipe.yaml
name: "Setup Project"
description: "Initialize a new web project"

steps:
  - name: "Create structure"
    prompt: "Create a standard web project structure with src/, tests/, and docs/"

  - name: "Install dependencies"
    prompt: "Set up package.json and install React, TypeScript, and Vite"

  - name: "Configure linting"
    prompt: "Add ESLint and Prettier configuration"

  - name: "Write README"
    prompt: "Create a comprehensive README.md with setup instructions"
```

---

## 세션 내 명령어

세션 중에 사용할 수 있는 특수 명령어:

### /plan - 계획 모드

```
G❯ /plan create a full-stack app
```

Goose가 단계별 계획을 세우고 승인을 요청합니다.

### /clear - 컨텍스트 초기화

```
G❯ /clear
```

대화 히스토리를 지우고 새로 시작합니다.

### /help - 도움말

```
G❯ /help
```

사용 가능한 명령어 목록을 표시합니다.

### /exit - 세션 종료

```
G❯ /exit
```

또는 `Ctrl+C`를 누릅니다.

---

## 고급 사용법

### 1. 환경 변수 사용

```bash
# API 키 환경 변수로 설정
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# Goose 실행
goose session
```

### 2. 설정 파일 직접 편집

```bash
# 설정 디렉토리 열기
cd ~/.config/goose

# Provider 설정
vim providers.yaml

# Extension 설정
vim extensions.yaml
```

### 3. 프로젝트별 힌트

```bash
# .goosehints 생성
cat > .goosehints << 'EOF'
# Project: My Web App

## Tech Stack
- React 18
- TypeScript
- Vite
- TailwindCSS

## Coding Standards
- Use functional components
- Prefer hooks over classes
- Write type-safe code
- Add JSDoc comments

## Testing
- Use Vitest
- Write unit tests for utils
- Integration tests for components

## Git
- Conventional commits
- Branch: feature/*, bugfix/*
EOF

# Goose가 자동으로 힌트를 읽음
goose session
```

### 4. 멀티 Provider 전환

```bash
# Provider 별로 세션 시작
goose session --provider anthropic
goose session --provider openai
goose session --provider tetrate
```

---

## TUI (Terminal UI)

### 인터랙티브 프롬프트

```
┌─────────────────────────────────────────────────────┐
│ Goose Session                                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│ User: create a Python script                       │
│                                                     │
│ Goose: I'll create a Python script for you.        │
│        What should it do?                          │
│                                                     │
│ [Working on: Creating main.py...]                  │
│                                                     │
├─────────────────────────────────────────────────────┤
│ G❯ _                                                │
└─────────────────────────────────────────────────────┘
```

### 승인 프롬프트

```
┌─────────────────────────────────────────────────────┐
│ Tool Execution Request                              │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Goose wants to execute:                            │
│   Command: rm -rf /tmp/cache                       │
│                                                     │
│ This will delete files. Continue?                  │
│                                                     │
│   [y] Yes    [n] No    [a] Always    [v] View      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 파이프라인 & 자동화

### 1. 스크립트에서 사용

```bash
#!/bin/bash

# 자동화 스크립트
echo "Running Goose automation..."

# Recipe 실행
goose run --recipe setup.yaml

# 결과 확인
if [ $? -eq 0 ]; then
    echo "Setup complete!"
else
    echo "Setup failed!"
    exit 1
fi
```

### 2. CI/CD 통합

```yaml
# .github/workflows/goose.yml
name: Goose Code Review

on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install Goose
        run: |
          curl -fsSL https://github.com/block/goose/releases/latest/download/install.sh | bash

      - name: Run Review
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          goose run --recipe code-review.yaml
```

### 3. Recipe로 반복 작업

```yaml
# daily-tasks.yaml
name: "Daily Development Tasks"

steps:
  - name: "Pull latest changes"
    prompt: "git pull origin main"

  - name: "Run tests"
    prompt: "Run all tests and report any failures"

  - name: "Update dependencies"
    prompt: "Check for outdated dependencies and suggest updates"

  - name: "Lint code"
    prompt: "Run linter and fix auto-fixable issues"
```

```bash
# 매일 실행
goose run --recipe daily-tasks.yaml
```

---

## 성능 최적화

### 1. 캐시 활용

```bash
# 환경 변수로 캐시 설정
export GOOSE_CACHE_ENABLED=true
export GOOSE_CACHE_TTL=3600  # 1시간

goose session
```

### 2. 빠른 모델 사용

```bash
# 간단한 작업에 빠른 모델
goose session --provider openai --model gpt-5-mini
```

### 3. 병렬 처리

```yaml
# parallel-tasks.yaml
name: "Parallel Processing"
parallel: true

steps:
  - name: "Task 1"
    prompt: "..."

  - name: "Task 2"
    prompt: "..."

  - name: "Task 3"
    prompt: "..."
```

---

## 문제 해결

### 세션이 응답하지 않음

```bash
# 로그 확인
goose logs --follow

# 세션 재시작
Ctrl+C
goose session -r
```

### API 키 오류

```bash
# 설정 재설정
goose configure
> Configure Providers
> [Provider 재선택]
```

### Extension 오류

```bash
# Extension 비활성화
goose configure
> Toggle Extensions
> [문제 Extension 비활성화]

# 로그 확인
goose logs --level error
```

---

## 단축키

| 키 | 동작 |
|----|------|
| `Ctrl+C` | 세션 종료 |
| `Ctrl+D` | 입력 완료 (빈 줄에서) |
| `↑` / `↓` | 명령어 히스토리 |
| `Tab` | 자동 완성 (향후 지원) |

---

## 다음 단계

CLI 사용법을 마스터했다면, 다음 장에서는 Desktop 앱을 살펴봅니다.

*다음 글에서는 Goose Desktop 앱의 기능과 Electron 구조를 분석합니다.*
