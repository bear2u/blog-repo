---
layout: post
title: "Entire CLI 완벽 가이드 (02) - 설치 및 시작하기"
date: 2026-02-11
permalink: /entire-cli-guide-02-installation/
author: Entire Team
categories: [AI 코딩, 개발 도구]
tags: [Entire, CLI, Installation, Setup, Quick Start]
original_url: "https://github.com/entireio/cli"
excerpt: "Entire CLI 설치부터 첫 세션 시작까지 단계별 가이드"
---

## 요구사항

Entire CLI를 사용하기 전에 다음 요구사항을 확인하세요.

### 필수 요구사항

| 항목 | 요구사항 | 확인 명령 |
|-----|---------|----------|
| **Git** | 최신 버전 | `git --version` |
| **운영체제** | macOS, Linux, Windows (WSL) | `uname -s` |
| **AI 에이전트** | Claude Code 또는 Gemini CLI | `claude --version` |

### Claude Code 설치

Entire는 Claude Code와 함께 사용할 때 가장 강력합니다.

```bash
# Claude Code 설치 (Homebrew)
brew install anthropics/claude/claude

# 인증
claude auth login
```

---

## 설치 방법

### 방법 1: Homebrew (권장)

macOS와 Linux에서 가장 간단한 방법입니다.

```bash
# Entire tap 추가
brew tap entireio/tap

# Entire CLI 설치
brew install entireio/tap/entire

# 설치 확인
entire version
```

**출력 예시:**
```
entire version v0.1.0
commit: a3f2b1c
```

### 방법 2: Go Install

Go가 설치되어 있다면 직접 빌드할 수 있습니다.

```bash
# 최신 버전 설치
go install github.com/entireio/cli/cmd/entire@latest

# PATH 확인
which entire

# 버전 확인
entire version
```

**참고:** `$GOPATH/bin`이 PATH에 포함되어 있어야 합니다.

```bash
# PATH에 추가 (필요시)
echo 'export PATH="$GOPATH/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### 방법 3: 소스에서 빌드

개발자나 최신 기능을 사용하고 싶다면 직접 빌드할 수 있습니다.

```bash
# 레포지토리 클론
git clone https://github.com/entireio/cli.git
cd cli

# mise 설치 (빌드 도구)
curl https://mise.run | sh

# 의존성 설치
mise install

# 빌드
mise run build

# 바이너리 이동
mv entire /usr/local/bin/

# 확인
entire version
```

---

## 첫 번째 프로젝트 설정

### 1. 프로젝트로 이동

```bash
cd your-project

# Git 저장소 확인
git status
```

**중요:** Entire는 Git 저장소에서만 작동합니다.

### 2. Entire 활성화

```bash
entire enable
```

**이 명령이 하는 일:**

1. **Git 훅 설치**
   - `prepare-commit-msg` - 커밋 메시지에 Checkpoint ID 추가
   - `post-commit` - 세션 메타데이터 저장
   - `pre-push` - `entire/checkpoints/v1` 브랜치 푸시

2. **AI 에이전트 훅 설정**
   - Claude Code의 `.claude/settings.json` 업데이트
   - SessionStart, UserPromptSubmit, Stop 훅 추가

3. **디렉토리 생성**
   - `.entire/` - 설정 및 임시 파일
   - `.entire/logs/` - 로그 파일
   - `.git/entire-sessions/` - 세션 상태 (공유)

**출력 예시:**
```
✓ Git hooks installed
✓ Claude Code hooks configured
✓ Entire enabled with strategy: manual-commit
```

### 3. 상태 확인

```bash
entire status
```

**출력 예시:**
```
Strategy: manual-commit
Status: No active session
Branch: main
Checkpoints: 0
```

---

## 첫 세션 만들기

### 1. Claude Code 시작

```bash
claude
```

### 2. 프롬프트 입력

```
Add a simple README.md file with project description
```

Claude가 파일을 생성하고 작업을 완료합니다.

### 3. 상태 확인

```bash
entire status
```

**출력 예시:**
```
Strategy: manual-commit
Status: Active session
Session ID: 2026-02-11-abc123de...
Branch: main
Checkpoints: 1 (uncommitted)
Modified files: 1
```

### 4. 변경사항 커밋

```bash
git add README.md
git commit -m "Add README"
```

Entire가 자동으로:
- 세션 메타데이터를 `entire/checkpoints/v1`에 저장
- Checkpoint ID를 커밋 메시지에 추가
- Shadow 브랜치 정리

### 5. 세션 확인

```bash
# 최근 커밋 확인
git log -1 --pretty=fuller

# 커밋 설명
entire explain HEAD
```

**출력 예시:**
```
Commit: a3f2b1c4
Message: Add README
Checkpoint ID: a3b2c4d5e6f7

Session Details:
- Session ID: 2026-02-11-abc123de...
- Start time: 2026-02-11 10:30:00
- Prompts: 1
- Modified files: 1
- Transcript available: Yes
```

---

## 전략 선택

Entire는 두 가지 전략을 지원합니다. 프로젝트에 맞는 전략을 선택하세요.

### Manual-Commit (기본)

**특징:**
- 사용자가 직접 커밋 생성
- 깔끔한 Git 히스토리
- main 브랜치에서 안전

**활성화:**
```bash
entire enable --strategy manual-commit
```

**언제 사용:**
- 대부분의 프로젝트
- 깔끔한 커밋 히스토리 원할 때
- main/master 브랜치에서 직접 작업

### Auto-Commit

**특징:**
- AI 응답마다 자동 커밋
- 세밀한 체크포인트
- 자동 진행

**활성화:**
```bash
entire enable --strategy auto-commit
```

**언제 사용:**
- 자동 커밋을 원하는 팀
- 모든 단계를 기록하고 싶을 때
- Feature 브랜치에서 작업

---

## 추가 설정 옵션

### Gemini CLI 사용

```bash
entire enable --agent gemini
```

### 로컬 설정 (Git에 커밋 안 함)

```bash
entire enable --local
```

### 자동 푸시 비활성화

```bash
entire enable --skip-push-sessions
```

### 텔레메트리 비활성화

```bash
entire enable --telemetry=false
```

### 강제 재설치

```bash
entire enable --force
```

---

## 설정 파일

Entire는 두 개의 설정 파일을 사용합니다.

### settings.json (프로젝트 설정)

위치: `.entire/settings.json`

팀과 공유되며 Git에 커밋됩니다.

```json
{
  "strategy": "manual-commit",
  "agent": "claude-code",
  "enabled": true
}
```

### settings.local.json (개인 설정)

위치: `.entire/settings.local.json`

개인 설정이며 `.gitignore`에 포함됩니다.

```json
{
  "log_level": "debug",
  "telemetry": false
}
```

**우선순위:** local 설정이 project 설정을 덮어씁니다.

---

## 접근성 모드

스크린 리더 사용자를 위한 접근성 모드를 지원합니다.

```bash
# 환경 변수 설정
export ACCESSIBLE=1

# Entire 활성화
entire enable
```

이 모드는:
- 인터랙티브 TUI 대신 간단한 텍스트 프롬프트 사용
- 모든 명령에서 작동

---

## 문제 해결

### "Not a git repository"

```bash
# Git 저장소 초기화
git init
git add .
git commit -m "Initial commit"

# Entire 활성화
entire enable
```

### "Claude Code not found"

```bash
# Claude Code 설치 확인
which claude

# 없다면 설치
brew install anthropics/claude/claude
```

### 훅이 작동하지 않음

```bash
# Entire 재설치
entire disable
entire enable --force

# 훅 확인
ls -la .git/hooks/
```

### PATH 문제

```bash
# Entire 위치 확인
which entire

# PATH에 추가
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

---

## 설치 확인 체크리스트

설치가 제대로 되었는지 확인하세요:

- [ ] `entire version` 명령이 작동함
- [ ] Git 저장소에서 `entire enable` 실행 성공
- [ ] `.entire/` 디렉토리가 생성됨
- [ ] `entire status` 명령이 작동함
- [ ] Claude Code와 통합 완료
- [ ] 첫 세션 생성 및 커밋 완료

---

## 다음 단계

설치가 완료되었습니다! 다음 챕터에서는:

- **핵심 개념** - Session, Checkpoint, Strategy 깊이 이해
- **일반적인 워크플로우** - 실제 사용 시나리오와 패턴
- **명령어 레퍼런스** - 모든 명령어 상세 설명

---

*다음 글에서는 Entire의 핵심 개념인 Session, Checkpoint, Strategy를 자세히 살펴봅니다.*
