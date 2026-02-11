---
layout: post
title: "Goose 완벽 가이드 (02) - 설치 및 시작"
date: 2026-02-11
permalink: /goose-guide-02-installation/
author: Block
categories: [AI 에이전트, 개발 도구]
tags: [Goose, Installation, Setup, LLM, Configuration]
original_url: "https://github.com/block/goose"
excerpt: "Goose 설치부터 첫 세션까지 완벽 가이드"
---

## 설치 방법

Goose는 **Desktop 앱**과 **CLI** 두 가지 인터페이스를 제공합니다. 원하는 방식을 선택하거나 둘 다 설치할 수 있습니다.

---

## macOS 설치

### Desktop 앱 설치

#### Intel Mac
```bash
# Intel 프로세서용 다운로드
curl -L -o goose-desktop-macos-x64.zip \
  https://github.com/block/goose/releases/latest/download/goose-desktop-macos-x64.zip

# 압축 해제
unzip goose-desktop-macos-x64.zip

# 실행
./Goose.app
```

#### Apple Silicon (M1/M2/M3)
```bash
# ARM 프로세서용 다운로드
curl -L -o goose-desktop-macos-arm64.zip \
  https://github.com/block/goose/releases/latest/download/goose-desktop-macos-arm64.zip

# 압축 해제
unzip goose-desktop-macos-arm64.zip

# 실행
open Goose.app
```

### CLI 설치

```bash
# 자동 설치 스크립트
curl -fsSL https://github.com/block/goose/releases/latest/download/install.sh | bash

# 설치 확인
goose --version
```

---

## Linux 설치

### Desktop 앱 설치 (Debian/Ubuntu)

```bash
# DEB 패키지 다운로드
curl -L -o goose-desktop.deb \
  https://github.com/block/goose/releases/latest/download/goose-desktop-linux-x64.deb

# 설치
sudo dpkg -i goose-desktop.deb

# 의존성 문제 해결 (필요시)
sudo apt-get install -f

# 앱 메뉴에서 실행
```

### CLI 설치

```bash
# 자동 설치 스크립트
curl -fsSL https://github.com/block/goose/releases/latest/download/install.sh | bash

# 설치 확인
goose --version
```

---

## Windows 설치

### Desktop 앱 설치

```powershell
# 다운로드 (PowerShell)
Invoke-WebRequest -Uri "https://github.com/block/goose/releases/latest/download/goose-desktop-windows-x64.zip" `
  -OutFile "goose-desktop.zip"

# 압축 해제
Expand-Archive -Path goose-desktop.zip -DestinationPath .

# 실행
.\Goose.exe
```

### CLI 설치

**Git Bash 또는 MSYS2에서 실행:**

```bash
curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash
```

**PowerShell에서 실행:**

```powershell
curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash
```

#### PATH 추가 (Windows)

설치 후 PATH 경고가 표시되면:

```powershell
# PowerShell에서 PATH 추가
$env:Path += ";$HOME\.goose\bin"

# 영구적으로 추가
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";$HOME\.goose\bin", "User")
```

---

## LLM 제공자 설정

Goose를 사용하려면 LLM 제공자 설정이 필요합니다.

### Desktop 앱에서 설정

1. Goose Desktop 실행
2. 환영 화면에서 제공자 선택
3. **Tetrate Agent Router** (추천) 선택
4. 브라우저에서 인증 완료

```
💡 무료 크레딧: $10 제공 (신규/기존 사용자 모두)
```

### CLI에서 설정

```bash
# 설정 시작
goose configure

# 메뉴에서 선택
# 1. Configure Providers
```

---

## 주요 LLM 제공자

### 1. Tetrate Agent Router (추천)

**장점:**
- 여러 AI 모델 접근
- 자동 failover
- 내장 rate limiting
- $10 무료 크레딧

**설정:**

```bash
goose configure
> Configure Providers
> Tetrate Agent Router Service

# 브라우저에서 자동 인증
```

### 2. Anthropic (Claude)

**설정:**

```bash
goose configure
> Configure Providers
> Anthropic

# API 키 입력
ANTHROPIC_API_KEY: sk-ant-...

# 모델 선택
> claude-sonnet-4-5 (추천)
```

### 3. OpenAI

**설정:**

```bash
goose configure
> Configure Providers
> OpenAI

# API 키 입력
OPENAI_API_KEY: sk-proj-...

# 모델 선택
> gpt-5 (추천)
> gpt-4.1
```

### 4. GitHub Copilot

**설정:**

```bash
goose configure
> Configure Providers
> GitHub Copilot

# 인증 코드 자동 복사됨
# 브라우저에서 인증 완료
```

### 5. Azure OpenAI

**설정:**

```bash
goose configure
> Configure Providers
> Azure OpenAI

# 설정 입력
AZURE_OPENAI_API_KEY: ...
AZURE_OPENAI_ENDPOINT: https://...
AZURE_OPENAI_DEPLOYMENT: ...
```

---

## 모델 선택 가이드

| 모델 | 용도 | 비용 | 속도 |
|------|------|------|------|
| **GPT-5** | 복잡한 작업, 최고 품질 | 높음 | 중간 |
| **Claude Sonnet 4.5** | 균형잡힌 성능 | 중간 | 빠름 |
| **GPT-5-mini** | 간단한 작업, 빠른 응답 | 낮음 | 매우 빠름 |
| **Gemini 2.5 Pro** | 멀티모달, 긴 컨텍스트 | 중간 | 빠름 |

---

## 첫 세션 시작

### Desktop 앱

1. 설정 완료 후 `Home` 버튼 클릭
2. 입력 필드에 요청 입력
3. Enter로 전송

```
create a simple todo app in Python
```

### CLI

#### 일반 세션

```bash
# 작업 디렉토리로 이동
cd ~/projects/my-project

# 세션 시작
goose session

# 프롬프트 입력
G❯ create a REST API using FastAPI
```

#### 웹 인터페이스 (CLI 사용자)

```bash
# 웹 인터페이스로 시작
goose web --open

# 브라우저가 자동으로 열림
# http://localhost:8080
```

---

## 기본 명령어

### 세션 관리

```bash
# 새 세션 시작
goose session

# 마지막 세션 재개
goose session -r

# 특정 세션 재개
goose session --resume <session-id>

# 세션 목록
goose session --list
```

### 설정 관리

```bash
# 설정 메뉴
goose configure

# 제공자 변경
goose configure
> Configure Providers

# 확장 추가
goose configure
> Add Extension

# 확장 토글
goose configure
> Toggle Extensions
```

### 진단

```bash
# 시스템 정보
goose doctor

# 로그 확인
goose logs

# 버전 확인
goose --version
```

---

## 첫 프로젝트 만들기

### 예제 1: Tic-Tac-Toe 게임

```bash
# 빈 디렉토리 생성
mkdir goose-demo
cd goose-demo

# 세션 시작
goose session

# 요청 입력
G❯ create an interactive browser-based tic-tac-toe game
   in javascript where a player competes against a bot
```

**Goose가 자동으로:**
1. HTML 파일 생성
2. JavaScript 게임 로직 작성
3. CSS 스타일 추가
4. 파일 저장

### 예제 2: REST API

```bash
goose session

G❯ create a REST API in Python using FastAPI with:
   - User CRUD operations
   - SQLite database
   - Authentication
   - API documentation
```

**Goose가 자동으로:**
1. 프로젝트 구조 생성
2. 의존성 파일 생성 (requirements.txt)
3. 모델 및 라우터 구현
4. 인증 시스템 추가
5. 테스트 작성

---

## 확장(Extension) 추가

확장을 추가하면 Goose의 기능을 확장할 수 있습니다.

### Computer Controller 추가 (추천)

**Desktop:**
1. 사이드바에서 `Extensions` 클릭
2. `Computer Controller` 토글
3. 세션으로 돌아가기

**CLI:**

```bash
# 설정 메뉴
goose configure
> Add Extension
> Built-in Extension
> Computer Controller

# Timeout 설정
Timeout (seconds): 300
```

**활용 예시:**

```bash
G❯ open the tic-tac-toe game in a browser
```

Goose가 자동으로 브라우저를 열고 HTML 파일을 로드합니다.

### Developer Extension

개발 도구 접근 권한을 제공합니다.

```bash
goose configure
> Add Extension
> Built-in Extension
> Developer

# 액세스 제어 설정
Allow shell commands: Yes
Allow file operations: Yes
```

---

## 설정 파일 위치

### macOS/Linux

```bash
~/.config/goose/
├── config.yaml          # 설정
├── providers.yaml       # LLM 제공자
├── extensions.yaml      # 확장
└── sessions/            # 세션 히스토리
```

### Windows

```powershell
%APPDATA%\goose\
├── config.yaml
├── providers.yaml
├── extensions.yaml
└── sessions\
```

---

## 고급 설정

### 멀티 모델 구성

다른 작업에 다른 모델을 사용할 수 있습니다:

```yaml
# ~/.config/goose/providers.yaml
providers:
  - name: primary
    type: anthropic
    model: claude-sonnet-4-5

  - name: fast
    type: openai
    model: gpt-5-mini

  - name: powerful
    type: openai
    model: gpt-5
```

**사용:**

```bash
# 기본 모델로 시작
goose session

# 빠른 모델 사용
goose session --provider fast

# 강력한 모델 사용
goose session --provider powerful
```

### 프로젝트별 힌트 설정

```bash
# 프로젝트 루트에 .goosehints 생성
cat > .goosehints << 'EOF'
# Goose Hints for this project

## Coding Standards
- Use Python 3.11+
- Follow PEP 8
- Write type hints
- Add docstrings

## Testing
- Use pytest
- Maintain >80% coverage

## Git
- Conventional commits
- Branch naming: feature/*, bugfix/*
EOF
```

---

## 문제 해결

### Windows: PATH 경고

```powershell
# 영구적으로 PATH 추가
[Environment]::SetEnvironmentVariable(
    "Path",
    $env:Path + ";$HOME\.goose\bin",
    "User"
)

# 터미널 재시작
```

### Windows: Keyring 오류

```bash
# Keyring 사용 안 함 선택
goose configure
> Do not store to keyring
```

### 모델 로드 실패

```bash
# 설정 재설정
goose configure
> Configure Providers

# API 키 재입력
```

### 느린 응답

```bash
# 더 빠른 모델로 변경
goose configure
> Configure Providers
> Select faster model (gpt-5-mini, claude-haiku)
```

---

## 다음 단계

설치와 설정을 완료했다면, 다음 장에서는 Goose의 내부 아키텍처를 깊이 있게 살펴봅니다.

*다음 글에서는 Goose의 아키텍처와 내부 구조를 상세히 분석합니다.*
