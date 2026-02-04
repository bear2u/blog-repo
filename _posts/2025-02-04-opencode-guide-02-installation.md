---
layout: post
title: "OpenCode 가이드 - 설치 가이드"
date: 2025-02-04
category: AI
tags: [opencode, installation, npm, homebrew, desktop-app]
series: opencode-guide
part: 2
author: anomalyco
original_url: https://github.com/anomalyco/opencode
---

## 설치 방법 개요

OpenCode는 다양한 방법으로 설치할 수 있습니다. 운영체제와 선호도에 따라 적합한 방법을 선택하세요.

## 패키지 매니저 설치

### npm (모든 플랫폼)

```bash
# npm 사용
npm i -g opencode-ai@latest

# bun 사용 (권장)
bun add -g opencode-ai@latest

# pnpm 사용
pnpm add -g opencode-ai@latest

# yarn 사용
yarn global add opencode-ai@latest
```

### Homebrew (macOS & Linux)

```bash
# anomalyco tap 사용 (항상 최신 버전)
brew install anomalyco/tap/opencode

# 공식 brew formula (업데이트 빈도 낮음)
brew install opencode
```

### Windows 패키지 매니저

```powershell
# Scoop
scoop install opencode

# Chocolatey
choco install opencode
```

### Linux 배포판

```bash
# Arch Linux (AUR)
paru -S opencode-bin

# 또는 yay 사용
yay -S opencode-bin
```

### mise (모든 OS)

```bash
mise use -g opencode
```

### Nix

```bash
# nixpkgs에서 직접 실행
nix run nixpkgs#opencode

# 최신 dev 브랜치
nix run github:anomalyco/opencode
```

## 스크립트 설치 (YOLO)

가장 간단한 설치 방법:

```bash
curl -fsSL https://opencode.ai/install | bash
```

### 설치 디렉토리 우선순위

스크립트는 다음 순서로 설치 경로를 결정합니다:

1. `$OPENCODE_INSTALL_DIR` - 커스텀 설치 디렉토리
2. `$XDG_BIN_DIR` - XDG 표준 경로
3. `$HOME/bin` - 표준 사용자 바이너리 디렉토리
4. `$HOME/.opencode/bin` - 기본 폴백

```bash
# 커스텀 경로 지정 예시
OPENCODE_INSTALL_DIR=/usr/local/bin curl -fsSL https://opencode.ai/install | bash

# XDG 표준 경로 사용
XDG_BIN_DIR=$HOME/.local/bin curl -fsSL https://opencode.ai/install | bash
```

## 데스크톱 앱 설치 (베타)

CLI 외에도 네이티브 데스크톱 앱을 제공합니다.

### 다운로드

[Releases 페이지](https://github.com/anomalyco/opencode/releases) 또는 [opencode.ai/download](https://opencode.ai/download)에서 직접 다운로드:

| 플랫폼 | 파일명 |
|--------|--------|
| macOS (Apple Silicon) | `opencode-desktop-darwin-aarch64.dmg` |
| macOS (Intel) | `opencode-desktop-darwin-x64.dmg` |
| Windows | `opencode-desktop-windows-x64.exe` |
| Linux | `.deb`, `.rpm`, 또는 AppImage |

### 패키지 매니저로 설치

```bash
# macOS (Homebrew Cask)
brew install --cask opencode-desktop

# Windows (Scoop)
scoop bucket add extras
scoop install extras/opencode-desktop
```

### 데스크톱 앱 요구사항

Tauri v2 기반으로 제작되어 다음이 필요합니다:

- **macOS**: 10.15 (Catalina) 이상
- **Windows**: Windows 10 이상
- **Linux**: GTK 3, WebKitGTK

## 업그레이드

### 이전 버전 제거

> **중요**: 0.1.x 버전이 설치되어 있다면 먼저 제거하세요.

```bash
# npm 제거
npm uninstall -g opencode-ai

# brew 제거
brew uninstall opencode

# 데이터 디렉토리 정리 (선택사항)
rm -rf ~/.opencode
rm -rf ~/.config/opencode
```

### 최신 버전으로 업그레이드

```bash
# npm
npm update -g opencode-ai

# brew
brew upgrade opencode
# 또는
brew upgrade anomalyco/tap/opencode
```

## 설치 확인

```bash
# 버전 확인
opencode --version

# 도움말 보기
opencode --help

# 첫 실행
opencode
```

## API 키 설정

OpenCode를 사용하려면 AI 프로바이더 API 키가 필요합니다.

### 환경 변수로 설정

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Google AI
export GOOGLE_API_KEY="..."

# Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://..."

# AWS Bedrock
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"
```

### 설정 파일로 지정

`~/.config/opencode/opencode.json`:

```json
{
  "provider": {
    "anthropic": {
      "apiKey": "sk-ant-..."
    }
  }
}
```

### OpenCode Zen 인증

유료 서비스인 OpenCode Zen을 사용하면 별도 API 키 없이 이용 가능합니다:

```bash
# 브라우저에서 인증
opencode auth
```

## 문제 해결

### PATH 설정

설치 후 `opencode` 명령을 찾을 수 없는 경우:

```bash
# bash/zsh
echo 'export PATH="$HOME/.opencode/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# fish
fish_add_path $HOME/.opencode/bin
```

### 권한 오류

```bash
# npm 전역 설치 권한 문제
sudo npm i -g opencode-ai@latest

# 또는 npm 디렉토리 소유권 변경
sudo chown -R $USER /usr/local/lib/node_modules
```

### Node.js 버전

Node.js 18 이상이 필요합니다:

```bash
# 버전 확인
node --version

# nvm으로 업그레이드
nvm install 20
nvm use 20
```

### 프록시 환경

프록시 뒤에서 사용하는 경우:

```bash
export HTTP_PROXY="http://proxy.example.com:8080"
export HTTPS_PROXY="http://proxy.example.com:8080"
```

## 개발 환경 설정

소스에서 직접 빌드하려면:

```bash
# 저장소 클론
git clone https://github.com/anomalyco/opencode
cd opencode

# 의존성 설치 (Bun 필요)
bun install

# 개발 모드 실행
bun run dev
```

### 필수 도구

- **Bun**: 1.3.5 이상
- **Turbo**: 빌드 시스템 (자동 설치)

## 다음 단계

설치가 완료되었다면, 다음 챕터에서 OpenCode의 아키텍처와 패키지 구조를 알아봅니다.

---

**이전 글**: [소개 및 주요 특징](/2025/02/04/opencode-guide-01-intro)

**다음 글**: [아키텍처](/2025/02/04/opencode-guide-03-architecture)
