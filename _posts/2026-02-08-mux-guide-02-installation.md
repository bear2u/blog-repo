---
layout: post
title: "Mux 완벽 가이드 (02) - 설치 및 시작"
date: 2026-02-08 00:00:00 +0900
categories: [AI 코딩, 개발 도구]
tags: [Mux, Electron, AI, 코딩 에이전트, 설치, 시작하기, 환경설정, API키]
author: cataclysm99
original_url: "https://github.com/coder/mux"
excerpt: "Mux 데스크톱 앱 설치부터 API 키 설정, 첫 프로젝트 추가까지 단계별 가이드"
permalink: /mux-guide-02-installation/
toc: true
related_posts:
  - /blog-repo/2026-02-08-mux-guide-01-introduction
  - /blog-repo/2026-02-08-mux-guide-03-workspaces
---

## 시스템 요구사항

Mux는 Electron 기반 데스크톱 애플리케이션으로, 다음 플랫폼을 지원합니다:

| 플랫폼 | 최소 요구사항 | 권장 사양 |
|--------|--------------|----------|
| **macOS** | macOS 10.15+ (Intel/Apple Silicon) | macOS 12+ |
| **Linux** | 64-bit, glibc 2.27+ | Ubuntu 20.04+, Fedora 32+ |
| **Windows** | Windows 10 64-bit | Windows 11 64-bit |
| **Node.js** | v20+ (CLI 사용 시) | v20 LTS |
| **메모리** | 4GB RAM | 8GB+ RAM |
| **디스크** | 500MB 여유 공간 | 2GB+ (프로젝트 포함) |

### 주요 의존성

- **Git**: 모든 플랫폼에서 필수 (worktree 런타임 사용 시)
- **SSH**: 원격 런타임 사용 시 필요
- **Docker**: Docker 런타임 사용 시 필요
- **Windows 사용자**: Git for Windows 필수 설치 (WSL 미지원)

```bash
# Git 버전 확인
git --version

# SSH 확인
ssh -V
```

---

## macOS 설치

### 1. DMG 파일 다운로드

[GitHub Releases](https://github.com/coder/mux/releases)에서 최신 버전 다운로드:

- **Intel Mac**: `macos-dmg-x64` 또는 `Mux-x.x.x-x64.dmg`
- **Apple Silicon (M1/M2/M3)**: `macos-dmg-arm64` 또는 `Mux-x.x.x-arm64.dmg`

### 2. 앱 설치

```bash
# 1. DMG 파일 열기
open Mux-*.dmg

# 2. Mux.app을 Applications 폴더로 드래그

# 3. 실행
open /Applications/Mux.app
```

### 3. 서명 및 공증

Mux는 Apple의 공식 서명 및 공증을 거쳐 보안 경고 없이 실행됩니다.

#### 개발 빌드 사용 시 (PR/브랜치 테스트)

```bash
# Gatekeeper 우회 (main 브랜치 외 빌드)
xattr -cr /Applications/Mux.app
codesign --force --deep --sign - /Applications/Mux.app
```

> **참고**: main 브랜치 빌드만 공식 서명됩니다.

---

## Linux 설치

### AppImage (권장)

```bash
# 1. AppImage 다운로드
wget https://github.com/coder/mux/releases/latest/download/Mux-*.AppImage

# 2. 실행 권한 부여
chmod +x Mux-*.AppImage

# 3. 실행
./Mux-*.AppImage
```

### 시스템 통합 (선택사항)

```bash
# 데스크톱 항목 생성
cat > ~/.local/share/applications/mux.desktop <<EOF
[Desktop Entry]
Type=Application
Name=Mux
Exec=/path/to/Mux-*.AppImage
Icon=/path/to/icon.png
Terminal=false
Categories=Development;
EOF

# 아이콘 추출 (AppImage 내부)
./Mux-*.AppImage --appimage-extract
cp squashfs-root/resources/app.asar.unpacked/dist/icon.png ~/.local/share/icons/mux.png
```

---

## Windows 설치 (Alpha)

> **경고**: Windows 지원은 현재 알파 단계입니다.

### 사전 준비

1. **Git for Windows 설치** (필수)
   ```powershell
   # Chocolatey로 설치
   choco install git

   # 또는 수동 다운로드
   # https://git-scm.com/download/win
   ```

2. **Mux 재시작** (Git 설치 후)

### 설치

```powershell
# 1. 설치 파일 다운로드
# https://github.com/coder/mux/releases/latest
# mux-x.x.x-x64.exe

# 2. 설치 프로그램 실행
.\mux-x.x.x-x64.exe

# 3. 시작 메뉴 또는 바탕화면에서 실행
```

### 알려진 제한사항

- WSL 미지원 (Git Bash 사용)
- SSH 런타임 안정성 제한적
- 일부 터미널 기능 제한

---

## CLI via npm (선택사항)

데스크톱 앱 없이 CLI만 사용하려면:

```bash
# npx로 즉시 실행 (설치 불필요)
npx mux run "Fix the failing tests"

# 글로벌 설치
npm install -g mux

# 설치 후 사용
mux run "Add authentication"
mux server --port 3000
```

### CLI 사용 사례

- **CI/CD 파이프라인**: GitHub Actions, GitLab CI
- **배치 작업**: 스크립트 기반 자동화
- **원격 서버**: SSH/Docker 런타임

---

## 초기 설정

### 1. 프로바이더 API 키 설정

첫 실행 시 Settings에서 API 키를 구성합니다:

```
Settings (⌘+, / Ctrl+,) → Providers
```

#### 지원되는 프로바이더

| 프로바이더 | 모델 | API 키 발급 |
|-----------|------|------------|
| **Anthropic** | Claude Opus 4.6, Sonnet 4.5, Haiku 4.5 | [console.anthropic.com](https://console.anthropic.com/) |
| **OpenAI** | GPT-5.2, Codex | [platform.openai.com](https://platform.openai.com/) |
| **Google** | Gemini 3 Pro/Flash | [aistudio.google.com](https://aistudio.google.com/) |
| **xAI** | Grok 4.1, Grok Code | [console.x.ai](https://console.x.ai/) |
| **DeepSeek** | DeepSeek Chat, Reasoner | [platform.deepseek.com](https://platform.deepseek.com/) |
| **OpenRouter** | 300+ 모델 | [openrouter.ai](https://openrouter.ai/) |
| **Ollama** | 로컬 LLM | 로컬 설치 (키 불필요) |

#### UI에서 설정

1. **Settings** 열기: `Cmd+,` (macOS) / `Ctrl+,` (Windows/Linux)
2. **Providers** 탭 선택
3. 원하는 프로바이더 확장
4. API 키 입력
5. 자동 유효성 검증

#### 환경 변수 (대안)

```bash
# ~/.bashrc 또는 ~/.zshrc
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export XAI_API_KEY="sk-xai-..."
```

#### 설정 파일 (고급)

```bash
# ~/.mux/providers.jsonc
{
  "anthropic": {
    "apiKey": "sk-ant-...",
    "baseUrl": "https://api.anthropic.com"  // 선택사항
  },
  "openai": {
    "apiKey": "sk-...",
    "orgId": "org-..."  // 선택사항
  },
  "ollama": {
    "baseUrl": "http://localhost:11434/api"
  }
}
```

### 2. 모델 선택

기본 모델은 `Claude Opus 4.6`입니다.

```
채팅 입력창 하단의 모델 pill 클릭
또는
Command Palette (⌘+Shift+P) → "Change Model"
```

#### 모델 전환 단축키

- **macOS**: `Cmd+/`
- **Windows/Linux**: `Ctrl+/`

#### 커스텀 모델 사용

```bash
# Command Palette에서
/model <provider:model_id>

# 예시
/model anthropic:claude-sonnet-4-5
/model openai:gpt-5.2-codex
/model ollama:llama3.1:70b
```

### 3. Ollama 로컬 LLM 설정 (선택사항)

```bash
# 1. Ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# 2. 모델 다운로드
ollama pull llama3.1:70b
ollama pull deepseek-coder:33b

# 3. Mux에서 사용
# Settings → Providers → Ollama
# Base URL: http://localhost:11434/api (기본값)

# 4. 모델 선택
/model ollama:llama3.1:70b
```

---

## 첫 프로젝트 추가

### 방법 1: 드래그 앤 드롭

1. Finder/탐색기에서 프로젝트 폴더 선택
2. Mux 좌측 사이드바로 드래그

### 방법 2: 메뉴 사용

```
File → Add Project → 디렉토리 선택
```

### 방법 3: Command Palette

```
⌘+Shift+P / Ctrl+Shift+P
→ "Add Project"
→ 디렉토리 선택
```

### 프로젝트 구조 예시

```
my-project/
├── .git/                 # Git 저장소 (worktree 런타임용)
├── .mux/
│   ├── agents/          # 커스텀 에이전트 정의
│   └── init             # 워크스페이스 초기화 훅
├── AGENTS.md            # 프로젝트 지침
├── src/
├── tests/
└── package.json
```

---

## 첫 워크스페이스 생성

### 워크스페이스란?

- 독립적인 채팅 세션
- 프로젝트별 격리된 실행 환경
- 병렬 작업 지원

### 생성 방법

#### 1. 좌측 사이드바에서

```
프로젝트 이름 클릭
→ 우측 "+" 버튼
→ 워크스페이스 이름 입력
→ 런타임 선택 (Local/Worktree/SSH)
```

#### 2. Command Palette

```
⌘+Shift+P
→ "New Workspace"
→ 프로젝트 선택
→ 설정
```

### 런타임 선택 가이드

| 런타임 | 사용 사례 | 격리 수준 | Git 필요 |
|--------|---------|----------|---------|
| **Local** | 빠른 일회성 작업 | 없음 (작업 디렉토리 직접 수정) | 선택사항 |
| **Worktree** | 병렬 기능 개발 | 파일시스템 격리 (Git 공유) | 필수 |
| **SSH** | 원격 서버 작업 | 완전 격리 | 원격 서버 |

### 워크스페이스 이름 규칙

```
feature-auth-x7k2    # 기능명 + 랜덤 접미사
fix-bug-p3m9         # 버그 수정
explore-arch-k1n4    # 탐색/분석
```

> **팁**: Mux는 자동으로 4자리 랜덤 접미사 추가 (충돌 방지)

---

## 빠른 시작 가이드

### 시나리오 1: 로컬 디렉토리에서 간단한 작업

```bash
# 1. 프로젝트 디렉토리로 이동
cd ~/projects/my-app

# 2. Mux 실행
open -a Mux  # macOS
./Mux.AppImage  # Linux
# Windows: 시작 메뉴에서 실행

# 3. 프로젝트 추가 (드래그 또는 Add Project)

# 4. Local 워크스페이스 생성
"quick-fix"

# 5. 채팅 시작
"Fix the TypeScript errors in src/utils.ts"
```

### 시나리오 2: 병렬 기능 개발 (Worktree)

```bash
# 1. Git 저장소 확인
cd ~/projects/my-app
git status

# 2. Mux에서 프로젝트 추가

# 3. Worktree 워크스페이스 생성
"feature-auth"  # 런타임: Worktree

# 4. 두 번째 워크스페이스 생성
"feature-payment"  # 런타임: Worktree

# 5. 병렬 작업
# 워크스페이스 1: "Implement OAuth2 authentication"
# 워크스페이스 2: "Add Stripe payment integration"
```

### 시나리오 3: 원격 서버 작업 (SSH)

```bash
# 1. SSH 키 설정
ssh-add ~/.ssh/id_ed25519

# 2. ~/.ssh/config 설정
cat >> ~/.ssh/config <<EOF
Host staging-server
  HostName 192.168.1.100
  User deploy
  IdentityFile ~/.ssh/id_ed25519
EOF

# 3. Mux에서 프로젝트 추가

# 4. SSH 워크스페이스 생성
# 런타임: SSH
# Host: staging-server

# 5. 원격 작업
"Update production database schema"
```

---

## 첫 대화 시작

### Plan 모드 (계획 수립)

```
⌘+Shift+M / Ctrl+Shift+M → Plan 모드 선택

사용자: "Add user authentication with JWT"

에이전트:
1. 저장소 분석 (기존 인증 코드 확인)
2. 계획 파일 작성 (~/.mux/plans/my-project/feature-auth-x7k2.md)
3. propose_plan 호출
4. 사용자 검토 → 승인/수정
5. Exec 모드 전환 → 구현
```

### Exec 모드 (즉시 실행)

```
⌘+Shift+M / Ctrl+Shift+M → Exec 모드 선택

사용자: "Fix the failing Jest tests"

에이전트:
1. 테스트 실행 (npm test)
2. 오류 분석
3. 파일 수정
4. 재검증
5. 커밋 (선택사항)
```

### Ask 모드 (질문 답변)

```
⌘+Shift+M / Ctrl+Shift+M → Ask 모드 선택

사용자: "Where is the database connection logic?"

에이전트:
1. Explore 서브에이전트 실행 (병렬)
2. 저장소 검색 (rg, file_read)
3. 결과 종합
4. 파일 경로 + 코드 스니펫 제공
```

---

## 초기 설정 체크리스트

### 필수 설정

- [ ] Mux 앱 설치 및 실행
- [ ] 최소 1개 프로바이더 API 키 설정
- [ ] 첫 프로젝트 추가
- [ ] 첫 워크스페이스 생성
- [ ] 모델 선택 (기본: Opus 4.6)

### 권장 설정

- [ ] 커맨드 팔레트 단축키 숙지 (`⌘+Shift+P`)
- [ ] 모드 전환 단축키 숙지 (`⌘+Shift+M`)
- [ ] Vim 모드 활성화 (선택사항, Settings → Vim Mode)
- [ ] Git 사용자 정보 확인 (`git config user.name`)
- [ ] 프로젝트 시크릿 설정 (API 키 등, 프로젝트 우클릭 → 🔑)

### 고급 설정

- [ ] Ollama 로컬 LLM 설치 (프라이버시 중시 시)
- [ ] SSH 원격 런타임 설정 (원격 서버 사용 시)
- [ ] VS Code 확장 설치 (워크스페이스 점프용)
- [ ] `.mux/init` 훅 작성 (의존성 자동 설치)
- [ ] `AGENTS.md` 작성 (프로젝트 지침)

---

## 다음 설정

### ~/.mux/config.json

```json
{
  "defaultModel": "anthropic:claude-opus-4-6",
  "defaultRuntime": "worktree",
  "telemetryEnabled": true,
  "vimMode": false,
  "theme": "dark"
}
```

### ~/.mux/providers.jsonc

```jsonc
{
  "anthropic": {
    "apiKey": "sk-ant-...",
    // "baseUrl": "https://api.anthropic.com"  // 선택사항
  },
  "openai": {
    "apiKey": "sk-...",
    // "orgId": "org-..."  // 조직 ID (선택사항)
  },
  "ollama": {
    "baseUrl": "http://localhost:11434/api"
  }
}
```

### 프로젝트별 설정: .mux/

```bash
my-project/
├── .mux/
│   ├── agents/
│   │   └── review.md      # 커스텀 에이전트
│   ├── init               # 초기화 훅 (chmod +x)
│   └── .muxignore         # 에이전트 무시 패턴
├── AGENTS.md              # 프로젝트 지침
└── AGENTS.local.md        # 개인 로컬 지침 (gitignore)
```

#### .mux/init 예시

```bash
#!/usr/bin/env bash
set -e

echo "Initializing workspace..."

# 의존성 설치
bun install

# 빌드
bun run build

# 테스트 실행 (선택사항)
# bun test

echo "Workspace ready!"
```

---

## 문제 해결

### API 키가 작동하지 않음

```bash
# 환경 변수 확인
echo $ANTHROPIC_API_KEY

# UI 설정 확인
Settings → Providers → [프로바이더] → API Key 재입력

# 유효성 검증
# UI에 녹색 체크 표시 확인
```

### 워크스페이스가 생성되지 않음

```bash
# Git 저장소 확인 (Worktree 런타임)
cd ~/projects/my-app
git status

# 초기화되지 않은 경우
git init

# 최소 1개 커밋 필요
git add .
git commit -m "Initial commit"
```

### SSH 런타임 연결 실패

```bash
# SSH 키 확인
ssh-add -l

# 수동 연결 테스트
ssh user@hostname

# ~/.ssh/config 설정
Host myserver
  HostName 192.168.1.100
  User deploy
  IdentityFile ~/.ssh/id_ed25519
```

### Windows: Git 미감지

```powershell
# Git 설치 확인
git --version

# PATH 확인
echo $env:PATH | Select-String "Git"

# Mux 재시작 (필수)
```

### 로그 확인

```bash
# macOS
~/Library/Logs/Mux/main.log

# Linux
~/.config/Mux/logs/main.log

# Windows
%APPDATA%\Mux\logs\main.log
```

---

## 성능 최적화 팁

### 1. 컨텍스트 관리

```bash
# 자동 압축 활성화
Settings → Costs → Auto-Compact 설정 (70%)

# 수동 압축
/compact  # AI 요약
/truncate # 단순 잘라내기
/clear    # 전체 삭제
```

### 2. 모델 선택 전략

| 작업 유형 | 권장 모델 | 이유 |
|----------|----------|------|
| 복잡한 리팩토링 | Opus 4.6 | 최고 품질 |
| 일반 코딩 | Sonnet 4.5 | 균형 (속도+품질) |
| 빠른 수정 | Haiku 4.5 | 최고 속도 |
| 로컬 작업 | Ollama (DeepSeek-Coder) | 프라이버시 |

### 3. 런타임 선택

```
Local: 빠른 일회성 작업
Worktree: 병렬 기능 개발 (파일 충돌 없음)
SSH: CPU 집약적 작업 (원격 서버 활용)
```

---

## 다음 단계

설치가 완료되었다면:

1. **[챕터 03: 워크스페이스 관리](/blog-repo/mux-guide-03-workspaces)** - Local/Worktree/SSH 런타임 심화
2. **[챕터 04: 에이전트 시스템](/blog-repo/mux-guide-04-agents)** - Plan/Exec 모드, 서브에이전트
3. **[챕터 05: 멀티모델 지원](/blog-repo/mux-guide-05-multimodel)** - 모델별 특징 및 비용 최적화

---

## 참고 자료

- [공식 설치 문서](https://mux.coder.com/install)
- [GitHub Releases](https://github.com/coder/mux/releases)
- [프로바이더 설정 가이드](https://mux.coder.com/config/providers)
- [CLI 참조](https://mux.coder.com/reference/cli)
- [Discord 커뮤니티](https://discord.gg/thkEdtwm8c)
