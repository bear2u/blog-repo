---
layout: post
title: "Superset 완벽 가이드 (2) - 설치 및 시작"
date: 2025-02-05
permalink: /superset-guide-02-installation/
author: Superset Team
categories: [AI 에이전트, Superset]
tags: [Superset, Installation, Bun, Turborepo, Electron]
original_url: "https://github.com/superset-sh/superset"
excerpt: "Superset을 설치하고 개발 환경을 구성하는 방법을 알아봅니다."
---

## 시스템 요구사항

Superset을 설치하기 전에 다음 요구사항을 확인하세요.

| 요구사항 | 버전 | 설명 |
|----------|------|------|
| **macOS** | 10.15+ | 현재 macOS만 공식 지원 |
| **Bun** | 1.0+ | 패키지 매니저 & 런타임 |
| **Git** | 2.20+ | Worktree 기능 필요 |
| **gh CLI** | 최신 | GitHub 연동용 |

---

## 사전 빌드 버전 설치 (권장)

가장 빠른 시작 방법은 공식 릴리즈를 다운로드하는 것입니다.

```bash
# GitHub 릴리즈 페이지에서 다운로드
open https://github.com/superset-sh/superset/releases/latest
```

또는 gh CLI로 직접 다운로드:

```bash
# 최신 릴리즈 다운로드
gh release download --repo superset-sh/superset --pattern "*.dmg"

# DMG 마운트 및 앱 설치
open Superset-*.dmg
```

---

## 소스에서 빌드

개발 또는 커스터마이징을 위해 소스에서 빌드할 수 있습니다.

### 1단계: 의존성 설치

```bash
# Bun 설치 (미설치 시)
curl -fsSL https://bun.sh/install | bash

# Git 버전 확인
git --version  # 2.20 이상 필요

# GitHub CLI 설치 (macOS)
brew install gh

# gh 인증
gh auth login
```

### 2단계: 레포지토리 클론

```bash
git clone https://github.com/superset-sh/superset.git
cd superset
```

### 3단계: 환경 변수 설정

```bash
# 환경 파일 복사
cp .env.example .env

# .env 파일 편집
nano .env
```

**필수 환경 변수:**

```bash
# .env 파일 내용 예시

# Database (Neon PostgreSQL)
DATABASE_URL="postgresql://user:pass@host/db"

# Authentication
AUTH_SECRET="your-auth-secret-key"
GITHUB_CLIENT_ID="your-github-client-id"
GITHUB_CLIENT_SECRET="your-github-client-secret"

# Optional: Skip validation for quick testing
# SKIP_ENV_VALIDATION=1
```

**빠른 로컬 테스트 (환경 검증 건너뛰기):**

```bash
# 환경 변수 없이 빠르게 테스트
export SKIP_ENV_VALIDATION=1
```

### 4단계: 의존성 설치 및 실행

```bash
# 의존성 설치
bun install

# 개발 서버 실행
bun run dev
```

### 5단계: 데스크탑 앱 빌드

```bash
# 프로덕션 빌드
bun run build

# 빌드된 앱 열기
open apps/desktop/release
```

---

## 개발 명령어

### 기본 명령어

```bash
# 전체 개발 서버 실행 (web + desktop + api)
bun dev

# 모든 앱 개발 서버
bun dev:all

# 특정 앱만 실행
bun dev:docs       # 문서 사이트
bun dev:cli        # CLI 도구
bun dev:marketing  # 마케팅 사이트
```

### 빌드 및 테스트

```bash
# 전체 빌드
bun build

# 테스트 실행
bun test

# 타입 체크
bun run typecheck
```

### 코드 품질

```bash
# 린트 검사 및 자동 수정
bun run lint:fix

# 포맷팅만
bun run format

# 포맷팅 검사 (CI용)
bun run format:check
```

### 데이터베이스

```bash
# 스키마 변경 적용
bun run db:push

# 데이터베이스 시드
bun run db:seed

# 마이그레이션 실행
bun run db:migrate

# Drizzle Studio 열기
bun run db:studio
```

### 정리

```bash
# root node_modules 정리
bun run clean

# 모든 워크스페이스 node_modules 정리
bun run clean:workspaces
```

---

## 환경 변수 상세 설명

### .env.example 파일 구조

```bash
# ===== Database =====
# Neon PostgreSQL 연결 URL
DATABASE_URL="postgresql://..."
NEON_ORG_ID="org_..."
NEON_PROJECT_ID="..."

# ===== Authentication =====
# NextAuth 시크릿 키
AUTH_SECRET="..."

# GitHub OAuth
GITHUB_CLIENT_ID="..."
GITHUB_CLIENT_SECRET="..."

# ===== Desktop App =====
# 포트 설정
DESKTOP_RENDERER_DEV_PORT=5177
DESKTOP_API_DEV_PORT=5178

# ===== API =====
# API 서버 설정
API_PORT=3001

# ===== Sentry (선택사항) =====
SENTRY_DSN="..."
SENTRY_AUTH_TOKEN="..."

# ===== Feature Flags =====
SKIP_ENV_VALIDATION=0
```

---

## 프로젝트 구조

설치 후 프로젝트 구조를 이해하면 개발에 도움이 됩니다.

```
superset/
├── apps/                    # 애플리케이션들
│   ├── desktop/            # Electron 데스크탑 앱
│   ├── web/                # 웹 앱 (app.superset.sh)
│   ├── api/                # API 백엔드
│   ├── marketing/          # 마케팅 사이트
│   ├── admin/              # 관리자 대시보드
│   ├── docs/               # 문서 사이트
│   ├── cli/                # CLI 도구
│   ├── mobile/             # 모바일 앱
│   └── streams/            # 스트리밍 서비스
│
├── packages/               # 공유 패키지들
│   ├── ui/                 # UI 컴포넌트 (shadcn/ui)
│   ├── db/                 # Drizzle ORM 스키마
│   ├── auth/               # 인증 모듈
│   ├── trpc/               # tRPC 설정
│   ├── mcp/                # MCP 서버
│   ├── shared/             # 공유 유틸리티
│   ├── local-db/           # 로컬 SQLite
│   ├── ai-chat/            # AI 채팅 모듈
│   ├── email/              # 이메일 템플릿
│   └── scripts/            # CLI 스크립트
│
├── tooling/                # 개발 도구
│   └── typescript-config/  # TypeScript 설정
│
├── scripts/                # 빌드 스크립트
├── docs/                   # 내부 문서
├── plans/                  # 계획 문서
│
├── package.json            # 루트 패키지 설정
├── turbo.jsonc             # Turborepo 설정
├── biome.jsonc             # Biome 린터 설정
├── bunfig.toml             # Bun 설정
└── .env.example            # 환경 변수 예시
```

---

## 문제 해결

### Bun 설치 실패

```bash
# Bun 재설치
curl -fsSL https://bun.sh/install | bash

# PATH 설정 확인
export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:$PATH"
```

### node_modules 충돌

```bash
# 전체 정리 후 재설치
bun run clean:workspaces
rm -rf node_modules
bun install
```

### 포트 충돌

```bash
# 사용 중인 포트 확인
lsof -i :5177
lsof -i :5178

# 프로세스 종료
kill -9 <PID>

# 또는 .env에서 포트 변경
DESKTOP_RENDERER_DEV_PORT=5277
```

### 데이터베이스 연결 실패

```bash
# DATABASE_URL 확인
echo $DATABASE_URL

# Neon 대시보드에서 연결 URL 확인
# https://console.neon.tech
```

### Electron 빌드 실패

```bash
# 캐시 정리
rm -rf apps/desktop/release
rm -rf apps/desktop/node_modules/.cache

# 재빌드
bun run build
```

---

## 개발 팁

### 1. Turbo 필터 사용

```bash
# 특정 앱만 실행
bun run dev --filter=@superset/desktop

# 의존성 포함 빌드
bun run build --filter=@superset/desktop...
```

### 2. 타입 검사 자주 실행

```bash
# 전체 타입 체크
bun run typecheck

# 특정 패키지만
cd packages/db && bun run typecheck
```

### 3. Biome로 코드 품질 유지

```bash
# 저장 시 자동 수정 (VSCode 설정)
# settings.json에 추가:
{
  "editor.codeActionsOnSave": {
    "quickfix.biome": true
  }
}
```

---

## 다음 단계

설치가 완료되면:

1. `bun dev`로 개발 서버 실행
2. 브라우저에서 앱 확인
3. 데스크탑 앱 빌드 후 실제 사용

---

*다음 글에서는 Superset의 전체 아키텍처를 분석합니다.*
