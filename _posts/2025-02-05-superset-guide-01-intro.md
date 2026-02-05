---
layout: post
title: "Superset 완벽 가이드 (1) - 소개 및 개요"
date: 2025-02-05
permalink: /superset-guide-01-intro/
author: Superset Team
categories: [AI 에이전트, Superset]
tags: [Superset, Claude Code, Coding Agent, Terminal, Electron, Worktree]
original_url: "https://github.com/superset-sh/superset"
excerpt: "CLI 코딩 에이전트를 병렬로 실행하는 터보차지 터미널 Superset을 소개합니다."
---

## Superset이란?

**Superset**은 **CLI 코딩 에이전트를 위한 터보차지 터미널**입니다. Claude Code, OpenAI Codex CLI 등 CLI 기반 코딩 에이전트를 병렬로 실행하고, Git Worktree를 활용해 각 태스크를 격리하여 개발 워크플로우를 10배 향상시킵니다.

```
┌─────────────────────────────────────────────┐
│            Superset의 핵심 가치             │
├─────────────────────────────────────────────┤
│  • 10개 이상 코딩 에이전트 병렬 실행        │
│  • Git Worktree로 태스크별 격리             │
│  • 에이전트 모니터링 및 알림                │
│  • 내장 Diff 뷰어 & 편집기                  │
│  • 원클릭 IDE 통합                          │
└─────────────────────────────────────────────┘
```

---

## 주요 특징

| 특징 | 설명 |
|------|------|
| **Parallel Execution** | 10개 이상의 코딩 에이전트를 동시에 실행 |
| **Worktree Isolation** | 각 태스크가 자체 브랜치와 작업 디렉토리를 가짐 |
| **Agent Monitoring** | 에이전트 상태 추적 및 변경사항 준비 시 알림 |
| **Built-in Diff Viewer** | 앱을 떠나지 않고 에이전트 변경사항 검토 및 편집 |
| **Workspace Presets** | 환경 설정, 의존성 설치 등 자동화 |
| **Universal Compatibility** | 터미널에서 실행되는 모든 CLI 에이전트 지원 |
| **Quick Context Switching** | 주의가 필요한 태스크 간 빠른 전환 |
| **IDE Integration** | 원클릭으로 즐겨쓰는 에디터에서 워크스페이스 열기 |

---

## 왜 Superset인가?

### 기존 방식의 한계

```
┌─────────────────────────────────────────────┐
│         기존 에이전트 사용의 문제점          │
├─────────────────────────────────────────────┤
│  ❌ 한 번에 하나의 에이전트만 실행 가능      │
│  ❌ 컨텍스트 스위칭 오버헤드 큼              │
│  ❌ 에이전트 간 작업 충돌 위험               │
│  ❌ 변경사항 추적 및 검토 어려움             │
│  ❌ 환경 설정 반복 작업                      │
└─────────────────────────────────────────────┘
```

### Superset의 해결책

```
┌─────────────────────────────────────────────┐
│          Superset이 제공하는 가치           │
├─────────────────────────────────────────────┤
│  ✅ 여러 에이전트를 동시에 병렬 실행        │
│  ✅ Worktree로 완벽한 격리 보장             │
│  ✅ 대시보드에서 모든 에이전트 상태 모니터링 │
│  ✅ 내장 Diff 뷰어로 즉시 변경사항 확인     │
│  ✅ 프리셋으로 환경 설정 자동화             │
└─────────────────────────────────────────────┘
```

---

## 지원 에이전트

| 에이전트 | 상태 |
|----------|------|
| [Claude Code](https://github.com/anthropics/claude-code) | 완벽 지원 |
| [OpenAI Codex CLI](https://github.com/openai/codex) | 완벽 지원 |
| [OpenCode](https://github.com/opencode-ai/opencode) | 완벽 지원 |
| 모든 CLI 에이전트 | 동작 가능 |

> **터미널에서 실행되면, Superset에서도 실행됩니다.**

---

## 시스템 요구사항

| 요구사항 | 상세 |
|----------|------|
| **OS** | macOS (Windows/Linux 미테스트) |
| **런타임** | [Bun](https://bun.sh/) v1.0+ |
| **버전 관리** | Git 2.20+ |
| **GitHub CLI** | [gh](https://cli.github.com/) |

---

## 빠른 시작

### 사전 빌드 버전 (권장)

**[macOS용 Superset 다운로드](https://github.com/superset-sh/superset/releases/latest)**

### 소스에서 빌드

```bash
# 1. 레포지토리 클론
git clone https://github.com/superset-sh/superset.git
cd superset

# 2. 환경 변수 설정 (옵션 선택)
# 옵션 A: 전체 설정
cp .env.example .env
# .env 파일 편집하여 값 입력

# 옵션 B: 환경 검증 건너뛰기 (빠른 로컬 테스트용)
export SKIP_ENV_VALIDATION=1

# 3. 의존성 설치 및 실행
bun install
bun run dev

# 4. 데스크탑 앱 빌드
bun run build
open apps/desktop/release
```

---

## 주요 단축키

### 워크스페이스 네비게이션

| 단축키 | 동작 |
|--------|------|
| `⌘1-9` | 워크스페이스 1-9로 전환 |
| `⌘⌥↑/↓` | 이전/다음 워크스페이스 |
| `⌘N` | 새 워크스페이스 |
| `⌘⇧N` | 빠른 워크스페이스 생성 |
| `⌘⇧O` | 프로젝트 열기 |

### 터미널

| 단축키 | 동작 |
|--------|------|
| `⌘T` | 새 탭 |
| `⌘W` | 패인/터미널 닫기 |
| `⌘D` | 오른쪽 분할 |
| `⌘⇧D` | 아래 분할 |
| `⌘K` | 터미널 클리어 |
| `⌘F` | 터미널 내 검색 |

### 레이아웃

| 단축키 | 동작 |
|--------|------|
| `⌘B` | 워크스페이스 사이드바 토글 |
| `⌘L` | 변경사항 패널 토글 |
| `⌘O` | 외부 앱에서 열기 |

---

## 기술 스택

```
┌─────────────────────────────────────────────┐
│              Superset 기술 스택             │
├─────────────────────────────────────────────┤
│  Electron    - 크로스 플랫폼 데스크탑 앱    │
│  React       - UI 프레임워크                │
│  TailwindCSS - 스타일링                     │
│  Bun         - 패키지 매니저 & 런타임       │
│  Turborepo   - 모노레포 빌드 시스템         │
│  Vite        - 프론트엔드 빌드 도구         │
│  Drizzle ORM - 데이터베이스 ORM             │
│  Neon        - PostgreSQL 호스팅            │
│  tRPC        - 타입 안전 API                │
│  Biome       - 린팅 & 포맷팅                │
└─────────────────────────────────────────────┘
```

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                       Superset                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐     ┌──────────────┐     ┌────────────┐  │
│   │   Desktop    │     │     Web      │     │    API     │  │
│   │  (Electron)  │     │  (Next.js)   │     │ (Next.js)  │  │
│   │              │     │              │     │            │  │
│   │  • Terminal  │     │  • Dashboard │     │  • tRPC    │  │
│   │  • Worktree  │     │  • Tasks     │     │  • MCP     │  │
│   │  • Diff View │     │  • Auth      │     │  • Sync    │  │
│   └──────┬───────┘     └──────┬───────┘     └─────┬──────┘  │
│          │                    │                    │         │
│          └────────────────────┼────────────────────┘         │
│                               │                              │
│                    ┌──────────┴──────────┐                   │
│                    │     Packages        │                   │
│                    ├────────────────────┤                    │
│                    │  ui     │ trpc     │                    │
│                    │  db     │ auth     │                    │
│                    │  mcp    │ shared   │                    │
│                    └─────────────────────┘                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 이 가이드에서 다루는 내용

| # | 제목 | 내용 |
|---|------|------|
| 01 | **소개 및 개요** (현재 글) | Superset이란? 주요 특징 |
| 02 | **설치 및 시작** | 빌드, 환경 설정, 요구사항 |
| 03 | **아키텍처 분석** | 모노레포 구조, 앱/패키지 |
| 04 | **Electron 앱** | main/renderer/preload 구조 |
| 05 | **Workspace & Worktree** | Git worktree 격리 시스템 |
| 06 | **터미널 관리** | node-pty, 병렬 실행 |
| 07 | **tRPC 라우터** | 내부 API 구조 |
| 08 | **MCP 서버** | 태스크/디바이스 관리 |
| 09 | **UI 컴포넌트** | React, stores, providers |
| 10 | **확장 및 커스터마이징** | 설정, 스크립트, 프리셋 |

---

## 라이선스

Superset은 **Apache 2.0 라이선스**로 배포됩니다.

---

## 커뮤니티

- **[Discord](https://discord.gg/cZeD9WYcV7)** - 팀 및 커뮤니티와 대화
- **[Twitter/X](https://x.com/superset_sh)** - 업데이트 및 발표
- **[GitHub Issues](https://github.com/superset-sh/superset/issues)** - 버그 리포트 및 기능 요청

---

*다음 글에서는 Superset의 설치 및 시작 방법을 살펴봅니다.*
