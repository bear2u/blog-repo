---
layout: page
title: Superset 가이드
permalink: /superset-guide/
icon: fas fa-terminal
---

# Superset 완벽 가이드

> **CLI 코딩 에이전트를 위한 터보차지 터미널**

**Superset**은 Claude Code, OpenAI Codex CLI 등 CLI 기반 코딩 에이전트를 병렬로 실행하고, Git Worktree로 각 태스크를 격리하여 개발 워크플로우를 10배 향상시키는 데스크탑 앱입니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/superset-guide-01-intro/) | Superset이란? 주요 특징, 지원 에이전트 |
| 02 | [설치 및 시작](/blog-repo/superset-guide-02-installation/) | 빌드, 환경 설정, 요구사항 |
| 03 | [아키텍처 분석](/blog-repo/superset-guide-03-architecture/) | 모노레포 구조, 설계 원칙, 코딩 컨벤션 |
| 04 | [Electron 앱](/blog-repo/superset-guide-04-electron/) | Main/Renderer/Preload, IPC 시스템 |
| 05 | [Workspace & Worktree](/blog-repo/superset-guide-05-workspace/) | Git Worktree 격리, 설정/정리 스크립트 |
| 06 | [터미널 관리](/blog-repo/superset-guide-06-terminal/) | node-pty, xterm.js, 탭/패인 관리 |
| 07 | [tRPC 라우터](/blog-repo/superset-guide-07-trpc/) | 타입 안전 API, 프로시저 구조 |
| 08 | [MCP 서버](/blog-repo/superset-guide-08-mcp/) | 태스크/디바이스 관리 도구 |
| 09 | [UI 컴포넌트](/blog-repo/superset-guide-09-ui/) | React, Zustand, shadcn/ui |
| 10 | [확장 및 커스터마이징](/blog-repo/superset-guide-10-customization/) | 설정, 프리셋, 훅, IDE 통합 |

---

## 주요 특징

- **병렬 실행** - 10개 이상의 코딩 에이전트를 동시에 실행
- **Worktree 격리** - 각 태스크가 자체 브랜치와 작업 디렉토리를 가짐
- **에이전트 모니터링** - 상태 추적 및 변경 준비 시 알림
- **내장 Diff 뷰어** - 에이전트 변경사항 즉시 검토 및 편집
- **워크스페이스 프리셋** - 환경 설정 자동화
- **유니버설 호환성** - 터미널에서 실행되는 모든 CLI 에이전트 지원

---

## 빠른 시작

```bash
# 사전 빌드 버전 다운로드 (권장)
# https://github.com/superset-sh/superset/releases/latest

# 또는 소스에서 빌드
git clone https://github.com/superset-sh/superset.git
cd superset

# 환경 설정
cp .env.example .env
# 또는 빠른 테스트: export SKIP_ENV_VALIDATION=1

# 설치 및 실행
bun install
bun run dev

# 데스크탑 앱 빌드
bun run build
open apps/desktop/release
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
│   └──────────────┘     └──────────────┘     └────────────┘  │
│          │                    │                    │         │
│          └────────────────────┼────────────────────┘         │
│                               │                              │
│                    ┌──────────┴──────────┐                   │
│                    │     Packages        │                   │
│                    │  ui | db | trpc     │                   │
│                    │  mcp | auth | ...   │                   │
│                    └─────────────────────┘                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 기술 | 용도 |
|------|------|
| Electron | 데스크탑 앱 프레임워크 |
| React | UI 프레임워크 |
| TailwindCSS v4 | 스타일링 |
| Bun + Turborepo | 패키지 관리 & 빌드 |
| tRPC | 타입 안전 API |
| Drizzle ORM | 데이터베이스 |
| node-pty | 터미널 관리 |
| xterm.js | 터미널 UI |

---

## 관련 링크

- [GitHub 저장소](https://github.com/superset-sh/superset)
- [공식 문서](https://docs.superset.sh)
- [Discord 커뮤니티](https://discord.gg/cZeD9WYcV7)
- [Twitter/X](https://x.com/superset_sh)

