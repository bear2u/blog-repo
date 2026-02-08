---
layout: page
title: Mux 가이드
permalink: /mux-guide/
icon: fas fa-layer-group
---

# Mux 완벽 가이드

> **병렬 에이전트 개발을 위한 코딩 멀티플렉서**

**Mux**는 Coder에서 개발한 데스크톱 & 브라우저 애플리케이션으로, 로컬 또는 원격 컴퓨팅에서 여러 AI 에이전트로 작업을 계획하고 실행할 수 있습니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/blog-repo/mux-guide-01-intro/) | 프로젝트 소개, 주요 특징, Claude Code 비교 |
| 02 | [설치 및 시작](/blog-repo/mux-guide-02-installation/) | 시스템 요구사항, 설치 방법, 초기 설정 |
| 03 | [워크스페이스 관리](/blog-repo/mux-guide-03-workspaces/) | Local/Worktree/SSH 런타임, Git 분기 추적 |
| 04 | [에이전트 시스템](/blog-repo/mux-guide-04-agents/) | Plan/Exec 모드, 병렬 워크플로우 |
| 05 | [멀티모델 지원](/blog-repo/mux-guide-05-multimodel/) | Claude/GPT/Grok, Ollama, OpenRouter |
| 06 | [VS Code 통합](/blog-repo/mux-guide-06-vscode-integration/) | 확장 설치, 워크스페이스 점프 |
| 07 | [고급 기능](/blog-repo/mux-guide-07-advanced-features/) | Compaction, Mode Prompts, Hooks |
| 08 | [개발 및 확장](/blog-repo/mux-guide-08-development/) | 개발 환경, 테스트, 커스터마이징 |

---

## 주요 특징

### 🔀 격리된 워크스페이스

- **Local** - 프로젝트 디렉토리에서 직접 실행
- **Worktree** - Git worktree로 병렬 개발
- **SSH** - 원격 서버에서 에이전트 실행

```
Project Root
├── main branch (Local)
├── feature-a (Worktree ~/.mux/src/project/feature-a)
└── feature-b (SSH remote:/workspace/project)
```

### 🤖 멀티모델 지원

```yaml
지원 모델:
  - Claude: sonnet-4-*, opus-4-*
  - OpenAI: gpt-5-*
  - X.AI: grok-*
  - Ollama: 로컬 LLM (llama3, codellama)
  - OpenRouter: 장거리 LLM 액세스
```

### ⚡ 효율적인 UI & 키바인딩

| 기능 | 단축키 (macOS) | 단축키 (Win/Linux) |
|------|----------------|-------------------|
| 커맨드 팔레트 | `Cmd+Shift+P` | `Ctrl+Shift+P` |
| 빠른 열기 | `Cmd+P` | `Ctrl+P` |
| 워크스페이스 전환 | `Cmd+K` | `Ctrl+K` |

### 🔌 VS Code 통합

```
VS Code → Mux Extension → Open Workspace
                ↓
         Mux Desktop App (해당 워크스페이스)
```

---

## 빠른 시작

### 1. 다운로드 & 설치

```bash
# macOS
# mux-*.dmg를 다운로드
# https://github.com/coder/mux/releases

# Applications 폴더로 이동
open -a Mux

# Linux
# mux-*.AppImage를 다운로드
chmod +x mux-*.AppImage
./mux-*.AppImage
```

### 2. API 키 설정

```
Settings → API Keys
  → Claude: sk-ant-api03-...
  → OpenAI: sk-proj-...
```

### 3. 첫 프로젝트 추가

```
Projects Sidebar → Add Project
  → 프로젝트 디렉토리 선택
  → Git 레포지토리 감지됨
```

### 4. 워크스페이스 생성

```
Project → New Workspace
  → Local / Worktree / SSH 선택
  → 브랜치 지정
  → 생성 완료!
```

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    Mux Desktop App                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Electron Main Process                                       │
│    ↓                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Config     │  │  Workspaces  │  │  Git Manager │      │
│  │ (~/.mux)     │  │   Manager    │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  React Renderer (Browser)                                    │
│    ↓                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Projects UI │  │  Agent Chat  │  │  Code Review │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  Agent Loop                                                  │
│    ↓                                                         │
│  User Input → Planning → Execution → Review → Iterate       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**핵심 컴포넌트:**

- **Main Process** - Electron 백엔드, Git/파일 시스템 관리
- **Renderer** - React UI, 채팅, 코드 리뷰
- **Agent Loop** - Plan/Exec 모드, 병렬 에이전트 실행
- **Config** - `~/.mux/config.json`, 프로젝트 설정
- **Sessions** - `~/.mux/sessions/<workspace>/chat.jsonl`

---

## 기술 스택

| 기술 | 용도 |
|------|------|
| Electron | 데스크톱 앱 프레임워크 |
| React | UI 컴포넌트 |
| TypeScript | 타입 안전 개발 |
| Bun | 패키지 매니저 & 런타임 |
| Vite | 빌드 도구 |
| Jest | 유닛 테스트 |
| Playwright | E2E 테스트 |

---

## Claude Code와의 비교

| 특징 | Claude Code | Mux |
|------|-------------|-----|
| **에이전트** | 단일 | 다중 (병렬) |
| **워크스페이스** | 단일 디렉토리 | Local/Worktree/SSH |
| **Git 관리** | 기본 | 중앙화된 분기 추적 |
| **플랫폼** | CLI + IDE | 데스크톱 앱 + VS Code |
| **모델** | Claude 전용 | 멀티모델 |
| **컨텍스트** | 수동 압축 | Opportunistic Compaction |

**언제 Mux를 사용할까?**

- ✅ 여러 기능을 병렬로 개발
- ✅ 원격 서버에서 에이전트 실행
- ✅ Git 브랜치 간 변경 사항 추적
- ✅ 여러 AI 모델 비교 필요

---

## 주요 워크플로우

### 1. 병렬 기능 개발

```
main (Local)
  ↓
feature-a (Worktree) → Agent-1: 구현
  ↓
feature-b (Worktree) → Agent-2: 구현
  ↓
Git Divergence UI: 변경 사항 추적
  ↓
병합 및 충돌 해결
```

### 2. 원격 서버 실행

```
Local Machine
  ↓
SSH 연결 → Remote Server (8-core, 32GB RAM)
  ↓
Mux Workspace → 강력한 에이전트 실행
  ↓
로컬에서 결과 리뷰
```

### 3. 모델 비교

```
Task: "새로운 API 엔드포인트 구현"
  ↓
Workspace-1 (Claude Sonnet 4) → 솔루션 A
Workspace-2 (GPT-5) → 솔루션 B
Workspace-3 (Grok Beta) → 솔루션 C
  ↓
최적의 솔루션 선택
```

---

## 스크린샷 갤러리

### 통합 코드 리뷰

<img src="https://github.com/coder/mux/raw/main/docs/img/code-review.webp" alt="Code Review" width="600" />

### Git 분기 추적

<img src="https://github.com/coder/mux/raw/main/docs/img/git-status.webp" alt="Git Status" width="600" />

### Mermaid 다이어그램

<img src="https://github.com/coder/mux/raw/main/docs/img/plan-mermaid.webp" alt="Mermaid Diagram" width="600" />

### 비용 추적

<img src="https://github.com/coder/mux/raw/main/docs/img/costs-tab.webp" alt="Costs Table" width="600" />

---

## 관련 링크

- **GitHub**: [https://github.com/coder/mux](https://github.com/coder/mux)
- **공식 문서**: [https://mux.coder.com](https://mux.coder.com)
- **Releases**: [https://github.com/coder/mux/releases](https://github.com/coder/mux/releases)
- **Discord**: [https://discord.gg/thkEdtwm8c](https://discord.gg/thkEdtwm8c)

---

## 라이선스

이 프로젝트는 [AGPL-3.0 라이선스](https://github.com/coder/mux/blob/main/LICENSE)로 배포됩니다.

Copyright (C) 2026 Coder Technologies, Inc.

---

*Mux로 여러 AI 에이전트를 병렬로 활용하여 개발 속도를 극대화하세요!* 🚀
