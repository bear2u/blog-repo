---
layout: post
title: "Mux 완벽 가이드 (01) - 소개 및 개요"
date: 2026-02-08
permalink: /mux-guide-01-intro/
author: Coder Technologies
categories: [AI 코딩, 개발 도구]
tags: [Mux, AI Agent, Coding Agent, Electron, Parallel Development, Multiplexer]
original_url: "https://github.com/coder/mux"
excerpt: "병렬 에이전트 개발을 위한 데스크톱 & 브라우저 애플리케이션"
---

## Mux란?

**Mux (Coding Agent Multiplexer)**는 병렬 에이전트 개발을 위한 데스크톱 & 브라우저 애플리케이션입니다. 로컬 또는 원격 컴퓨팅에서 여러 AI 에이전트로 작업을 계획하고 실행할 수 있습니다.

```
┌─────────────────────────────────────────────────────────────┐
│                          Mux                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User → Planning → Multiple Agents → Parallel Execution     │
│           ↓                                                  │
│      Workspaces (Local/Worktree/SSH)                         │
│           ↓                                                  │
│      Git Divergence Tracking                                 │
│           ↓                                                  │
│      Results & Code Review                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

<p><img src="https://github.com/coder/mux/raw/main/docs/img/mux-demo.gif" alt="mux product demo" width="100%" /></p>

---

## 주요 특징

### 🔀 격리된 워크스페이스

**중앙화된 Git 분기 관리와 함께 작업을 분리합니다:**

| 런타임 타입 | 설명 | 사용 사례 |
|------------|------|-----------|
| **Local** | 프로젝트 디렉토리에서 직접 실행 | 빠른 테스트, 단일 작업 |
| **Worktree** | 로컬 머신의 git worktree | 병렬 기능 개발 |
| **SSH** | SSH를 통한 원격 서버 실행 | 원격 컴퓨팅, 팀 협업 |

```
Project Root
├── main branch (Local)
├── feature-a (Worktree ~/.mux/src/project/feature-a)
└── feature-b (SSH remote:/workspace/project)
```

### 🤖 멀티모델 지원

**여러 AI 모델을 동시에 활용:**

```yaml
지원 모델:
  - Claude: sonnet-4-*, opus-4-*
  - OpenAI: gpt-5-*
  - X.AI: grok-*
  - Ollama: 로컬 LLM (llama3, codellama, etc.)
  - OpenRouter: 장거리 LLM 액세스
```

### 🔌 VS Code 통합

**VS Code에서 직접 Mux 워크스페이스로 점프:**

```
VS Code → Mux Extension → Open Workspace
                ↓
         Mux Desktop App (해당 워크스페이스)
```

### ⚡ 효율적인 에이전트 관리

- **Plan/Exec 모드** - Claude Code에서 영감을 받은 플래닝 워크플로우
- **Vim 입력** - Vim 키바인딩 지원
- **Rich Markdown** - Mermaid 다이어그램, LaTeX 지원
- **Opportunistic Compaction** - 컨텍스트 크기 자동 관리
- **Mode Prompts** - 모드별 커스텀 프롬프트

---

## Claude Code와의 비교

Mux는 Claude Code에서 영감을 받았지만, 병렬 에이전트 워크플로우에 특화되었습니다.

| 특징 | Claude Code | Mux |
|------|-------------|-----|
| **에이전트 수** | 단일 에이전트 | 다중 에이전트 (병렬) |
| **워크스페이스** | 단일 디렉토리 | Local/Worktree/SSH |
| **Git 관리** | 기본 | 중앙화된 분기 추적 |
| **플랫폼** | CLI + IDE 통합 | 데스크톱 앱 + VS Code |
| **컨텍스트 관리** | 수동 압축 | Opportunistic Compaction |
| **모델** | Claude 전용 | 멀티모델 (Claude, GPT, Grok, Ollama) |

**언제 Mux를 사용할까?**

- ✅ 여러 기능을 병렬로 개발
- ✅ 원격 서버에서 에이전트 실행
- ✅ Git 브랜치 간 변경 사항 추적 필요
- ✅ 여러 AI 모델 비교 필요

**언제 Claude Code를 사용할까?**

- ✅ 단일 작업에 집중
- ✅ IDE 통합 선호
- ✅ Claude 모델만 사용

---

## 핵심 UX 요소

### 1. 사이드바 (왼쪽 패널)

```
┌────────────────┐
│  Projects      │ ← 프로젝트 목록
├────────────────┤
│  • project-a   │
│    - main      │ ← 브랜치/워크스페이스
│    - feature-1 │
│  • project-b   │
└────────────────┘
```

### 2. Git 분기 UI

```
┌─────────────────────────────────────┐
│  Git Divergence                     │
├─────────────────────────────────────┤
│  main ←→ feature-1: +12 -5          │ ← 변경 사항 추적
│  Conflicts: src/app.ts (2)          │ ← 충돌 감지
└─────────────────────────────────────┘
```

### 3. 에이전트 상태

```
┌─────────────────────────────────────┐
│  Agent Status                       │
├─────────────────────────────────────┤
│  🟢 Agent-1: Executing tests...     │
│  🟡 Agent-2: Planning refactor...   │
│  🔴 Agent-3: Error (click to view)  │
└─────────────────────────────────────┘
```

---

## 스크린샷 갤러리

### 통합 코드 리뷰

<img src="https://github.com/coder/mux/raw/main/docs/img/code-review.webp" alt="Code Review" width="600" />

더 빠른 반복을 위한 통합 코드 리뷰 UI.

### Git 분기 추적

<img src="https://github.com/coder/mux/raw/main/docs/img/git-status.webp" alt="Git Status" width="600" />

변경 사항과 잠재적 충돌을 실시간으로 추적.

### Mermaid 다이어그램

<img src="https://github.com/coder/mux/raw/main/docs/img/plan-mermaid.webp" alt="Mermaid Diagram" width="600" />

에이전트의 복잡한 제안을 Mermaid 다이어그램으로 시각화.

### 비용 추적

<img src="https://github.com/coder/mux/raw/main/docs/img/costs-tab.webp" alt="Costs Table" width="600" />

비용 및 토큰 소비량을 실시간으로 모니터링.

---

## 시작하기 전에

### 요구사항

```yaml
OS: macOS, Linux (Windows 지원 예정)
Runtime: Node.js 18+ (Bun 권장)
Git: 2.25+
메모리: 8GB+ RAM 권장
저장소: 2GB+ (워크스페이스 및 모델 캐시)
```

### 빠른 시작

```bash
# 1. 다운로드
# https://github.com/coder/mux/releases

# 2. 설치 (macOS)
# mux-*.dmg를 다운로드하고 Applications 폴더로 이동

# 3. 실행
open -a Mux

# 4. API 키 설정
# Settings → API Keys → Claude/OpenAI 키 입력
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

---

## 프로젝트 정보

| 항목 | 내용 |
|------|------|
| **프로젝트명** | Mux |
| **개발사** | Coder Technologies, Inc. |
| **버전** | 0.16.0 |
| **라이선스** | AGPL-3.0 |
| **GitHub** | https://github.com/coder/mux |
| **문서** | https://mux.coder.com |
| **Discord** | https://discord.gg/thkEdtwm8c |

---

## 다음 단계

**이 가이드에서는:**
- ✅ Mux의 핵심 개념 이해
- ✅ 주요 특징 및 Claude Code와의 비교
- ✅ 아키텍처 및 워크플로우 파악

**다음 글에서는:**
- 📦 상세한 설치 과정 (macOS/Linux)
- 🔧 초기 설정 및 API 키 구성
- 🚀 첫 워크스페이스 생성 및 에이전트 실행

---

*Mux로 여러 AI 에이전트를 병렬로 활용하여 개발 속도를 극대화하세요!* 🚀
