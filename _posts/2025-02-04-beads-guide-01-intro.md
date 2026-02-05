---
layout: post
title: "Beads 완벽 가이드 (1) - 소개 및 개요"
date: 2025-02-04
permalink: /beads-guide-01-intro/
author: Steve Yegge
categories: [AI 에이전트, Beads]
tags: [Beads, Issue Tracker, Git, AI Agent, Go]
original_url: "https://github.com/steveyegge/beads"
excerpt: "AI 에이전트를 위한 분산 Git 기반 이슈 트래커 Beads를 소개합니다."
---

## Beads란?

**Beads (bd)**는 **AI 코딩 에이전트를 위한 분산, Git 기반 그래프 이슈 트래커**입니다. 복잡한 마크다운 계획 대신 의존성 인식 그래프를 제공하여, 에이전트가 컨텍스트를 잃지 않고 장기 작업을 처리할 수 있게 합니다.

```
┌─────────────────────────────────────────┐
│            Beads의 핵심 철학             │
├─────────────────────────────────────────┤
│  • Git이 데이터베이스                    │
│  • 에이전트 최적화된 설계                │
│  • 충돌 없는 분산 작업                   │
│  • 보이지 않는 인프라                    │
└─────────────────────────────────────────┘
```

---

## 주요 특징

| 특징 | 설명 |
|------|------|
| **Git as Database** | 이슈를 `.beads/`에 JSONL로 저장. 코드처럼 버전 관리, 브랜치, 머지 |
| **Agent-Optimized** | JSON 출력, 의존성 추적, 자동 ready 작업 감지 |
| **Zero Conflict** | 해시 기반 ID(`bd-a1b2`)로 다중 에이전트/브랜치 작업 시 충돌 방지 |
| **Invisible Infrastructure** | SQLite 로컬 캐시로 속도 향상, 백그라운드 데몬으로 자동 동기화 |
| **Compaction** | 시맨틱 "메모리 감쇠"로 오래된 완료 작업을 요약하여 컨텍스트 윈도우 절약 |

---

## 빠른 시작

```bash
# Beads CLI 설치 (시스템 전역 - 프로젝트에 클론하지 않음)
curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash

# 또는 npm/Homebrew/Go로 설치
npm install -g @beads/bd
brew install beads
go install github.com/steveyegge/beads/cmd/bd@latest

# 프로젝트에서 초기화
cd your-project
bd init

# 에이전트에게 알려주기
echo "Use 'bd' for task tracking" >> AGENTS.md
```

---

## 필수 명령어

| 명령어 | 동작 |
|--------|------|
| `bd ready` | 열린 차단이 없는 작업 목록 표시 |
| `bd create "Title" -p 0` | P0 작업 생성 |
| `bd dep add <child> <parent>` | 작업 연결 (blocks, related, parent-child) |
| `bd show <id>` | 작업 상세 및 감사 추적 보기 |
| `bd list` | 모든 이슈 목록 |
| `bd close <id>` | 작업 완료 |
| `bd sync` | Git과 동기화 |

---

## 계층적 ID 시스템

Beads는 에픽을 위한 계층적 ID를 지원합니다:

```
bd-a3f8       (Epic)
├── bd-a3f8.1     (Task)
│   └── bd-a3f8.1.1   (Sub-task)
└── bd-a3f8.2     (Task)
```

---

## 작업 모드

### Stealth Mode

```bash
# 로컬에서만 사용, 메인 레포에 커밋하지 않음
bd init --stealth
```

공유 프로젝트에서 개인적으로 사용하기에 적합합니다.

### Contributor vs Maintainer

| 역할 | 명령어 | 설명 |
|------|--------|------|
| **Contributor** | `bd init --contributor` | 계획 이슈를 별도 레포(`~/.beads-planning`)로 라우팅 |
| **Maintainer** | 자동 감지 | SSH URL 또는 인증된 HTTPS로 자동 감지 |

---

## 프로젝트 구조

```
beads/
├── cmd/bd/              # CLI 명령어
├── internal/
│   ├── types/           # 핵심 데이터 타입
│   ├── storage/         # 스토리지 레이어
│   │   └── sqlite/      # SQLite 구현
│   ├── daemon/          # 백그라운드 데몬
│   ├── rpc/             # RPC 프로토콜
│   ├── autoimport/      # 자동 임포트
│   ├── export/          # 자동 익스포트
│   └── molecules/       # 분자/위습 시스템
├── integrations/        # 외부 통합
├── examples/            # 예제
└── docs/                # 문서
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **언어** | Go 1.24+ |
| **데이터베이스** | SQLite (로컬 캐시) |
| **데이터 형식** | JSONL (Git 추적) |
| **통신** | Unix Domain Socket, Named Pipes (Windows) |
| **빌드** | Makefile, goreleaser |
| **테스트** | Go test, golangci-lint |

---

## 지원 플랫폼

- macOS
- Linux
- Windows
- FreeBSD

---

## 왜 GitHub Issues가 아닌가?

| 기능 | Beads | GitHub Issues |
|------|-------|---------------|
| **의존성 타입** | 4가지 (blocks, related, parent-child, discovered-from) | blocks/blocked-by만 |
| **Ready 감지** | `bd ready`로 10ms 내 오프라인 계산 | 없음 (커스텀 GraphQL 필요) |
| **오프라인 작업** | Git-first, 완전 오프라인 지원 | 네트워크 필수 |
| **브랜치 스코프** | 브랜치별 이슈 상태 | 레포 전역 |
| **충돌 해결** | 해시 ID로 자동 | 수동 |
| **JSON API** | 모든 명령어 `--json` 지원 | 혼합된 출력 |

---

## 이 가이드에서 다루는 내용

1. **소개 및 개요** (현재 글)
2. **아키텍처** - 3계층 데이터 모델
3. **CLI 명령어** - 상세 명령어 사용법
4. **데이터 모델** - Issue, Dependency, JSONL 스키마
5. **데몬 시스템** - 백그라운드 동기화
6. **동기화 메커니즘** - Import/Export
7. **의존성 관리** - Ready 작업 감지
8. **Molecules & Wisps** - 템플릿 워크플로우
9. **확장 및 통합** - MCP, 외부 연동
10. **활용 가이드** - 베스트 프랙티스

---

## 라이선스

MIT License

---

*다음 글에서는 Beads의 3계층 데이터 모델 아키텍처를 살펴봅니다.*
