---
layout: page
title: Entire CLI 가이드
permalink: /entire-cli-guide/
icon: fas fa-terminal
---

# Entire CLI 완벽 가이드

> **Git 워크플로우에 통합되어 AI 에이전트 세션을 자동으로 캡처하는 도구**

**Entire CLI**는 Claude Code, Gemini CLI 등의 AI 코딩 도구와 함께 사용하여 코드가 어떻게 작성되었는지에 대한 완전한 기록을 커밋과 함께 저장합니다.

---

## 목차

### 기본 개념 (5챕터)

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/blog-repo/entire-cli-guide-01-intro/) | Entire CLI란? 주요 특징, 사용 이유 |
| 02 | [설치 및 시작하기](/blog-repo/entire-cli-guide-02-installation/) | 요구사항, 설치 방법, 빠른 시작 |
| 03 | [핵심 개념](/blog-repo/entire-cli-guide-03-concepts/) | Session, Checkpoint, Strategy 이해 |
| 04 | [일반적인 워크플로우](/blog-repo/entire-cli-guide-04-workflow/) | Enable → Work → Rewind → Resume |
| 05 | [명령어 레퍼런스](/blog-repo/entire-cli-guide-05-commands/) | 전체 명령어 개요 및 상세 설명 |

### Strategy 시스템 (3챕터)

| # | 제목 | 내용 |
|---|------|------|
| 06 | [Strategy 개요](/blog-repo/entire-cli-guide-06-strategy-overview/) | Manual-commit vs Auto-commit 비교 |
| 07 | [Manual-Commit Strategy](/blog-repo/entire-cli-guide-07-manual-commit/) | Shadow Branch 메커니즘, 기본 전략 |
| 08 | [Auto-Commit Strategy](/blog-repo/entire-cli-guide-08-auto-commit/) | 자동 커밋 전략, 세밀한 체크포인트 |

### Session & Checkpoint (4챕터)

| # | 제목 | 내용 |
|---|------|------|
| 09 | [Session 관리](/blog-repo/entire-cli-guide-09-session-management/) | Session 생성, 추적, 조회 |
| 10 | [Checkpoint 시스템](/blog-repo/entire-cli-guide-10-checkpoint-system/) | Temporary vs Committed Checkpoint |
| 11 | [Checkpoint ID 연결](/blog-repo/entire-cli-guide-11-checkpoint-linking/) | 12-hex-char ID, Bidirectional Linking |
| 12 | [Multi-Session 처리](/blog-repo/entire-cli-guide-12-multi-session/) | 동시 세션, Conflict 처리 |

### 아키텍처 상세 (5챕터)

| # | 제목 | 내용 |
|---|------|------|
| 13 | [Git 통합](/blog-repo/entire-cli-guide-13-git-integration/) | Git Hooks, Worktree 지원 |
| 14 | [Storage 구조](/blog-repo/entire-cli-guide-14-storage-structure/) | Shadow Branch, Metadata Branch |
| 15 | [Claude Code Hooks](/blog-repo/entire-cli-guide-15-claude-hooks/) | SessionStart, UserPromptSubmit, Stop |
| 16 | [Subagent Tracking](/blog-repo/entire-cli-guide-16-subagent-tracking/) | Task, TodoWrite 체크포인트 |
| 17 | [Logging 시스템](/blog-repo/entire-cli-guide-17-logging-system/) | 구조화된 로깅, 프라이버시 |

### 고급 기능 (4챕터)

| # | 제목 | 내용 |
|---|------|------|
| 18 | [Rewind 메커니즘](/blog-repo/entire-cli-guide-18-rewind-mechanism/) | 체크포인트로 되돌리기 |
| 19 | [Resume 기능](/blog-repo/entire-cli-guide-19-resume-feature/) | 이전 세션 복원 |
| 20 | [Auto-Summarization](/blog-repo/entire-cli-guide-20-auto-summarization/) | AI 기반 자동 요약 |
| 21 | [Token Usage Tracking](/blog-repo/entire-cli-guide-21-token-tracking/) | 사용량 추적 및 분석 |

### 개발 및 확장 (4챕터)

| # | 제목 | 내용 |
|---|------|------|
| 22 | [개발 환경 설정](/blog-repo/entire-cli-guide-22-development-setup/) | mise, Go, 테스트 실행 |
| 23 | [코드 구조](/blog-repo/entire-cli-guide-23-code-structure/) | 패키지 구성, 주요 파일 |
| 24 | [Agent 통합](/blog-repo/entire-cli-guide-24-agent-integration/) | Gemini CLI, 새 Agent 추가 |
| 25 | [Contributing](/blog-repo/entire-cli-guide-25-contributing/) | 기여 가이드, 테스트, PR 프로세스 |

---

## 주요 특징

- **Git 훅 기반 자동 캡처** - 별도의 수동 작업 없이 모든 세션 자동 기록
- **체크포인트 시스템** - 언제든지 이전 상태로 되돌리기 가능
- **다중 전략 지원** - Manual-commit과 Auto-commit 전략 선택
- **AI 에이전트 통합** - Claude Code, Gemini CLI 지원
- **Worktree 지원** - 병렬 작업을 위한 완벽한 격리
- **동시 세션 처리** - 여러 세션을 하나의 커밋으로 병합
- **토큰 사용량 추적** - 비용 분석 및 최적화
- **Auto-Summarization** - AI 기반 세션 요약 자동 생성

---

## 빠른 시작

```bash
# 설치
brew tap entireio/tap
brew install entireio/tap/entire

# 또는 Go로 설치
go install github.com/entireio/cli/cmd/entire@latest

# 프로젝트에서 활성화
cd your-project
entire enable

# Claude Code로 작업
claude "Add user authentication"

# 커밋
git commit -m "Add auth system"

# 상태 확인
entire status

# 필요시 되돌리기
entire rewind
```

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                        Entire CLI                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Git Workflow                                                │
│       │                                                      │
│       ├─ enable ────► Git Hooks                             │
│       │               ├─ prepare-commit-msg                  │
│       │               ├─ post-commit                         │
│       │               └─ pre-push                            │
│       │                                                      │
│       ├─ work ─────► AI Agent Hooks                         │
│       │               ├─ SessionStart                        │
│       │               ├─ UserPromptSubmit                    │
│       │               ├─ Stop                                │
│       │               ├─ PreToolUse[Task]                    │
│       │               ├─ PostToolUse[Task]                   │
│       │               └─ PostToolUse[TodoWrite]              │
│       │                                                      │
│       ├─ commit ───► Condensation                           │
│       │               └─ entire/checkpoints/v1              │
│       │                                                      │
│       ├─ rewind ───► Checkpoint Restore                     │
│       │                                                      │
│       └─ resume ───► Session Restore                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 기술 | 용도 |
|------|------|
| **Go 1.25.x** | 주 언어 |
| **Cobra** | CLI 프레임워크 |
| **Huh** | TUI (Terminal UI) |
| **go-git/go-git/v5** | Git 라이브러리 |
| **mise** | 빌드 도구 및 버전 관리 |
| **golangci-lint** | 린팅 |
| **slog** | 구조화된 로깅 |

---

## Storage 구조

### Session State
```
.git/entire-sessions/<session-id>.json
```

### Temporary Checkpoints
```
entire/<commit[:7]>-<worktree[:6]> (Shadow Branch)
└── .entire/metadata/<session-id>/
    ├── full.jsonl
    ├── prompt.txt
    ├── context.md
    └── tasks/<tool-use-id>/
```

### Committed Checkpoints
```
entire/checkpoints/v1 (Metadata Branch)
└── <id[:2]>/<id[2:]>/
    ├── metadata.json
    ├── 0/  # Session 1
    │   ├── metadata.json
    │   ├── full.jsonl
    │   ├── prompt.txt
    │   └── context.md
    └── 1/  # Session 2 (concurrent)
```

---

## 사용 사례

### 1. **시간 여행 디버깅**
```bash
# 문제 발생
claude "Refactor authentication"
# ... 버그 발생 ...

# 이전 상태로 되돌리기
entire rewind
# → 작동하던 버전으로 복원

# 다시 시도
claude "Refactor authentication with tests"
```

### 2. **컨텍스트 복원**
```bash
# Feature 브랜치에서 작업 중
git checkout feature/payment
claude "Add Stripe integration"

# 긴급 버그 수정
git checkout main
git checkout -b hotfix/auth

# 원래 작업으로 복귀
entire resume feature/payment
claude --session <session-id>
# → 이전 컨텍스트와 함께 계속
```

### 3. **팀 협업**
```bash
# 동료의 커밋 이해하기
git log
entire explain a3f2b1c4

# 출력: 전체 세션 기록
# - 프롬프트
# - AI 응답
# - 파일 변경사항
# - 토큰 사용량
```

### 4. **비용 분석**
```bash
# 프로젝트 전체 토큰 사용량
entire stats --all

# 출력:
# Total tokens: 150,000
# Input: 100,000
# Output: 50,000
# Estimated cost: $15.00
```

---

## Strategy 비교

| 항목 | Manual-Commit | Auto-Commit |
|-----|---------------|-------------|
| **코드 커밋** | 사용자가 직접 | 자동 생성 |
| **체크포인트 빈도** | 커밋 시 | AI 응답마다 |
| **Git 히스토리** | 깔끔 | 많은 커밋 |
| **main 브랜치** | 안전 | 주의 필요 |
| **Rewind** | 항상 가능 | 제한적 |
| **적합한 용도** | 대부분의 워크플로우 | 자동 커밋 원하는 팀 |

---

## 요구사항

- **Git** - 최신 버전
- **macOS or Linux** - Windows는 WSL 사용
- **Claude Code** 또는 **Gemini CLI** - AI 에이전트

---

## 관련 링크

- [GitHub 저장소](https://github.com/entireio/cli)
- [공식 문서](https://github.com/entireio/cli#readme)
- [Claude Code 문서](https://docs.anthropic.com/en/docs/claude-code)
- [Gemini CLI 문서](https://github.com/google-gemini/gemini-cli)
- [Contributing 가이드](https://github.com/entireio/cli/blob/main/CONTRIBUTING.md)
- [Discord 커뮤니티](https://discord.gg/4WXDu2Ph)

---

## 라이선스

MIT License - [LICENSE](https://github.com/entireio/cli/blob/main/LICENSE)

---

## 기여하기

Entire CLI는 오픈 소스 프로젝트입니다. 기여를 환영합니다!

- [Issue 제출](https://github.com/entireio/cli/issues)
- [Pull Request](https://github.com/entireio/cli/pulls)
- [Discussion](https://github.com/entireio/cli/discussions)

자세한 내용은 [Contributing 가이드](/blog-repo/entire-cli-guide-25-contributing/)를 참조하세요.

---

**전체 25개 챕터로 구성된 완벽한 Entire CLI 가이드입니다. 각 챕터를 순서대로 읽으면서 Entire CLI를 마스터하세요!** 🚀
