---
layout: post
title: "Ralph 가이드 05 - CLI 명령어"
date: 2025-02-04
categories: [AI, Claude Code, Ralph]
tags: [ralph, cli, commands, terminal]
series: "ralph-guide"
permalink: /ralph-guide-05-commands/
---

# CLI 명령어

## 명령어 개요

| 명령어 | 용도 | 사용 시점 |
|--------|------|-----------|
| `ralph` | 자율 개발 루프 실행 | 개발 시작 |
| `ralph-enable` | 기존 프로젝트에 Ralph 활성화 | 초기 설정 |
| `ralph-import` | PRD로 새 프로젝트 생성 | 요구사항 문서가 있을 때 |
| `ralph-setup` | 새 프로젝트 초기화 | 처음부터 시작 |

## ralph - 메인 명령어

### 기본 사용법

```bash
ralph [options]
```

### 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--monitor` | tmux 모니터링 대시보드와 함께 실행 | 비활성 |
| `--status` | 현재 세션 상태 확인 | - |
| `--stop` | 실행 중인 Ralph 세션 중지 | - |
| `--resume` | 이전 세션 재개 | - |
| `--verbose` | 상세 로그 출력 | 비활성 |
| `--dry-run` | 실제 변경 없이 시뮬레이션 | 비활성 |

### 실행 예시

```bash
# 기본 실행 (백그라운드 없음)
ralph

# 모니터링 대시보드와 함께 실행
ralph --monitor

# 현재 상태 확인
ralph --status

# 실행 중인 세션 중지
ralph --stop

# 이전 세션 재개
ralph --resume

# 상세 로그와 함께 실행
ralph --verbose

# 테스트 실행 (변경 없음)
ralph --dry-run
```

### --monitor 모드

```bash
ralph --monitor
```

tmux 세션을 생성하여 실시간 모니터링:

```
┌────────────────────────────────┬───────────────────────────────┐
│                                │                               │
│  Ralph Loop Output             │  Monitoring Dashboard         │
│                                │                               │
│  [10:30:15] Reading PROMPT.md  │  Session: my-project-abc123   │
│  [10:30:16] Found 5 tasks      │  Status: RUNNING              │
│  [10:30:17] Starting Task 1    │  Loop: 3 / 100                │
│  [10:30:45] Task 1 complete    │  Tasks: 2/5 complete          │
│  [10:30:46] Running tests...   │  API Calls: 45/100            │
│  [10:31:02] Tests passed       │  Errors: 0                    │
│  [10:31:03] Starting Task 2    │  Circuit: CLOSED              │
│                                │                               │
│                                │  Last Activity: 10:31:03      │
│                                │                               │
└────────────────────────────────┴───────────────────────────────┘
```

**tmux 키 바인딩:**
- `Ctrl+B, D` - 세션 분리 (Ralph 계속 실행)
- `Ctrl+B, [` - 스크롤 모드
- `Ctrl+B, 0/1` - 패널 전환

## ralph-enable - 프로젝트 활성화

### 기본 사용법

```bash
cd existing-project
ralph-enable [options]
```

### 옵션

| 옵션 | 설명 |
|------|------|
| `--from-issues` | GitHub Issues에서 작업 가져오기 |
| `--from-file <path>` | 파일에서 작업 가져오기 |
| `--non-interactive` | 대화형 모드 비활성화 |

### 인터랙티브 위저드

```bash
ralph-enable
```

```
Ralph Enable Wizard
==================

Phase 1: Environment Detection
------------------------------
✓ Detected project type: typescript
✓ Detected package manager: npm
✓ Git repository: yes
✓ Found tsconfig.json
✓ Found package.json with scripts

Phase 2: Task Source Selection
------------------------------
Where should Ralph get initial tasks from?

[1] GitHub Issues (fetches open issues)
[2] Existing PRD/Requirements file
[3] Start with empty task list
[4] Import from markdown file

Enter choice (1-4): 2

Enter path to requirements file: ./docs/requirements.md
Found 12 actionable items. Import all? (y/n): y

Phase 3: Configuration
------------------------------
Max API calls per hour [100]:
Session timeout (seconds) [3600]:
Enable circuit breaker? [Y/n]: y

Creating .ralph/ directory structure...

Phase 4: File Generation
------------------------------
✓ Created: .ralph/PROMPT.md (with project context)
✓ Created: .ralph/fix_plan.md (12 tasks imported)
✓ Created: .ralph/AGENT.md (npm scripts detected)
✓ Created: .ralph/specs/
✓ Created: .ralph/logs/
✓ Created: .ralphrc

Ralph is now enabled for this project.
Review .ralph/PROMPT.md and .ralph/fix_plan.md, then run:

  ralph --monitor
```

### 비대화형 모드

```bash
# GitHub Issues에서 자동 가져오기
ralph-enable --from-issues --non-interactive

# 파일에서 가져오기
ralph-enable --from-file requirements.md --non-interactive
```

## ralph-import - PRD 가져오기

### 기본 사용법

```bash
ralph-import <prd-file> <project-name> [options]
```

### 옵션

| 옵션 | 설명 |
|------|------|
| `--type <type>` | 프로젝트 타입 지정 (node, python, etc.) |
| `--no-git` | Git 초기화 건너뛰기 |

### 사용 예시

```bash
# PRD로 새 프로젝트 생성
ralph-import requirements.md my-new-app

# 프로젝트 타입 지정
ralph-import spec.md backend-api --type python

# Git 없이 생성
ralph-import ideas.md prototype --no-git
```

### 동작 과정

```
1. 새 디렉토리 생성 (my-new-app/)
2. Git 저장소 초기화
3. PRD 파일 분석
   - 요구사항 추출
   - 작업 목록 생성
   - 프로젝트 타입 감지
4. .ralph/ 디렉토리 생성
5. PROMPT.md 생성 (PRD 컨텍스트)
6. fix_plan.md 생성 (추출된 작업)
7. 기본 프로젝트 구조 생성
```

### PRD 형식 가이드

```markdown
# Product Requirements Document

## Overview
[프로젝트 설명]

## Goals
- Goal 1
- Goal 2

## Features
### Feature 1: [기능명]
- Requirement 1.1
- Requirement 1.2

### Feature 2: [기능명]
- Requirement 2.1

## Technical Requirements
- Language: TypeScript
- Framework: Express
- Database: PostgreSQL

## Non-Functional Requirements
- Response time < 200ms
- 99.9% uptime
```

## ralph-setup - 새 프로젝트

### 기본 사용법

```bash
ralph-setup <project-name> [options]
```

### 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--type <type>` | 프로젝트 타입 | 대화형 선택 |
| `--template <name>` | 템플릿 사용 | 없음 |

### 프로젝트 타입

| 타입 | 설명 | 생성되는 파일 |
|------|------|--------------|
| `node` | Node.js | package.json, src/, tests/ |
| `typescript` | TypeScript | tsconfig.json, src/, tests/ |
| `python` | Python | requirements.txt, src/, tests/ |
| `rust` | Rust | Cargo.toml, src/ |
| `go` | Go | go.mod, cmd/, pkg/ |

### 사용 예시

```bash
# 대화형 설정
ralph-setup my-project

# 타입 지정
ralph-setup api-server --type typescript

# 템플릿 사용
ralph-setup web-app --template express-api
```

## 유틸리티 명령어

### 상태 확인

```bash
# 현재 세션 상태
ralph --status
```

출력:
```
Ralph Session Status
====================
Session ID: my-project-abc123
Status: RUNNING
Started: 2024-01-15 10:30:00
Duration: 1h 23m

Progress:
  Loop: 15
  Tasks: 8/12 complete
  API Calls: 67/100 this hour

Circuit Breaker: CLOSED
Last Error: None
```

### 로그 확인

```bash
# 최신 로그 보기
tail -f .ralph/logs/session_latest.log

# 특정 세션 로그
cat .ralph/logs/session_abc123.log
```

### 세션 관리

```bash
# 세션 중지
ralph --stop

# 세션 재개
ralph --resume

# 강제 리셋
rm .ralph/status.json
ralph
```

## 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `RALPH_HOME` | Ralph 설치 디렉토리 | `~/.local/share/ralph` |
| `RALPH_CONFIG` | 전역 설정 파일 | `~/.ralphrc` |
| `RALPH_LOG_LEVEL` | 로그 레벨 | `info` |
| `CLAUDE_MODEL` | 사용할 Claude 모델 | 기본 모델 |

```bash
# 환경 변수 설정 예시
export RALPH_LOG_LEVEL=debug
export CLAUDE_MODEL=claude-3-opus-20240229
ralph --monitor
```

---

**이전 장:** [핵심 개념](/ralph-guide-04-concepts/) | **다음 장:** [구성 및 설정](/ralph-guide-06-configuration/)
