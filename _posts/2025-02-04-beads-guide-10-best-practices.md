---
layout: post
title: "Beads 완벽 가이드 (10) - 활용 가이드 및 결론"
date: 2025-02-04
permalink: /beads-guide-10-best-practices/
author: Steve Yegge
categories: [AI]
tags: [Beads, Best Practices, Workflow, AI Agent]
original_url: "https://github.com/steveyegge/beads"
excerpt: "Beads 활용 베스트 프랙티스와 AI 에이전트 워크플로우 가이드입니다."
---

## AI 에이전트 워크플로우

### 기본 작업 루프

AI 에이전트가 Beads를 사용하는 기본 패턴:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Work Loop                               │
│                                                                  │
│   1. bd ready --json          Ready 작업 조회                   │
│          │                                                       │
│          ▼                                                       │
│   2. 최고 우선순위 작업 선택                                    │
│          │                                                       │
│          ▼                                                       │
│   3. bd update <id> --status in_progress                        │
│          │                                                       │
│          ▼                                                       │
│   4. 작업 수행 (코드 작성, 테스트 등)                           │
│          │                                                       │
│          ├──────── 성공 ─────────▶ bd close <id>                │
│          │                                                       │
│          └──────── 차단됨 ───────▶ bd create "blocker"          │
│                                    bd dep add <new> <current>   │
│                                    bd update <current> blocked   │
└─────────────────────────────────────────────────────────────────┘
```

### AGENT_INSTRUCTIONS.md 예시

```markdown
# AI 에이전트 지침

## 작업 관리

이 프로젝트는 `bd` (Beads)를 사용합니다.

### 시작 시
1. `bd ready --json`으로 Ready 작업 확인
2. 가장 높은 우선순위 작업 선택
3. `bd update <id> --status in_progress`로 시작 표시

### 작업 중
- 새로운 작업 발견 시: `bd create "title" -p 2`
- 의존성 발견 시: `bd dep add <child> <parent> --blocks`
- 차단됨: `bd update <id> --status blocked`

### 완료 시
- `bd close <id> --reason "구현 완료"`
- `bd ready`로 다음 작업 확인
```

---

## 우선순위 전략

### 5단계 우선순위

| 우선순위 | 의미 | 응답 시간 |
|----------|------|----------|
| **P0** | Critical - 시스템 다운, 데이터 손실 | 즉시 |
| **P1** | High - 주요 기능 장애 | 당일 |
| **P2** | Medium - 일반 작업 (기본값) | 이번 스프린트 |
| **P3** | Low - 개선사항 | 다음 스프린트 |
| **P4** | Backlog - 언젠가 | 미정 |

### 우선순위 할당 가이드

```bash
# P0: 프로덕션 장애
bd create "Fix database connection failure" -p 0

# P1: 중요 기능
bd create "Add user authentication" -p 1

# P2: 일반 작업 (기본값)
bd create "Refactor API handlers"

# P3: 낮은 우선순위
bd create "Update README" -p 3

# P4: 백로그
bd create "Consider dark mode" -p 4
```

---

## 의존성 관리 패턴

### Epic 분해

```bash
# Epic 생성
bd create "User Authentication System" --type epic -p 1

# Sub-tasks 생성
bd create "Design auth flow" --parent bd-epic
bd create "Implement login" --parent bd-epic
bd create "Implement logout" --parent bd-epic
bd create "Add password reset" --parent bd-epic

# 의존성 추가
bd dep add bd-design bd-implement --blocks
bd dep add bd-implement bd-test --blocks
```

### 발견된 작업 처리

```bash
# 작업 중 새로운 이슈 발견
bd create "Fix edge case in validation" \
  --discovered-from bd-current
```

---

## 협업 패턴

### 다중 에이전트 시나리오

```
┌─────────────────────────────────────────────────────────────────┐
│                 Multi-Agent Workflow                             │
│                                                                  │
│   Agent-1 (Frontend)        Agent-2 (Backend)                   │
│         │                         │                              │
│         ▼                         ▼                              │
│   bd ready                   bd ready                            │
│         │                         │                              │
│         ▼                         ▼                              │
│   bd-ui-task (P1)           bd-api-task (P1)                    │
│         │                         │                              │
│         └─────────────────────────┘                              │
│                     │                                            │
│                     ▼                                            │
│              bd sync (자동)                                      │
│                     │                                            │
│                     ▼                                            │
│              Git Repository                                      │
│              (issues.jsonl)                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 작업 분배

```bash
# 담당자 지정
bd update bd-frontend --assignee agent-1
bd update bd-backend --assignee agent-2

# 담당자별 Ready 작업
bd ready --assignee agent-1
```

---

## 컨텍스트 관리

### Compaction (시맨틱 감쇠)

오래된 완료 작업을 요약하여 컨텍스트 윈도우 절약:

```bash
# 30일 이상 된 완료 작업 압축
bd compact --older-than 30d

# 드라이런
bd compact --dry-run

# 결과: 50개 이슈 → 5개 요약 이슈
```

### 이슈 요약 생성

```bash
# 현재 상태 요약
bd summary

# 출력:
# === Beads Summary ===
# Open: 15 (P0: 1, P1: 3, P2: 8, P3: 3)
# In Progress: 4
# Blocked: 2
# Ready: 9
#
# Recent Activity:
# - 3 issues closed today
# - 2 new issues created
# - 1 priority escalation (P2 → P0)
```

---

## 문제 해결

### 일반적인 문제

#### 데몬 연결 실패

```bash
# 데몬 상태 확인
bd daemons status

# 데몬 재시작
bd daemons restart

# 데몬 없이 작업
bd ready --no-daemon
```

#### 동기화 문제

```bash
# 상태 진단
bd doctor

# 강제 재임포트
bd import --force -i .beads/issues.jsonl

# 강제 익스포트
bd export --force -o .beads/issues.jsonl
```

#### Git 충돌

```bash
# JSONL 충돌 시
git checkout --ours .beads/issues.jsonl  # 로컬 우선
# 또는
git checkout --theirs .beads/issues.jsonl  # 원격 우선

# 재임포트
bd import -i .beads/issues.jsonl
```

### 디버그 모드

```bash
# 상세 로깅
BEADS_LOG_LEVEL=debug bd ready

# 데몬 로그 확인
tail -f .beads/daemon.log
```

---

## 베스트 프랙티스

### Do's ✓

- **작은 작업 단위**: 하나의 이슈는 1-2시간 내 완료 가능하게
- **명확한 제목**: "Fix bug" ❌ → "Fix null pointer in auth middleware" ✓
- **의존성 활용**: 블로킹 관계를 명시하여 작업 순서 보장
- **우선순위 준수**: P0 작업은 즉시 처리
- **정기적 동기화**: `bd sync` 자주 실행

### Don'ts ✗

- **거대 Epic 생성**: 100개 이상의 sub-task 피하기
- **순환 의존성**: A → B → A 형태 의존성
- **중복 이슈**: 동일한 작업에 여러 이슈
- **방치된 in_progress**: 시작한 작업은 완료하거나 blocked로 변경

---

## 활용 시나리오

### 1. 개인 프로젝트

```bash
bd init --stealth  # 로컬만, Git 커밋 안함
bd create "MVP features" --type epic
bd molecule apply feature-dev --var feature_name="Core API"
```

### 2. 팀 프로젝트

```bash
bd init --branch beads-metadata  # 별도 sync 브랜치
# 각 팀원이 자신의 작업 관리
bd ready --assignee $(whoami)
```

### 3. 오픈소스 기여

```bash
bd init --contributor  # 별도 레포로 라우팅
bd gh import --repo owner/repo --issue 123
# 작업 후
bd gh export bd-local --repo owner/repo
```

---

## 마무리

Beads는 AI 에이전트를 위해 설계된 분산 이슈 트래커입니다. 핵심 장점:

- **Git-first**: 특별한 서버 없이 코드와 함께 이동
- **의존성 인식**: `bd ready`로 즉시 시작 가능한 작업만 조회
- **충돌 없음**: 해시 기반 ID로 다중 에이전트/브랜치 작업 가능
- **빠른 쿼리**: SQLite 로컬 캐시로 밀리초 단위 응답
- **확장 가능**: MCP, GitHub, Jira 등 다양한 통합

---

## 리소스

- **GitHub**: [github.com/steveyegge/beads](https://github.com/steveyegge/beads)
- **문서**: [beads.dev](https://beads.dev) (가정)
- **MCP 서버**: `beads-mcp` 패키지
- **라이선스**: MIT

---

*이 가이드 시리즈가 Beads를 이해하고 활용하는 데 도움이 되길 바랍니다.*
