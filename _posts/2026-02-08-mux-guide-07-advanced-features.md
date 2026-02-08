---
layout: post
title: "Mux 완벽 가이드 (07) - 고급 기능"
date: 2026-02-08 00:00:00 +0900
categories: [AI 코딩, 개발 도구]
tags: [Mux, Compaction, ModePrompts, InstructionFiles, Hooks, 프로젝트시크릿, 커맨드팔레트]
author: cataclysm99
original_url: "https://github.com/coder/mux"
excerpt: "Opportunistic Compaction, Mode Prompts, Instruction Files, Hooks, 프로젝트 시크릿 등 고급 기능 완벽 가이드"
permalink: /mux-guide-07-advanced-features/
toc: true
related_posts:
  - /blog-repo/2026-02-08-mux-guide-06-vscode-integration
  - /blog-repo/2026-02-08-mux-guide-08-development
---

## Opportunistic Compaction (컨텍스트 압축)

### 개념

Mux는 대화가 길어질수록 컨텍스트 윈도우를 효율적으로 관리하기 위해 여러 압축 전략을 제공합니다.

```
원본 히스토리 (50,000 토큰):
├── 사용자: "Add OAuth2"
├── 에이전트: [플랜 작성... 15,000 토큰]
├── 사용자: "Looks good"
├── 에이전트: [구현... 30,000 토큰]
└── 사용자: "Test it"

압축 후 (5,000 토큰):
└── Summary:
    - Implemented OAuth2 with Google provider
    - Files: src/auth/google.ts, src/routes/auth.ts
    - Tests passing
```

### 압축 방법 비교

| 방법 | 속도 | 컨텍스트 보존 | 비용 | 가역성 |
|------|------|--------------|------|--------|
| **Start Here** | 즉시 | 지능적 | 무료 | ✓ 가능 |
| **/compact** | 느림 (AI 사용) | 지능적 | 토큰 비용 | ✗ 불가능 |
| **/clear** | 즉시 | 없음 | 무료 | ✗ 불가능 |
| **/truncate** | 즉시 | 시간순 | 무료 | ✗ 불가능 |
| **Auto-Compaction** | 자동 | 지능적 | 토큰 비용 | ✗ 불가능 |

---

## Start Here (Opportunistic Compaction)

### 개념

**이미 잘 구조화된 컨텐츠**를 컨텍스트 시작점으로 사용하여 즉시 압축합니다.

### 사용 위치

#### 1. 플랜 메시지

```
Plan 모드:
├── propose_plan 출력
│   ├── ## Context
│   ├── ## Evidence
│   └── ## Implementation
│
└── [🎯 Start Here] 버튼

클릭 시:
→ 플랜 내용만 남기고 이전 히스토리 삭제
→ 새로운 대화 시작점
```

#### 2. 최종 Assistant 메시지

```
Exec 모드:
└── 에이전트 최종 응답
    ├── "Implemented OAuth2..."
    ├── "Files modified: ..."
    ├── "Tests passing"
    └── [🎯 Start Here] 버튼

클릭 시:
→ 이 메시지만 남기고 이전 삭제
→ 작업 결과만 보존
```

### 워크플로우

```
1. Plan 모드: 플랜 작성
2. propose_plan 출력
3. [🎯 Start Here] 클릭
4. 이전 조사/탐색 히스토리 삭제
5. 플랜만 남김 (깔끔한 시작점)
6. Exec 모드: 플랜 기반 구현
```

### 가역성

```
Start Here는 유일한 가역 압축 방법

1. [🎯 Start Here] 클릭
2. 미리보기 표시
   - 새 시작점: [플랜 또는 메시지]
   - 삭제될 메시지: [이전 히스토리]
3. 확인 또는 취소
4. 확인 시 영구 삭제
```

---

## Manual Compaction Commands

### /compact (AI 요약)

#### 기본 사용

```bash
/compact
```

**동작**:
1. AI 모델이 전체 대화 분석
2. 중요 정보 추출 및 요약
3. 원본 히스토리 교체
4. 컨텍스트 크기 감소

#### 옵션

```bash
# 요약 크기 제한 (토큰)
/compact -t 5000

# 압축 모델 선택
/compact -m haiku

# 조합
/compact -m haiku -t 8000
```

#### 자동 계속 메시지

```bash
# 단일 라인
/compact
Continue implementing the auth system

# 멀티 라인
/compact
Now let's refactor the middleware to use the new auth context.
Make sure to add tests for the error cases.
```

**동작**:
1. 압축 완료
2. 자동으로 "Continue implementing..." 메시지 전송
3. 에이전트가 작업 계속

### /clear (전체 삭제)

```bash
/clear
```

**효과**:
- 모든 대화 히스토리 즉시 삭제
- 복구 불가능
- 완전히 새로운 대화 시작

### /truncate (단순 잘라내기)

```bash
# 50% 삭제 (오래된 메시지부터)
/truncate 50

# 75% 삭제
/truncate 75

# 전체 삭제 (= /clear)
/truncate 100
```

**특징**:
- AI 사용 안 함 (빠름)
- 시간순 보존
- 비용 없음

#### OpenAI 제한사항

```
/truncate는 OpenAI 모델에서 작동하지 않음

원인: Responses API 서버 측 상태 관리
대안:
- /compact 사용
- /clear 사용
- 자동 truncation (기본 활성화)
```

---

## Auto-Compaction (자동 압축)

### Usage-Based (사용량 기반)

```
설정: Costs → Context Usage

┌───────────────────────────────────┐
│  Context Usage                    │
├───────────────────────────────────┤
│  Current: 140,000 / 200,000 (70%) │
│  ███████████████████░░░░░░░       │
│                                   │
│  Auto-Compact Threshold: 70%      │
│  [────────────────██──────]        │
└───────────────────────────────────┘
```

#### 동작

```
1. 컨텍스트 사용량 모니터링
2. 70% (기본값) 도달 시 경고
   "Auto-Compact in 12% usage"

3. 사용자가 메시지 전송
   → 70% 이상이면 자동 압축
   → 압축 완료 후 메시지 자동 전송
```

#### 설정

```
1. Costs 탭 → Context Usage
2. 파란 마커 드래그 (0-90%)
3. 모델별 저장
4. 100% = 비활성화
```

#### Force-Compaction

```
스트리밍 중 70% + 5% 초과 시:
1. 스트리밍 중단
2. 자동 압축
3. 대화 자동 재개
```

### Idle-Based (유휴 기반)

```bash
# 24시간 후 자동 압축
/idle 24

# 48시간 후 자동 압축
/idle 48

# 비활성화
/idle off
```

#### 동작

```
1. 워크스페이스별 마지막 활동 추적
2. 설정 시간(예: 24시간) 경과
3. 비활성 워크스페이스 자동 압축
4. 💤📦 배지 표시
```

#### 조건

```
압축 대상:
- 비활성 시간 >= 설정 시간
- 스트리밍 중 아님
- 이미 압축되지 않음

제외:
- 활성 워크스페이스
- 스트리밍 중
- 이미 압축됨
```

---

## Mode Prompts (모드별 프롬프트)

> **레거시**: 에이전트 시스템으로 통합 권장

### AGENTS.md에서 설정

```markdown
<!-- ~/projects/my-app/AGENTS.md -->

## Model: sonnet

Be terse and to the point.
Focus on code quality over verbosity.

## Model: opus

Provide detailed explanations and rationale.
Consider edge cases and security implications.

## Tool: bash

- Use `rg` instead of `grep` for searching
- Use `fd` instead of `find` for file listing
- Prefer modern Unix tools (bat, exa, etc.)

## Tool: file_edit_replace_string

- Run `make fmt` after editing files
- Verify syntax before saving

## Tool: status_set

- Set status URL to the Pull Request once opened
```

### Scoped Instructions

```markdown
<!-- AGENTS.md -->

## Model: gpt

Focus on:
- Concise code generation
- Minimal comments
- Follow existing patterns

## Model: gemini-3-pro

Leverage the large context:
- Read entire project before suggesting changes
- Consider all related files
- Provide comprehensive analysis
```

---

## Instruction Files (.muxignore, AGENTS.md)

### AGENTS.md 계층

```
프로젝트 우선순위:

1. <workspace>/AGENTS.md         (최우선)
2. <workspace>/AGENT.md
3. <workspace>/CLAUDE.md
4. ~/.mux/AGENTS.md              (글로벌)
5. Built-in 지침                 (최하위)
```

### AGENTS.local.md

```markdown
<!-- ~/projects/my-app/AGENTS.local.md -->

# 개인 로컬 설정 (gitignored)

## Model: opus

Always explain your reasoning step-by-step.
I prefer verbose explanations.

## Tool: bash

Use verbose flags (-v) for all commands.
```

#### .gitignore 추가

```bash
# .gitignore
AGENTS.local.md
```

### HTML 주석 지원

```markdown
<!-- ~/projects/my-app/AGENTS.md -->

<!-- 이 주석은 에이전트에게 전송되지 않음 -->

<!--
프로젝트 노트 (에디터 전용):
- 이 프로젝트는 TypeScript 4.5 사용
- Node.js 20 필요
- Bun 사용 (npm 아님)
-->

<!-- 실제 지침 시작 -->

## General Rules

Always use TypeScript strict mode.
Prefer functional programming patterns.

<!-- 여기도 주석 (에이전트에게 안 보임) -->
```

---

## Hooks (Pre/Post 훅)

### Init Hook (.mux/init)

#### 기본 예시

```bash
#!/usr/bin/env bash
# .mux/init

set -e

echo "Initializing workspace..."

# 의존성 설치
bun install

# 빌드
bun run build

# 환경 변수 복사 (선택사항)
if [ -f "../.env.example" ]; then
  cp "../.env.example" "$PWD/.env"
fi

echo "Workspace ready!"
```

#### 실행 권한

```bash
chmod +x .mux/init
```

#### 동작

```
워크스페이스 생성 시:
1. 워크스페이스 디렉토리 생성
2. Git 설정 (Worktree 런타임)
3. .mux/init 실행 (백그라운드)
4. 출력 스트리밍 (UI 상단 배너)
5. 성공/실패 상태 표시
```

#### 환경 변수

```bash
#!/usr/bin/env bash
# .mux/init

echo "Runtime: $MUX_RUNTIME"
echo "Project: $MUX_PROJECT_PATH"
echo "Workspace: $MUX_WORKSPACE_NAME"

# 런타임별 동작
if [ "$MUX_RUNTIME" = "local" ]; then
  echo "Running on local machine"
  # 로컬 전용 설정
elif [ "$MUX_RUNTIME" = "ssh" ]; then
  echo "Running on SSH remote"
  # SSH 전용 설정
fi

# 의존성 설치
bun install

# 개발 서버 준비
bun run build
```

#### SSH 워크스페이스

```bash
#!/usr/bin/env bash
# .mux/init

set -e

# SSH 워크스페이스에서 실행됨
echo "Remote workspace initialization"

# 원격 서버에 특화된 설정
export PATH="/opt/custom/bin:$PATH"

# 원격 의존성
sudo apt-get update -qq
sudo apt-get install -y build-essential

# 프로젝트 빌드
npm install
npm run build
```

### Tool Hooks (고급)

> **참고**: [Hooks 문서](https://mux.coder.com/hooks/tools)

```bash
# .mux/hooks/pre-file-edit.sh
#!/usr/bin/env bash
# 파일 편집 전 실행

# 파일 백업
cp "$MUX_FILE_PATH" "$MUX_FILE_PATH.backup"

# .mux/hooks/post-bash.sh
#!/usr/bin/env bash
# Bash 명령 실행 후

# 로그 저장
echo "$MUX_BASH_OUTPUT" >> ~/.mux/bash-log.txt
```

---

## 프로젝트 시크릿 관리

### 개념

프로젝트별 환경 변수를 안전하게 저장하고 에이전트에게 자동 주입합니다.

```
~/.mux/secrets.json (평문 저장)
└── {
      "my-app": {
        "GH_TOKEN": "ghp_abc123...",
        "DATABASE_URL": "postgresql://...",
        "API_KEY": "sk-..."
      }
    }
```

### 설정 방법

```
1. 프로젝트 우클릭 (또는 호버)
2. 🔑 아이콘 클릭
3. Project Secrets 모달 열림
4. 키-값 쌍 추가
   - Name: GH_TOKEN
   - Value: ghp_abc123...
5. Save
```

### 사용 (에이전트)

```bash
# 에이전트가 Bash 도구 사용 시 자동 주입

# 예시 1: GitHub API
gh api /user  # GH_TOKEN 자동 사용

# 예시 2: 환경 변수 참조
echo $DATABASE_URL
# postgresql://user:pass@localhost/db

# 예시 3: 스크립트
node deploy.js  # API_KEY 환경 변수 읽음
```

### 보안 고려사항

```
저장 위치: ~/.mux/secrets.json
암호화: 없음 (평문)
권한: 사용자 전용 (600)

권장:
- 중요 시크릿: 최소한으로
- CI/CD: 별도 관리
- 로컬 개발: 안전하게 사용
```

---

## Agentic Git Identity (에이전트 Git 신원)

### 개념

에이전트 커밋을 사람 커밋과 구별하기 위한 별도 Git 신원 설정

```
사람 커밋:
Author: John Doe <john@example.com>
Committer: John Doe <john@example.com>

에이전트 커밋:
Author: John Doe (Agent) <john+ai@example.com>
Committer: John Doe (Agent) <john+ai@example.com>
```

### 설정 단계

#### 1. GitHub 계정 생성 (선택사항)

```
GitHub: yourname-agent
Email: yourname+ai@example.com

또는

동일 계정, 다른 이메일 사용
```

#### 2. Classic Token 생성

```
https://github.com/settings/tokens

New Token (Classic)
→ Scopes: repo
→ Generate
→ 토큰 복사 (ghp_...)
```

#### 3. 프로젝트 시크릿 설정

```
Mux → 프로젝트 → 🔑

추가:
- GIT_AUTHOR_NAME: "Your Name (Agent)"
- GIT_AUTHOR_EMAIL: "yourname+ai@example.com"
- GIT_COMMITTER_NAME: "Your Name (Agent)"
- GIT_COMMITTER_EMAIL: "yourname+ai@example.com"
```

#### 4. GitHub 인증 설정

```bash
# GitHub CLI 설치
brew install gh  # macOS
winget install GitHub.cli  # Windows

# 인증 설정
gh auth setup-git

# 또는 수동 설정
git config --global credential.https://github.com.helper '!gh auth git-credential'
```

### 대안: Co-Author Attribution

```bash
# .git/hooks/prepare-commit-msg
#!/bin/bash

COMMIT_MSG_FILE=$1
COMMIT_SOURCE=$2

# Mux에서만 실행
if [ -z "$MUX_RUNTIME" ]; then
  exit 0
fi

# Merge 커밋 제외
if [ "$COMMIT_SOURCE" = "merge" ]; then
  exit 0
fi

# Co-author 추가
if ! grep -q "Co-authored-by:" "$COMMIT_MSG_FILE"; then
  echo "" >> "$COMMIT_MSG_FILE"
  echo "Co-authored-by: AI Assistant <ai@example.com>" >> "$COMMIT_MSG_FILE"
fi
```

```bash
chmod +x .git/hooks/prepare-commit-msg
```

### 비교

| 방법 | 장점 | 단점 |
|------|------|------|
| **별도 계정** | 완전 분리, 브랜치 보호 규칙 | GitHub 계정 추가 필요 |
| **Co-author Hook** | 단일 계정, 명확한 귀속 | 분리 덜함, 저장소별 훅 |

---

## 커맨드 팔레트 (Cmd+Shift+P)

### 기본 커맨드

```
⌘+Shift+P / Ctrl+Shift+P

주요 커맨드:
- Add Project
- New Workspace
- Change Model
- Change Agent (Switch Mode)
- Delete Workspace
- Refresh Workspaces
- Open Settings
```

### 빠른 전환 (Cmd+P)

```
⌘+P / Ctrl+P

빠른 토글:
- 파일 검색 (프로젝트 내)
- 워크스페이스 검색
- 커맨드 검색 (>)
```

### 슬래시 명령어

```
채팅 입력창에 "/" 입력
→ 자동완성 목록:

/compact [-m <model>] [-t <tokens>]
/clear
/truncate <percentage>
/model <model>
/idle <hours|off>

예시:
/compact -m haiku -t 5000
/model opus
/idle 24
```

### 키바인딩

| 기능 | macOS | Windows/Linux |
|------|-------|---------------|
| **Command Palette** | `⌘+Shift+P` | `Ctrl+Shift+P` |
| **Quick Open** | `⌘+P` | `Ctrl+P` |
| **Settings** | `⌘+,` | `Ctrl+,` |
| **Agent/Mode 전환** | `⌘+Shift+M` | `Ctrl+Shift+M` |
| **Model 전환** | `⌘+/` | `Ctrl+/` |
| **Workspace 1-9** | `⌘+1~9` | `Ctrl+1~9` |

---

## 고급 워크플로우

### 워크플로우 1: Plan → Compact → Exec

```
1. Plan 모드: 플랜 작성
   → propose_plan

2. [🎯 Start Here] 클릭
   → 플랜만 남기고 탐색 히스토리 삭제

3. Exec 모드: 구현
   → 플랜 기반으로 깔끔하게 시작

4. 구현 완료 후 다시 압축
   → /compact
   → 결과만 요약
```

### 워크플로우 2: Multi-Workspace + Agentic Identity

```
워크스페이스 1 (에이전트):
- Agentic Git Identity 설정
- Exec 모드로 기능 구현
- 자동 커밋 (Agent 신원)

워크스페이스 2 (사용자):
- Local 런타임
- 수동 코드 리뷰
- 수동 커밋 (사용자 신원)

GitHub:
- Commits 탭에서 에이전트/사용자 구별
- 에이전트 커밋만 선택적 리뷰
```

### 워크플로우 3: Cost Optimization

```
Plan 모드: Claude Opus (최고 품질)
→ 플랜 작성

[🎯 Start Here]
→ 플랜만 보존

Exec 모드: Claude Sonnet (균형)
→ 구현

Explore 서브에이전트: Claude Haiku (속도)
→ 탐색

Auto-Compact: 70%
→ 자동 압축

결과:
- 총 비용: $5.00
- Opus: $1.50 (플랜만)
- Sonnet: $2.50 (구현)
- Haiku: $1.00 (탐색)
```

---

## 다음 단계

고급 기능을 마스터했다면:

1. **[챕터 08: 개발 및 확장](/blog-repo/mux-guide-08-development)** - Mux 자체 개발 및 커스터마이징
2. **실전 프로젝트** - 실제 프로젝트에 Mux 적용
3. **커뮤니티 참여** - Discord, GitHub Issues, PR 기여

---

## 참고 자료

- [Compaction 문서](https://mux.coder.com/workspaces/compaction/)
- [Instruction Files](https://mux.coder.com/agents/instruction-files)
- [Init Hooks](https://mux.coder.com/hooks/init)
- [Project Secrets](https://mux.coder.com/config/project-secrets)
- [Agentic Git Identity](https://mux.coder.com/config/agentic-git-identity)
