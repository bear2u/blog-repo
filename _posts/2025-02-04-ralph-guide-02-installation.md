---
layout: post
title: "Ralph 가이드 02 - 설치 및 시작"
date: 2025-02-04
categories: [AI, Claude Code, Ralph]
tags: [ralph, installation, setup, quickstart]
permalink: /ralph-guide-02-installation/
---

# 설치 및 시작

## 사전 요구사항

### 필수 요구사항

| 요구사항 | 최소 버전 | 확인 방법 |
|----------|-----------|-----------|
| Claude Code CLI | 최신 | `claude --version` |
| Node.js | 18+ | `node --version` |
| Git | 2.0+ | `git --version` |
| Bash | 4.0+ | `bash --version` |

### 선택적 요구사항

| 요구사항 | 용도 |
|----------|------|
| tmux | 모니터링 대시보드 |
| jq | JSON 처리 (일부 기능) |

## 글로벌 설치

Ralph를 한 번 설치하면 모든 프로젝트에서 사용할 수 있습니다.

```bash
# 1. 저장소 클론
git clone https://github.com/frankbria/ralph-claude-code.git
cd ralph-claude-code

# 2. 설치 스크립트 실행
./install.sh

# 3. 설치 확인
ralph --version
ralph-enable --help
```

### 설치 스크립트가 하는 일

```bash
# install.sh 내부 동작
#!/bin/bash

# 1. 글로벌 bin 디렉토리 생성
mkdir -p ~/.local/bin

# 2. 스크립트 복사
cp scripts/* ~/.local/bin/

# 3. PATH 설정 (필요한 경우)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# 4. 권한 설정
chmod +x ~/.local/bin/ralph*
```

### 수동 설치

설치 스크립트 없이 수동 설치:

```bash
# 필요한 파일만 복사
cp scripts/ralph ~/.local/bin/
cp scripts/ralph-enable ~/.local/bin/
cp scripts/ralph-import ~/.local/bin/
cp scripts/ralph-setup ~/.local/bin/
cp lib/* ~/.local/lib/ralph/

# 실행 권한 부여
chmod +x ~/.local/bin/ralph*
```

## 프로젝트 초기화

### 방법 1: ralph-enable (기존 프로젝트)

```bash
cd my-existing-project
ralph-enable
```

인터랙티브 위저드가 실행됩니다:

```
Ralph Enable Wizard
==================

Phase 1: Environment Detection
------------------------------
Detected project type: typescript
Detected package manager: npm
Git repository: yes

Phase 2: Task Source Selection
------------------------------
Where should Ralph get initial tasks from?
[1] GitHub Issues
[2] Existing PRD/Requirements file
[3] Start with empty task list
[4] Import from markdown file

Enter choice (1-4): 3

Phase 3: Configuration
------------------------------
Creating .ralph/ directory structure...

Phase 4: File Generation
------------------------------
Created: .ralph/PROMPT.md
Created: .ralph/fix_plan.md
Created: .ralph/AGENT.md
Created: .ralph/specs/
Created: .ralph/logs/
Created: .ralphrc

Ralph is now enabled for this project.
Run 'ralph --monitor' to start development.
```

### 방법 2: ralph-import (PRD 가져오기)

기존 요구사항 문서가 있는 경우:

```bash
# 기존 PRD로 새 프로젝트 생성
ralph-import requirements.md my-new-app
cd my-new-app
```

ralph-import가 하는 일:
1. 새 디렉토리 생성
2. git 초기화
3. PRD 분석하여 fix_plan.md 생성
4. PROMPT.md에 컨텍스트 설정

### 방법 3: ralph-setup (새 프로젝트)

처음부터 새로 시작:

```bash
ralph-setup my-project
cd my-project
```

## 생성되는 파일 구조

```
my-project/
├── .ralph/
│   ├── PROMPT.md        # 프로젝트 비전과 지침
│   ├── fix_plan.md      # 작업 체크리스트
│   ├── AGENT.md         # 빌드/테스트 명령어
│   ├── specs/           # 상세 스펙 문서
│   │   └── stdlib/      # 표준 패턴 문서
│   ├── logs/            # 실행 로그
│   └── status.json      # 런타임 상태
└── .ralphrc             # 프로젝트 설정
```

## Quick Start: 첫 번째 프로젝트

단계별로 Todo CLI 앱을 만들어 봅시다.

### Step 1: 프로젝트 생성

```bash
mkdir todo-cli
cd todo-cli
npm init -y
git init
```

### Step 2: Ralph 활성화

```bash
ralph-enable
# -> 옵션 3 선택 (빈 작업 목록)
```

### Step 3: PROMPT.md 작성

```markdown
# Ralph Development Instructions

## Context
You are Ralph, an autonomous AI development agent building a CLI
todo application in Node.js.

## Current Objectives
1. Create a command-line todo app with add, list, complete, delete
2. Store todos in ~/.todos.json
3. Use commander.js for argument parsing
4. Write unit tests with Jest

## Key Principles
- Keep the code simple and readable
- Use async/await for file operations
- Provide clear error messages
```

### Step 4: fix_plan.md 작성

```markdown
# Fix Plan - Todo CLI

## Priority 1: Core Structure
- [ ] Set up package.json with dependencies
- [ ] Create src/index.js entry point
- [ ] Create src/storage.js for file operations

## Priority 2: Commands
- [ ] Implement `todo add "task"` command
- [ ] Implement `todo list` command
- [ ] Implement `todo complete <id>` command
- [ ] Implement `todo delete <id>` command

## Priority 3: Polish
- [ ] Add --help documentation
- [ ] Handle edge cases
- [ ] Write unit tests
```

### Step 5: Ralph 실행

```bash
ralph --monitor
```

### Step 6: 결과 확인

```bash
# 생성된 파일 확인
ls -la src/

# 테스트 실행
npm test

# CLI 사용해보기
node src/index.js add "Learn Ralph"
node src/index.js list
```

## 설치 문제 해결

### PATH 문제

```bash
# ralph 명령어를 찾을 수 없는 경우
export PATH="$HOME/.local/bin:$PATH"

# 영구 적용
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 권한 문제

```bash
# 실행 권한 확인
ls -la ~/.local/bin/ralph*

# 권한 부여
chmod +x ~/.local/bin/ralph*
```

### Claude Code 연결 문제

```bash
# Claude Code 상태 확인
claude --version

# 인증 확인
claude auth status

# 재인증
claude auth login
```

---

**이전 장:** [소개](/ralph-guide-01-intro/) | **다음 장:** [파일 구조](/ralph-guide-03-files/)
