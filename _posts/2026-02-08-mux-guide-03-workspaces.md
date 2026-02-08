---
layout: post
title: "Mux 완벽 가이드 (03) - 워크스페이스 관리"
date: 2026-02-08 00:00:00 +0900
categories: [AI 코딩, 개발 도구]
tags: [Mux, 워크스페이스, Git, Worktree, SSH, 원격개발, 병렬작업]
author: cataclysm99
original_url: "https://github.com/coder/mux"
excerpt: "Local, Worktree, SSH 런타임을 활용한 격리된 병렬 개발 환경 구축 가이드"
permalink: /mux-guide-03-workspaces/
toc: true
related_posts:
  - /blog-repo/2026-02-08-mux-guide-02-installation
  - /blog-repo/2026-02-08-mux-guide-04-agents
---

## 워크스페이스 개념

워크스페이스는 Mux의 핵심 개념으로, **독립적인 개발 환경과 채팅 세션**을 제공합니다.

### 핵심 특징

```
┌─────────────────────────────────────┐
│  프로젝트: my-app                    │
├─────────────────────────────────────┤
│  워크스페이스 1: feature-auth-x7k2  │ ← Local 런타임
│  - 채팅 히스토리 (독립)             │
│  - 작업 디렉토리: ~/projects/my-app │
├─────────────────────────────────────┤
│  워크스페이스 2: fix-bug-p3m9       │ ← Worktree 런타임
│  - 채팅 히스토리 (독립)             │
│  - 작업 디렉토리: ~/.mux/src/...    │
├─────────────────────────────────────┤
│  워크스페이스 3: deploy-staging-k1n4│ ← SSH 런타임
│  - 채팅 히스토리 (독립)             │
│  - 작업 디렉토리: user@remote:/app  │
└─────────────────────────────────────┘
```

### 주요 특징

| 특성 | 설명 |
|------|------|
| **독립 세션** | 각 워크스페이스는 별도의 채팅 히스토리 유지 |
| **병렬 실행** | 여러 워크스페이스에서 동시 작업 가능 |
| **런타임 선택** | Local/Worktree/SSH 중 선택 |
| **Git 통합** | 브랜치 상태 추적 UI 제공 |
| **컨텍스트 격리** | 각 워크스페이스의 대화 내용 독립 |

---

## 런타임 비교

### 개요 표

| 런타임 | 격리 수준 | Git 필요 | 파일 충돌 위험 | 사용 사례 |
|--------|----------|---------|--------------|----------|
| **Local** | 없음 | 선택사항 | 높음 | 빠른 일회성 작업 |
| **Worktree** | 파일시스템 | 필수 | 없음 | 병렬 기능 개발 |
| **SSH** | 완전 격리 | 원격 서버 | 없음 | 원격 서버 작업, 보안 격리 |

### 런타임 선택 결정 트리

```
작업 시작
    │
    ├─ 빠른 수정/탐색? ──→ Local
    │
    ├─ 병렬 기능 개발? ──→ Worktree
    │
    ├─ 원격 서버 작업? ──→ SSH
    │
    └─ 보안 격리 필요? ──→ SSH
```

---

## Local 런타임

### 개념

프로젝트 디렉토리에서 **직접 작업**하는 런타임입니다. 파일시스템 격리가 없으며, 에이전트가 실제 작업 디렉토리를 수정합니다.

```
~/projects/my-app/        ← 프로젝트 디렉토리
└── (모든 파일 직접 수정)
```

### 사용 사례

#### 1. 빠른 수정

```bash
# 시나리오: 테스트 실패 긴급 수정
워크스페이스: quick-fix-tests
런타임: Local

프롬프트: "Fix the failing Jest tests in src/__tests__/"
```

#### 2. 탐색 및 분석

```bash
# 시나리오: 코드베이스 이해
워크스페이스: explore-architecture
런타임: Local

프롬프트: "Explain the database connection flow"
```

#### 3. 기존 변경사항과 함께 작업

```bash
# 시나리오: 로컬에 미커밋 변경사항이 있는 상태
git status
# modified:   src/utils.ts (수동 작업 중)

워크스페이스: enhance-utils
런타임: Local

프롬프트: "Add input validation to the parseData function"
# 에이전트가 기존 변경사항과 함께 작업
```

### 주의사항

#### 동시 실행 경고

```
⚠️ Warning: Another local workspace is actively streaming
   Workspace: feature-auth-x7k2
   Running them simultaneously may cause conflicts
```

> **해결책**: 하나의 워크스페이스만 활성화하거나 Worktree 런타임 사용

#### 작업 디렉토리 직접 수정

```bash
# 에이전트가 수정한 파일이 즉시 작업 디렉토리에 반영됨
git status
# modified:   src/auth.ts  ← 에이전트 수정
# modified:   src/utils.ts ← 사용자 수정

# 주의: git add -p로 선택적 스테이징 권장
git add -p src/auth.ts
```

### 파일시스템

```
작업 디렉토리 = 워크스페이스 디렉토리
추가 디렉토리 생성 없음
```

---

## Worktree 런타임

### 개념

[Git Worktree](https://git-scm.com/docs/git-worktree)를 사용하여 **별도 디렉토리에서 작업**합니다. 각 워크스페이스가 독립된 파일시스템을 가지지만 `.git` 디렉토리는 공유합니다.

```
~/projects/my-app/.git/         ← 공유 Git 데이터베이스
│
├── ~/projects/my-app/          ← 메인 체크아웃
│
├── ~/.mux/src/my-app-main/feature-auth-x7k2/  ← 워크스페이스 1
│
└── ~/.mux/src/my-app-main/fix-bug-p3m9/       ← 워크스페이스 2
```

### Git Worktree 원리

#### 1. 공유 저장소, 독립 작업 트리

```bash
# 메인 저장소
~/projects/my-app/.git/
└── (모든 커밋, 브랜치 정보)

# Worktree 1
~/.mux/src/my-app-main/feature-auth-x7k2/
├── .git → ~/projects/my-app/.git/worktrees/feature-auth-x7k2
└── src/, tests/, ... (독립 파일)

# Worktree 2
~/.mux/src/my-app-main/fix-bug-p3m9/
├── .git → ~/projects/my-app/.git/worktrees/fix-bug-p3m9
└── src/, tests/, ... (독립 파일)
```

#### 2. 커밋 즉시 공유

```bash
# 워크스페이스 1에서 커밋
cd ~/.mux/src/my-app-main/feature-auth-x7k2
git add src/auth.ts
git commit -m "Add OAuth2 support"

# 메인 저장소에서 즉시 확인 가능
cd ~/projects/my-app
git log --all --graph
# * abc123 (feature-auth-x7k2) Add OAuth2 support
```

### 사용 사례

#### 1. 병렬 기능 개발

```bash
# 워크스페이스 1: 인증 기능
워크스페이스: feature-auth-x7k2
런타임: Worktree
브랜치: feature/oauth2

프롬프트: "Implement OAuth2 authentication with Google provider"

# 워크스페이스 2: 결제 기능 (동시 작업)
워크스페이스: feature-payment-p3m9
런타임: Worktree
브랜치: feature/stripe

프롬프트: "Integrate Stripe payment gateway"
```

#### 2. 버그 수정과 기능 개발 병행

```bash
# 긴급 버그 수정
워크스페이스: hotfix-login-k1n4
런타임: Worktree
브랜치: hotfix/login-timeout

프롬프트: "Fix login timeout issue"

# 동시에 기능 개발 계속
워크스페이스: feature-dashboard-m7p2
런타임: Worktree
브랜치: feature/admin-dashboard

프롬프트: "Continue implementing admin dashboard charts"
```

### 파일시스템 레이아웃

```
~/.mux/src/
└── <project-name>/
    ├── <workspace-1>/
    │   ├── .git → 메인 저장소 링크
    │   └── (프로젝트 파일)
    ├── <workspace-2>/
    │   ├── .git → 메인 저장소 링크
    │   └── (프로젝트 파일)
    └── ...
```

#### 실제 예시

```bash
~/.mux/src/
└── my-app-main/
    ├── feature-auth-x7k2/
    │   ├── .git
    │   ├── src/
    │   ├── tests/
    │   └── package.json
    ├── fix-bug-p3m9/
    │   ├── .git
    │   ├── src/
    │   ├── tests/
    │   └── package.json
    └── explore-arch-k1n4/
        └── ...
```

### 브랜치 관리

#### 자유로운 브랜치 전환

```bash
# 워크스페이스는 브랜치에 고정되지 않음
# 에이전트가 필요에 따라 브랜치 전환 가능

# 예시 1: 새 브랜치 생성
워크스페이스 내부:
git checkout -b feature/new-feature

# 예시 2: 기존 브랜치로 전환
git checkout main

# 예시 3: Detached HEAD
git checkout abc123
```

#### 브랜치 제약 설정 (AGENTS.md)

```markdown
<!-- ~/projects/my-app/AGENTS.md -->

## Git Policy

- Always create feature branches from `main`
- Never commit directly to `main` or `develop`
- Branch naming: `feature/`, `fix/`, `hotfix/`

## Tool: bash

Before creating commits:
1. Ensure you're on a feature branch
2. Run tests: `npm test`
3. Run lint: `npm run lint`
```

### 브랜치 충돌 방지

```bash
# Git 제한: 한 브랜치는 하나의 worktree에서만 체크아웃 가능
# 워크스페이스 1
git checkout feature/auth

# 워크스페이스 2 (동일 브랜치 시도)
git checkout feature/auth
# 오류: fatal: 'feature/auth' is already checked out at '...'

# 해결책: 다른 브랜치 사용
git checkout feature/payment
```

---

## SSH 런타임

### 개념

SSH를 통해 **원격 서버에서 작업**합니다. 로컬과 완전히 격리되며, 프롬프트 인젝션 위험을 원격 머신으로 제한합니다.

```
┌─────────────────────────────────┐
│  로컬 머신 (사용자)              │
│  - Mux 앱 실행                   │
│  - 채팅 UI                       │
│  - API 키 (로컬에만 저장)        │
└──────────────┬──────────────────┘
               │ SSH
               ▼
┌─────────────────────────────────┐
│  원격 서버 (user@remote)         │
│  - Git 아카이브 동기화           │
│  - 에이전트 명령 실행            │
│  - 프로젝트 시크릿 (선택적)      │
└─────────────────────────────────┘
```

### 위협 모델

Mux는 원격 호스트를 **잠재적으로 적대적**으로 취급합니다.

#### 로컬에서 전송되지 않는 것

- ❌ 로컬 SSH 키
- ❌ 로컬 환경 변수
- ❌ 로컬 Git 자격 증명
- ❌ 로컬 API 키 (Anthropic, OpenAI 등)

#### 원격으로 전송되는 것

- ✅ Git 아카이브 (프로젝트 코드)
- ✅ 프로젝트 시크릿 (명시적 설정 시)

### 사용 사례

#### 1. 보안 격리

```bash
# 시나리오: 신뢰할 수 없는 코드베이스 작업
# 프롬프트 인젝션 위험을 원격 머신으로 제한

워크스페이스: audit-third-party-k1n4
런타임: SSH (sandbox-server)

프롬프트: "Analyze the security of this third-party library"
```

#### 2. 고성능 작업

```bash
# 시나리오: CPU 집약적 빌드 또는 테스트
# 로컬 노트북 배터리 절약

워크스페이스: build-production-m7p2
런타임: SSH (build-server)

프롬프트: "Run full integration test suite"
```

#### 3. 원격 서버 관리

```bash
# 시나리오: 프로덕션 서버 직접 작업

워크스페이스: deploy-staging-x7k2
런타임: SSH (staging.example.com)

프롬프트: "Update Nginx configuration for HTTPS"
```

### SSH 호스트 설정

#### 1. ~/.ssh/config

```bash
# ~/.ssh/config
Host build-server
  HostName 192.168.1.100
  User deploy
  IdentityFile ~/.ssh/id_ed25519
  Port 22

Host staging
  HostName staging.example.com
  User ubuntu
  IdentityFile ~/.ssh/staging_key
  ForwardAgent no  # 보안: 에이전트 포워딩 비활성화
```

#### 2. Mux에서 사용

```
New Workspace
→ Runtime: SSH
→ Host: build-server  ← ~/.ssh/config 별칭 사용
```

### 인증 방법

#### 1. 로컬 기본 키 (자동)

```bash
# SSH가 자동으로 확인하는 위치
~/.ssh/id_rsa
~/.ssh/id_ecdsa
~/.ssh/id_ed25519
```

#### 2. SSH 에이전트

```bash
# 키 추가
ssh-add ~/.ssh/my_key_ecdsa

# 확인
ssh-add -l
```

#### 3. 명시적 설정 (~/.ssh/config)

```bash
Host myserver
  HostName 192.168.1.100
  User root
  IdentityFile ~/.ssh/specific_key
```

### 프로젝트 시크릿 (SSH)

원격 서버에 안전하게 시크릿 전달:

```
프로젝트 우클릭 → 🔑 Project Secrets

# 추가 예시
GH_TOKEN=ghp_abc123...
DEPLOY_KEY=ssh-rsa AAAA...
DATABASE_URL=postgresql://...
```

#### 원격 서버에서 사용

```bash
# SSH 워크스페이스 내부
echo $GH_TOKEN  # ghp_abc123...

# 에이전트가 자동으로 사용
gh api /user  # GH_TOKEN 환경 변수 사용
```

### Coder 워크스페이스 통합

[Coder](https://coder.com) 사용 시:

```
Runtime: SSH
Host: coder.<workspace-name>

# Coder SSH config 자동 설정됨
```

---

## 워크스페이스 전환 및 관리

### UI에서 전환

```
좌측 사이드바
→ 프로젝트 확장
→ 워크스페이스 클릭
```

#### 키보드 단축키

```
Cmd+1, Cmd+2, ... (macOS)
Ctrl+1, Ctrl+2, ... (Windows/Linux)

# 워크스페이스 1~9 빠른 전환
```

### Command Palette

```
Cmd+Shift+P / Ctrl+Shift+P
→ "Switch Workspace"
→ 검색 및 선택
```

### 워크스페이스 삭제

```
워크스페이스 우클릭
→ "Delete Workspace"

# 또는
Command Palette → "Delete Workspace"
```

#### 삭제 시 동작

| 런타임 | 파일 삭제 | 채팅 히스토리 | Git 커밋 |
|--------|----------|--------------|---------|
| **Local** | 없음 | 삭제 | 보존 |
| **Worktree** | 워크트리 디렉토리 삭제 | 삭제 | 보존 (공유 저장소) |
| **SSH** | 원격 디렉토리 삭제 | 삭제 | 원격 저장소 상태 유지 |

---

## Git 분기 추적 UI

Mux는 각 워크스페이스의 Git 상태를 실시간으로 추적합니다.

### Git 상태 표시

#### 사이드바 표시

```
my-app
  ├── feature-auth-x7k2 [feature/oauth2 ↑2 ↓1]
  │   - 2 commits ahead, 1 commit behind main
  │
  ├── fix-bug-p3m9 [main =]
  │   - Up to date with origin/main
  │
  └── hotfix-login-k1n4 [hotfix/login ↑1]
      - 1 commit ahead, not pushed
```

#### 아이콘 의미

| 아이콘 | 의미 |
|-------|------|
| `↑2` | 2개 커밋 앞섬 (로컬 → 원격) |
| `↓1` | 1개 커밋 뒤짐 (원격 → 로컬) |
| `=` | 동기화됨 |
| `*` | Uncommitted changes |
| `?` | Untracked files |

### Git 분기 UI 패널

```
┌─────────────────────────────────────┐
│  Git Divergence                     │
├─────────────────────────────────────┤
│  Current: feature/oauth2            │
│  Tracking: origin/main              │
│                                     │
│  Ahead:  2 commits                  │
│    - abc123 Add OAuth2 routes       │
│    - def456 Add Google provider     │
│                                     │
│  Behind: 1 commit                   │
│    - ghi789 Fix database migration  │
│                                     │
│  Actions:                           │
│  [Pull] [Push] [View Diff]          │
└─────────────────────────────────────┘
```

### 충돌 감지

```
⚠️ Potential Conflict Detected

Workspace: feature-auth-x7k2
File: src/auth.ts

- Modified in your workspace (uncommitted)
- Modified in origin/main (1 commit behind)

Recommended actions:
1. Commit your changes
2. Pull latest from origin/main
3. Resolve conflicts
```

---

## 변경사항 리뷰 워크플로우

### 워크플로우 1: 에이전트 편집만

```bash
# 시나리오: 에이전트가 파일 수정, 사용자가 직접 커밋

# 1. 에이전트 작업 완료
워크스페이스: feature-auth-x7k2

# 2. 워크스페이스 디렉토리로 이동
cd ~/.mux/src/my-app-main/feature-auth-x7k2  # Worktree
# 또는
cd ~/projects/my-app  # Local

# 3. 변경사항 확인
git status
git diff

# 4. 선택적 스테이징
git add -p src/auth.ts

# 5. 커밋
git commit -m "Add OAuth2 authentication"
```

### 워크플로우 2: 에이전트 커밋

```bash
# 시나리오: 에이전트가 직접 커밋

프롬프트: "Implement OAuth2 and commit the changes"

에이전트 응답:
1. 파일 수정
2. git add src/auth.ts
3. git commit -m "Add OAuth2 authentication"
4. agent_report (완료 보고)

# 사용자 리뷰 (메인 체크아웃)
cd ~/projects/my-app
git show feature-auth-x7k2  # 워크스페이스 브랜치 확인
git log --all --graph

# 승인 시 푸시
git push origin feature-auth-x7k2
```

### 워크플로우 3: 에이전트가 PR 생성

```bash
# 시나리오: 에이전트가 커밋 + 푸시 + PR 생성

프롬프트: "Implement OAuth2, commit, push, and open a pull request"

에이전트 응답:
1. 파일 수정
2. git add src/auth.ts
3. git commit -m "Add OAuth2 authentication"
4. git push origin feature/oauth2
5. gh pr create --title "Add OAuth2" --body "..."
6. agent_report (PR URL 포함)

# 사용자 리뷰
# GitHub/GitLab UI에서 PR 검토
```

> **권장**: [Agentic Git Identity](/blog-repo/mux-guide-07-advanced-features#agentic-git-identity) 설정으로 에이전트 커밋 구별

---

## 기능 리뷰 (UI, 동작)

### Detached HEAD로 리뷰

```bash
# 시나리오: 워크스페이스 브랜치를 메인 체크아웃에서 테스트

# 1. 에이전트가 커밋 완료
워크스페이스: feature-auth-x7k2 [feature/oauth2]

# 2. 메인 체크아웃에서 detached HEAD로 체크아웃
cd ~/projects/my-app
git checkout --detach feature-auth-x7k2

# 3. 테스트
npm install
npm run dev  # 앱 실행 및 UI 확인
npm test     # 테스트 실행

# 4. 승인 후 원래 브랜치로 복귀
git checkout main
```

> **팁**: Detached HEAD는 브랜치 잠금 우회 (Worktree 제한)

### 워크스페이스 디렉토리에서 직접 실행

```bash
# 시나리오: 빠른 반복 테스트

# 워크스페이스 디렉토리로 이동
cd ~/.mux/src/my-app-main/feature-auth-x7k2

# 개발 서버 실행
npm run dev

# 브라우저에서 확인
# http://localhost:3000

# 에이전트에게 추가 수정 요청
프롬프트: "Change the login button color to blue"

# 핫 리로드로 즉시 확인
```

---

## 고급 워크스페이스 패턴

### 패턴 1: Explore + Exec 분리

```bash
# 워크스페이스 1: 읽기 전용 탐색
워크스페이스: explore-codebase-k1n4
런타임: Local
에이전트: Ask 모드

프롬프트: "Explain the authentication flow"

# 워크스페이스 2: 구현
워크스페이스: feature-auth-x7k2
런타임: Worktree
에이전트: Exec 모드

프롬프트: "Implement OAuth2 based on the analysis from explore-codebase"
```

### 패턴 2: 버전별 테스트

```bash
# 워크스페이스 1: 현재 버전
워크스페이스: test-v1-m7p2
런타임: Worktree
브랜치: main

프롬프트: "Run integration tests"

# 워크스페이스 2: 새 버전
워크스페이스: test-v2-k1n4
런타임: Worktree
브랜치: feature/v2-refactor

프롬프트: "Run integration tests and compare with v1"
```

### 패턴 3: 보안 감사 + 수정

```bash
# 워크스페이스 1: 보안 감사 (SSH 격리)
워크스페이스: security-audit-x7k2
런타임: SSH (sandbox-server)
에이전트: Explore (읽기 전용)

프롬프트: "Audit the codebase for security vulnerabilities"

# 워크스페이스 2: 수정 (로컬)
워크스페이스: fix-security-p3m9
런타임: Worktree
에이전트: Exec

프롬프트: "Fix the SQL injection issue in src/db/query.ts"
```

---

## 워크스페이스 설정 파일

### 세션 저장 위치

```bash
~/.mux/sessions/
└── <workspace-id>/
    ├── chat.jsonl          # 채팅 히스토리
    ├── metadata.json       # 워크스페이스 메타데이터
    └── compaction-state.json  # 압축 상태
```

### 플랜 파일 위치

```bash
~/.mux/plans/
└── <project-name>/
    └── <workspace-name>.md  # Plan 모드 플랜 파일
```

---

## 문제 해결

### Worktree 생성 실패

```bash
# 오류: fatal: not a git repository

# 해결책: Git 초기화
cd ~/projects/my-app
git init
git add .
git commit -m "Initial commit"
```

### SSH 연결 실패

```bash
# 오류: Permission denied (publickey)

# 해결책 1: SSH 키 확인
ssh-add -l

# 해결책 2: 수동 연결 테스트
ssh user@hostname

# 해결책 3: ~/.ssh/config 확인
cat ~/.ssh/config
```

### 브랜치 충돌

```bash
# 오류: 'feature/auth' is already checked out

# 해결책 1: 다른 브랜치 사용
git checkout -b feature/auth-v2

# 해결책 2: 기존 워크스페이스 확인
git worktree list
```

### 디스크 공간 부족 (Worktree)

```bash
# 워크트리 정리
cd ~/projects/my-app
git worktree prune

# Mux 세션 정리
rm -rf ~/.mux/sessions/<old-workspace-id>
```

---

## 성능 최적화

### Worktree 의존성 공유

```bash
# 문제: 각 워크트리마다 node_modules 재설치
du -sh ~/.mux/src/my-app-main/*/node_modules
# 500MB x 5 워크스페이스 = 2.5GB

# 해결책: 심볼릭 링크 (주의: 일부 도구 호환성 문제)
ln -s ~/projects/my-app/node_modules ~/.mux/src/.../node_modules

# 또는 pnpm/yarn workspaces 사용
```

### SSH 아카이브 캐싱

```bash
# Mux는 Git 아카이브를 원격으로 전송
# 대용량 저장소는 초기 동기화 느림

# 최적화: .gitignore 적극 활용
echo "node_modules/" >> .gitignore
echo "dist/" >> .gitignore
echo ".mux/" >> .gitignore
```

---

## 다음 단계

워크스페이스 관리를 마스터했다면:

1. **[챕터 04: 에이전트 시스템](/blog-repo/mux-guide-04-agents)** - Plan/Exec 모드, 서브에이전트 활용
2. **[챕터 06: VS Code 통합](/blog-repo/mux-guide-06-vscode-integration)** - 워크스페이스 점프 기능
3. **[챕터 07: 고급 기능](/blog-repo/mux-guide-07-advanced-features)** - Agentic Git Identity, 프로젝트 시크릿

---

## 참고 자료

- [워크스페이스 문서](https://mux.coder.com/workspaces/)
- [Local 런타임](https://mux.coder.com/runtime/local)
- [Worktree 런타임](https://mux.coder.com/runtime/worktree)
- [SSH 런타임](https://mux.coder.com/runtime/ssh)
- [Git Worktree 공식 문서](https://git-scm.com/docs/git-worktree)
