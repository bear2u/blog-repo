---
layout: post
title: "oh-my-claudecode 완벽 가이드 (02) - 설치 및 빠른 시작"
date: 2026-02-09
permalink: /oh-my-claudecode-guide-02-quick-start/
author: Yeachan Heo
categories: [AI 코딩, CLI]
tags: [Claude Code, Multi-Agent, Orchestration, AI, CLI, Autopilot, Ultrawork]
original_url: "https://github.com/Yeachan-Heo/oh-my-claudecode"
excerpt: "3단계만으로 oh-my-claudecode를 설치하고 첫 멀티-에이전트 작업을 실행하는 완벽 가이드입니다."
---

## 요구사항

oh-my-claudecode를 사용하기 전에 다음이 필요합니다:

### 1. Claude Code CLI

Claude Code 명령줄 인터페이스가 설치되어 있어야 합니다.

**설치 확인:**
```bash
claude --version
```

**아직 설치하지 않았다면:**
```bash
# macOS/Linux
curl -fsSL https://claude.ai/install.sh | sh

# Windows (PowerShell)
irm https://claude.ai/install.ps1 | iex
```

### 2. Claude 구독 또는 API 키

다음 중 하나가 필요합니다:

**옵션 A: Claude 구독**
- Claude Pro 구독 (월 $20)
- Claude Max 구독 (월 $200)

**옵션 B: Anthropic API 키**
- [Anthropic Console](https://console.anthropic.com/)에서 API 키 발급
- 종량제 요금 (사용량에 따라 청구)

**API 키 설정:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. 시스템 요구사항

- **OS**: macOS, Linux, 또는 Windows (WSL2 권장)
- **Node.js**: 16.x 이상 (플러그인 설치용)
- **Git**: 버전 관리용
- **디스크 공간**: 최소 100MB

## 설치

oh-my-claudecode는 단 3단계로 설치됩니다.

### Step 1: Plugin Marketplace 추가

Claude Code의 플러그인 마켓플레이스를 활성화합니다:

```bash
claude plugin marketplace add
```

**예상 출력:**
```
✓ Plugin marketplace enabled
✓ Fetching available plugins...
✓ Found 47 plugins
```

### Step 2: Plugin 설치

oh-my-claudecode 플러그인을 설치합니다:

```bash
claude plugin install oh-my-claudecode
```

**예상 출력:**
```
✓ Downloading oh-my-claudecode v1.0.0
✓ Installing dependencies...
✓ Configuring 32 agents...
✓ Setting up execution modes...
✓ oh-my-claudecode installed successfully!
```

### Step 3: 초기 설정 실행

설치 마법사로 기본 설정을 완료합니다:

```bash
claude omc-setup
```

**설정 과정:**

```
Welcome to oh-my-claudecode setup!

? Select your default execution mode: (Use arrow keys)
  ❯ Autopilot (Fast, autonomous workflows)
    Ultrawork (Maximum parallelism)
    Ralph (Persistent execution)
    Ecomode (Budget-conscious)

? Enable real-time HUD statusline? (Y/n) Y

? Default cost optimization: (Use arrow keys)
  ❯ Balanced (Recommended)
    Performance (Faster, higher cost)
    Economy (Slower, lower cost)

? Multi-AI integration:
  [ ] Gemini CLI
  [ ] Codex CLI
  (Press space to select, enter to confirm)

✓ Configuration saved to ~/.claude/omc.config.json
✓ oh-my-claudecode is ready to use!
```

## 설치 확인

설치가 정상적으로 완료되었는지 확인합니다:

```bash
claude omc-status
```

**정상 출력:**
```
oh-my-claudecode Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Version:        1.0.0
Status:         ✓ Active
Active Agents:  0/32
Default Mode:   Autopilot

Available Modes:
  ✓ Autopilot     Fast autonomous workflows
  ✓ Ultrawork     Maximum parallelism
  ✓ Ralph         Persistent execution
  ✓ Ultrapilot    Multi-component systems
  ✓ Ecomode       Budget-conscious
  ✓ Swarm         Coordinated parallel tasks
  ✓ Pipeline      Sequential processing

Multi-AI Integration:
  ✗ Gemini CLI    Not configured
  ✗ Codex CLI     Not configured

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ready to orchestrate!
```

## 첫 작업 실행

이제 oh-my-claudecode를 사용해 첫 프로젝트를 만들어봅시다!

### 예제: REST API 구축

간단한 REST API를 자동으로 구축해봅니다:

```bash
cd ~/projects
mkdir my-api && cd my-api

# OMC로 전체 API 자동 생성
claude chat
```

Claude Code 대화형 모드에서:

```
> autopilot: build a REST API with Node.js and Express that has:
> - User authentication (JWT)
> - CRUD operations for tasks
> - Rate limiting
> - Input validation
> - Error handling
> - Swagger documentation
```

**실행 과정:**

```
╔═══════════════════════════════════════════════════════════╗
║  OMC Autopilot Mode                                       ║
╠═══════════════════════════════════════════════════════════╣
║  Analyzing requirements...                                ║
║  ✓ Identified 6 major components                          ║
║  ✓ Planning execution strategy                            ║
║                                                            ║
║  Deploying agents:                                        ║
║  [1] Architect    - Designing system structure            ║
║  [2] Backend      - Setting up Express server             ║
║  [3] Database     - Creating database schema              ║
║  [4] Security     - Implementing authentication           ║
║  [5] Testing      - Writing test suite                    ║
║  [6] Documentation - Generating Swagger docs              ║
║                                                            ║
║  Progress: ████████████████░░░░  78%                      ║
║  Est. Time Remaining: 1m 15s                              ║
╚═══════════════════════════════════════════════════════════╝
```

**완료 후 결과:**

```
✓ Task completed successfully!

Created files:
  package.json
  src/
    server.js
    config/
      database.js
    models/
      User.js
      Task.js
    controllers/
      authController.js
      taskController.js
    middleware/
      auth.js
      rateLimit.js
      validator.js
      errorHandler.js
    routes/
      auth.js
      tasks.js
    utils/
      jwt.js
  tests/
    auth.test.js
    tasks.test.js
  swagger.yaml
  README.md

Next steps:
1. Install dependencies: npm install
2. Configure environment: cp .env.example .env
3. Run migrations: npm run migrate
4. Start server: npm start
5. View API docs: http://localhost:3000/api-docs
```

**프로젝트 실행:**

```bash
npm install
cp .env.example .env
# .env 파일에서 DATABASE_URL과 JWT_SECRET 설정
npm run migrate
npm start
```

**API 테스트:**

```bash
# 사용자 등록
curl -X POST http://localhost:3000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john",
    "email": "john@example.com",
    "password": "securepass123"
  }'

# 로그인
curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john@example.com",
    "password": "securepass123"
  }'

# 태스크 생성 (JWT 토큰 필요)
curl -X POST http://localhost:3000/api/tasks \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "title": "Complete OMC guide",
    "description": "Write comprehensive documentation",
    "status": "pending"
  }'
```

## 업데이트

oh-my-claudecode를 최신 버전으로 업데이트하는 방법:

### 자동 업데이트

```bash
claude plugin update oh-my-claudecode
```

**예상 출력:**
```
✓ Checking for updates...
✓ Found version 1.1.0 (current: 1.0.0)
✓ Downloading update...
✓ Installing new agents...
✓ Migrating configuration...
✓ oh-my-claudecode updated successfully!

New features in 1.1.0:
  • 5 new specialized agents
  • Improved parallelization algorithm
  • 20% faster execution
  • Enhanced cost optimization
```

### 모든 플러그인 업데이트

```bash
claude plugin update --all
```

### 특정 버전 설치

```bash
claude plugin install oh-my-claudecode@1.0.5
```

## 문제 해결

설치나 실행 중 문제가 발생하면 `doctor` 명령어를 사용하세요.

### 진단 실행

```bash
claude omc-doctor
```

**진단 보고서:**

```
oh-my-claudecode Health Check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

System Requirements:
  ✓ Claude Code CLI    v2.1.0
  ✓ Node.js            v18.17.0
  ✓ Git                v2.39.0
  ✓ Disk Space         2.3 GB available

Plugin Status:
  ✓ oh-my-claudecode   v1.0.0 installed
  ✓ Configuration      Valid
  ✓ 32 Agents          All loaded

API Connection:
  ✓ Anthropic API      Connected
  ✓ API Key            Valid
  ✓ Rate Limits        Normal

Common Issues:
  ✗ Warning: Low disk space (< 1GB)
    → Recommendation: Free up disk space

Performance:
  ✓ Agent response time: 234ms (Good)
  ✓ Parallelization:     Working
  ✓ HUD statusline:      Active

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall Health: Good
```

### 일반적인 문제와 해결책

#### 문제 1: "Agent not responding"

**증상:**
```
Error: Agent timeout after 30s
```

**해결:**
```bash
# 에이전트 재시작
claude omc-restart

# 또는 전체 재설정
claude plugin reinstall oh-my-claudecode
```

#### 문제 2: "API rate limit exceeded"

**증상:**
```
Error: Rate limit exceeded. Retry after 60s
```

**해결:**
```bash
# Ecomode로 전환하여 요청 수 감소
> eco: [your task]

# 또는 설정에서 요청 간격 조정
claude omc-config set request-interval 2000
```

#### 문제 3: "Configuration file corrupted"

**증상:**
```
Error: Invalid configuration format
```

**해결:**
```bash
# 설정 초기화
claude omc-setup --reset

# 백업에서 복구
cp ~/.claude/omc.config.json.backup ~/.claude/omc.config.json
```

#### 문제 4: "Agent conflict detected"

**증상:**
```
Warning: Multiple agents working on same file
```

**해결:**
```bash
# 자동 해결 (권장)
claude omc-resolve-conflicts

# 수동 해결
# 1. 진행 중인 작업 확인
claude omc-status

# 2. 충돌하는 에이전트 중단
claude omc-stop-agent [agent-id]
```

## 선택적 Multi-AI 통합

oh-my-claudecode는 다른 AI 도구와 통합하여 더 강력한 기능을 제공합니다.

### Gemini CLI 통합

Google의 Gemini를 추가 에이전트로 사용:

**설치:**
```bash
# Gemini CLI 설치
npm install -g @google/gemini-cli

# API 키 설정
export GEMINI_API_KEY="your-gemini-api-key"

# OMC에 통합
claude omc-integrate gemini
```

**사용:**
```bash
# Gemini 에이전트를 포함한 작업
> autopilot --with-gemini: analyze this codebase for optimization opportunities
```

**이점:**
- Claude와 Gemini의 상호 보완적 강점 활용
- 더 다양한 관점의 코드 분석
- 특정 작업에 최적화된 모델 선택

### Codex CLI 통합

OpenAI Codex를 코드 생성에 활용:

**설치:**
```bash
# Codex CLI 설치
npm install -g openai-codex-cli

# API 키 설정
export OPENAI_API_KEY="your-openai-api-key"

# OMC에 통합
claude omc-integrate codex
```

**사용:**
```bash
# Codex 에이전트를 포함한 작업
> autopilot --with-codex: generate boilerplate code for a React component library
```

**이점:**
- Codex의 뛰어난 코드 생성 능력 활용
- 보일러플레이트 생성 가속화
- 다양한 프로그래밍 언어 지원 강화

### 멀티-AI 협업 예제

여러 AI를 동시에 활용:

```bash
> ultrapilot --with-all: build a full-stack app with:
> - React frontend (Codex 특화)
> - Python backend (Claude 특화)
> - Performance optimization (Gemini 특화)
> - Comprehensive testing (모든 AI 협업)
```

**실행 과정:**
```
[Claude Agent]  → Backend API 설계 및 구현
[Codex Agent]   → React 컴포넌트 생성
[Gemini Agent]  → 성능 병목 지점 분석 및 최적화
[All Agents]    → 통합 테스트 작성 및 실행
```

## 다음 단계

이제 oh-my-claudecode를 설치하고 첫 작업을 실행했습니다! 다음 챕터에서 7가지 실행 모드를 자세히 알아봅시다:

- **[챕터 3: 실행 모드 상세](/oh-my-claudecode-guide-03-execution-modes/)** - Autopilot, Ultrawork, Ralph 등 모든 모드 완벽 가이드

### 추천 학습 경로

1. **기본 모드 익히기**
   - Autopilot으로 간단한 프로젝트 생성
   - Ecomode로 비용 최적화 경험
   - Ralph로 복잡한 작업 완료

2. **고급 모드 탐험**
   - Ultrawork로 병렬 처리 최대화
   - Ultrapilot으로 대규모 시스템 구축
   - Swarm과 Pipeline으로 특수 워크플로우 구현

3. **실전 프로젝트**
   - 실제 프로젝트에 OMC 적용
   - 팀 협업에 OMC 통합
   - 커스텀 워크플로우 개발

## 참고 자료

- GitHub 저장소: [https://github.com/Yeachan-Heo/oh-my-claudecode](https://github.com/Yeachan-Heo/oh-my-claudecode)
- 문제 해결 가이드: [https://github.com/Yeachan-Heo/oh-my-claudecode/wiki/Troubleshooting](https://github.com/Yeachan-Heo/oh-my-claudecode/wiki/Troubleshooting)
- 커뮤니티 포럼: [링크 추가 필요]
- 비디오 튜토리얼: [링크 추가 필요]
