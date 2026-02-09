---
layout: post
title: "oh-my-claudecode 완벽 가이드 (05) - 고급 활용 및 유틸리티"
date: 2026-02-09
permalink: /oh-my-claudecode-guide-05-advanced/
author: Yeachan Heo
categories: [AI 코딩, CLI]
tags: [Claude Code, Multi-Agent, Orchestration, AI, CLI, Autopilot, Ultrawork]
original_url: "https://github.com/Yeachan-Heo/oh-my-claudecode"
excerpt: "Rate Limit Wait 유틸리티, Multi-AI Orchestration, CLI 명령어, 성능 모니터링, 베스트 프랙티스, 문제 해결 가이드 등 oh-my-claudecode의 고급 활용법을 마스터합니다."
---

## Rate Limit Wait 유틸리티

Claude API는 사용량 제한이 있습니다. OMC의 `omc wait` 유틸리티는 제한에 도달했을 때 자동으로 처리합니다.

### 기본 사용법

#### 1. 상태 확인 (omc wait)

현재 Rate Limit 상태를 확인합니다:

```bash
$ omc wait
```

**출력 예시:**

```
┌─────────────────────────────────────────────────────┐
│ Claude API Rate Limit Status                        │
├─────────────────────────────────────────────────────┤
│ Status: Limited                                     │
│                                                     │
│ Current Usage:                                      │
│ ├─ Requests: 5,000 / 5,000 (100%)                  │
│ ├─ Tokens:   1,000,000 / 1,000,000 (100%)          │
│ └─ Reset in: 14 minutes 32 seconds                 │
│                                                     │
│ Recommendations:                                    │
│ 1. Use 'omc wait --start' to auto-resume           │
│ 2. Switch to Ecomode to reduce token usage         │
│ 3. Take a break and let the limit reset            │
└─────────────────────────────────────────────────────┘
```

정상 상태:

```
┌─────────────────────────────────────────────────────┐
│ Claude API Rate Limit Status                        │
├─────────────────────────────────────────────────────┤
│ Status: OK                                          │
│                                                     │
│ Current Usage:                                      │
│ ├─ Requests: 2,345 / 5,000 (46.9%)                 │
│ ├─ Tokens:   456,789 / 1,000,000 (45.7%)           │
│ └─ Estimated time until limit: 3h 24m              │
│                                                     │
│ You're good to go!                                  │
└─────────────────────────────────────────────────────┘
```

#### 2. 자동 재개 데몬 (omc wait --start)

Rate Limit이 리셋될 때 자동으로 Claude Code를 재개합니다:

```bash
$ omc wait --start
```

**출력:**

```
┌─────────────────────────────────────────────────────┐
│ Auto-Resume Daemon Started                          │
├─────────────────────────────────────────────────────┤
│ Monitoring: Claude API Rate Limits                  │
│ Target: tmux session 'claude-code'                  │
│                                                     │
│ Status: Waiting for rate limit reset...            │
│ Reset at: 2026-02-09 15:42:00 (in 14m 32s)        │
│                                                     │
│ When limits reset, the daemon will:                │
│ 1. Send notification                               │
│ 2. Resume Claude Code session                      │
│ 3. Continue your last task                         │
│                                                     │
│ Daemon PID: 12345                                   │
│ Log file: ~/.omc/logs/wait-daemon.log              │
└─────────────────────────────────────────────────────┘

Daemon running in background.
You can safely close this terminal.
```

**작동 흐름:**

```
사용자
  ↓
[Rate Limit 도달]
  ↓
$ omc wait --start
  ↓
[Daemon 시작]
  ↓
[API 상태 모니터링]
  ↓
[리셋 감지]
  ↓
[알림 전송]
  ↓
[Claude Code 재개]
  ↓
[이전 작업 계속]
```

**실제 시나리오:**

```bash
# 1. 대규모 리팩토링 작업 시작
$ claude-code
> ralph: refactor entire codebase to use async/await

# 2. 작업 중 Rate Limit 도달
[Error] Rate limit exceeded. Reset in 15 minutes.

# 3. 자동 재개 데몬 활성화
$ omc wait --start
Daemon started. Will resume in 15 minutes.

# 4. 다른 작업을 하러 감
# (커피 마시기, 회의 참석 등)

# 5. 15분 후 자동으로:
# - 시스템 알림 표시
# - Claude Code 세션 재개
# - 리팩토링 작업 계속
```

#### 3. 데몬 중지 (omc wait --stop)

자동 재개 데몬을 중지합니다:

```bash
$ omc wait --stop
```

**출력:**

```
┌─────────────────────────────────────────────────────┐
│ Auto-Resume Daemon Stopped                          │
├─────────────────────────────────────────────────────┤
│ Daemon PID: 12345 terminated                        │
│                                                     │
│ Session Statistics:                                 │
│ ├─ Total wait time: 47 minutes                     │
│ ├─ Auto-resumes: 3 times                           │
│ └─ Tasks resumed: 2 tasks                          │
│                                                     │
│ Logs saved to: ~/.omc/logs/wait-daemon.log         │
└─────────────────────────────────────────────────────┘
```

### tmux 통합

`omc wait --start`는 tmux 세션 감지 및 제어를 위해 tmux가 필요합니다.

#### tmux 설치

```bash
# Ubuntu/Debian
sudo apt install tmux

# macOS
brew install tmux

# CentOS/RHEL
sudo yum install tmux
```

#### tmux 세션 설정

Claude Code를 tmux 세션에서 실행:

```bash
# 새 tmux 세션 시작
$ tmux new -s claude-code

# 세션 내에서 Claude Code 실행
$ claude-code

# 세션 분리 (Detach): Ctrl+B, D
# 세션 재연결: tmux attach -t claude-code
```

#### 자동 tmux 설정

OMC가 자동으로 tmux 세션을 생성하도록 설정:

```bash
# ~/.omc/config.json
{
  "wait": {
    "tmux": {
      "autoCreate": true,
      "sessionName": "claude-code",
      "startCommand": "claude-code"
    }
  }
}
```

이제 `omc wait --start`가 tmux 세션을 자동으로 관리합니다:

```bash
$ omc wait --start

Auto-Resume Daemon:
├─ tmux session not found
├─ Creating new session: 'claude-code'
├─ Starting Claude Code in session
└─ Monitoring for rate limit reset
```

### 고급 설정

#### 1. 알림 커스터마이즈

```json
// ~/.omc/config.json
{
  "wait": {
    "notifications": {
      "enabled": true,
      "sound": true,
      "methods": ["desktop", "terminal", "slack"],
      "slack": {
        "webhook": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        "channel": "#dev-notifications"
      }
    }
  }
}
```

알림 예시:

**Desktop Notification:**
```
┌─────────────────────────────────┐
│ OMC Auto-Resume                 │
├─────────────────────────────────┤
│ Rate limit reset detected!      │
│ Resuming Claude Code session... │
│                                 │
│ Task: Refactor codebase         │
│ Progress: 67% complete          │
└─────────────────────────────────┘
```

**Slack Message:**
```
🤖 OMC Auto-Resume Alert
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rate limit has been reset.
Claude Code session resumed automatically.

Current Task: Refactor codebase to async/await
Progress: 67% (8/12 modules complete)
Estimated completion: 23 minutes

Dashboard: http://localhost:3000/omc-dashboard
```

#### 2. 다중 세션 관리

여러 프로젝트를 동시에 모니터링:

```bash
# 프로젝트 A
$ tmux new -s project-a
$ cd /path/to/project-a
$ omc wait --start --session project-a

# 프로젝트 B
$ tmux new -s project-b
$ cd /path/to/project-b
$ omc wait --start --session project-b

# 모든 세션 상태 확인
$ omc wait --list
```

**출력:**

```
Active Auto-Resume Sessions:
┌──────────────┬───────────┬──────────────┬─────────┐
│ Session      │ Project   │ Status       │ ETA     │
├──────────────┼───────────┼──────────────┼─────────┤
│ project-a    │ /path/to/a│ Waiting      │ 5m 23s  │
│ project-b    │ /path/to/b│ Waiting      │ 12m 45s │
│ experiment   │ /path/to/c│ Active       │ N/A     │
└──────────────┴───────────┴──────────────┴─────────┘
```

#### 3. 예약 실행

특정 시간에 작업을 자동 시작:

```bash
# 오후 2시에 자동 시작
$ omc wait --schedule "14:00" --command "autopilot: run integration tests"

# 매일 오전 9시에 자동 실행
$ omc wait --schedule "09:00 daily" --command "plan: review code quality"
```

## CLI 명령어 전체

OMC는 강력한 CLI 도구를 제공합니다.

### omc-analytics

분석 및 메트릭 도구입니다.

#### 기본 명령어

```bash
# 토큰 사용량 분석
$ omc-analytics tokens

# 비용 분석
$ omc-analytics cost

# 성능 메트릭
$ omc-analytics performance

# 세션 히스토리
$ omc-analytics sessions

# 전체 대시보드
$ omc-analytics dashboard
```

#### 고급 쿼리

```bash
# 특정 기간의 비용
$ omc-analytics cost --from "2026-02-01" --to "2026-02-09"

# 특정 모드의 통계
$ omc-analytics performance --mode ralph

# CSV로 내보내기
$ omc-analytics tokens --export tokens.csv

# 그래프 생성
$ omc-analytics cost --graph --output cost-chart.png
```

#### 실시간 모니터링

```bash
# 실시간 토큰 사용량
$ omc-analytics live

# 출력:
Real-time OMC Analytics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Updated: 2026-02-09 14:23:45 (refresh every 5s)

Current Session: #143
├─ Mode: Ultrapilot
├─ Active Agents: 4/32
├─ Duration: 12m 34s
├─ Tokens: 45,230 (input) + 34,567 (output)
├─ Cost: $8.23
└─ ETA: 5m 12s

Token Rate: 1,234 tokens/min
Cost Rate: $0.65/min
Estimated Final Cost: $11.50

Press Ctrl+C to exit
```

### omc-cli

프로젝트 관리 및 설정 도구입니다.

#### 프로젝트 관리

```bash
# 프로젝트 초기화
$ omc-cli init

# 프로젝트 정보
$ omc-cli info

# 설정 보기
$ omc-cli config list

# 설정 변경
$ omc-cli config set <key> <value>

# 설정 초기화
$ omc-cli config reset
```

#### 스킬 관리

```bash
# 스킬 목록
$ omc-cli skills list

# 스킬 생성
$ omc-cli skills create

# 스킬 수정
$ omc-cli skills edit <skill-name>

# 스킬 삭제
$ omc-cli skills delete <skill-name>

# 스킬 가져오기
$ omc-cli skills import <file.yaml>

# 스킬 내보내기
$ omc-cli skills export <skill-name> -o <file.yaml>
```

#### 세션 관리

```bash
# 세션 목록
$ omc-cli sessions list

# 세션 상세 정보
$ omc-cli sessions show <session-id>

# 세션 복원
$ omc-cli sessions restore <session-id>

# 세션 삭제
$ omc-cli sessions delete <session-id>

# 모든 세션 정리
$ omc-cli sessions clean --older-than 30d
```

### doctor (문제 해결)

OMC 설치 및 설정을 진단하고 수정합니다.

#### 기본 진단

```bash
$ omc-cli doctor
```

**출력:**

```
OMC Doctor - System Diagnostics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Checking OMC Installation...

✓ Claude Code installed (v2.3.0)
✓ OMC plugin installed (v1.5.2)
✓ Node.js version OK (v18.17.0)
✓ npm version OK (v9.8.1)

Checking Configuration...

✓ Config file exists (~/.omc/config.json)
✓ Config is valid JSON
✓ All required fields present
✗ Cache directory has permission issues
  → Run: chmod 755 ~/.omc/cache

Checking Dependencies...

✓ tmux installed (v3.2a)
✓ git installed (v2.34.1)
✗ jq not installed (optional)
  → Run: sudo apt install jq

Checking Claude API...

✓ API key configured
✓ API key is valid
✓ Rate limits: OK (2,345/5,000 requests)
✗ Token limit: WARNING (890,000/1,000,000 tokens)
  → Consider using Ecomode

Checking Project Setup...

✓ Project initialized
✓ .omc directory exists
✓ Skills loaded (12 skills)
✗ Cache corrupted
  → Run: omc-cli cache clear

Overall Status: 3 issues found
Action Required: Run suggested fixes above
```

#### 자동 수정

```bash
# 자동으로 문제 수정
$ omc-cli doctor --fix

Fixing issues...
├─ Fixing cache directory permissions... ✓
├─ Installing jq... ✓
├─ Clearing corrupted cache... ✓
└─ All issues resolved!

Run 'omc-cli doctor' again to verify.
```

#### 특정 항목 진단

```bash
# API 연결만 확인
$ omc-cli doctor --check api

# 캐시 진단
$ omc-cli doctor --check cache

# 설정 검증
$ omc-cli doctor --check config

# 전체 상세 진단
$ omc-cli doctor --verbose
```

#### 캐시 관리

```bash
# 캐시 정보
$ omc-cli cache info

Cache Statistics:
├─ Location: ~/.omc/cache
├─ Size: 245.7 MB
├─ Files: 1,234 files
├─ Last cleanup: 3 days ago
└─ Recommended action: No action needed

# 캐시 정리
$ omc-cli cache clean

# 캐시 완전 삭제
$ omc-cli cache clear

# 캐시 재구성
$ omc-cli cache rebuild
```

## Multi-AI Orchestration

여러 AI 모델을 함께 사용하여 더 나은 결과를 얻습니다.

### Gemini CLI 통합

Google의 Gemini 모델을 Claude와 함께 사용합니다.

#### 설치

```bash
# Gemini CLI 설치
$ npm install -g @google/gemini-cli

# API 키 설정
$ gemini-cli config set apiKey YOUR_GEMINI_API_KEY

# OMC에 Gemini 통합 활성화
$ omc-cli config set integrations.gemini.enabled true
```

#### 사용 사례 1: 디자인 리뷰

Gemini의 1M 토큰 컨텍스트를 활용한 디자인 검증:

```bash
# Claude로 UI 구현
> autopilot: create dashboard with charts and tables

# Gemini로 디자인 리뷰 (대용량 컨텍스트)
> omc-cli gemini review-design --scope all

Gemini Design Review:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Context Analyzed: 847,234 tokens (entire codebase)
Review Time: 15 seconds

Consistency Issues Found:
1. Color Palette Inconsistency (12 occurrences)
   ├─ Dashboard uses #3B82F6
   ├─ Settings uses #2563EB
   └─ Recommendation: Standardize to #3B82F6

2. Spacing Inconsistency (8 occurrences)
   ├─ Most components use 16px padding
   ├─ 3 components use 20px padding
   └─ Recommendation: Standardize to 16px

3. Button Styles (5 variations)
   ├─ Primary: 3 different styles found
   ├─ Secondary: 2 different styles found
   └─ Recommendation: Use design system

Overall Design Score: 78/100
Estimated fix time: 45 minutes
```

#### 사용 사례 2: UI 일관성 검증

```bash
# Gemini로 전체 UI 일관성 체크
> omc-cli gemini validate-ui

UI Consistency Report:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Components Analyzed: 47
Pages Analyzed: 12
Total Lines: 15,432

Color Usage:
├─ Primary Color: #3B82F6 (89% consistency)
├─ Secondary Color: #6B7280 (92% consistency)
├─ Accent Color: #10B981 (67% consistency) ⚠
└─ Recommendation: Standardize accent color

Typography:
├─ Font Family: Inter (100% consistency) ✓
├─ Heading Sizes: 4 variations (should be 3) ⚠
└─ Line Heights: Mostly consistent (95%) ✓

Spacing System:
├─ Uses Tailwind spacing (87% adherence)
├─ Custom values found: 23 instances ⚠
└─ Recommendation: Stick to Tailwind scale

Component Patterns:
✓ Buttons: Consistent
✓ Inputs: Consistent
✗ Modals: 3 different implementations
✗ Cards: 2 different shadow styles
```

### Codex CLI 통합

OpenAI Codex를 아키텍처 검증에 활용합니다.

#### 설치

```bash
# Codex CLI 설치
$ npm install -g @openai/codex

# API 키 설정
$ codex config set apiKey YOUR_OPENAI_API_KEY

# OMC에 Codex 통합 활성화
$ omc-cli config set integrations.codex.enabled true
```

#### 사용 사례 1: 아키텍처 검증

```bash
# Claude로 아키텍처 설계
> arch: design microservices architecture for e-commerce

# Codex로 아키텍처 검증
> omc-cli codex validate-architecture

Codex Architecture Validation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Architecture Pattern Detected: Microservices

Strengths:
✓ Clear service boundaries
✓ Proper API Gateway usage
✓ Database per service pattern
✓ Event-driven communication

Potential Issues:

1. Service Dependency Complexity
   Severity: Medium
   ├─ Payment service depends on 4 other services
   ├─ Risk: Cascading failures
   └─ Recommendation: Introduce circuit breaker

2. Data Consistency
   Severity: High
   ├─ No saga pattern for distributed transactions
   ├─ Risk: Data inconsistency across services
   └─ Recommendation: Implement Saga or 2PC

3. Service Discovery
   Severity: Low
   ├─ Hardcoded service URLs found
   ├─ Risk: Difficult to scale
   └─ Recommendation: Use service mesh (Istio/Linkerd)

Overall Architecture Score: 82/100

Comparison with Industry Patterns:
├─ Netflix OSS: 78% similarity
├─ AWS Best Practices: 85% similarity
└─ Microservices.io Patterns: 90% similarity
```

#### 사용 사례 2: 코드 리뷰

```bash
# Codex로 코드 품질 리뷰
> omc-cli codex review-code --file src/payment/processor.js

Codex Code Review:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: src/payment/processor.js
Lines: 234
Complexity: Medium

Code Quality Score: 76/100

Issues Found:

1. Error Handling (Line 45)
   Severity: High
   ├─ Empty catch block
   ├─ Errors silently ignored
   └─ Fix: Log errors and handle gracefully

2. Memory Leak Risk (Line 89)
   Severity: Medium
   ├─ Event listener not removed
   ├─ Risk: Memory accumulation
   └─ Fix: Use removeEventListener in cleanup

3. Performance (Line 123)
   Severity: Low
   ├─ Nested loops with O(n²) complexity
   ├─ Risk: Slow for large datasets
   └─ Fix: Use Map for O(n) lookup

4. Security (Line 167)
   Severity: Critical
   ├─ API key exposed in client code
   ├─ Risk: Key compromise
   └─ Fix: Move to server-side environment

Recommendations:
1. Add input validation
2. Implement retry logic
3. Use async/await consistently
4. Add unit tests (coverage: 0%)
```

### Cross-validation 워크플로우

여러 AI를 사용한 교차 검증 워크플로우입니다.

#### 1. 설계 → 구현 → 검증 파이프라인

```bash
# Step 1: Claude로 아키텍처 설계
> arch: design payment processing system

# Step 2: Codex로 설계 검증
> omc-cli codex validate-architecture

# Step 3: Claude로 구현
> ultrapilot: implement the validated architecture

# Step 4: Gemini로 코드 리뷰 (대용량 컨텍스트)
> omc-cli gemini review-code --scope all

# Step 5: Codex로 보안 검증
> omc-cli codex security-audit

# Step 6: Claude로 테스트 작성
> ultraqa: create comprehensive test suite
```

#### 2. 자동화된 교차 검증

설정 파일로 자동 교차 검증:

```yaml
# .omc/workflows/cross-validation.yaml
name: Cross-Validation Workflow
description: Multi-AI validation pipeline

steps:
  - name: Design
    agent: claude
    mode: arch
    task: Design system architecture

  - name: Validate Design
    agent: codex
    command: validate-architecture
    requires: Design

  - name: Implement
    agent: claude
    mode: ultrapilot
    task: Implement validated design
    requires: Validate Design

  - name: UI Review
    agent: gemini
    command: review-design
    requires: Implement

  - name: Code Review
    agent: codex
    command: review-code
    requires: Implement

  - name: Security Audit
    agent: codex
    command: security-audit
    requires: Implement

  - name: Final Tests
    agent: claude
    mode: ultraqa
    task: Create comprehensive test suite
    requires: [UI Review, Code Review, Security Audit]

notifications:
  on_failure: slack
  on_success: email
```

실행:

```bash
$ omc-cli workflow run cross-validation

Cross-Validation Workflow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/7] Design (Claude) ........................ ✓ (2m 34s)
[2/7] Validate Design (Codex) ................ ✓ (45s)
[3/7] Implement (Claude) ..................... ✓ (15m 23s)
[4/7] UI Review (Gemini) ..................... ✓ (1m 12s)
[5/7] Code Review (Codex) .................... ✓ (2m 45s)
[6/7] Security Audit (Codex) ................. ✓ (1m 30s)
[7/7] Final Tests (Claude) ................... ✓ (8m 15s)

All steps completed successfully! ✓
Total time: 32m 24s
Total cost: $18.45

Detailed reports saved to:
├─ reports/architecture-validation.md
├─ reports/ui-review.md
├─ reports/code-review.md
├─ reports/security-audit.md
└─ reports/test-coverage.md
```

### 비용 고려사항

Multi-AI orchestration의 비용 분석:

```
Monthly Cost Estimation (Active Development)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Claude Pro: $20/month
├─ Primary development
├─ Code generation
└─ Testing

Gemini Pro: $20/month (1M tokens)
├─ Design review
├─ Large codebase analysis
└─ UI consistency checks

OpenAI Plus: $20/month (for Codex access)
├─ Architecture validation
├─ Code review
└─ Security audits

Total: ~$60/month

Value Proposition:
├─ 24/7 expert-level reviews
├─ Multiple perspectives on design
├─ Comprehensive validation
└─ ROI: Prevents bugs worth 10-100x cost
```

## Performance 모니터링

작업 성능을 추적하고 최적화합니다.

### 에이전트 추적

각 에이전트의 성능을 모니터링:

```bash
$ omc-analytics agents

Agent Performance Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌────────────────┬──────────┬─────────┬─────────┬──────────┐
│ Agent          │ Tasks    │ Success │ Avg Time│ Tokens   │
├────────────────┼──────────┼─────────┼─────────┼──────────┤
│ Architect      │ 23       │ 95.7%   │ 8m 45s  │ 45,230   │
│ Frontend       │ 45       │ 97.8%   │ 5m 12s  │ 38,567   │
│ Backend        │ 38       │ 94.7%   │ 6m 34s  │ 42,890   │
│ Database       │ 15       │ 100%    │ 3m 23s  │ 12,456   │
│ Testing        │ 67       │ 97.0%   │ 4m 56s  │ 28,901   │
│ DevOps         │ 12       │ 91.7%   │ 12m 34s │ 56,789   │
│ Security       │ 8        │ 100%    │ 7m 45s  │ 34,567   │
└────────────────┴──────────┴─────────┴─────────┴──────────┘

Top Performers:
1. Database Agent: 100% success, fastest avg time
2. Testing Agent: 97% success, most tasks completed
3. Frontend Agent: 97.8% success, good balance

Need Improvement:
1. DevOps Agent: Longest avg time (12m 34s)
   → Consider breaking complex tasks
2. Architect Agent: Lower success rate (95.7%)
   → May need better task descriptions
```

### 디버깅 도구

문제 진단 도구:

```bash
# 실패한 작업 분석
$ omc-cli debug failures

Recent Failures Analysis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Failures: 12 (last 7 days)
Success Rate: 94.2% (206/218 tasks)

Failure Breakdown:

1. Dependency Issues (5 failures)
   ├─ Missing packages
   ├─ Version conflicts
   └─ Fix: Pre-validate dependencies

2. API Errors (4 failures)
   ├─ Rate limits
   ├─ Timeouts
   └─ Fix: Implement retry logic

3. Test Failures (3 failures)
   ├─ Flaky tests
   ├─ Environment issues
   └─ Fix: Stabilize test environment

Common Patterns:
├─ 67% of failures occur during peak hours
├─ 42% are retryable errors
└─ Average recovery time: 3m 45s

# 특정 작업 디버그
$ omc-cli debug session 143

Session #143 Debug Information:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Task: Implement user authentication
Status: Failed at 87% completion
Duration: 23m 45s
Tokens Used: 67,890

Failure Point:
├─ Step: Testing authentication flow
├─ Error: Test timeout after 30s
├─ Root Cause: Database connection not mocked
└─ Fix Applied: Added database mock

Timeline:
├─ [00:00] Task started
├─ [03:23] Dependencies installed ✓
├─ [08:45] Code generation complete ✓
├─ [15:12] Unit tests written ✓
├─ [21:34] Integration tests started
├─ [23:45] Test timeout ✗
└─ [26:12] Fixed and completed ✓

Logs: ~/.omc/logs/session-143.log
```

### 최적화 전략

성능 개선 제안:

```bash
$ omc-cli optimize suggest

Optimization Suggestions:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on your usage patterns, here are recommendations:

1. Use Ecomode for Simple Tasks (Potential savings: $45/mo)
   ├─ Current: 45% of tasks use Opus
   ├─ Recommended: Use Haiku for 30% of those
   └─ Impact: 20% cost reduction, minimal quality impact

2. Increase Parallelization (Time savings: 35%)
   ├─ Current: Average 2.3 parallel agents
   ├─ Recommended: Increase to 4-5 for complex tasks
   └─ Impact: 35% faster completion

3. Optimize Context Size (Cost savings: 15%)
   ├─ Current: Average 12,345 tokens per request
   ├─ Recommended: Use focused context (8,000 tokens)
   └─ Impact: 15% token reduction

4. Enable Caching (Speed improvement: 40%)
   ├─ Current: Cache hit rate 34%
   ├─ Recommended: Increase cache size, enable smart caching
   └─ Impact: 40% faster for repeated operations

5. Batch Similar Tasks (Efficiency gain: 25%)
   ├─ Pattern detected: Multiple similar CRUD endpoints
   ├─ Recommended: Use 'ultrawork' to batch create
   └─ Impact: 25% faster, 20% cheaper

Apply all suggestions? [Y/n]
```

## 베스트 프랙티스

프로젝트를 효율적으로 관리하는 방법입니다.

### 프로젝트 구조화

#### 권장 디렉토리 구조

```
my-project/
├─ .omc/
│  ├─ config.json              # OMC 설정
│  ├─ skills/                  # 커스텀 스킬
│  │  ├─ react-component.yaml
│  │  └─ api-endpoint.yaml
│  ├─ workflows/               # 자동화 워크플로우
│  │  ├─ cross-validation.yaml
│  │  └─ deployment.yaml
│  └─ cache/                   # 에이전트 캐시
├─ docs/
│  ├─ architecture.md          # 아키텍처 문서
│  ├─ api.md                   # API 문서
│  └─ omc-sessions/            # OMC 세션 로그
├─ src/
├─ tests/
└─ README.md
```

#### 설정 파일 예시

```json
// .omc/config.json
{
  "project": {
    "name": "my-awesome-project",
    "type": "fullstack",
    "languages": ["typescript", "python"],
    "frameworks": ["react", "fastapi"]
  },
  "modes": {
    "default": "autopilot",
    "testing": "ultraqa",
    "deployment": "devops"
  },
  "budget": {
    "daily": 50,
    "weekly": 200,
    "alerts": true
  },
  "integrations": {
    "gemini": {
      "enabled": true,
      "use_for": ["design-review", "large-context"]
    },
    "codex": {
      "enabled": true,
      "use_for": ["architecture-validation", "security-audit"]
    }
  },
  "cache": {
    "enabled": true,
    "size": "1GB",
    "ttl": "7d"
  },
  "parallelization": {
    "max_agents": 8,
    "auto_detect": true
  }
}
```

### 모드 선택 전략

#### 작업별 최적 모드

```
단순 기능 추가 → autopilot
├─ 빠른 구현
├─ 일반적인 품질
└─ 비용 효율적

복잡한 프로젝트 → ultrapilot
├─ 병렬 실행
├─ 빠른 완성
└─ 높은 비용

100% 완성 필요 → ralph
├─ 자동 재시도
├─ 오류 복구
└─ 긴 실행 시간

다수의 유사 작업 → ultrawork
├─ 최대 병렬화
├─ 일관된 품질
└─ 시간 절약

예산 제약 → eco
├─ Haiku 모델 사용
├─ 비용 30-50% 절감
└─ 약간의 시간 증가

계획 수립 → plan
├─ 실행 없음
├─ 상세 계획만
└─ 최소 비용

작업 나열 → list
├─ 작업 분해만
├─ 체크리스트 생성
└─ 무료 (거의)
```

#### 단계별 접근법

대규모 프로젝트를 효율적으로 진행:

```bash
# Phase 1: 계획 (Plan 모드)
> plan: design and plan an e-commerce platform
# 비용: ~$2
# 결과: 상세 계획, 작업 분해

# Phase 2: 아키텍처 (Arch 모드)
> arch: implement the architecture from the plan
# 비용: ~$15
# 결과: 프로젝트 구조, 설정

# Phase 3: 병렬 개발 (Ultrapilot 모드)
> ultrapilot: implement all features from the plan
# 비용: ~$80
# 결과: 전체 기능 구현

# Phase 4: 품질 보증 (UltraQA 모드)
> ultraqa: comprehensive testing of all features
# 비용: ~$25
# 결과: 완전한 테스트 커버리지

# Phase 5: 배포 (DevOps 모드)
> devops: set up CI/CD and deploy to production
# 비용: ~$15

# Total: ~$137 for complete e-commerce platform
# Time: 1-2 days (vs weeks of manual work)
```

### 비용 최적화 팁

#### 1. 스마트 모드 전환

```bash
# BAD: 모든 작업에 ultrapilot 사용
> ultrapilot: fix typo in README
# 비용: $5 (과도함)

# GOOD: 간단한 작업은 autopilot
> autopilot: fix typo in README
# 비용: $0.20

# BAD: 단순 리팩토링에 ralph
> ralph: rename variable
# 비용: $8 (불필요)

# GOOD: 단순 작업은 eco
> eco: rename variable
# 비용: $0.15
```

#### 2. 컨텍스트 최적화

```bash
# BAD: 전체 프로젝트를 컨텍스트로
> autopilot: fix bug in user-service.js
# (모든 파일 로드, 50,000 tokens)

# GOOD: 관련 파일만 지정
> autopilot: fix bug in src/services/user-service.js (focus on this file only)
# (필요한 파일만, 5,000 tokens)
# 절감: 90%
```

#### 3. 배치 처리

```bash
# BAD: 개별 작업으로 실행
> autopilot: add login endpoint
> autopilot: add register endpoint
> autopilot: add logout endpoint
# 비용: $3 × 3 = $9

# GOOD: 배치로 한 번에
> ultrawork: add login, register, and logout endpoints
# 비용: $6
# 절감: 33%
```

#### 4. 캐싱 활용

```bash
# 캐싱 활성화
$ omc-cli config set cache.enabled true
$ omc-cli config set cache.size 2GB

# 반복 작업이 40% 빠르고 저렴해짐
```

### 성능 튜닝

#### 병렬화 최적화

```json
// .omc/config.json
{
  "parallelization": {
    "max_agents": 8,           // CPU 코어 수에 맞춤
    "auto_detect": true,       // 자동 의존성 감지
    "aggressive": false,       // 안전한 병렬화만
    "timeout": 300,            // 5분 타임아웃
    "retry_failed": true       // 실패 시 재시도
  }
}
```

## 문제 해결 가이드

일반적인 문제와 해결 방법입니다.

### 일반적인 문제

#### 1. Rate Limit 도달

**증상:**
```
Error: Rate limit exceeded
Reset time: 14 minutes
```

**해결:**
```bash
# 자동 대기 및 재개
$ omc wait --start

# 또는 Ecomode로 전환
> eco: continue previous task
```

#### 2. 메모리 부족

**증상:**
```
Error: JavaScript heap out of memory
```

**해결:**
```bash
# Node.js 메모리 증가
$ export NODE_OPTIONS="--max-old-space-size=8192"

# 또는 컨텍스트 크기 감소
$ omc-cli config set context.max_size 50000
```

#### 3. 캐시 손상

**증상:**
```
Error: Invalid cache entry
Warning: Cache checksum mismatch
```

**해결:**
```bash
# 캐시 재구축
$ omc-cli cache rebuild

# 또는 완전 초기화
$ omc-cli cache clear
$ omc-cli doctor --fix
```

#### 4. 에이전트 충돌

**증상:**
```
Error: Agent conflict detected
Multiple agents modifying the same file
```

**해결:**
```bash
# 병렬 에이전트 수 감소
$ omc-cli config set parallelization.max_agents 4

# 또는 순차 모드로 전환
$ omc-cli config set parallelization.aggressive false
```

### 고급 문제 해결

#### 디버그 모드 활성화

```bash
# 상세 로그 활성화
$ omc-cli config set debug.enabled true
$ omc-cli config set debug.level verbose

# 이제 모든 작업이 상세 로그 생성
> autopilot: test task

# 로그 확인
$ tail -f ~/.omc/logs/debug.log
```

#### 세션 복구

```bash
# 중단된 세션 복구
$ omc-cli sessions restore <session-id>

# 마지막 세션 자동 복구
$ omc-cli sessions restore --last
```

## 향후 로드맵

OMC의 개발 계획입니다.

### 단기 (1-3개월)

- **더 많은 에이전트**: 10개의 새로운 전문 에이전트
  - Mobile (iOS/Android native)
  - Game Development
  - Blockchain/Smart Contracts
  - Embedded Systems

- **개선된 UI**: VSCode 확장 프로그램
  - 그래픽 대시보드
  - 실시간 시각화
  - 드래그 앤 드롭 워크플로우

- **더 나은 통합**:
  - GitHub Copilot 연동
  - Cursor AI 지원
  - JetBrains IDE 플러그인

### 중기 (3-6개월)

- **팀 협업 기능**:
  - 공유 스킬 라이브러리
  - 팀 분석 대시보드
  - 비용 할당

- **고급 AI 통합**:
  - GPT-4 Turbo
  - Claude Opus 2.0
  - Gemini Ultra

- **엔터프라이즈 기능**:
  - SSO 지원
  - 감사 로그
  - 규정 준수 리포트

### 장기 (6-12개월)

- **자율 에이전트**:
  - 완전 자율 개발 모드
  - 자동 버그 감지 및 수정
  - 프로액티브 최적화

- **에이전트 마켓플레이스**:
  - 커뮤니티 에이전트 공유
  - 유료 프리미엄 에이전트
  - 에이전트 평가 시스템

- **AI 학습 플랫폼**:
  - 프로젝트별 맞춤 에이전트
  - 팀 스타일 학습
  - 지속적 개선

## 커뮤니티 리소스

### 공식 채널

- **GitHub**: [https://github.com/Yeachan-Heo/oh-my-claudecode](https://github.com/Yeachan-Heo/oh-my-claudecode)
- **Discord**: [OMC 커뮤니티 서버]
- **Twitter**: [@ohmyclaudecode]

### 기여 방법

#### 코드 기여

```bash
# 1. Fork 및 Clone
$ git clone https://github.com/YOUR_USERNAME/oh-my-claudecode.git

# 2. 브랜치 생성
$ git checkout -b feature/my-new-feature

# 3. 개발 (OMC로!)
> ultrapilot: implement my new feature

# 4. 테스트
> ultraqa: test my new feature

# 5. PR 생성
$ git push origin feature/my-new-feature
```

#### 스킬 공유

```bash
# 스킬 내보내기
$ omc-cli skills export "My Awesome Skill" -o skill.yaml

# GitHub에 업로드
# Community Skills Repository에 PR

# 다른 사용자가 사용
$ omc-cli skills import https://raw.githubusercontent.com/.../skill.yaml
```

#### 버그 리포트

GitHub Issues에 다음 정보 포함:

```markdown
**Bug Description**
[Clear description of the issue]

**Environment**
- OMC Version: 1.5.2
- Claude Code Version: 2.3.0
- OS: Ubuntu 22.04
- Node.js: v18.17.0

**Steps to Reproduce**
1. Run command: `...`
2. ...

**Expected Behavior**
[What should happen]

**Actual Behavior**
[What actually happened]

**Logs**
```
[Paste relevant logs from ~/.omc/logs/]
```

**Additional Context**
[Screenshots, config files, etc.]
```

## 결론

oh-my-claudecode는 AI 코딩의 미래입니다:

- **Zero Learning Curve**: 자연어로 즉시 사용
- **Multi-Agent Orchestration**: 32개 전문 에이전트의 협업
- **Automatic Parallelization**: 3-5배 빠른 개발
- **Cost Optimization**: 30-50% 비용 절감
- **Multi-AI Support**: Gemini, Codex와 통합

이 가이드를 통해 OMC를 마스터하고 생산성을 극대화하세요!

## 전체 가이드 시리즈

- **[챕터 1: 소개 및 개요](/oh-my-claudecode-guide-01-intro/)** - OMC 소개, 핵심 개념, 주요 특징
- **[챕터 2: 설치 및 빠른 시작](/oh-my-claudecode-guide-02-quick-start/)** - 3단계 설치, 첫 작업 실행
- **[챕터 3: 실행 모드 상세](/oh-my-claudecode-guide-03-execution-modes/)** - 7가지 실행 모드 완벽 가이드
- **[챕터 4: 핵심 기능 및 도구](/oh-my-claudecode-guide-04-features/)** - 32개 에이전트, 스마트 라우팅, HUD
- **[챕터 5: 고급 활용 및 유틸리티](/oh-my-claudecode-guide-05-advanced/)** - 본 문서

## 참고 자료

- GitHub 저장소: [https://github.com/Yeachan-Heo/oh-my-claudecode](https://github.com/Yeachan-Heo/oh-my-claudecode)
- Claude Code 문서: [https://docs.anthropic.com/claude/docs/claude-code](https://docs.anthropic.com/claude/docs/claude-code)
- Gemini CLI: [https://www.npmjs.com/package/@google/gemini-cli](https://www.npmjs.com/package/@google/gemini-cli)
- OpenAI Codex: [https://openai.com/blog/openai-codex](https://openai.com/blog/openai-codex)
- 이슈 트래커: [https://github.com/Yeachan-Heo/oh-my-claudecode/issues](https://github.com/Yeachan-Heo/oh-my-claudecode/issues)
