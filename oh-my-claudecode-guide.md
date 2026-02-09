---
layout: page
title: oh-my-claudecode 가이드
permalink: /oh-my-claudecode-guide/
icon: fas fa-robot
---

# oh-my-claudecode 완벽 가이드

> **Multi-agent orchestration for Claude Code. Zero learning curve.**

**oh-my-claudecode (OMC)**는 Claude Code를 위한 멀티-에이전트 오케스트레이션 시스템입니다. 자연어 명령만으로 32개의 전문 에이전트가 자동으로 협업하여 복잡한 작업을 완수합니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/blog-repo/oh-my-claudecode-guide-01-intro/) | OMC란?, 7가지 주요 특징, 32개 전문 에이전트 |
| 02 | [설치 및 빠른 시작](/blog-repo/oh-my-claudecode-guide-02-quick-start/) | 3단계 설치, 첫 작업 실행, Multi-AI 통합 |
| 03 | [실행 모드 상세](/blog-repo/oh-my-claudecode-guide-03-execution-modes/) | 7가지 실행 모드, Magic Keywords, 성능 비교 |
| 04 | [핵심 기능 및 도구](/blog-repo/oh-my-claudecode-guide-04-features/) | 32개 에이전트, HUD, 스킬 학습, 분석 도구 |
| 05 | [고급 활용 및 유틸리티](/blog-repo/oh-my-claudecode-guide-05-advanced/) | Rate Limit Wait, CLI 도구, Multi-AI, 베스트 프랙티스 |

---

## 주요 특징

- **🎯 Zero Configuration** - 설치 후 바로 사용 가능, 별도 설정 불필요
- **💬 Natural Language** - 명령어 암기 불필요, 자연어로 요청
- **⚡ Automatic Parallelization** - 복잡한 작업 자동 분산 처리
- **🔄 Persistent Execution** - 검증 완료까지 포기하지 않음
- **💰 Cost Optimization** - 스마트 모델 라우팅으로 30-50% 비용 절감
- **🧠 Learn from Experience** - 문제 해결 패턴 자동 추출 및 재사용
- **📊 Real-time HUD** - 상태줄에서 실시간 오케스트레이션 메트릭 확인

---

## 빠른 시작

### 설치 (3단계)

```bash
# Step 1: Marketplace 추가
/plugin marketplace add https://github.com/Yeachan-Heo/oh-my-claudecode

# Step 2: 플러그인 설치
/plugin install oh-my-claudecode

# Step 3: 초기 설정
/oh-my-claudecode:omc-setup
```

### 첫 작업 실행

```
autopilot: build a REST API for managing tasks
```

끝입니다! 나머지는 모두 자동입니다.

---

## 7가지 실행 모드

```
┌────────────────────────────────────────────────────────────────┐
│                  OMC Execution Modes Overview                   │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Autopilot    ━━━━━━━━━━━━━━━━▶  Full autonomous workflows     │
│  Ultrawork    ━━━━━━━━━━━━━━━━▶  Maximum parallelism (3.3x)    │
│  Ralph        ━━━━━━━━━━━━━━━━▶  Persistent (100% completion)  │
│  Ultrapilot   ━━━━━━━━━━━━━━━━▶  Multi-component (3-5x faster) │
│  Ecomode      ━━━━━━━━━━━━━━━━▶  30-50% cost savings           │
│  Swarm        ━━━━━━━━━━━━━━━━▶  Coordinated parallel          │
│  Pipeline     ━━━━━━━━━━━━━━━━▶  Sequential multi-stage        │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

| 모드 | 속도 | 사용 사례 | Magic Keyword |
|------|------|-----------|---------------|
| **Autopilot** | Fast | 풀스택 자율 워크플로우 | `autopilot` |
| **Ultrawork** | 3.3x faster | 최대 병렬 처리 | `ulw` |
| **Ralph** | Persistent | 반드시 완료해야 하는 작업 | `ralph` |
| **Ultrapilot** | 3-5x faster | 멀티 컴포넌트 시스템 | (자동) |
| **Ecomode** | Fast + 저렴 | 예산 고려 프로젝트 | `eco` |
| **Swarm** | Coordinated | 독립적인 병렬 작업 | (자동) |
| **Pipeline** | Sequential | 다단계 순차 처리 | (자동) |

---

## 32개 전문 에이전트

### Architecture 에이전트
- **arch**: 시스템 아키텍처 설계
- **ralph**: 끈질긴 구현 (Ultrawork 포함)
- **ralplan**: 반복적 계획 합의
- **system-architect**: 전체 시스템 구조

### Research 에이전트
- **deepsearch**: 심층 코드베이스 분석
- **researcher**: 기술 조사 및 벤치마킹
- **doc-analyzer**: 문서 분석 및 요약

### Design 에이전트
- **designer**: UI/UX 디자인
- **ux-specialist**: 사용자 경험 최적화
- **design-validator**: 디자인 일관성 검증

### Testing 에이전트
- **tdd**: TDD 주도 개발
- **ultraqa**: 포괄적 QA
- **e2e-tester**: E2E 테스트
- **security-auditor**: 보안 감사

### Data Science 에이전트
- **data-scientist**: 데이터 분석
- **ml-engineer**: ML 모델 개발
- **stats-analyzer**: 통계 분석

### DevOps 에이전트
- **devops-engineer**: CI/CD 파이프라인
- **ci-cd-specialist**: 배포 자동화

*... 그 외 12개 에이전트*

---

## Magic Keywords

자연어만으로도 충분하지만, 명시적 제어를 원한다면:

| Keyword | 효과 | 예시 |
|---------|------|------|
| `autopilot` | 완전 자율 실행 | `autopilot: build a todo app` |
| `ralph` | 지속성 모드 (Ultrawork 포함) | `ralph: refactor auth` |
| `ulw` | 최대 병렬화 | `ulw fix all errors` |
| `eco` | 토큰 효율 실행 | `eco: migrate database` |
| `plan` | 계획 인터뷰 | `plan the API` |
| `ralplan` | 반복적 계획 합의 | `ralplan this feature` |

---

## HUD Statusline

실시간 오케스트레이션 메트릭을 상태줄에서 확인:

```
[OMC] ⚡ Autopilot | 🤖 3 agents | 💬 2.3K tokens | ⏱️ 45s
```

- **실행 모드**: 현재 활성 모드
- **활성 에이전트 수**: 동시 실행 중인 에이전트
- **토큰 사용량**: 실시간 토큰 카운터
- **경과 시간**: 작업 진행 시간

---

## 스마트 모델 라우팅

**30-50% 비용 절감** ⚡

| 작업 유형 | 모델 | 비용 | 예시 |
|----------|------|------|------|
| 단순 작업 | Haiku | 저렴 | 코드 포맷, 단순 리팩토링 |
| 복잡한 추론 | Opus | 비쌈 | 아키텍처 설계, 복잡한 버그 |
| 중간 작업 | Sonnet | 중간 | 일반 개발 작업 |

OMC가 자동으로 최적 모델 선택 → **비용 최적화 + 성능 보장**

---

## 성능 비교

### 속도 향상

| 모드 | 단일 에이전트 | OMC | 속도 향상 |
|------|-------------|-----|----------|
| Autopilot | 10분 | 10분 | 1x (기준) |
| Ultrawork | 10분 | 3분 | **3.3x** |
| Ultrapilot | 15분 | 3-5분 | **3-5x** |

### 비용 절감

| 프로젝트 | Claude Code 단독 | OMC Ecomode | 절감 |
|---------|-----------------|------------|------|
| Todo App | $2.50 | $1.25 | **50%** |
| REST API | $5.00 | $3.00 | **40%** |
| 풀스택 앱 | $15.00 | $9.00 | **40%** |

---

## Rate Limit Wait

Claude Code 세션 Rate Limit 시 자동 재개:

```bash
# 상태 확인
omc wait

# 자동 재개 데몬 시작
omc wait --start

# 데몬 중지
omc wait --stop
```

**요구사항**: tmux (세션 감지용)

---

## Multi-AI Orchestration (선택사항)

외부 AI 제공자로 교차 검증 및 디자인 일관성:

| 제공자 | 설치 | 용도 |
|-------|------|------|
| **Gemini CLI** | `npm install -g @google/gemini-cli` | 디자인 리뷰, UI 일관성 (1M 토큰 컨텍스트) |
| **Codex CLI** | `npm install -g @openai/codex` | 아키텍처 검증, 코드 리뷰 교차 확인 |

**비용**: 3개 Pro 플랜 (Claude + Gemini + ChatGPT) ~$60/월

**선택사항**: OMC는 이들 없이도 완벽히 작동합니다.

---

## 스킬 학습 시스템

OMC는 여러분의 작업에서 학습합니다:

1. **패턴 추출**: 성공적인 문제 해결 방법 자동 추출
2. **재사용 가능한 워크플로우**: 반복 작업 템플릿화
3. **자동 생성 스킬**: 추출된 패턴을 스킬로 변환
4. **지속적 개선**: 사용할수록 더 똑똑해짐

```bash
# 추출된 스킬 확인
/oh-my-claudecode:skill-list

# 특정 스킬 사용
/oh-my-claudecode:use-skill refactoring-pattern-1
```

---

## Analytics & Cost Tracking

```bash
# 토큰 사용량 분석
omc-analytics tokens

# 비용 분석
omc-analytics cost --by-mode

# 성능 메트릭
omc-analytics performance
```

**출력 예시**:
```
📊 Token Usage (Last 30 days)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Autopilot:  125K tokens  ($3.75)
Ultrawork:   85K tokens  ($2.55)
Ecomode:     45K tokens  ($0.90)  ← 50% savings!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:      255K tokens  ($7.20)
```

---

## CLI 도구

| 명령어 | 설명 |
|--------|------|
| `oh-my-claudecode` | 메인 CLI |
| `omc` | 단축 명령어 |
| `omc-analytics` | 분석 도구 |
| `omc wait` | Rate limit 대기 |
| `/oh-my-claudecode:doctor` | 문제 해결 |

---

## 요구사항

- **필수**:
  - [Claude Code](https://docs.anthropic.com/claude-code) CLI
  - Claude Max/Pro 구독 또는 Anthropic API 키

- **선택사항**:
  - tmux (Rate Limit Wait 기능용)
  - Gemini CLI (Multi-AI 교차 검증용)
  - Codex CLI (아키텍처 검증용)

---

## 베스트 프랙티스

### 1. 모드 선택 전략

```
빠른 프로토타입 → Autopilot
최대 속도 필요 → Ultrawork
반드시 완성 → Ralph
비용 중요 → Ecomode
멀티 컴포넌트 → Ultrapilot
```

### 2. 비용 최적화 팁

- ✅ **Ecomode 활용**: 30-50% 절감
- ✅ **컨텍스트 최적화**: 불필요한 파일 제외
- ✅ **배치 처리**: 유사 작업 그룹화
- ✅ **스킬 재사용**: 학습된 패턴 활용

### 3. 성능 튜닝

- ⚡ **Ultrawork**: 독립 작업 병렬 처리
- ⚡ **Ralph**: 복잡하고 긴 작업
- ⚡ **Pipeline**: 순차 의존성 있는 작업

---

## 문제 해결

### Rate Limit 발생

```bash
# 자동 재개 설정
omc wait --start
```

### 플러그인 캐시 문제

```bash
/oh-my-claudecode:doctor
```

### 업데이트 후 문제

```bash
# 플러그인 재설치
/plugin install oh-my-claudecode

# 설정 재실행
/oh-my-claudecode:omc-setup
```

---

## 라이선스 및 기여

**라이선스**: MIT License

**기여 방법**:
- ⭐ Star the repo
- 🐛 버그 리포트
- 💡 기능 제안
- 📝 코드 기여
- 💖 [Sponsor](https://github.com/sponsors/Yeachan-Heo)

---

## 관련 링크

- [GitHub 저장소](https://github.com/Yeachan-Heo/oh-my-claudecode)
- [공식 문서](https://yeachan-heo.github.io/oh-my-claudecode-website)
- [Full Reference](https://github.com/Yeachan-Heo/oh-my-claudecode/blob/main/docs/REFERENCE.md)
- [Migration Guide](https://github.com/Yeachan-Heo/oh-my-claudecode/blob/main/docs/MIGRATION.md)
- [Performance Monitoring](https://github.com/Yeachan-Heo/oh-my-claudecode/blob/main/docs/PERFORMANCE-MONITORING.md)
- [NPM Package](https://www.npmjs.com/package/oh-my-claude-sisyphus)

---

## 영감을 받은 프로젝트

- [oh-my-opencode](https://github.com/code-yeongyu/oh-my-opencode)
- [claude-hud](https://github.com/ryanjoachim/claude-hud)
- [Superpowers](https://github.com/NexTechFusion/Superpowers)
- [everything-claude-code](https://github.com/affaan-m/everything-claude-code)

---

<div align="center">

**Zero learning curve. Maximum power.**

*Don't learn Claude Code. Just use OMC.*

</div>

---

*작성일: 2026년 2월 9일*
*저자: Yeachan Heo*
