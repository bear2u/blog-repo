---
layout: post
title: "Mux 완벽 가이드 (05) - 멀티모델 지원"
date: 2026-02-08 00:00:00 +0900
categories: [AI 코딩, 개발 도구]
tags: [Mux, AI모델, Claude, GPT, Grok, Ollama, OpenRouter, 비용최적화]
author: cataclysm99
original_url: "https://github.com/coder/mux"
excerpt: "Claude, GPT, Grok, Ollama 등 다양한 AI 모델 설정 및 비용 최적화 전략"
permalink: /mux-guide-05-multimodel/
toc: true
related_posts:
  - /blog-repo/2026-02-08-mux-guide-04-agents
  - /blog-repo/2026-02-08-mux-guide-06-vscode-integration
---

## 지원되는 모델

Mux는 여러 프로바이더의 최신 프론티어 모델을 지원합니다.

### First-Class 모델 (공식 지원)

| 모델 | ID | 별칭 | 기본값 | 컨텍스트 | 특징 |
|------|-----|------|--------|---------|------|
| **Claude Opus 4.6** | `anthropic:claude-opus-4-6` | `opus` | ✓ | 200K | 최고 품질, 복잡한 추론 |
| **Claude Sonnet 4.5** | `anthropic:claude-sonnet-4-5` | `sonnet` | | 200K | 균형잡힌 성능 |
| **Claude Haiku 4.5** | `anthropic:claude-haiku-4-5` | `haiku` | | 200K | 최고 속도 |
| **GPT-5.2** | `openai:gpt-5.2` | `gpt` | | 128K | 범용 |
| **GPT-5.2 Pro** | `openai:gpt-5.2-pro` | `gpt-pro` | | 128K | 향상된 추론 |
| **GPT-5.2 Codex** | `openai:gpt-5.2-codex` | `codex` | | 128K | 코딩 특화 |
| **GPT-5.3 Codex** | `openai:gpt-5.3-codex` | `codex-5.3` | | 128K | 최신 Codex |
| **GPT-5.1 Codex Mini** | `openai:gpt-5.1-codex-mini` | `codex-mini` | | 64K | 빠른 코딩 |
| **Gemini 3 Pro** | `google:gemini-3-pro-preview` | `gemini`, `gemini-3` | | 1M | 대용량 컨텍스트 |
| **Gemini 3 Flash** | `google:gemini-3-flash-preview` | `gemini-3-flash` | | 1M | 빠른 대용량 처리 |
| **Grok 4.1 Fast** | `xai:grok-4-1-fast` | `grok`, `grok-4` | | 128K | 실시간 웹 검색 |
| **Grok Code Fast** | `xai:grok-code-fast-1` | `grok-code` | | 128K | 코딩 + 웹 검색 |

> **팁**: 별칭 사용으로 빠른 전환 가능 (`/model opus`, `/model haiku`)

---

## 모델 설정 및 API 키 관리

### UI에서 설정

```
1. Settings 열기 (⌘+, / Ctrl+,)
2. Providers 탭 선택
3. 프로바이더 확장 (Anthropic, OpenAI, Google, xAI 등)
4. API 키 입력
5. 자동 유효성 검증 (녹색 체크 표시)
```

### 환경 변수

```bash
# ~/.bashrc 또는 ~/.zshrc

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_ORG_ID="org-..."  # 선택사항

# Google
export GOOGLE_API_KEY="..."
export GOOGLE_GENERATIVE_AI_API_KEY="..."  # 대안

# xAI
export XAI_API_KEY="sk-xai-..."

# OpenRouter
export OPENROUTER_API_KEY="sk-or-v1-..."

# DeepSeek
export DEEPSEEK_API_KEY="sk-..."
```

### 설정 파일

```bash
# ~/.mux/providers.jsonc
{
  "anthropic": {
    "apiKey": "sk-ant-...",
    "baseUrl": "https://api.anthropic.com"  // 선택사항
  },
  "openai": {
    "apiKey": "sk-...",
    "orgId": "org-...",  // 선택사항
    "baseUrl": "https://api.openai.com/v1"  // 선택사항
  },
  "google": {
    "apiKey": "..."
  },
  "xai": {
    "apiKey": "sk-xai-...",
    "searchParameters": { "mode": "auto" }  // Grok 웹 검색 설정
  },
  "openrouter": {
    "apiKey": "sk-or-v1-...",
    "order": ["Cerebras", "Fireworks"],  // 프로바이더 우선순위
    "allow_fallbacks": true
  },
  "deepseek": {
    "apiKey": "sk-..."
  },
  "ollama": {
    "baseUrl": "http://localhost:11434/api"
  }
}
```

---

## Ollama 로컬 LLM 설정

### 설치

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS (Homebrew)
brew install ollama

# Windows
# https://ollama.com/download/windows
```

### 모델 다운로드

```bash
# 코딩 특화 모델
ollama pull deepseek-coder:33b
ollama pull codellama:70b
ollama pull llama3.1:70b

# 경량 모델
ollama pull deepseek-coder:7b
ollama pull llama3.1:8b

# 모델 목록 확인
ollama list
```

### Mux 설정

```
Settings → Providers → Ollama
Base URL: http://localhost:11434/api (기본값)
```

### 사용

```
/model ollama:deepseek-coder:33b
/model ollama:llama3.1:70b
```

#### 원격 Ollama 서버

```jsonc
// ~/.mux/providers.jsonc
{
  "ollama": {
    "baseUrl": "http://your-server:11434/api"
  }
}
```

---

## OpenRouter 통합

OpenRouter는 300+ 모델에 단일 API로 접근할 수 있는 서비스입니다.

### 설정

```bash
# API 키 발급: https://openrouter.ai/

# 환경 변수
export OPENROUTER_API_KEY="sk-or-v1-..."

# 또는 설정 파일
# ~/.mux/providers.jsonc
{
  "openrouter": {
    "apiKey": "sk-or-v1-..."
  }
}
```

### 모델 사용

```
/model openrouter:anthropic/claude-opus-4-6
/model openrouter:meta-llama/llama-3.3-405b-instruct
/model openrouter:google/gemini-pro-2.0
```

### Provider Routing (고급)

```jsonc
// ~/.mux/providers.jsonc
{
  "openrouter": {
    "apiKey": "sk-or-v1-...",

    // 프로바이더 우선순위
    "order": ["Cerebras", "Fireworks", "Together"],

    // 실패 시 대체 허용
    "allow_fallbacks": true,

    // 특정 프로바이더만 사용
    "only": ["Cerebras"],

    // 특정 프로바이더 제외
    "ignore": ["SlowProvider"],

    // 데이터 수집 정책
    "data_collection": "deny"  // "allow" 또는 "deny"
  }
}
```

> **참고**: [OpenRouter Provider Routing](https://openrouter.ai/docs/features/provider-routing)

---

## 모델별 특징 및 사용 시나리오

### Claude (Anthropic)

#### Opus 4.6

```yaml
컨텍스트: 200K 토큰
강점:
  - 최고 수준의 추론 능력
  - 복잡한 리팩토링
  - 아키텍처 설계
  - 보안 감사
약점:
  - 상대적으로 느림
  - 높은 비용

사용 시나리오:
  - 대규모 리팩토링
  - 복잡한 버그 수정
  - 아키텍처 설계
  - Plan 모드 (플랜 작성)
```

#### Sonnet 4.5

```yaml
컨텍스트: 200K 토큰
강점:
  - 균형잡힌 속도/품질
  - 일반적인 코딩 작업
  - 비용 효율적
약점:
  - Opus보다 추론력 낮음

사용 시나리오:
  - 일반 기능 개발
  - 버그 수정
  - 테스트 작성
  - Exec 모드 (기본)
```

#### Haiku 4.5

```yaml
컨텍스트: 200K 토큰
강점:
  - 최고 속도
  - 최저 비용
  - 간단한 작업에 최적
약점:
  - 복잡한 추론 제한적

사용 시나리오:
  - 간단한 수정
  - 코드 탐색 (Ask 모드)
  - 반복 작업
  - 탐색 서브에이전트
```

### GPT (OpenAI)

#### GPT-5.2 Codex

```yaml
컨텍스트: 128K 토큰
강점:
  - 코딩 특화 학습
  - 다양한 언어 지원
  - 빠른 코드 생성
약점:
  - Claude Opus보다 추론력 낮음

사용 시나리오:
  - 빠른 프로토타이핑
  - 코드 생성
  - API 래퍼 작성
```

#### GPT-5.1 Codex Mini

```yaml
컨텍스트: 64K 토큰
강점:
  - 매우 빠름
  - 저비용
약점:
  - 짧은 컨텍스트

사용 시나리오:
  - 간단한 함수 작성
  - 빠른 수정
  - 단일 파일 작업
```

### Gemini (Google)

#### Gemini 3 Pro

```yaml
컨텍스트: 1M 토큰 (1,000,000!)
강점:
  - 초대용량 컨텍스트
  - 전체 코드베이스 분석 가능
  - 멀티모달 (이미지, 비디오)
약점:
  - 코딩 특화 아님

사용 시나리오:
  - 대규모 코드베이스 분석
  - 전체 프로젝트 리팩토링
  - 문서 전체 컨텍스트
```

#### Gemini 3 Flash

```yaml
컨텍스트: 1M 토큰
강점:
  - Gemini Pro보다 빠름
  - 저비용
  - 대용량 컨텍스트 유지
약점:
  - Pro보다 품질 낮음

사용 시나리오:
  - 대용량 로그 분석
  - 전체 프로젝트 탐색
  - 문서 검색
```

### Grok (xAI)

#### Grok 4.1 Fast

```yaml
컨텍스트: 128K 토큰
강점:
  - 실시간 웹 검색 통합
  - 최신 정보 참조
약점:
  - 코딩 특화 아님

사용 시나리오:
  - 최신 라이브러리 정보
  - API 문서 검색
  - 오류 메시지 검색
```

#### Grok Code Fast

```yaml
컨텍스트: 128K 토큰
강점:
  - 코딩 + 웹 검색
  - 최신 문서 참조
약점:
  - 다른 Codex보다 느림

사용 시나리오:
  - 신규 프레임워크 학습
  - 최신 베스트 프랙티스
```

---

## 모델 선택 전략

### 작업 유형별 권장 모델

| 작업 | 1순위 | 2순위 | 로컬 (Ollama) |
|------|-------|-------|---------------|
| **복잡한 리팩토링** | Opus 4.6 | GPT-5.2 Pro | DeepSeek-Coder:33b |
| **일반 기능 개발** | Sonnet 4.5 | GPT-5.2 Codex | DeepSeek-Coder:33b |
| **빠른 수정** | Haiku 4.5 | Codex Mini | DeepSeek-Coder:7b |
| **코드 탐색** | Haiku 4.5 | Gemini Flash | Llama3.1:8b |
| **대규모 분석** | Gemini 3 Pro | Opus 4.6 | Llama3.1:70b |
| **최신 정보 필요** | Grok Code | Grok 4.1 | - |
| **문서 작성** | Sonnet 4.5 | GPT-5.2 | Llama3.1:70b |
| **보안 감사** | Opus 4.6 | GPT-5.2 Pro | DeepSeek-Coder:33b |

### 비용/품질 트레이드오프

```
높은 비용, 최고 품질
├── Claude Opus 4.6
├── GPT-5.2 Pro
├── Claude Sonnet 4.5
├── GPT-5.2 Codex
├── Gemini 3 Pro
├── Claude Haiku 4.5
├── GPT-5.1 Codex Mini
├── Gemini 3 Flash
└── Ollama (무료, 로컬)
낮은 비용, 낮은 품질
```

### 속도 순위

```
최고 속도
├── Claude Haiku 4.5
├── GPT-5.1 Codex Mini
├── Gemini 3 Flash
├── Claude Sonnet 4.5
├── GPT-5.2 Codex
├── Gemini 3 Pro
├── GPT-5.2 Pro
└── Claude Opus 4.6
느림
```

---

## 비용 추적

### Costs 탭

```
사이드바 → Costs 탭

┌───────────────────────────────────┐
│  Session Costs                    │
├───────────────────────────────────┤
│  Model: Claude Opus 4.6           │
│  Input:  125,000 tokens ($1.25)   │
│  Output: 15,000 tokens  ($0.75)   │
│  Total:  $2.00                    │
├───────────────────────────────────┤
│  Context Usage: 45% ████░░░░░     │
│  Auto-Compact at 70%              │
└───────────────────────────────────┘
```

### 비용 분석

```
Costs → Statistics

┌───────────────────────────────────┐
│  Total Costs (Last 30 Days)       │
├───────────────────────────────────┤
│  Claude Opus 4.6:    $125.50      │
│  Claude Sonnet 4.5:  $45.20       │
│  Claude Haiku 4.5:   $8.30        │
│  GPT-5.2 Codex:      $32.10       │
│  Ollama:             $0.00        │
├───────────────────────────────────┤
│  Total:              $211.10      │
└───────────────────────────────────┘
```

### 컨텍스트 사용량

```
┌───────────────────────────────────┐
│  Context Usage                    │
├───────────────────────────────────┤
│  Current: 90,000 / 200,000 (45%)  │
│  ████████████░░░░░░░░░░░░░        │
│                                   │
│  Auto-Compact Threshold: 70%      │
│  [────────────────██──────]        │
│                                   │
│  Warning appears at: 58%          │
└───────────────────────────────────┘
```

---

## 비용 최적화 전략

### 전략 1: 모델 스위칭

```
Plan 모드: Claude Opus 4.6 (최고 품질 계획)
Exec 모드: Claude Sonnet 4.5 (균형)
Explore 서브에이전트: Claude Haiku 4.5 (속도)

# Command Palette에서 빠른 전환
⌘+/ (macOS) / Ctrl+/ (Windows/Linux)
```

### 전략 2: 에이전트별 기본 모델

```markdown
<!-- .mux/agents/fast-exec.md -->
---
name: Fast Exec
base: exec
ai:
  model: haiku
  thinkingLevel: low
---

Use this for simple, repetitive tasks.
```

### 전략 3: 컨텍스트 압축

```
자동 압축 활성화:
Settings → Costs → Auto-Compact: 70%

수동 압축:
/compact  (중요 정보 보존)

Start Here:
대화 중간에 플랜/결과만 남기고 히스토리 교체
```

### 전략 4: Ollama 활용

```bash
# 민감한 코드 → Ollama (로컬)
/model ollama:deepseek-coder:33b

# 탐색 작업 → Ollama
워크스페이스: explore-codebase
모델: ollama:llama3.1:70b

# 문서 작성 → Ollama
워크스페이스: write-docs
모델: ollama:llama3.1:70b
```

### 전략 5: OpenRouter 저가 모델

```
OpenRouter 저가 모델:
- meta-llama/llama-3.3-70b-instruct
- mistralai/mistral-large-2
- google/gemma-2-27b

/model openrouter:meta-llama/llama-3.3-70b-instruct
```

---

## 모델 선택 및 전환

### UI에서 전환

```
채팅 입력창 하단
→ 모델 pill 클릭 (예: "Opus 4.6")
→ 드롭다운에서 선택
```

### 키보드 단축키

```
⌘+/ (macOS) / Ctrl+/ (Windows/Linux)
→ 모델 순환 (Opus → Sonnet → Haiku → ...)
```

### Command Palette

```
⌘+Shift+P / Ctrl+Shift+P
→ "Change Model"
→ 검색 및 선택
```

### 슬래시 명령어

```
/model opus
/model sonnet
/model haiku
/model gpt
/model codex
/model gemini
/model grok
/model ollama:deepseek-coder:33b
/model openrouter:anthropic/claude-opus-4-6
```

---

## 고급 프로바이더 설정

### Azure OpenAI

```bash
# 환경 변수
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT="gpt-5-deployment"
export AZURE_OPENAI_API_VERSION="2024-02-15"

# OpenAI 프로바이더가 자동으로 Azure 백엔드 사용
```

### AWS Bedrock

```jsonc
// ~/.mux/providers.jsonc
{
  "bedrock": {
    "region": "us-east-1",

    // 인증 방법 (우선순위)
    // 1. Bearer Token (단일 API 키)
    "bearerToken": "...",  // 또는 AWS_BEARER_TOKEN_BEDROCK 환경 변수

    // 2. 명시적 자격 증명
    "accessKeyId": "AKIA...",
    "secretAccessKey": "...",

    // 3. AWS 자격 증명 체인 (자동)
    // - ~/.aws/credentials
    // - AWS SSO
    // - EC2/ECS IAM 역할
  }
}
```

#### AWS SSO 사용

```bash
# SSO 로그인
aws sso login --profile my-profile

# Mux는 자동으로 SSO 자격 증명 사용
# 설정 파일에 region만 지정
```

### GitHub Copilot

```bash
# 환경 변수
export GITHUB_COPILOT_TOKEN="..."

# 또는 설정 파일
# ~/.mux/providers.jsonc
{
  "github-copilot": {
    "apiKey": "..."
  }
}
```

---

## xAI Grok 웹 검색 설정

### 기본 설정 (자동)

```jsonc
// ~/.mux/providers.jsonc
{
  "xai": {
    "apiKey": "sk-xai-...",
    "searchParameters": { "mode": "auto" }  // 기본값
  }
}
```

### 고급 설정

```jsonc
{
  "xai": {
    "apiKey": "sk-xai-...",
    "searchParameters": {
      "mode": "always",  // "auto", "always", "never"
      "region": "US",    // 지역 필터
      "time_filter": "day"  // "hour", "day", "week", "month"
    }
  }
}
```

> **참고**: [xAI Search Orchestration](https://docs.x.ai/docs/resources/search)

---

## 문제 해결

### 모델이 표시되지 않음

```bash
# 원인: API 키 미설정 또는 유효하지 않음

# 해결책:
Settings → Providers → [프로바이더] → API Key 재입력
→ 녹색 체크 표시 확인
```

### Ollama 연결 실패

```bash
# Ollama 서비스 확인
ollama list

# 서비스 재시작
# macOS/Linux
ollama serve

# Base URL 확인
Settings → Providers → Ollama
→ Base URL: http://localhost:11434/api
```

### 비용이 예상보다 높음

```bash
# Costs 탭에서 분석
Costs → Statistics

# 컨텍스트 압축
/compact

# 자동 압축 활성화
Settings → Costs → Auto-Compact: 70%

# 저가 모델 사용
/model haiku
/model ollama:deepseek-coder:33b
```

---

## 다음 단계

멀티모델 지원을 마스터했다면:

1. **[챕터 06: VS Code 통합](/blog-repo/mux-guide-06-vscode-integration)** - IDE 워크플로우 최적화
2. **[챕터 07: 고급 기능](/blog-repo/mux-guide-07-advanced-features)** - Mode Prompts, 프로젝트 시크릿
3. **비용 최적화 실험** - 작업별 최적 모델 찾기

---

## 참고 자료

- [Models 문서](https://mux.coder.com/config/models)
- [Providers 문서](https://mux.coder.com/config/providers)
- [Ollama 공식 사이트](https://ollama.com/)
- [OpenRouter 문서](https://openrouter.ai/docs)
- [Anthropic 가격 정책](https://www.anthropic.com/pricing)
- [OpenAI 가격 정책](https://openai.com/pricing)
