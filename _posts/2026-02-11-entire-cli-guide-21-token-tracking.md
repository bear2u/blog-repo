---
layout: post
title: "Entire CLI 완벽 가이드 (21) - Token Usage Tracking"
date: 2026-02-11
permalink: /entire-cli-guide-21-token-tracking/
author: Entire Team
categories: [AI 코딩, 개발 도구]
tags: [Entire, Token, Usage, Analytics, Cost]
original_url: "https://github.com/entireio/cli"
excerpt: "Entire CLI의 Token Usage Tracking - AI 사용량 추적 및 비용 분석 완벽 가이드"
---

## 개요

**Token Usage Tracking**은 AI 에이전트의 토큰 사용량을 추적하여 비용을 분석하고 최적화할 수 있게 합니다. 각 세션의 토큰 사용량이 체크포인트 메타데이터에 저장되어 나중에 조회할 수 있습니다.

```
┌─────────────────────────────────────────────────────┐
│              Token Usage Tracking                    │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Session Transcript          Token Usage             │
│         │                          ▲                 │
│         ▼                          │                 │
│  ┌──────────────┐                  │                 │
│  │ API Response │                  │                 │
│  │ {            │                  │                 │
│  │   usage: {   │──────────────────┘                 │
│  │     input_tokens: 1500                            │
│  │     cache_creation_tokens: 200                    │
│  │     cache_read_tokens: 800                        │
│  │     output_tokens: 500                            │
│  │   }                                               │
│  │ }                                                 │
│  └──────────────┘                                    │
│                                                      │
│  Saved to:                                           │
│  entire/checkpoints/v1/<id>/metadata.json            │
│                                                      │
│  {                                                   │
│    "token_usage": {                                  │
│      "input_tokens": 1500,                           │
│      "cache_creation_tokens": 200,                   │
│      "cache_read_tokens": 800,                       │
│      "output_tokens": 500,                           │
│      "api_call_count": 3                             │
│    }                                                 │
│  }                                                   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## TokenUsage 구조

### 데이터 타입

```go
type TokenUsage struct {
    InputTokens          int `json:"input_tokens"`
    CacheCreationTokens  int `json:"cache_creation_tokens"`
    CacheReadTokens      int `json:"cache_read_tokens"`
    OutputTokens         int `json:"output_tokens"`
    APICallCount         int `json:"api_call_count"`
}
```

### 필드 설명

| 필드 | 설명 | Claude API | 비용 영향 |
|-----|------|-----------|----------|
| **InputTokens** | 입력 토큰 수 | `usage.input_tokens` | 높음 |
| **CacheCreationTokens** | 캐시 생성 토큰 | `usage.cache_creation_input_tokens` | 높음 |
| **CacheReadTokens** | 캐시 읽기 토큰 | `usage.cache_read_input_tokens` | 낮음 (90% 할인) |
| **OutputTokens** | 출력 토큰 수 | `usage.output_tokens` | 매우 높음 |
| **APICallCount** | API 호출 횟수 | - | 간접적 |

---

## 토큰 수집

### Claude Code Transcript

Claude Code의 transcript에 토큰 정보가 포함됩니다.

**full.jsonl 예시:**

```jsonl
{"type":"user","content":"Add user authentication"}
{"type":"assistant","content":"I'll implement authentication...","usage":{"input_tokens":1500,"cache_creation_input_tokens":200,"cache_read_input_tokens":800,"output_tokens":500}}
{"type":"tool_use","tool":"Edit","input":"..."}
{"type":"tool_result","tool":"Edit","output":"..."}
{"type":"assistant","content":"Done.","usage":{"input_tokens":2000,"cache_creation_input_tokens":0,"cache_read_input_tokens":1500,"output_tokens":100}}
```

### Transcript 파싱

```go
type Message struct {
    Type    string      `json:"type"`
    Content string      `json:"content"`
    Usage   *TokenUsage `json:"usage,omitempty"`
}

func extractTokenUsage(transcriptBytes []byte) TokenUsage {
    total := TokenUsage{}
    scanner := bufio.NewScanner(bytes.NewReader(transcriptBytes))

    for scanner.Scan() {
        var msg Message
        json.Unmarshal(scanner.Bytes(), &msg)

        if msg.Usage != nil {
            total.InputTokens += msg.Usage.InputTokens
            total.CacheCreationTokens += msg.Usage.CacheCreationTokens
            total.CacheReadTokens += msg.Usage.CacheReadTokens
            total.OutputTokens += msg.Usage.OutputTokens
            total.APICallCount++
        }
    }

    return total
}
```

---

## 저장 위치

### Checkpoint Metadata

Token usage는 체크포인트 메타데이터에 저장됩니다.

**CommittedMetadata (세션별):**

```json
{
  "session_id": "2026-02-11-abc123...",
  "checkpoint_id": "a3b2c4d5e6f7",
  "token_usage": {
    "input_tokens": 1500,
    "cache_creation_tokens": 200,
    "cache_read_tokens": 800,
    "output_tokens": 500,
    "api_call_count": 3
  }
}
```

**CheckpointSummary (통합):**

```json
{
  "checkpoint_id": "a3b2c4d5e6f7",
  "checkpoints_count": 3,
  "token_usage": {
    "input_tokens": 5000,
    "cache_creation_tokens": 500,
    "cache_read_tokens": 3000,
    "output_tokens": 1200,
    "api_call_count": 10
  }
}
```

Multi-session checkpoint의 경우 모든 세션의 토큰이 합산됩니다.

---

## 비용 계산

### Claude API 요금

**Opus 4.6 (2026년 2월 기준):**

| 토큰 타입 | 요금 (per million tokens) |
|----------|-------------------------|
| Input | $15.00 |
| Cache Creation | $18.75 |
| Cache Read | $1.50 (90% 할인) |
| Output | $75.00 |

**Sonnet 4.5:**

| 토큰 타입 | 요금 (per million tokens) |
|----------|-------------------------|
| Input | $3.00 |
| Cache Creation | $3.75 |
| Cache Read | $0.30 (90% 할인) |
| Output | $15.00 |

### 비용 계산 공식

```go
func calculateCost(usage TokenUsage, model string) float64 {
    rates := getRates(model)

    cost := 0.0
    cost += float64(usage.InputTokens) / 1_000_000 * rates.Input
    cost += float64(usage.CacheCreationTokens) / 1_000_000 * rates.CacheCreation
    cost += float64(usage.CacheReadTokens) / 1_000_000 * rates.CacheRead
    cost += float64(usage.OutputTokens) / 1_000_000 * rates.Output

    return cost
}

type Rates struct {
    Input         float64
    CacheCreation float64
    CacheRead     float64
    Output        float64
}

func getRates(model string) Rates {
    switch model {
    case "opus-4.6":
        return Rates{15.00, 18.75, 1.50, 75.00}
    case "sonnet-4.5":
        return Rates{3.00, 3.75, 0.30, 15.00}
    default:
        return Rates{3.00, 3.75, 0.30, 15.00} // Default to Sonnet
    }
}
```

---

## Token Usage 조회

### explain 명령어

```bash
entire explain <commit-hash>

# 출력:
Checkpoint: a3b2c4d5e6f7
Session: 2026-02-11-abc123...
Commit: Add user authentication

Token Usage:
  Input tokens:          1,500
  Cache creation:          200
  Cache read:              800
  Output tokens:           500
  API calls:                 3

Estimated cost (Sonnet 4.5): $0.014
```

### status 명령어

현재 세션의 토큰 사용량 표시:

```bash
entire status

# 출력:
Current session: 2026-02-11-abc123...
Strategy: manual-commit
Checkpoints: 2 uncommitted

Token usage (current session):
  Input tokens:          3,500
  Cache creation:          500
  Cache read:            2,000
  Output tokens:         1,000
  API calls:                 7

Estimated cost: $0.031
```

---

## 통계 및 분석

### 프로젝트 전체 통계

```bash
# 모든 체크포인트의 토큰 사용량 합산
entire stats --all

# 출력:
Total checkpoints: 50

Token usage (all sessions):
  Input tokens:        150,000
  Cache creation:       20,000
  Cache read:           80,000
  Output tokens:        30,000
  API calls:               250

Total estimated cost: $1.23
```

### 날짜별 통계

```bash
# 이번 주 통계
entire stats --since 7d

# 출력:
Checkpoints: 12 (last 7 days)

Token usage:
  Input tokens:         15,000
  Output tokens:         3,000

Daily average:
  Checkpoints:              1.7
  Input tokens:           2,143
  Cost:                  $0.053
```

### 세션별 비교

```bash
# 토큰 사용량이 많은 세션 찾기
entire stats --top 5

# 출력:
Top 5 sessions by token usage:

1. 2026-02-10-xyz789... (5,000 tokens, $0.045)
   "Implement complex authentication flow"

2. 2026-02-09-abc123... (4,500 tokens, $0.041)
   "Refactor database layer"

3. 2026-02-08-def456... (4,200 tokens, $0.038)
   "Add comprehensive test suite"
```

---

## Condensation 시 집계

### 세션별 토큰 추출

```go
func (s *ManualCommitStrategy) Condense(sessionID string) error {
    // 1. Transcript 읽기
    transcriptBytes := ReadTranscript(sessionID)

    // 2. Token usage 추출
    usage := extractTokenUsage(transcriptBytes)

    // 3. Metadata에 저장
    metadata := CommittedMetadata{
        SessionID:   sessionID,
        TokenUsage:  &usage,
    }

    SaveMetadata(checkpointID, metadata)

    return nil
}
```

### 다중 세션 합산

```go
func aggregateTokenUsage(sessions []CommittedMetadata) TokenUsage {
    total := TokenUsage{}

    for _, session := range sessions {
        if session.TokenUsage != nil {
            total.InputTokens += session.TokenUsage.InputTokens
            total.CacheCreationTokens += session.TokenUsage.CacheCreationTokens
            total.CacheReadTokens += session.TokenUsage.CacheReadTokens
            total.OutputTokens += session.TokenUsage.OutputTokens
            total.APICallCount += session.TokenUsage.APICallCount
        }
    }

    return total
}
```

---

## 최적화 인사이트

### 1. 캐시 효율성

캐시 읽기 비율을 계산하여 효율성을 측정합니다.

```go
func calculateCacheEfficiency(usage TokenUsage) float64 {
    totalInput := usage.InputTokens + usage.CacheCreationTokens + usage.CacheReadTokens
    if totalInput == 0 {
        return 0
    }

    return float64(usage.CacheReadTokens) / float64(totalInput) * 100
}
```

**예시:**

```
Cache efficiency: 45.7%
  - Input tokens:       1,500
  - Cache creation:       200
  - Cache read:         1,500  ← 45.7% of total
  - Total input:        3,200

Estimated savings: $0.012 (from cache reads)
```

### 2. Output Token 비율

Output 토큰이 전체 비용의 몇 %인지 계산:

```go
func calculateOutputRatio(usage TokenUsage) float64 {
    cost := calculateCost(usage, "sonnet-4.5")
    outputCost := float64(usage.OutputTokens) / 1_000_000 * 15.00

    return outputCost / cost * 100
}
```

**예시:**

```
Output token cost: 72.3% of total
  → Consider shorter responses or fewer iterations
```

### 3. API Call 효율성

API 호출당 평균 토큰 수:

```go
func calculateTokensPerCall(usage TokenUsage) float64 {
    if usage.APICallCount == 0 {
        return 0
    }

    totalTokens := usage.InputTokens + usage.OutputTokens
    return float64(totalTokens) / float64(usage.APICallCount)
}
```

**예시:**

```
Average tokens per API call: 1,234
  - API calls: 7
  - Total tokens: 8,638

Tip: Batch operations to reduce API calls
```

---

## 실습 예제

### 1. 세션 비용 추적

```bash
# 세션 시작
claude "Implement user authentication"

# 작업 진행...

# 현재 사용량 확인
entire status

# 출력:
Token usage (current session):
  Input tokens:          2,500
  Output tokens:           600
  Estimated cost:      $0.018

# 커밋
git commit -m "Add authentication"

# 최종 사용량 확인
entire explain HEAD

# 출력:
Token usage:
  Input tokens:          3,500
  Output tokens:         1,000
  API calls:                 5
  Final cost:          $0.026
```

### 2. 프로젝트 비용 분석

```bash
# 전체 프로젝트 통계
entire stats --all

# 월별 비용 추세
entire stats --since 30d --group-by month

# 가장 비싼 세션 찾기
entire stats --top 10 --sort-by cost
```

### 3. 최적화 기회 발견

```bash
# 캐시 효율성 분석
entire stats --cache-analysis

# 출력:
Cache efficiency analysis:

Sessions with low cache efficiency:
1. 2026-02-10-xyz... (15.2% cache hits)
   → Consider using longer context windows

2. 2026-02-09-abc... (8.7% cache hits)
   → Session has many unique prompts

Average cache efficiency: 42.5%
Potential savings with 60% target: $0.15/month
```

---

## 할당량 및 경고

### 할당량 설정

```json
{
  "token_limits": {
    "daily_limit": 100000,
    "monthly_limit": 2000000,
    "session_limit": 10000
  }
}
```

### 경고 시스템

```go
func checkTokenLimit(usage TokenUsage, limits TokenLimits) error {
    totalTokens := usage.InputTokens + usage.OutputTokens

    if totalTokens > limits.SessionLimit {
        return fmt.Errorf("session token limit exceeded: %d > %d", totalTokens, limits.SessionLimit)
    }

    return nil
}
```

**경고 메시지:**

```
Warning: Session token usage is high (8,500 / 10,000)
  Consider committing and starting a new session.
```

---

## Export 및 리포팅

### CSV Export

```bash
entire stats --export csv --output token_usage.csv

# token_usage.csv:
date,session_id,input_tokens,output_tokens,cost
2026-02-11,abc123...,3500,1000,0.026
2026-02-10,def456...,5000,1500,0.038
```

### JSON Export

```bash
entire stats --export json --output token_usage.json

# token_usage.json:
[
  {
    "date": "2026-02-11",
    "session_id": "2026-02-11-abc123...",
    "token_usage": {
      "input_tokens": 3500,
      "output_tokens": 1000
    },
    "cost": 0.026
  }
]
```

---

## 다음 단계

Token Usage Tracking을 이해했습니다! 다음 챕터에서는:

- **개발 환경 설정** - mise, Go, 테스트 실행
- **코드 구조** - 패키지 구성, 주요 파일
- **Agent 통합** - Gemini CLI, 새 Agent 추가

---

*다음 글에서는 Entire CLI 개발 환경 설정을 살펴봅니다.*
