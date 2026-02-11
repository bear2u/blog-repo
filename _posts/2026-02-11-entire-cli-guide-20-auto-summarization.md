---
layout: post
title: "Entire CLI 완벽 가이드 (20) - Auto-Summarization"
date: 2026-02-11
permalink: /entire-cli-guide-20-auto-summarization/
author: Entire Team
categories: [AI 코딩, 개발 도구]
tags: [Entire, Summarization, AI, Claude, Checkpoint]
original_url: "https://github.com/entireio/cli"
excerpt: "Entire CLI의 AI 기반 Auto-Summarization - 세션 자동 요약 완벽 가이드"
---

## 개요

**Auto-Summarization**은 AI를 사용하여 세션 transcript를 자동으로 요약하는 기능입니다. 긴 세션을 간단한 설명으로 변환하여 나중에 빠르게 이해할 수 있게 합니다.

```
┌─────────────────────────────────────────────────────┐
│              Auto-Summarization Flow                 │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Session Transcript              Summary             │
│  (1000+ lines)                   (2-3 sentences)     │
│         │                              ▲             │
│         ▼                              │             │
│  ┌──────────────────┐                  │             │
│  │ Condense         │                  │             │
│  │ - User prompts   │                  │             │
│  │ - Tool calls     │                  │             │
│  │ - File edits     │                  │             │
│  └────────┬─────────┘                  │             │
│           │                            │             │
│           ▼                            │             │
│  ┌──────────────────┐                  │             │
│  │ Claude API       │──────────────────┘             │
│  │ Haiku 3.5        │                                │
│  └──────────────────┘                                │
│                                                      │
│  "Implemented user authentication with JWT tokens    │
│   and added login/logout endpoints"                  │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Summarization 패키지

### 위치

```
cmd/entire/cli/summarize/
├── summarize.go       # Core 로직
├── claude.go          # Claude API 통합
├── summarize_test.go  # 테스트
└── claude_test.go
```

### Generator 인터페이스

```go
type Generator interface {
    Generate(ctx context.Context, input Input) (*checkpoint.Summary, error)
}
```

**구현체:**

- `ClaudeGenerator` - Claude Haiku 3.5 사용 (기본)
- 향후 확장 가능 (GPT, Gemini 등)

---

## Summary 구조

### checkpoint.Summary

```go
type Summary struct {
    Description string   `json:"description"`  // 2-3 문장 요약
    KeyChanges  []string `json:"key_changes"`  // 주요 변경사항 목록
}
```

**예시:**

```json
{
  "description": "Implemented user authentication with JWT tokens and added login/logout endpoints. Created middleware for token validation and updated API routes to use authentication.",
  "key_changes": [
    "Added JWT token generation and validation",
    "Created auth middleware",
    "Implemented login and logout endpoints",
    "Updated API routes with authentication"
  ]
}
```

---

## Condensed Transcript

### 목적

전체 transcript (1000+ lines)를 요약 가능한 크기로 압축합니다.

```
Full Transcript (1000 lines)
    ↓
Condensed Transcript (50-100 lines)
    ↓
Summary (2-3 sentences)
```

### Entry 타입

```go
type Entry struct {
    Type       EntryType  // user, assistant, tool
    Content    string     // 텍스트 내용
    ToolName   string     // 도구 이름 (tool 타입)
    ToolDetail string     // 도구 상세 (tool 타입)
}

type EntryType string

const (
    EntryTypeUser      EntryType = "user"
    EntryTypeAssistant EntryType = "assistant"
    EntryTypeTool      EntryType = "tool"
)
```

### Condensation 로직

```go
func BuildCondensedTranscriptFromBytes(transcriptBytes []byte) ([]Entry, error) {
    entries := []Entry{}
    scanner := bufio.NewScanner(bytes.NewReader(transcriptBytes))

    for scanner.Scan() {
        var msg transcript.Message
        json.Unmarshal(scanner.Bytes(), &msg)

        switch msg.Type {
        case "user":
            // 사용자 프롬프트 추가
            entries = append(entries, Entry{
                Type:    EntryTypeUser,
                Content: msg.Content,
            })

        case "assistant":
            // Assistant 응답 (간략화)
            entries = append(entries, Entry{
                Type:    EntryTypeAssistant,
                Content: summarizeAssistantResponse(msg),
            })

        case "tool_use":
            // 도구 호출 (메타데이터만)
            entries = append(entries, Entry{
                Type:       EntryTypeTool,
                ToolName:   msg.ToolName,
                ToolDetail: getToolDetail(msg),
            })
        }
    }

    return entries, nil
}
```

### 도구별 처리

```go
// Minimal detail tools - 상세 정보 생략
var minimalDetailTools = map[string]bool{
    "Read":  true,  // 파일 경로만
    "Write": true,  // 파일 경로만
    "Edit":  true,  // 파일 경로만
}

func getToolDetail(msg transcript.Message) string {
    if minimalDetailTools[msg.ToolName] {
        // 파일 경로만 반환
        return extractFilePath(msg.Input)
    }

    // 전체 input 반환 (Bash, Grep 등)
    return msg.Input
}
```

**예시:**

```go
// Read 도구
Entry{
    Type:       "tool",
    ToolName:   "Read",
    ToolDetail: "src/auth.go",  // 전체 내용 생략
}

// Bash 도구
Entry{
    Type:       "tool",
    ToolName:   "Bash",
    ToolDetail: "npm test",  // 전체 명령어 포함
}
```

---

## Claude Generator

### API 통합

```go
type ClaudeGenerator struct {
    APIKey string
    Model  string  // "claude-haiku-3.5-20250219"
}

func (g *ClaudeGenerator) Generate(ctx context.Context, input Input) (*checkpoint.Summary, error) {
    // 1. Prompt 생성
    prompt := buildSummarizationPrompt(input)

    // 2. Claude API 호출
    response := callClaudeAPI(prompt)

    // 3. JSON 파싱
    var summary checkpoint.Summary
    json.Unmarshal(response.Content, &summary)

    return &summary, nil
}
```

### 모델 선택

**Claude Haiku 3.5** (기본)

- **빠름** - 1-2초 응답
- **저렴** - 요약에 충분한 성능
- **정확** - 코드 변경사항 이해

### API Key 설정

```bash
# 환경 변수
export ANTHROPIC_API_KEY=sk-ant-...

# 또는 설정 파일
{
  "summarize": {
    "enabled": true,
    "api_key": "sk-ant-..."
  }
}
```

---

## Summarization Prompt

### 구조

```
You are a code review assistant. Summarize this development session.

## Transcript
[Condensed transcript entries...]

## Files Modified
- src/auth.go
- src/middleware/auth.go
- tests/auth_test.go

## Instructions
- Provide a 2-3 sentence description
- List 3-5 key changes
- Focus on WHAT was done, not HOW

## Output Format (JSON)
{
  "description": "...",
  "key_changes": ["...", "..."]
}
```

### Prompt 생성 로직

```go
func buildSummarizationPrompt(input Input) string {
    var sb strings.Builder

    // Header
    sb.WriteString("You are a code review assistant. Summarize this development session.\n\n")

    // Transcript
    sb.WriteString("## Transcript\n")
    for _, entry := range input.Transcript {
        switch entry.Type {
        case EntryTypeUser:
            sb.WriteString(fmt.Sprintf("User: %s\n", entry.Content))
        case EntryTypeAssistant:
            sb.WriteString(fmt.Sprintf("Assistant: %s\n", entry.Content))
        case EntryTypeTool:
            sb.WriteString(fmt.Sprintf("Tool: %s - %s\n", entry.ToolName, entry.ToolDetail))
        }
    }

    // Files
    if len(input.FilesTouched) > 0 {
        sb.WriteString("\n## Files Modified\n")
        for _, file := range input.FilesTouched {
            sb.WriteString(fmt.Sprintf("- %s\n", file))
        }
    }

    // Instructions
    sb.WriteString("\n## Instructions\n")
    sb.WriteString("- Provide a 2-3 sentence description\n")
    sb.WriteString("- List 3-5 key changes\n")
    sb.WriteString("- Focus on WHAT was done, not HOW\n")

    // Format
    sb.WriteString("\n## Output Format (JSON)\n")
    sb.WriteString(`{"description":"...","key_changes":["...","..."]}`)

    return sb.String()
}
```

---

## 자동 요약 활성화

### Settings 설정

```json
{
  "summarize": {
    "enabled": true,
    "api_key": "sk-ant-..."
  }
}
```

### Enable 시 설정

```bash
entire enable --summarize

# 또는
entire enable --summarize --api-key sk-ant-...
```

**프롬프트:**

```
Enable auto-summarization? [y/N] y

Enter your Anthropic API key (or leave blank to set later):
sk-ant-...

Auto-summarization enabled. Summaries will be generated at commit time.
```

---

## 요약 생성 시점

### 1. Condensation 시 (Manual-Commit)

커밋 시 자동으로 요약을 생성합니다.

```bash
git commit -m "Add authentication"

# PostCommit hook:
# → Condensation 시작
# → Transcript 압축
# → Claude API 호출
# → Summary 저장
```

**저장 위치:**

```
entire/checkpoints/v1/<cpid[:2]>/<cpid[2:]>/0/
├── metadata.json        # Summary 포함
├── full.jsonl
├── prompt.txt
└── context.md
```

**metadata.json:**

```json
{
  "checkpoint_id": "a3b2c4d5e6f7",
  "session_id": "2026-02-11-abc123...",
  "summary": {
    "description": "Implemented user authentication with JWT tokens",
    "key_changes": ["Added JWT validation", "Created auth middleware"]
  }
}
```

### 2. 수동 생성 (explain --generate)

```bash
entire explain --generate

# 현재 세션의 요약 즉시 생성
```

---

## explain 명령어 통합

### --generate Flag

```bash
# 커밋의 요약 생성
entire explain <commit-hash> --generate

# 현재 세션의 요약 생성
entire explain --generate
```

**동작:**

1. Checkpoint 메타데이터 조회
2. Transcript 로드
3. Condensation
4. Claude API 호출
5. Summary 출력

**예시:**

```bash
entire explain HEAD --generate

# 출력:
Generating summary...

Description:
  Implemented user authentication with JWT tokens and added login/logout
  endpoints. Created middleware for token validation and updated API routes
  to use authentication.

Key Changes:
  - Added JWT token generation and validation
  - Created auth middleware
  - Implemented login and logout endpoints
  - Updated API routes with authentication
```

---

## 에러 처리

### API Key 누락

```bash
entire explain --generate

# 출력:
Error: ANTHROPIC_API_KEY not set
Set it via environment variable or settings.json:
  export ANTHROPIC_API_KEY=sk-ant-...
  or
  {
    "summarize": {
      "api_key": "sk-ant-..."
    }
  }
```

### API 호출 실패

```go
func (g *ClaudeGenerator) Generate(ctx context.Context, input Input) (*checkpoint.Summary, error) {
    response, err := callClaudeAPI(prompt)
    if err != nil {
        // Fallback: 프롬프트 첫 줄 사용
        if len(input.Transcript) > 0 && input.Transcript[0].Type == EntryTypeUser {
            return &checkpoint.Summary{
                Description: input.Transcript[0].Content,
            }, nil
        }
        return nil, fmt.Errorf("failed to generate summary: %w", err)
    }
    return response, nil
}
```

### JSON 파싱 실패

Claude 응답이 유효한 JSON이 아니면 fallback:

```go
var summary checkpoint.Summary
if err := json.Unmarshal(response.Content, &summary); err != nil {
    // Fallback: 전체 응답을 description으로
    return &checkpoint.Summary{
        Description: string(response.Content),
    }, nil
}
```

---

## 성능 최적화

### 1. Transcript Condensation

전체 transcript 대신 압축된 버전만 전송:

```
Before: 1000 lines → 100,000 tokens
After:  100 lines  → 10,000 tokens

API 비용 90% 절감
```

### 2. 캐싱

동일한 checkpoint는 재생성하지 않음:

```go
func shouldGenerateSummary(checkpointID id.CheckpointID) bool {
    metadata := ReadMetadata(checkpointID)
    return metadata.Summary == nil
}
```

### 3. Batch Processing

여러 세션을 한 번에 요약:

```go
func summarizeBatch(checkpoints []id.CheckpointID) error {
    for _, cpID := range checkpoints {
        summary := Generate(cpID)
        SaveSummary(cpID, summary)
    }
}
```

---

## 실습 예제

### 1. Auto-summarization 활성화

```bash
# Entire enable 시 설정
entire enable --summarize

# API key 입력
Enter your Anthropic API key: sk-ant-...

# 작업 진행
claude "Add user authentication"
git commit -m "Add auth"

# → Summary 자동 생성됨
```

### 2. 기존 커밋에 요약 추가

```bash
# 과거 커밋의 요약 생성
entire explain HEAD~3 --generate

# 출력:
Description:
  Fixed bug in login endpoint where tokens were not properly validated.
  Updated middleware to check token expiration.

Key Changes:
  - Fixed token validation logic
  - Added expiration check
  - Updated error messages
```

### 3. 여러 커밋 요약

```bash
# 최근 5개 커밋 요약
for commit in $(git log -5 --format=%H); do
  entire explain $commit --generate
done
```

---

## 고급 기능

### Custom Generator

다른 LLM 사용:

```go
type CustomGenerator struct {
    APIKey string
    Model  string
}

func (g *CustomGenerator) Generate(ctx context.Context, input Input) (*checkpoint.Summary, error) {
    // OpenAI, Gemini 등 다른 API 호출
    return &checkpoint.Summary{
        Description: "...",
        KeyChanges:  []string{"..."},
    }, nil
}
```

### 요약 템플릿 커스터마이징

```go
type SummaryTemplate struct {
    Header       string
    Instructions string
    Format       string
}

func (t *SummaryTemplate) Build(input Input) string {
    // 커스텀 프롬프트 생성
}
```

---

## 비용 및 할당량

### Claude API 비용

**Haiku 3.5:**

- Input: $0.80 / million tokens
- Output: $4.00 / million tokens

**평균 요약:**

- Input: ~10,000 tokens ($0.008)
- Output: ~200 tokens ($0.0008)
- **총: ~$0.01 per summary**

### 월간 예상 비용

```
10 commits/day × 30 days = 300 summaries/month
300 × $0.01 = $3/month
```

---

## 다음 단계

Auto-Summarization을 이해했습니다! 다음 챕터에서는:

- **Token Usage Tracking** - 사용량 추적 및 분석
- **개발 환경 설정** - mise, Go, 테스트 실행
- **코드 구조** - 패키지 구성, 주요 파일

---

*다음 글에서는 Entire CLI의 Token Usage Tracking을 살펴봅니다.*
