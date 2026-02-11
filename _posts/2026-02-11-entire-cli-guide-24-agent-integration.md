---
layout: post
title: "Entire CLI 완벽 가이드 (24) - Agent 통합"
date: 2026-02-11
permalink: /entire-cli-guide-24-agent-integration/
author: Entire Team
categories: [AI 코딩, 개발 도구, Agent]
tags: [Entire, Agent, Claude Code, Gemini CLI, Integration]
original_url: "https://github.com/entireio/cli"
excerpt: "Entire CLI의 Agent 통합 - Gemini CLI 통합과 새로운 Agent 추가 완벽 가이드"
---

## 개요

Entire CLI는 **Agent 추상화 계층**을 통해 여러 AI 코딩 도구와 통합됩니다. 현재 **Claude Code**와 **Gemini CLI**를 지원하며, 쉽게 새로운 Agent를 추가할 수 있습니다.

```
┌─────────────────────────────────────────────────────┐
│              Agent Abstraction Layer                 │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Entire CLI                                          │
│       │                                              │
│       ▼                                              │
│  ┌─────────────┐                                     │
│  │   Agent     │  (Interface)                        │
│  │  Interface  │                                     │
│  └─────┬───────┘                                     │
│        │                                             │
│        ├─────────────┬─────────────┬────────────     │
│        ▼             ▼             ▼                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │  Claude  │  │  Gemini  │  │  Custom  │           │
│  │   Code   │  │   CLI    │  │  Agent   │           │
│  └──────────┘  └──────────┘  └──────────┘           │
│                                                      │
│  - Hooks        - Hooks       - Hooks               │
│  - Sessions     - Sessions    - Sessions            │
│  - Transcript   - Transcript  - Transcript          │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Agent 인터페이스

### 정의

```go
package agent

type Agent interface {
    // 기본 정보
    Name() string

    // 훅 설치
    SetupHooks(repoPath string) error

    // Session 관리
    GetSessionDir(repoPath string) (string, error)
    GetTranscriptPath(sessionID string) (string, error)
    ReadSession(sessionRef string) (*AgentSession, error)
    WriteSession(session *AgentSession) error

    // ID 변환
    GenerateSessionID(agentSessionID string) string
    ExtractAgentSessionID(sessionID string) string

    // 명령어
    FormatResumeCommand(agentSessionID string) string
}
```

### AgentSession 구조

```go
type AgentSession struct {
    SessionID  string   // Agent의 고유 세션 ID
    AgentName  string   // Agent 이름
    RepoPath   string   // Repository 경로
    SessionRef string   // Session log 파일 경로
    NativeData []byte   // 원본 transcript 데이터
}
```

---

## Claude Code Agent

### 구현

```
cmd/entire/cli/agent/claudecode/
├── agent.go        # Agent 인터페이스 구현
├── transcript.go   # Transcript 파싱
└── types.go        # 데이터 타입
```

### 주요 특징

**1. Session ID 형식:**

```
Entire: 2026-02-11-abc123de-f456-7890-abcd-ef1234567890
Claude: abc123de-f456-7890-abcd-ef1234567890 (UUID만)
```

**2. Session 디렉토리:**

```
.claude/sessions/<uuid>/
├── full.jsonl      # 전체 transcript
├── prompt.txt      # 사용자 프롬프트
└── context.md      # 생성된 컨텍스트
```

**3. Hooks:**

```
.claude/hooks/
├── SessionStart.sh
├── UserPromptSubmit.sh
├── Stop.sh
└── SessionStop.sh
```

**Hook 스크립트 예시:**

```bash
#!/bin/bash
# .claude/hooks/SessionStart.sh

SESSION_ID="$1"

# Entire CLI 호출
entire hook session-start \
    --session-id "$SESSION_ID" \
    --agent claude-code
```

### 구현 예시

```go
package claudecode

type ClaudeCodeAgent struct{}

func (a *ClaudeCodeAgent) Name() string {
    return "claude-code"
}

func (a *ClaudeCodeAgent) SetupHooks(repoPath string) error {
    hooksDir := filepath.Join(repoPath, ".claude", "hooks")
    os.MkdirAll(hooksDir, 0o750)

    hooks := map[string]string{
        "SessionStart":      sessionStartHook,
        "UserPromptSubmit":  userPromptSubmitHook,
        "Stop":              stopHook,
        "SessionStop":       sessionStopHook,
    }

    for name, content := range hooks {
        path := filepath.Join(hooksDir, name+".sh")
        os.WriteFile(path, []byte(content), 0o755)
    }

    return nil
}

func (a *ClaudeCodeAgent) GetSessionDir(repoPath string) (string, error) {
    return filepath.Join(repoPath, ".claude", "sessions"), nil
}

func (a *ClaudeCodeAgent) GenerateSessionID(agentSessionID string) string {
    // 현재 날짜 + UUID
    date := time.Now().Format("2006-01-02")
    return fmt.Sprintf("%s-%s", date, agentSessionID)
}

func (a *ClaudeCodeAgent) ExtractAgentSessionID(sessionID string) string {
    // "2026-02-11-uuid" → "uuid"
    parts := strings.SplitN(sessionID, "-", 4)
    if len(parts) == 4 {
        return parts[3]
    }
    return sessionID
}

func (a *ClaudeCodeAgent) FormatResumeCommand(agentSessionID string) string {
    return fmt.Sprintf("claude resume %s", agentSessionID)
}
```

---

## Gemini CLI Agent

### 구현

```
cmd/entire/cli/agent/geminicli/
├── agent.go        # Agent 인터페이스 구현
├── transcript.go   # Transcript 파싱
└── types.go        # 데이터 타입
```

### 주요 특징

**1. Session ID 형식:**

```
Entire: 2026-02-11-abc123de-f456-7890-abcd-ef1234567890
Gemini: 2026-02-11-abc123de-f456-7890-abcd-ef1234567890 (동일)
```

**2. Session 디렉토리:**

```
.gemini/sessions/
└── 2026-02-11-abc123de-f456-7890-abcd-ef1234567890.jsonl
```

**3. Hooks:**

```
.gemini/hooks/
├── session_start.sh
├── user_prompt_submit.sh
├── stop.sh
└── session_stop.sh
```

### Transcript 형식 차이

**Claude Code:**

```jsonl
{"type":"user","content":"Add login"}
{"type":"assistant","content":"I'll add login...","usage":{...}}
{"type":"tool_use","tool":"Edit","input":{...}}
{"type":"tool_result","tool":"Edit","output":"..."}
```

**Gemini CLI:**

```jsonl
{"role":"user","parts":[{"text":"Add login"}]}
{"role":"model","parts":[{"text":"I'll add login..."}],"usage_metadata":{...}}
{"role":"function","parts":[{"function_call":{"name":"edit","args":{...}}}]}
{"role":"function","parts":[{"function_response":{"name":"edit","response":{...}}}]}
```

### Transcript 파싱

```go
package geminicli

type GeminiMessage struct {
    Role          string             `json:"role"`
    Parts         []Part             `json:"parts"`
    UsageMetadata *GeminiUsageMetadata `json:"usage_metadata,omitempty"`
}

type Part struct {
    Text             string            `json:"text,omitempty"`
    FunctionCall     *FunctionCall     `json:"function_call,omitempty"`
    FunctionResponse *FunctionResponse `json:"function_response,omitempty"`
}

func ParseTranscript(data []byte) ([]transcript.Message, error) {
    // Gemini 형식 → Entire 표준 형식으로 변환
    messages := []transcript.Message{}

    scanner := bufio.NewScanner(bytes.NewReader(data))
    for scanner.Scan() {
        var msg GeminiMessage
        json.Unmarshal(scanner.Bytes(), &msg)

        messages = append(messages, convertToStandardMessage(msg))
    }

    return messages, nil
}
```

---

## 새 Agent 추가하기

### 1. 디렉토리 생성

```bash
mkdir -p cmd/entire/cli/agent/myagent
cd cmd/entire/cli/agent/myagent
```

### 2. agent.go 구현

```go
package myagent

import (
    "github.com/entireio/cli/cmd/entire/cli/agent"
)

type MyAgent struct{}

func (a *MyAgent) Name() string {
    return "myagent"
}

func (a *MyAgent) SetupHooks(repoPath string) error {
    // Hook 스크립트 설치
    hooksDir := filepath.Join(repoPath, ".myagent", "hooks")
    os.MkdirAll(hooksDir, 0o750)

    // Hook 파일 작성...

    return nil
}

func (a *MyAgent) GetSessionDir(repoPath string) (string, error) {
    return filepath.Join(repoPath, ".myagent", "sessions"), nil
}

func (a *MyAgent) GetTranscriptPath(sessionID string) (string, error) {
    sessionDir, _ := a.GetSessionDir(repoPath)
    return filepath.Join(sessionDir, sessionID+".jsonl"), nil
}

func (a *MyAgent) ReadSession(sessionRef string) (*agent.AgentSession, error) {
    data, err := os.ReadFile(sessionRef)
    if err != nil {
        return nil, err
    }

    sessionID := filepath.Base(sessionRef)
    sessionID = strings.TrimSuffix(sessionID, ".jsonl")

    return &agent.AgentSession{
        SessionID:  sessionID,
        AgentName:  a.Name(),
        SessionRef: sessionRef,
        NativeData: data,
    }, nil
}

func (a *MyAgent) WriteSession(session *agent.AgentSession) error {
    return os.WriteFile(session.SessionRef, session.NativeData, 0o644)
}

func (a *MyAgent) GenerateSessionID(agentSessionID string) string {
    date := time.Now().Format("2006-01-02")
    return fmt.Sprintf("%s-%s", date, agentSessionID)
}

func (a *MyAgent) ExtractAgentSessionID(sessionID string) string {
    // Session ID에서 Agent 고유 ID 추출
    parts := strings.SplitN(sessionID, "-", 4)
    if len(parts) == 4 {
        return parts[3]
    }
    return sessionID
}

func (a *MyAgent) FormatResumeCommand(agentSessionID string) string {
    return fmt.Sprintf("myagent resume %s", agentSessionID)
}
```

### 3. transcript.go 구현

```go
package myagent

import (
    "github.com/entireio/cli/cmd/entire/cli/transcript"
)

type MyAgentMessage struct {
    // Agent의 transcript 형식
}

func ParseTranscript(data []byte) ([]transcript.Message, error) {
    // Agent transcript → 표준 형식 변환
    messages := []transcript.Message{}

    // 파싱 로직...

    return messages, nil
}
```

### 4. Registry에 등록

```go
// cmd/entire/cli/agent/registry.go

import (
    "github.com/entireio/cli/cmd/entire/cli/agent/myagent"
)

func init() {
    RegisterAgent(&myagent.MyAgent{})
}
```

### 5. Enable 시 사용

```bash
entire enable --agent myagent
```

---

## Hook 통합

### Hook 스크립트 생성

각 Agent는 자신의 hook 디렉토리에 스크립트를 설치합니다.

**예시 (MyAgent):**

```bash
# .myagent/hooks/session_start.sh

#!/bin/bash
set -euo pipefail

SESSION_ID="$1"
REPO_ROOT="$2"

# Entire CLI 호출
entire hook session-start \
    --session-id "$SESSION_ID" \
    --agent myagent \
    --repo-path "$REPO_ROOT"
```

### Hook Handler 구현

```go
// cmd/entire/cli/hooks_myagent_handlers.go

func handleMyAgentSessionStart(ctx context.Context, args []string) error {
    sessionID := args[0]
    repoPath := args[1]

    // Session 시작 로직...

    return nil
}
```

### Hook 등록

```go
// cmd/entire/cli/hooks.go

func RegisterHookHandlers() {
    // Claude Code hooks
    hooks["claude-code/SessionStart"] = handleClaudeCodeSessionStart

    // Gemini CLI hooks
    hooks["gemini/session_start"] = handleGeminiSessionStart

    // MyAgent hooks
    hooks["myagent/session_start"] = handleMyAgentSessionStart
}
```

---

## Transcript 표준화

### 표준 Message 형식

```go
package transcript

type Message struct {
    Type      string      `json:"type"`       // user, assistant, tool_use, tool_result
    Content   string      `json:"content"`
    ToolName  string      `json:"tool_name,omitempty"`
    ToolInput string      `json:"tool_input,omitempty"`
    Usage     *TokenUsage `json:"usage,omitempty"`
}

type TokenUsage struct {
    InputTokens          int `json:"input_tokens"`
    CacheCreationTokens  int `json:"cache_creation_tokens"`
    CacheReadTokens      int `json:"cache_read_tokens"`
    OutputTokens         int `json:"output_tokens"`
}
```

### 변환 예시

**Gemini → 표준:**

```go
func convertGeminiToStandard(msg GeminiMessage) transcript.Message {
    switch msg.Role {
    case "user":
        return transcript.Message{
            Type:    "user",
            Content: extractText(msg.Parts),
        }

    case "model":
        return transcript.Message{
            Type:    "assistant",
            Content: extractText(msg.Parts),
            Usage:   convertUsage(msg.UsageMetadata),
        }

    case "function":
        if hasFunctionCall(msg.Parts) {
            return transcript.Message{
                Type:      "tool_use",
                ToolName:  extractFunctionName(msg.Parts),
                ToolInput: extractFunctionArgs(msg.Parts),
            }
        }
        return transcript.Message{
            Type:    "tool_result",
            Content: extractFunctionResponse(msg.Parts),
        }
    }
}
```

---

## 테스트

### Agent 테스트

```go
package myagent_test

func TestMyAgent_SetupHooks(t *testing.T) {
    t.Parallel()

    tmpDir := t.TempDir()
    agent := &myagent.MyAgent{}

    err := agent.SetupHooks(tmpDir)
    assert.NoError(t, err)

    // Hook 파일 확인
    hooksDir := filepath.Join(tmpDir, ".myagent", "hooks")
    assert.DirExists(t, hooksDir)

    hookFile := filepath.Join(hooksDir, "session_start.sh")
    assert.FileExists(t, hookFile)
}

func TestMyAgent_SessionID(t *testing.T) {
    t.Parallel()

    agent := &myagent.MyAgent{}

    // Generate
    agentID := "abc123de-f456-7890-abcd-ef1234567890"
    fullID := agent.GenerateSessionID(agentID)
    assert.Contains(t, fullID, agentID)
    assert.Contains(t, fullID, time.Now().Format("2006-01-02"))

    // Extract
    extracted := agent.ExtractAgentSessionID(fullID)
    assert.Equal(t, agentID, extracted)
}
```

### Transcript 파싱 테스트

```go
func TestParseTranscript(t *testing.T) {
    t.Parallel()

    input := `{"role":"user","parts":[{"text":"Hello"}]}
{"role":"model","parts":[{"text":"Hi"}],"usage_metadata":{...}}`

    messages, err := myagent.ParseTranscript([]byte(input))
    assert.NoError(t, err)
    assert.Len(t, messages, 2)

    assert.Equal(t, "user", messages[0].Type)
    assert.Equal(t, "Hello", messages[0].Content)

    assert.Equal(t, "assistant", messages[1].Type)
    assert.Equal(t, "Hi", messages[1].Content)
}
```

---

## 실습: Cursor Agent 추가

### 1. Cursor의 Session 구조 파악

```bash
# Cursor의 세션 디렉토리 확인
ls ~/.cursor/sessions/

# Transcript 형식 확인
cat ~/.cursor/sessions/<session-id>.json
```

### 2. Agent 구현

```go
package cursor

type CursorAgent struct{}

func (a *CursorAgent) Name() string {
    return "cursor"
}

func (a *CursorAgent) GetSessionDir(repoPath string) (string, error) {
    home, _ := os.UserHomeDir()
    return filepath.Join(home, ".cursor", "sessions"), nil
}

// 나머지 메서드 구현...
```

### 3. Hooks 설정

```bash
# Cursor는 hook을 지원하지 않을 수 있음
# → Polling 방식으로 구현
```

### 4. 등록 및 테스트

```bash
entire enable --agent cursor

# 테스트
cursor "Add authentication"
git commit -m "Add auth"
entire explain HEAD
```

---

## 다음 단계

Agent 통합을 이해했습니다! 마지막 챕터에서는:

- **Contributing** - 기여 가이드, 테스트, PR 프로세스

---

*다음 글에서는 Entire CLI 프로젝트에 기여하는 방법을 살펴봅니다.*
