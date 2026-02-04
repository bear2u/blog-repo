---
layout: post
title: "OpenCode 가이드 - 에이전트 시스템"
date: 2025-02-04
category: AI
tags: [opencode, agents, ai-agent, build-agent, plan-agent]
series: opencode-guide
part: 4
author: anomalyco
original_url: https://github.com/anomalyco/opencode
---

## 에이전트 시스템 개요

OpenCode의 에이전트 시스템은 다양한 작업 모드를 지원합니다. 각 에이전트는 고유한 권한과 동작 방식을 가지며, 사용자는 Tab 키로 에이전트를 전환할 수 있습니다.

## 기본 내장 에이전트

### build 에이전트

**기본 에이전트**로, 모든 도구에 대한 권한이 설정되어 있습니다.

```typescript
// agent.ts에서 정의
build: {
  name: "build",
  description: "The default agent. Executes tools based on configured permissions.",
  mode: "primary",
  native: true,
  permission: {
    "*": "allow",
    doom_loop: "ask",
    question: "allow",
    plan_enter: "allow"
  }
}
```

**특징:**
- 파일 편집 허용
- Bash 명령 실행 허용
- 파일 읽기/쓰기 허용
- 사용자 질문 기능 허용
- Plan 모드 진입 허용

### plan 에이전트

**읽기 전용 에이전트**로, 코드베이스 분석과 계획 수립에 적합합니다.

```typescript
plan: {
  name: "plan",
  description: "Plan mode. Disallows all edit tools.",
  mode: "primary",
  native: true,
  permission: {
    edit: { "*": "deny" },
    plan_exit: "allow",
    question: "allow"
  }
}
```

**특징:**
- 파일 편집 불가 (기본)
- `.opencode/plans/*.md` 파일만 편집 가능
- Bash 명령 실행 전 확인 요청
- 코드베이스 탐색에 이상적

### general 서브에이전트

복잡한 검색이나 멀티스텝 작업을 위한 **서브에이전트**입니다.

```typescript
general: {
  name: "general",
  description: "General-purpose agent for researching complex questions",
  mode: "subagent",
  native: true,
  permission: {
    todoread: "deny",
    todowrite: "deny"
  }
}
```

**사용 방법:**

```
@general 이 코드베이스의 인증 시스템 구조를 분석해줘
```

### explore 서브에이전트

코드베이스 탐색에 특화된 **빠른 서브에이전트**입니다.

```typescript
explore: {
  name: "explore",
  description: "Fast agent specialized for exploring codebases",
  mode: "subagent",
  native: true,
  permission: {
    "*": "deny",
    grep: "allow",
    glob: "allow",
    read: "allow",
    webfetch: "allow",
    websearch: "allow"
  }
}
```

**특징:**
- 검색 도구만 허용
- 파일 수정 불가
- 빠른 코드베이스 탐색

## 에이전트 전환

### Tab 키 전환

TUI에서 Tab 키를 누르면 사용 가능한 에이전트 목록이 표시됩니다:

```
┌─────────────────────────────────┐
│  Select Agent                   │
├─────────────────────────────────┤
│  > build  (current)             │
│    plan                         │
│    custom-agent                 │
└─────────────────────────────────┘
```

### 멘션으로 서브에이전트 호출

메시지에서 `@` 멘션으로 서브에이전트를 호출할 수 있습니다:

```
@general API 엔드포인트들을 찾아서 문서화해줘

@explore src/ 디렉토리에서 React 컴포넌트 패턴을 분석해줘
```

## 커스텀 에이전트 생성

### 설정 파일 방식

`.opencode/agents/` 디렉토리에 JSON 파일로 정의:

```json
// .opencode/agents/reviewer.json
{
  "name": "reviewer",
  "description": "코드 리뷰 전용 에이전트",
  "mode": "primary",
  "prompt": "당신은 코드 리뷰어입니다. 다음 관점에서 코드를 분석하세요:\n1. 코드 품질\n2. 보안 취약점\n3. 성능 이슈\n4. 가독성",
  "permission": {
    "edit": { "*": "deny" },
    "bash": "ask"
  }
}
```

### opencode.json 방식

프로젝트 루트의 `opencode.json`에서 정의:

```json
{
  "agent": {
    "docs-writer": {
      "name": "docs-writer",
      "description": "문서 작성 전용 에이전트",
      "mode": "primary",
      "prompt": "당신은 기술 문서 작성자입니다.",
      "permission": {
        "edit": {
          "*": "deny",
          "docs/**": "allow",
          "*.md": "allow"
        }
      }
    }
  }
}
```

### 동적 에이전트 생성

AI를 통해 에이전트를 자동 생성할 수도 있습니다:

```typescript
// Agent.generate() 사용
const newAgent = await Agent.generate({
  description: "TypeScript 전문가 에이전트",
  model: { providerID: "anthropic", modelID: "claude-sonnet-4-20250514" }
})

// 결과
{
  identifier: "typescript-expert",
  whenToUse: "TypeScript 관련 질문이나 코드 작성 시",
  systemPrompt: "당신은 TypeScript 전문가입니다..."
}
```

## 에이전트 속성

### 모드 (mode)

| 모드 | 설명 |
|------|------|
| `primary` | 기본 에이전트, Tab으로 전환 가능 |
| `subagent` | 서브에이전트, @멘션으로 호출 |
| `all` | 둘 다 가능 |

### 권한 (permission)

```typescript
interface Permission {
  // 도구별 권한
  edit?: PermissionRule
  bash?: PermissionRule
  read?: PermissionRule

  // 특수 권한
  doom_loop?: "allow" | "ask" | "deny"
  external_directory?: Record<string, PermissionRule>
  question?: "allow" | "deny"
  plan_enter?: "allow" | "deny"
  plan_exit?: "allow" | "deny"
}

type PermissionRule =
  | "allow"
  | "ask"
  | "deny"
  | Record<string, "allow" | "ask" | "deny">
```

### 모델 오버라이드

특정 에이전트에 다른 AI 모델을 지정할 수 있습니다:

```json
{
  "agent": {
    "fast-agent": {
      "name": "fast",
      "model": "anthropic/claude-haiku-3-5-20241022",
      "description": "빠른 응답용 경량 에이전트"
    }
  }
}
```

### 온도 설정

```json
{
  "agent": {
    "creative": {
      "name": "creative",
      "temperature": 0.9,
      "description": "창의적 작업용 에이전트"
    }
  }
}
```

## 숨겨진 시스템 에이전트

OpenCode는 내부적으로 몇 가지 숨겨진 에이전트를 사용합니다:

```typescript
// 대화 요약 에이전트
compaction: {
  name: "compaction",
  hidden: true,
  prompt: PROMPT_COMPACTION
}

// 제목 생성 에이전트
title: {
  name: "title",
  hidden: true,
  temperature: 0.5,
  prompt: PROMPT_TITLE
}

// 요약 에이전트
summary: {
  name: "summary",
  hidden: true,
  prompt: PROMPT_SUMMARY
}
```

## 에이전트 비활성화

```json
{
  "agent": {
    "plan": {
      "disable": true
    }
  }
}
```

## 기본 에이전트 변경

```json
{
  "default_agent": "reviewer"
}
```

## 에이전트 색상 지정

TUI에서 표시되는 에이전트 색상을 지정할 수 있습니다:

```json
{
  "agent": {
    "custom": {
      "name": "custom",
      "color": "#10b981"
    }
  }
}
```

## 최대 스텝 제한

에이전트의 최대 동작 횟수를 제한할 수 있습니다:

```json
{
  "agent": {
    "limited": {
      "name": "limited",
      "steps": 10,
      "description": "최대 10단계만 수행하는 에이전트"
    }
  }
}
```

## 다음 단계

다음 챕터에서는 에이전트가 사용하는 내장 도구들을 자세히 살펴봅니다.

---

**이전 글**: [아키텍처](/opencode-guide-03-architecture/)

**다음 글**: [내장 도구](/opencode-guide-05-tools/)
