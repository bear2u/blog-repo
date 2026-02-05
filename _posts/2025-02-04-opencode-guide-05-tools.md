---
layout: post
title: "OpenCode 가이드 - 내장 도구"
date: 2025-02-04
categories: [AI]
tags: [opencode, tools, edit, bash, grep, ai-tools]
author: anomalyco
original_url: https://github.com/anomalyco/opencode
---

## 내장 도구 개요

OpenCode는 AI 에이전트가 코드베이스와 상호작용할 수 있도록 다양한 내장 도구를 제공합니다. 각 도구는 특정 작업에 최적화되어 있으며, 권한 시스템으로 제어됩니다.

## 파일 조작 도구

### edit (파일 편집)

가장 핵심적인 도구로, 파일 내용을 수정합니다.

```typescript
// 도구 파라미터
interface EditParams {
  file_path: string      // 수정할 파일 경로
  old_string: string     // 대체할 기존 문자열
  new_string: string     // 새로운 문자열
  replace_all?: boolean  // 모든 일치 항목 대체 (기본: false)
}
```

**사용 예:**

```
"이 함수의 이름을 processData에서 transformData로 변경해줘"

→ edit 도구 호출:
  file_path: "src/utils.ts"
  old_string: "function processData("
  new_string: "function transformData("
```

**특징:**
- 정확한 문자열 매칭 필요
- `old_string`이 유일하지 않으면 실패
- 들여쓰기 보존 필수

### read (파일 읽기)

파일 내용을 읽어옵니다.

```typescript
interface ReadParams {
  file_path: string   // 읽을 파일 경로
  offset?: number     // 시작 줄 번호
  limit?: number      // 읽을 줄 수 (기본: 2000)
  pages?: string      // PDF 페이지 범위 (예: "1-5")
}
```

**지원 형식:**
- 텍스트 파일 (코드, 설정 등)
- 이미지 파일 (PNG, JPG 등) - 시각적으로 표시
- PDF 파일 - 페이지 지정 가능
- Jupyter 노트북 (.ipynb)

### write (파일 작성)

새 파일을 생성하거나 기존 파일을 완전히 덮어씁니다.

```typescript
interface WriteParams {
  file_path: string   // 파일 경로
  content: string     // 파일 내용
}
```

**주의:** 기존 파일을 덮어쓰므로, 부분 수정은 `edit`를 사용해야 합니다.

### multiedit (다중 편집)

한 파일 내에서 여러 위치를 동시에 수정합니다.

```typescript
interface MultiEditParams {
  file_path: string
  edits: Array<{
    old_string: string
    new_string: string
  }>
}
```

## 검색 도구

### glob (파일 패턴 검색)

글로브 패턴으로 파일을 찾습니다.

```typescript
interface GlobParams {
  pattern: string   // 글로브 패턴 (예: "**/*.tsx")
  path?: string     // 검색 시작 디렉토리
}
```

**예:**

```bash
# React 컴포넌트 찾기
**/*.tsx

# 테스트 파일 찾기
**/*.test.ts

# 특정 디렉토리 내 설정 파일
src/**/*.config.js
```

### grep (텍스트 검색)

ripgrep 기반의 강력한 텍스트 검색입니다.

```typescript
interface GrepParams {
  pattern: string           // 정규식 패턴
  path?: string             // 검색 경로
  glob?: string             // 파일 필터 (예: "*.js")
  type?: string             // 파일 타입 (예: "ts", "py")
  output_mode?: "content" | "files_with_matches" | "count"
  context?: number          // 컨텍스트 줄 수
  multiline?: boolean       // 멀티라인 매칭
  "-i"?: boolean            // 대소문자 무시
}
```

**예:**

```
"useState를 사용하는 모든 파일을 찾아줘"

→ grep 도구 호출:
  pattern: "useState"
  type: "tsx"
  output_mode: "files_with_matches"
```

### codesearch (코드 검색)

LSP 기반 코드 심볼 검색입니다.

```typescript
interface CodeSearchParams {
  query: string   // 검색할 심볼 이름
}
```

**검색 대상:**
- 클래스, 인터페이스
- 함수, 메서드
- 변수, 상수
- 타입 정의

### ls (디렉토리 목록)

디렉토리 내용을 나열합니다.

```typescript
interface LsParams {
  path: string       // 디렉토리 경로
  depth?: number     // 탐색 깊이 (기본: 1)
}
```

## 실행 도구

### bash (쉘 명령)

쉘 명령을 실행합니다.

```typescript
interface BashParams {
  command: string           // 실행할 명령
  timeout?: number          // 타임아웃 (ms, 최대 600000)
  run_in_background?: boolean  // 백그라운드 실행
  description?: string      // 명령 설명
}
```

**안전 장치:**
- 기본 120초 타임아웃
- 샌드박스 모드 지원
- 권한 시스템 적용

**예:**

```bash
# 테스트 실행
bun test

# 빌드
npm run build

# Git 상태 확인
git status
```

### task (서브에이전트 작업)

서브에이전트에게 복잡한 작업을 위임합니다.

```typescript
interface TaskParams {
  description: string       // 작업 설명
  prompt: string           // 상세 지시
  subagent_type: string    // 에이전트 타입
  model?: string           // 사용할 모델
  run_in_background?: boolean
}
```

## 웹 도구

### webfetch (웹 페이지 가져오기)

URL의 내용을 가져와 분석합니다.

```typescript
interface WebFetchParams {
  url: string       // 가져올 URL
  prompt: string    // 분석 지시
}
```

**특징:**
- HTML을 마크다운으로 변환
- AI가 내용 요약/분석
- 15분 캐시

### websearch (웹 검색)

웹 검색을 수행합니다.

```typescript
interface WebSearchParams {
  query: string                    // 검색어
  allowed_domains?: string[]       // 허용 도메인
  blocked_domains?: string[]       // 차단 도메인
}
```

## LSP 도구

### lsp (Language Server 연동)

LSP 서버와 상호작용합니다.

```typescript
interface LspParams {
  action: "definition" | "references" | "hover" | "diagnostics"
  file: string
  line?: number
  character?: number
}
```

**지원 기능:**
- 정의로 이동
- 참조 찾기
- 호버 정보
- 진단 (오류/경고)

## 계획 도구

### plan-enter / plan-exit

Plan 모드 진입/종료를 제어합니다.

```typescript
// Plan 모드 진입
{
  tool: "plan-enter"
}

// Plan 모드 종료 (승인 요청)
{
  tool: "plan-exit",
  allowedPrompts: [
    { tool: "Bash", prompt: "run tests" }
  ]
}
```

## 질문 도구

### question (사용자 질문)

사용자에게 질문하여 정보를 수집합니다.

```typescript
interface QuestionParams {
  questions: Array<{
    question: string      // 질문 내용
    header: string        // 헤더 (12자 이내)
    options: Array<{
      label: string       // 옵션 레이블
      description: string // 옵션 설명
    }>
    multiSelect?: boolean // 다중 선택 허용
  }>
}
```

## TODO 도구

### todoread / todowrite

작업 목록을 관리합니다.

```typescript
// TODO 읽기
interface TodoReadParams {
  // 파라미터 없음
}

// TODO 작성
interface TodoWriteParams {
  tasks: Array<{
    subject: string      // 작업 제목
    description: string  // 작업 설명
    status: "pending" | "in_progress" | "completed"
  }>
}
```

## 스킬 도구

### skill (스킬 실행)

등록된 스킬을 실행합니다.

```typescript
interface SkillParams {
  skill: string     // 스킬 이름
  args?: string     // 인자
}
```

## 도구 권한 설정

각 도구의 권한을 설정 파일에서 제어할 수 있습니다:

```json
{
  "permission": {
    "edit": {
      "*": "allow",
      "*.env": "deny",
      "node_modules/**": "deny"
    },
    "bash": "ask",
    "read": {
      "*": "allow",
      ".env*": "ask"
    }
  }
}
```

## 도구 확장

MCP를 통해 추가 도구를 연결할 수 있습니다:

```json
{
  "mcp": {
    "database": {
      "type": "local",
      "command": ["npx", "@modelcontextprotocol/server-postgres"]
    }
  }
}
```

## 다음 단계

다음 챕터에서는 다양한 AI 프로바이더 설정 방법을 알아봅니다.

---

**이전 글**: [에이전트 시스템](/opencode-guide-04-agents/)

**다음 글**: [AI 프로바이더](/opencode-guide-06-providers/)
