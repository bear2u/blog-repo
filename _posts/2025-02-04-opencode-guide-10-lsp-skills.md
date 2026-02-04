---
layout: post
title: "OpenCode 가이드 - LSP & 스킬 시스템"
date: 2025-02-04
category: AI
tags: [opencode, lsp, language-server, skills, automation]
series: opencode-guide
part: 10
author: anomalyco
original_url: https://github.com/anomalyco/opencode
---

## LSP (Language Server Protocol) 통합

OpenCode의 가장 큰 차별점 중 하나는 **기본 내장 LSP 지원**입니다. LSP를 통해 AI가 코드를 더 정확하게 이해하고 분석할 수 있습니다.

## LSP 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    OpenCode                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌───────────────────────────────────────────────┐    │
│   │              LSP Manager                       │    │
│   ├───────────────────────────────────────────────┤    │
│   │  ┌────────────┐  ┌────────────┐               │    │
│   │  │ TypeScript │  │   Python   │  ...          │    │
│   │  │   Server   │  │   Server   │               │    │
│   │  └─────┬──────┘  └─────┬──────┘               │    │
│   │        │               │                       │    │
│   │        └───────┬───────┘                       │    │
│   │                ▼                               │    │
│   │         ┌────────────┐                        │    │
│   │         │ LSP Client │                        │    │
│   │         └────────────┘                        │    │
│   └───────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 기본 제공 LSP 서버

OpenCode는 다음 언어의 LSP를 자동으로 감지하고 연결합니다:

| 언어 | LSP 서버 | 확장자 |
|------|----------|--------|
| TypeScript/JavaScript | typescript-language-server | `.ts`, `.tsx`, `.js`, `.jsx` |
| Python | pyright | `.py` |
| Go | gopls | `.go` |
| Rust | rust-analyzer | `.rs` |
| 실험적: Python | ty | `.py` |

## LSP 기능

### 정의로 이동

```typescript
// LSP.definition() 사용
const definitions = await LSP.definition({
  file: "src/utils.ts",
  line: 10,
  character: 15
})
```

### 참조 찾기

```typescript
// LSP.references() 사용
const references = await LSP.references({
  file: "src/utils.ts",
  line: 10,
  character: 15
})
```

### 호버 정보

```typescript
// LSP.hover() 사용
const hoverInfo = await LSP.hover({
  file: "src/utils.ts",
  line: 10,
  character: 15
})
```

### 심볼 검색

```typescript
// 워크스페이스 심볼 검색
const symbols = await LSP.workspaceSymbol("MyClass")

// 문서 심볼
const docSymbols = await LSP.documentSymbol("file:///src/utils.ts")
```

### 진단 정보

```typescript
// 파일 진단 (오류, 경고)
const diagnostics = await LSP.diagnostics()

// 결과
{
  "src/utils.ts": [
    {
      severity: 1,  // ERROR
      range: { start: { line: 10, character: 5 }, ... },
      message: "Property 'foo' does not exist"
    }
  ]
}
```

### 호출 계층

```typescript
// 들어오는 호출
const incoming = await LSP.incomingCalls({
  file: "src/utils.ts",
  line: 10,
  character: 15
})

// 나가는 호출
const outgoing = await LSP.outgoingCalls({
  file: "src/utils.ts",
  line: 10,
  character: 15
})
```

## LSP 설정

### LSP 전체 비활성화

```json
{
  "lsp": false
}
```

### 특정 LSP 비활성화

```json
{
  "lsp": {
    "pyright": {
      "disabled": true
    }
  }
}
```

### 커스텀 LSP 추가

```json
{
  "lsp": {
    "my-custom-lsp": {
      "command": ["my-lsp-server", "--stdio"],
      "extensions": [".custom", ".mycustom"],
      "env": {
        "MY_LSP_CONFIG": "value"
      },
      "initialization": {
        "customOption": true
      }
    }
  }
}
```

### 실험적 LSP (ty)

Python용 실험적 LSP 서버 ty 활성화:

```bash
export OPENCODE_EXPERIMENTAL_LSP_TY=1
```

## 스킬 시스템

스킬은 **재사용 가능한 프롬프트/지시사항**입니다. Claude Code의 SKILL.md와 호환됩니다.

## 스킬 구조

### SKILL.md 형식

```markdown
---
name: deploy
description: 프로덕션 배포 자동화
---

# 배포 스킬

배포를 수행할 때 다음 단계를 따르세요:

## 전제 조건 확인
1. 모든 테스트 통과 확인
2. 빌드 성공 확인
3. 환경 변수 설정 확인

## 배포 절차
1. `npm run build` 실행
2. `npm run deploy` 실행
3. 배포 성공 여부 확인

## 롤백
문제 발생 시 `npm run rollback` 실행
```

### 프론트매터 필수 필드

| 필드 | 설명 | 필수 |
|------|------|------|
| `name` | 스킬 이름 | ✓ |
| `description` | 스킬 설명 | ✓ |

## 스킬 위치

OpenCode는 다음 위치에서 스킬을 검색합니다:

### 1. 외부 스킬 디렉토리

Claude Code와 호환되는 경로:

```
~/.claude/skills/*/SKILL.md
~/.agents/skills/*/SKILL.md
.claude/skills/*/SKILL.md
.agents/skills/*/SKILL.md
```

### 2. OpenCode 스킬 디렉토리

```
~/.config/opencode/skill/*/SKILL.md
~/.config/opencode/skills/*/SKILL.md
.opencode/skill/*/SKILL.md
.opencode/skills/*/SKILL.md
```

### 3. 추가 스킬 경로

설정 파일에서 지정:

```json
{
  "skills": {
    "paths": [
      "./custom-skills",
      "~/shared-skills"
    ]
  }
}
```

## 스킬 사용

### TUI에서 호출

```
/deploy

또는

이 프로젝트를 deploy 스킬을 사용해서 배포해줘
```

### 에이전트에서 자동 사용

에이전트는 관련 스킬을 자동으로 참조할 수 있습니다.

## 스킬 예제

### 테스팅 스킬

```markdown
# .opencode/skills/testing/SKILL.md
---
name: testing
description: 테스트 작성 및 실행 가이드
---

# 테스트 스킬

## 테스트 작성 규칙
- Jest 사용
- 파일명: `*.test.ts`
- describe/it 구조 사용

## 테스트 실행
```bash
npm test
```

## 커버리지 확인
```bash
npm run test:coverage
```
```

### 코드 리뷰 스킬

```markdown
# .opencode/skills/review/SKILL.md
---
name: review
description: 코드 리뷰 체크리스트
---

# 코드 리뷰 스킬

다음 관점에서 코드를 검토하세요:

## 보안
- SQL 인젝션 취약점
- XSS 취약점
- 인증/인가 문제

## 성능
- N+1 쿼리
- 불필요한 계산
- 메모리 누수

## 가독성
- 네이밍 컨벤션
- 함수 길이
- 주석 품질
```

### Git 스킬

```markdown
# .opencode/skills/git/SKILL.md
---
name: git
description: Git 작업 가이드
---

# Git 스킬

## 커밋 메시지 규칙
```
<type>(<scope>): <description>

Types:
- feat: 새 기능
- fix: 버그 수정
- docs: 문서 변경
- refactor: 리팩토링
```

## 브랜치 전략
- main: 프로덕션
- develop: 개발
- feature/*: 기능 브랜치
```

## 외부 스킬 비활성화

```bash
export OPENCODE_DISABLE_EXTERNAL_SKILLS=1
```

## 스킬과 명령

### 커스텀 명령

`.opencode/commands/` 디렉토리에 명령을 정의할 수 있습니다:

```json
// .opencode/commands/build.json
{
  "name": "build",
  "description": "프로젝트 빌드",
  "prompt": "프로젝트를 빌드하고 결과를 보고해주세요"
}
```

### 명령 실행

```
/build
```

## 플러그인

스킬보다 더 복잡한 확장은 플러그인으로 구현합니다:

```javascript
// .opencode/plugins/custom.js
export default {
  name: "custom-plugin",
  version: "1.0.0",

  async onInit() {
    console.log("Plugin initialized")
  },

  async onMessage(message) {
    // 메시지 처리
  },

  async onTool(call) {
    // 도구 호출 처리
  }
}
```

```json
// opencode.json
{
  "plugin": [".opencode/plugins/custom.js"]
}
```

## 스킬 vs 에이전트 vs 플러그인

| 기능 | 스킬 | 에이전트 | 플러그인 |
|------|------|----------|----------|
| 형식 | Markdown | JSON | JavaScript |
| 목적 | 지시사항/가이드 | 권한/동작 모드 | 기능 확장 |
| 복잡도 | 낮음 | 중간 | 높음 |
| 코드 실행 | ✗ | ✗ | ✓ |

## 마무리

이것으로 OpenCode 가이드의 모든 챕터를 완료했습니다. OpenCode는 강력하고 유연한 오픈소스 AI 코딩 에이전트로, 다양한 AI 프로바이더와 도구를 지원합니다.

### 추가 리소스

- [GitHub 저장소](https://github.com/anomalyco/opencode)
- [공식 문서](https://opencode.ai/docs)
- [Discord 커뮤니티](https://discord.gg/opencode)
- [OpenCode Zen](https://opencode.ai/zen)

---

**이전 글**: [TUI & 데스크톱 앱](/opencode-guide-09-tui-desktop/)

**시리즈 완료!** [목차로 돌아가기](/opencode-guide/)
