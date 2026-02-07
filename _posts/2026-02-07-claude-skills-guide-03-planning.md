---
layout: post
title: "Claude Skills 완벽 가이드 (03) - 계획 및 설계"
date: 2026-02-07
permalink: /claude-skills-guide-03-planning/
author: Anthropic
categories: [AI 에이전트, 개발 도구]
tags: [Claude, Skills, Planning, Design, Use Cases]
original_url: "https://github.com/corca-ai/claude-plugins/tree/main/references/anthropic-skills-guide"
excerpt: "스킬을 만들기 전 무엇을 고려해야 하는가? 유스케이스 정의부터 카테고리 선택까지"
---

## 스킬 만들기 전 질문들

### 1. 구체적인 유스케이스가 있는가?

❌ **나쁜 출발점:**
- "프로젝트 관리에 도움이 되는 스킬"
- "생산성 향상을 위한 스킬"

✅ **좋은 출발점:**
- "Linear에서 프로젝트 만들 때마다 매번 동일한 5단계를 거침 → 자동화하고 싶음"
- "Figma 파일을 받으면 항상 개발 가이드 문서 생성 → 스킬로 만들면 좋겠음"

**시작하기 전 2-3개의 구체적 예시를 적어보세요.**

---

### 2. 어떤 도구가 필요한가?

스킬은 다음 도구를 사용할 수 있습니다:

#### A) 내장 도구
- ✅ 파일 읽기/쓰기
- ✅ 코드 실행 (Python, Bash 등)
- ✅ 웹 검색
- ✅ 이미지 생성

#### B) MCP 서버
- ✅ 외부 서비스 연결 (Notion, GitHub, Slack 등)
- ✅ 데이터베이스 액세스
- ✅ 커스텀 API

**체크리스트:**
```
필요한 도구:
[ ] 내장 도구만으로 충분
[ ] MCP 서버 필요
[ ] MCP 서버가 이미 존재함
[ ] MCP 서버를 새로 만들어야 함
```

---

## 스킬 카테고리

### 1. Document Creation Skills (문서 생성)

**언제 사용:**
- 반복적인 문서 생성 작업
- 일관된 포맷 필요
- 여러 소스의 데이터 결합

**예시:**
```yaml
---
name: api-documentation-generator
description: Generates API documentation from OpenAPI specs. Use when user
  provides .yaml/.json OpenAPI file or asks for "API docs", "endpoint
  documentation", or "API reference".
---

# API Documentation Generator

## Instructions

### Step 1: Parse OpenAPI Spec
Read the provided OpenAPI specification file.
Validate schema version (2.0, 3.0, 3.1).

### Step 2: Extract Endpoints
For each endpoint:
- HTTP method and path
- Parameters (query, path, body)
- Response schemas
- Authentication requirements

### Step 3: Generate Markdown
Create structured documentation:
- Overview section
- Authentication guide
- Endpoint reference (one section per endpoint)
- Schema definitions
- Examples for each endpoint

### Step 4: Add Code Examples
Generate request examples in:
- cURL
- Python (requests library)
- JavaScript (fetch API)

## Output Format

Save as `API_REFERENCE.md` with:
- Table of contents
- Syntax highlighted code blocks
- Response status code tables
```

---

### 2. Workflow Automation Skills (워크플로우 자동화)

**언제 사용:**
- 여러 단계를 순서대로 실행
- 다수의 도구/서비스 조율
- 매번 동일한 절차 반복

**예시:**
```yaml
---
name: sprint-planning
description: End-to-end sprint planning workflow for Linear. Creates sprint,
  generates tasks from specifications, assigns team members, and sets up
  milestones. Use when user says "plan sprint", "create Linear sprint",
  or "set up next iteration".
---

# Sprint Planning Workflow

## Prerequisites

- Linear MCP server configured
- Team structure defined in Linear
- Previous sprint completed (optional)

## Workflow Steps

### Step 1: Review Previous Sprint
```bash
# Fetch last sprint data
mcp-tool call linear list_sprints --limit 1 --status completed
```

Analyze:
- Completion rate
- Unfinished tasks
- Team velocity

### Step 2: Create New Sprint
Ask user for:
- Sprint name (default: "Sprint [N+1]")
- Duration (default: 2 weeks)
- Start date (default: next Monday)

```bash
mcp-tool call linear create_sprint \
  --name "Sprint 23" \
  --start-date "2025-01-20" \
  --end-date "2025-02-03"
```

### Step 3: Generate Tasks
From specifications provided:
- Break down into sub-tasks
- Estimate story points
- Assign priority (P0-P3)

```bash
for task in tasks:
  mcp-tool call linear create_issue \
    --title "${task.title}" \
    --description "${task.description}" \
    --estimate ${task.points} \
    --priority ${task.priority}
```

### Step 4: Assign Team Members
Based on:
- Workload balance
- Skill matching
- Previous assignments

### Step 5: Set Up Milestones
Create checkpoints at:
- Week 1: Design review
- Week 2: Implementation complete
- End of sprint: Ready for QA

## Success Criteria

✓ Sprint created in Linear
✓ All tasks assigned
✓ Total story points within team capacity
✓ Milestones configured
✓ Team notified via Slack
```

---

### 3. MCP Enhancement Skills (MCP 강화)

**언제 사용:**
- MCP 서버는 있지만 사용법이 복잡
- 특정 워크플로우에 최적화 필요
- 여러 MCP 서버를 조율

**예시:**
```yaml
---
name: figma-to-code
description: Converts Figma designs to React components using Figma MCP.
  Analyzes design structure, generates component code, and creates storybook
  stories. Use when user shares Figma link or asks to "convert design to code",
  "generate components from Figma", or "create React from design".
---

# Figma to Code Converter

## Requirements

- Figma MCP server configured
- Figma file URL with view access
- Node.js and npm installed

## Instructions

### Step 1: Fetch Figma Design
```bash
mcp-tool call figma get_file --file-key FILE_KEY
```

Extract:
- Component structure
- Layout properties
- Text styles
- Color tokens
- Spacing values

### Step 2: Analyze Component Hierarchy
Identify:
- Reusable components
- Component variants
- Nested structures
- Auto-layout patterns

### Step 3: Generate React Code

For each component:
```typescript
// Example output structure
interface ButtonProps {
  variant: 'primary' | 'secondary' | 'tertiary';
  size: 'small' | 'medium' | 'large';
  children: React.ReactNode;
  onClick?: () => void;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'medium',
  children,
  onClick
}) => {
  return (
    <button
      className={`btn btn-${variant} btn-${size}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
};
```

### Step 4: Extract Design Tokens
Create `tokens.css`:
```css
:root {
  --color-primary: #1a73e8;
  --color-secondary: #34a853;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --radius-sm: 4px;
}
```

### Step 5: Generate Storybook Stories
```typescript
import { Button } from './Button';

export default {
  title: 'Components/Button',
  component: Button,
};

export const Primary = {
  args: {
    variant: 'primary',
    children: 'Click me',
  },
};
```

## Output Structure

```
components/
├── Button/
│   ├── Button.tsx
│   ├── Button.css
│   └── Button.stories.tsx
├── Card/
│   ├── Card.tsx
│   ├── Card.css
│   └── Card.stories.tsx
└── tokens.css
```

## Troubleshooting

**Error:** "Component too complex"
→ Break into smaller sub-components

**Error:** "Missing Figma access"
→ Verify file permissions in Figma
```

---

## 설계 체크리스트

### 명확성
- [ ] 유스케이스가 구체적인가?
- [ ] 성공 기준이 정의되었는가?
- [ ] 예상 입력/출력이 명확한가?

### 범위
- [ ] 하나의 명확한 목적이 있는가?
- [ ] 너무 광범위하지 않은가?
- [ ] 다른 스킬과 겹치지 않는가?

### 실행 가능성
- [ ] 필요한 도구가 모두 사용 가능한가?
- [ ] MCP 서버가 필요하면 존재하는가?
- [ ] 환경 의존성이 명확히 문서화되었는가?

### 품질
- [ ] 에러 처리가 포함되었는가?
- [ ] 예시가 충분한가?
- [ ] 문서가 이해하기 쉬운가?

---

## 다음 단계

계획이 완료되었다면 이제 스킬을 구조화할 차례입니다:

1. SKILL.md 파일 구조 설계
2. YAML frontmatter 작성
3. 명령어 작성
4. 테스트 케이스 정의

---

*다음 글에서는 스킬의 파일 구조와 YAML frontmatter 작성법을 다룹니다.*
