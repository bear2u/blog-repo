---
layout: post
title: "Claude Skills 완벽 가이드 (07) - 배포 및 공유"
date: 2026-02-07
permalink: /claude-skills-guide-07-distribution/
author: Anthropic
categories: [AI 에이전트, 개발 도구]
tags: [Claude, Skills, Distribution, API, GitHub, Deployment]
original_url: "https://github.com/corca-ai/claude-plugins/tree/main/references/anthropic-skills-guide"
excerpt: "스킬을 사용자에게 전달하는 방법 - API, GitHub, 조직 배포"
---

## 스킬이 MCP 통합을 더 완벽하게 만드는 이유

커넥터를 비교할 때, 사용자는 **스킬이 있는 MCP**를 선택합니다. 왜냐하면:

- ✅ 더 빠른 가치 실현
- ✅ 낮은 학습 곡선
- ✅ 일관된 사용 경험
- ✅ 베스트 프랙티스 내장

**스킬 = MCP 연결의 경쟁 우위**

---

## 현재 배포 모델 (2026년 1월 기준)

### 개인 사용자

**스킬을 받는 방법:**

1. 스킬 폴더 다운로드
2. 필요시 ZIP 압축
3. Claude.ai에 업로드
   - Settings > Capabilities > Skills
4. 또는 Claude Code 스킬 디렉토리에 배치

---

### 조직 레벨 배포

**2025년 12월 18일 출시**

**기능:**
- ✅ 관리자가 조직 전체에 스킬 배포
- ✅ 자동 업데이트
- ✅ 중앙 집중식 관리
- ✅ 버전 관리

**이점:**
- 팀 전체가 동일한 워크플로우 사용
- IT 관리자가 승인된 스킬만 배포
- 일관된 베스트 프랙티스 보장

---

## 오픈 스탠다드

**Agent Skills는 오픈 스탠다드로 공개되었습니다.**

MCP와 마찬가지로, Anthropic는 스킬이 도구와 플랫폼 간에 이식 가능해야 한다고 믿습니다.

**목표:**
- 동일한 스킬이 Claude 또는 다른 AI 플랫폼에서 작동
- 플랫폼 간 호환성
- 개방형 생태계

**참고:**
일부 스킬은 특정 플랫폼의 기능을 최대한 활용하도록 설계되었습니다. 이 경우 `compatibility` 필드에 명시하세요.

```yaml
compatibility: Optimized for Claude Code. Uses Bash tool for git operations.
```

---

## API를 통한 스킬 사용

**프로그래매틱 사용 사례:**
- 애플리케이션 빌드
- 에이전트 시스템 구축
- 자동화된 워크플로우
- 프로덕션 배포

---

### 주요 기능

```
/v1/skills 엔드포인트
├── 스킬 목록 조회
├── 스킬 관리
└── 버전 제어

Messages API
├── container.skills 파라미터로 스킬 추가
└── 프로그래매틱 스킬 실행

Claude Console
├── 버전 관리
└── 스킬 모니터링

Claude Agent SDK
└── 커스텀 에이전트 빌드
```

---

### 사용 사례별 추천 플랫폼

| 사용 사례 | 추천 플랫폼 |
|----------|-----------|
| 최종 사용자가 직접 스킬 사용 | Claude.ai / Claude Code |
| 개발 중 수동 테스트 및 반복 | Claude.ai / Claude Code |
| 개인적, 임시 워크플로우 | Claude.ai / Claude Code |
| 프로그래매틱 스킬 사용 | API |
| 대규모 프로덕션 배포 | API |
| 자동화된 파이프라인 및 에이전트 | API |

---

### API 사용 요구사항

**필수:** Code Execution Tool 베타

스킬이 실행되려면 안전한 환경이 필요하며, Code Execution Tool 베타가 이를 제공합니다.

**관련 문서:**
- Skills API Quickstart
- Create Custom Skills
- Skills in the Agent SDK

---

## 권장 배포 방법 (현재)

### 1. GitHub에 호스팅

**레포지토리 구조:**
```
your-skill-repo/
├── README.md                 # 사람을 위한 문서 (레포 루트)
├── CHANGELOG.md              # 버전 히스토리
├── LICENSE                   # 라이선스 파일
├── your-skill/               # 실제 스킬 폴더
│   ├── SKILL.md
│   ├── scripts/
│   ├── references/
│   └── assets/
└── examples/                 # 사용 예시 스크린샷
    ├── screenshot1.png
    └── screenshot2.png
```

**README.md 예시:**
```markdown
# ProjectHub Skill for Claude

End-to-end project setup automation for ProjectHub.

## Features

- ✅ Automated workspace creation
- ✅ Template application
- ✅ Team member assignment
- ✅ Milestone setup

## Installation

### For Claude.ai Users

1. Download the skill:
   ```bash
   git clone https://github.com/yourcompany/projecthub-skill
   cd projecthub-skill
   ```

2. Create ZIP file:
   ```bash
   zip -r projecthub-skill.zip projecthub-skill/
   ```

3. Upload to Claude:
   - Open [Claude.ai](https://claude.ai)
   - Go to Settings > Capabilities > Skills
   - Click "Upload skill"
   - Select `projecthub-skill.zip`

4. Enable the skill:
   - Toggle on "ProjectHub Setup"
   - Ensure ProjectHub MCP server is connected

### For Claude Code Users

1. Clone to skills directory:
   ```bash
   cd ~/.claude/skills
   git clone https://github.com/yourcompany/projecthub-skill
   ```

2. Restart Claude Code

## Usage

### Quick Start

Ask Claude:
```
"Set up a new project workspace in ProjectHub for Q4 planning"
```

Claude will:
1. Create workspace structure
2. Apply project template
3. Set up milestones
4. Assign team members
5. Configure notifications

### Examples

[Include screenshots here]

## Requirements

- ProjectHub MCP server configured
- Valid ProjectHub account with admin access
- Claude.ai Pro or Claude Code

## Support

- Issues: [GitHub Issues](https://github.com/yourcompany/projecthub-skill/issues)
- Docs: [ProjectHub Docs](https://docs.projecthub.com/claude-skill)
- Email: support@projecthub.com
```

---

### 2. MCP 문서에 링크

**MCP README.md에 스킬 섹션 추가:**

```markdown
# ProjectHub MCP Server

## What is this?

The ProjectHub MCP server gives Claude access to your ProjectHub data.

## Why use it with the ProjectHub Skill?

**MCP alone:**
- ✅ Claude can read/write ProjectHub data
- ❌ You explain the workflow each time
- ❌ Inconsistent results

**MCP + Skill:**
- ✅ Claude can read/write ProjectHub data
- ✅ Built-in workflow automation
- ✅ Consistent, reliable results
- ✅ Best practices included

## Installation

### 1. Install MCP Server

[MCP installation instructions...]

### 2. Install ProjectHub Skill (Recommended)

Get the skill: [projecthub-skill](https://github.com/yourcompany/projecthub-skill)

With the skill, you can say:
- "Set up a new project workspace"
- "Create Q4 planning project"

Without the skill, you'd need to:
- Manually explain each step
- Specify exact tool calls
- Handle errors yourself
```

---

### 3. 설치 가이드 작성

**빠른 설치 플로우:**

```markdown
## 5분 안에 시작하기

### Step 1: MCP Server (2분)
```bash
npx @projecthub/mcp-server
```
Enter your API key when prompted.

### Step 2: Skill (2분)
```bash
# Download
curl -L https://github.com/yourcompany/projecthub-skill/releases/latest/download/projecthub-skill.zip -o skill.zip

# Upload to Claude.ai
# Settings > Skills > Upload > Select skill.zip
```

### Step 3: Test (1분)
Ask Claude:
```
"Create a new project in ProjectHub"
```

Done! 🎉
```

---

## 스킬 포지셔닝

**사용자가 가치를 이해하도록 설명하세요.**

---

### 결과에 초점, 기능 아님

✅ **좋은 예시:**
> "The ProjectHub skill enables teams to set up complete project workspaces in seconds — including pages, databases, and templates — instead of spending 30 minutes on manual setup."

❌ **나쁜 예시:**
> "The ProjectHub skill is a folder containing YAML frontmatter and Markdown instructions that calls our MCP server tools."

---

### MCP + Skills 스토리 강조

```markdown
## Why ProjectHub MCP + Skills?

### MCP Server (The Connection)
Our MCP server gives Claude access to your ProjectHub data.
- Read projects
- Create pages
- Update databases

### ProjectHub Skill (The Knowledge)
Our skill teaches Claude your team's workflow.
- How to structure projects
- When to use templates
- Best practices for team collaboration

### Together = AI-Powered Project Management
- Ask once, get complete setup
- Consistent quality every time
- 10x faster than manual setup
```

---

### Before/After 비교

```markdown
## Before (MCP Only)

User: "Create a project"
Claude: "What name? Which template? Which team members? ..."
[15 messages later...]
Result: ✓ Project created (but inconsistent structure)

## After (MCP + Skill)

User: "Create a Q4 planning project"
Claude:
- Analyzes project type
- Applies Q4 planning template
- Assigns default team
- Sets quarterly milestones
- Configures notifications

Result: ✓ Complete project in 30 seconds
```

---

## 릴리스 체크리스트

### GitHub 준비
- [ ] 레포지토리 공개
- [ ] README.md 작성 (설치, 사용법, 예시)
- [ ] LICENSE 파일 추가
- [ ] CHANGELOG.md 생성
- [ ] 스크린샷 추가
- [ ] GitHub Release 생성

### 문서
- [ ] MCP 문서에 스킬 섹션 추가
- [ ] 설치 가이드 작성
- [ ] 사용 예시 포함
- [ ] 트러블슈팅 섹션 추가

### 테스트
- [ ] 신규 사용자가 설치 가능한지 확인
- [ ] 모든 플랫폼에서 테스트 (Claude.ai, Claude Code)
- [ ] MCP 서버와 함께 작동 확인
- [ ] 예시 쿼리 모두 작동 확인

### 마케팅
- [ ] 발표 블로그 포스트 작성
- [ ] 소셜 미디어 공유
- [ ] 커뮤니티에 공지
- [ ] 파트너에게 알림

---

## API 배포 (고급)

**프로덕션 환경:**

```typescript
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

// Create message with skill
const message = await anthropic.messages.create({
  model: 'claude-3-5-sonnet-20241022',
  max_tokens: 1024,
  messages: [{
    role: 'user',
    content: 'Create a new project in ProjectHub'
  }],
  container: {
    skills: ['projecthub-setup']  // Skill ID
  }
});
```

**버전 관리:**
```bash
# Claude Console에서 스킬 버전 관리
claude-cli skill:version projecthub-setup 1.0.0 --stable
claude-cli skill:version projecthub-setup 1.1.0 --beta
```

---

## 다음 단계

스킬이 배포되었다면:

1. 사용자 피드백 수집
2. Under/over-triggering 모니터링
3. 사용 패턴 분석
4. 지속적 개선

---

*다음 글에서는 실전에서 검증된 스킬 패턴들을 살펴봅니다.*
