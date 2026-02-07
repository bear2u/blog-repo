---
layout: post
title: "Claude Skills 완벽 가이드 (02) - 기초 개념"
date: 2026-02-07
permalink: /claude-skills-guide-02-fundamentals/
author: Anthropic
categories: [AI 에이전트, 개발 도구]
tags: [Claude, Skills, Progressive Disclosure, MCP, Composability]
original_url: "https://github.com/corca-ai/claude-plugins/tree/main/references/anthropic-skills-guide"
excerpt: "스킬이란 무엇이며 어떻게 작동하는가? 핵심 설계 원칙 이해하기"
---

## 스킬이란?

**스킬(Skill)**은 Claude가 특정 작업이나 워크플로우를 처리하는 방법을 가르치는 명령어 세트입니다. 간단한 폴더 형태로 패키징됩니다.

### 스킬 폴더 구성

```
my-skill/
├── SKILL.md          # 필수: 명령어 및 YAML frontmatter
├── scripts/          # 선택: Python, Bash 등 실행 가능한 코드
├── references/       # 선택: 필요시 로드되는 문서
└── assets/           # 선택: 템플릿, 폰트, 아이콘 등
```

---

## 핵심 설계 원칙

### 1. Progressive Disclosure (점진적 공개)

스킬은 **3단계 시스템**을 사용하여 토큰 사용량을 최소화하면서 전문 지식을 유지합니다.

```
┌─────────────────────────────────────────────────┐
│ Level 1: YAML Frontmatter                       │
│ ✓ 항상 시스템 프롬프트에 로드됨                      │
│ ✓ 스킬이 언제 사용되어야 하는지만 알려줌              │
│ ✓ 최소한의 정보만 포함                             │
└─────────────────────────────────────────────────┘
              ↓ (관련성 있을 때만)
┌─────────────────────────────────────────────────┐
│ Level 2: SKILL.md 본문                          │
│ ✓ Claude가 스킬이 관련 있다고 판단할 때 로드        │
│ ✓ 전체 명령어와 가이드 포함                         │
│ ✓ 워크플로우 단계 및 베스트 프랙티스                │
└─────────────────────────────────────────────────┘
              ↓ (필요시만)
┌─────────────────────────────────────────────────┐
│ Level 3: 링크된 파일들                           │
│ ✓ Claude가 필요에 따라 탐색하고 발견              │
│ ✓ references/ 디렉토리의 추가 문서                │
│ ✓ 상세한 API 가이드, 예제 등                      │
└─────────────────────────────────────────────────┘
```

**왜 중요한가?**
- 불필요한 토큰 사용 방지
- 컨텍스트 윈도우 효율적 활용
- 필요한 정보만 적시에 로드

---

### 2. Composability (구성 가능성)

Claude는 **여러 스킬을 동시에 로드**할 수 있습니다.

**설계 시 고려사항:**
- ✅ 다른 스킬과 잘 작동해야 함
- ✅ 유일한 기능이라고 가정하지 말 것
- ✅ 명확한 경계와 책임 정의
- ✅ 충돌하지 않는 트리거 조건

**예시:**
```yaml
# ✅ 좋은 예: 명확한 범위
description: Manages Linear sprint planning. Use when user mentions
  "sprint", "Linear tasks", or "project planning".

# ❌ 나쁜 예: 너무 광범위
description: Helps with all project management tasks.
```

---

### 3. Portability (이식성)

**동일한 스킬이 모든 플랫폼에서 작동:**
- ✅ Claude.ai
- ✅ Claude Code
- ✅ Claude API

**한 번 만들면 어디서나 사용 가능**

단, 환경 의존성은 `compatibility` 필드에 명시:

```yaml
compatibility: Requires Python 3.8+, npm, and network access
```

---

## MCP 개발자를 위한 Skills

### MCP를 이미 만들었다면?

축하합니다! **어려운 부분은 끝났습니다.**

스킬은 MCP 위의 **지식 레이어**입니다.

### 주방 비유

```
┌──────────────────────────────────────┐
│         MCP (전문 주방)               │
│                                      │
│  • 도구 (Tools)                      │
│  • 재료 (Data)                       │
│  • 장비 (Equipment)                  │
│                                      │
│  = 액세스와 기능 제공                  │
└──────────────────────────────────────┘
              ↓ 사용 방법
┌──────────────────────────────────────┐
│      Skills (레시피)                  │
│                                      │
│  • 단계별 지침                         │
│  • 베스트 프랙티스                      │
│  • 워크플로우                          │
│                                      │
│  = 가치 창출 방법 가르침                │
└──────────────────────────────────────┘
```

---

## MCP vs Skills 비교

| 측면 | MCP (연결성) | Skills (지식) |
|-----|-------------|--------------|
| **역할** | Claude를 서비스에 연결 | 서비스 사용 방법 가르침 |
| **제공** | 실시간 데이터 액세스 | 워크플로우와 베스트 프랙티스 |
| **초점** | **할 수 있는** 것 | **어떻게 해야 하는** 것 |
| **예시** | Notion 데이터베이스 읽기 | Notion으로 프로젝트 설정하기 |

---

## Skills가 MCP 사용자에게 중요한 이유

### 스킬 없이 (MCP만)

**문제점:**
- ❌ MCP는 연결했지만 다음에 뭘 해야 할지 모름
- ❌ "통합으로 X를 어떻게 하나요?" 지원 티켓 폭증
- ❌ 매 대화마다 처음부터 설명
- ❌ 프롬프트마다 다른 일관성 없는 결과
- ❌ 실제 문제는 워크플로우 가이드 부족인데 커넥터 탓으로 오인

### 스킬 사용 (MCP + Skills)

**해결:**
- ✅ 필요시 사전 구축된 워크플로우 자동 활성화
- ✅ 일관되고 신뢰할 수 있는 도구 사용
- ✅ 모든 상호작용에 베스트 프랙티스 임베딩
- ✅ 낮은 학습 곡선
- ✅ 사용자 만족도 향상

---

## 실전 예제

### MCP만 있을 때

**사용자:** "Linear에서 새 프로젝트 만들어줘"

**Claude (MCP만):**
```
Linear MCP 도구를 사용할 수 있습니다.
어떤 프로젝트 이름을 사용하시겠습니까?
어떤 팀에 할당하시겠습니까?
마일스톤은 어떻게 설정하시겠습니까?
...
```

➡️ **15번의 왕복 대화, 일관성 없는 결과**

### MCP + Skills

**사용자:** "Linear에서 새 프로젝트 만들어줘"

**Claude (MCP + Skills):**
```
Linear 프로젝트 설정 워크플로우를 시작합니다.

✓ 현재 팀 구조 분석 완료
✓ 프로젝트 템플릿 적용
✓ 마일스톤 자동 생성
✓ 팀 멤버 할당
✓ 알림 설정 완료

프로젝트 "Q4 Planning"이 생성되었습니다.
링크: linear.app/team/PROJECT-123
```

➡️ **2번의 확인만으로 완료, 일관된 품질**

---

## 함께 사용하는 방법

### 1. MCP 먼저 만들기

```typescript
// MCP Server 예시
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [{
    name: "create_project",
    description: "Creates a new project in Linear",
    inputSchema: { /* ... */ }
  }]
}));
```

### 2. Skills로 워크플로우 추가

```yaml
---
name: linear-project-setup
description: End-to-end Linear project setup including team assignment,
  milestones, and notifications. Use when user says "create project",
  "set up Linear project", or "new sprint".
---

# Linear Project Setup

## Workflow

### Step 1: Analyze Current State
Call MCP tool: `list_teams`
Call MCP tool: `list_projects`

### Step 2: Create Project
Call MCP tool: `create_project`
Parameters from user input + defaults

### Step 3: Setup Milestones
Call MCP tool: `create_milestone`
Apply standard sprint cadence

### Step 4: Assign Team
Call MCP tool: `add_team_members`
Based on project type

### Step 5: Configure Notifications
Call MCP tool: `setup_notifications`
Standard notification rules
```

---

## 다음 단계

이제 스킬의 기초 개념을 이해했습니다. 다음 챕터에서는:

- 구체적인 유스케이스 정의하기
- 스킬 카테고리 선택하기
- 성공 기준 설정하기

---

*다음 글에서는 스킬 계획 및 설계 방법을 살펴봅니다.*
