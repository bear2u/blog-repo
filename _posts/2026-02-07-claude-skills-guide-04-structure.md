---
layout: post
title: "Claude Skills 완벽 가이드 (04) - 스킬 구조 및 YAML"
date: 2026-02-07
permalink: /claude-skills-guide-04-structure/
author: Anthropic
categories: [AI 에이전트, 개발 도구]
tags: [Claude, Skills, YAML, File Structure, Frontmatter]
original_url: "https://github.com/corca-ai/claude-plugins/tree/main/references/anthropic-skills-guide"
excerpt: "스킬 폴더 구조와 YAML frontmatter 완벽 가이드"
---

## 스킬 파일 구조

```
your-skill-name/
├── SKILL.md              # 필수 - 메인 스킬 파일
├── scripts/              # 선택 - 실행 가능한 코드
│   ├── process_data.py
│   └── validate.sh
├── references/           # 선택 - 참조 문서
│   ├── api-guide.md
│   └── examples/
└── assets/               # 선택 - 템플릿 등
    └── report-template.md
```

---

## 중요 규칙

### SKILL.md 네이밍

✅ **반드시 지켜야 할 것:**
- 파일명은 정확히 `SKILL.md` (대소문자 구분)
- 다른 이름 사용 불가

❌ **잘못된 예시:**
- `SKILL.MD` (확장자 대문자)
- `skill.md` (파일명 소문자)
- `Skill.md` (혼합)
- `SKILLS.md` (복수형)

---

### 스킬 폴더 네이밍

✅ **올바른 형식: kebab-case**
```
notion-project-setup     ✅
linear-sprint-planner    ✅
figma-to-code           ✅
```

❌ **잘못된 형식:**
```
Notion Project Setup    ❌ (공백)
notion_project_setup    ❌ (언더스코어)
NotionProjectSetup      ❌ (대문자)
notionProjectSetup      ❌ (카멜케이스)
```

---

### README.md 금지

**중요:** 스킬 폴더 내부에 `README.md`를 포함하지 마세요.

- ✅ 모든 문서는 `SKILL.md` 또는 `references/`에
- ❌ `README.md`는 스킬 폴더 안에 넣지 말 것

**참고:** GitHub에서 배포할 때는 레포지토리 루트에 README.md를 두는 것이 좋습니다 (사용자용). 이는 스킬 폴더 외부입니다.

---

## YAML Frontmatter

YAML frontmatter는 **Claude가 스킬을 로드할지 결정**하는 핵심입니다.

### 최소 필수 형식

```yaml
---
name: your-skill-name
description: What it does. Use when user asks to [specific phrases].
---
```

이것만 있으면 시작할 수 있습니다.

---

## 필드 상세 설명

### name (필수)

**형식:** kebab-case만 사용

```yaml
# ✅ 올바른 예시
name: notion-project-setup
name: linear-sprint-planner
name: api-doc-generator

# ❌ 잘못된 예시
name: Notion Project Setup  # 공백
name: notion_project_setup  # 언더스코어
name: NotionProjectSetup    # 대문자
```

**규칙:**
- 폴더명과 일치해야 함
- 공백 없음
- 대문자 없음

---

### description (필수)

**두 가지를 반드시 포함:**
1. 스킬이 **무엇을 하는지**
2. **언제 사용하는지** (트리거 조건)

```yaml
# ✅ 좋은 예시 - WHAT + WHEN
description: Analyzes Figma design files and generates developer handoff
  documentation. Use when user uploads .fig files, asks for "design specs",
  "component documentation", or "design-to-code handoff".

# ✅ 좋은 예시 - 구체적 트리거
description: End-to-end Linear sprint planning including task creation, team
  assignment, and milestone setup. Use when user says "plan sprint",
  "create Linear sprint", or "set up next iteration".

# ❌ 나쁜 예시 - WHEN이 없음
description: Creates sophisticated multi-page documentation systems.

# ❌ 나쁜 예시 - 너무 모호함
description: Helps with projects.
```

**제약사항:**
- 최대 1024자
- XML 태그 (`<` `>`) 사용 금지
- 사용자가 말할 수 있는 구체적 문구 포함
- 관련 파일 형식이 있다면 명시

---

### license (선택)

오픈소스로 공개하는 경우 사용

```yaml
license: MIT
# 또는
license: Apache-2.0
```

---

### compatibility (선택)

환경 요구사항을 명시합니다.

```yaml
# 예시 1: 플랫폼 제한
compatibility: Requires Claude Code. Uses Bash tool for git operations.

# 예시 2: 시스템 의존성
compatibility: Requires Python 3.8+, npm, and network access

# 예시 3: MCP 서버 필요
compatibility: Requires Figma MCP server configured with valid API token
```

**길이:** 1-500자

---

### metadata (선택)

커스텀 키-값 쌍을 자유롭게 추가할 수 있습니다.

```yaml
metadata:
  author: Your Company
  version: 1.0.0
  mcp-server: server-name
  category: productivity
  tags: [project-management, automation]
  documentation: https://example.com/docs
  support: support@example.com
```

**권장 필드:**
- `author`: 제작자/회사명
- `version`: 버전 관리
- `mcp-server`: 필요한 MCP 서버 이름

---

## 전체 예시

### 최소 구성

```yaml
---
name: simple-skill
description: Does X. Use when user asks for Y.
---

# Simple Skill

Your instructions here...
```

---

### 완전한 구성

```yaml
---
name: linear-sprint-planner
description: End-to-end Linear sprint planning including task creation, team
  assignment, milestone setup, and notification configuration. Use when user
  says "plan sprint", "create Linear sprint", "set up iteration", or
  "organize Linear tasks".
license: MIT
compatibility: Requires Linear MCP server with admin access
metadata:
  author: Linear Team
  version: 2.1.0
  mcp-server: linear
  category: project-management
  tags: [linear, sprint, agile, project-management]
  documentation: https://linear.app/docs/skills
  support: skills@linear.app
---

# Linear Sprint Planner

## Overview
This skill automates the entire sprint planning workflow...

[나머지 명령어들...]
```

---

## 보안 제약사항

### 금지 사항

❌ **XML 태그 사용 금지**
```yaml
# ❌ 금지
description: Creates <documents> for <users>

# ✅ 사용
description: Creates documents for users
```

**이유:** Frontmatter는 Claude의 시스템 프롬프트에 나타납니다. 악의적 내용이 명령어를 주입할 수 있습니다.

---

❌ **'claude' 또는 'anthropic' 이름 사용 금지**
```yaml
# ❌ 금지
name: claude-helper
name: anthropic-tool

# ✅ 사용
name: project-helper
name: automation-tool
```

**이유:** 예약된 네임스페이스입니다.

---

### 허용 사항

✅ **표준 YAML 타입**
- 문자열 (strings)
- 숫자 (numbers)
- 불리언 (booleans)
- 리스트 (lists)
- 객체 (objects)

✅ **커스텀 metadata 필드**
```yaml
metadata:
  custom-field: value
  another-field: 123
  list-field: [item1, item2]
```

✅ **긴 설명 (최대 1024자)**
```yaml
description: >
  This skill does many things including
  task A, task B, and task C. It uses
  multiple lines for clarity. Use when...
```

---

## allowed-tools 필드 (선택)

스킬이 사용할 수 있는 도구를 제한합니다.

```yaml
# Python과 npm만 허용
allowed-tools: "Bash(python:*) Bash(npm:*) WebFetch"

# 모든 Bash 명령 허용 (기본값)
allowed-tools: "Bash(*)"

# 특정 명령만 허용
allowed-tools: "Bash(git:*) Bash(docker:*)"
```

**사용 시나리오:**
- 보안이 중요한 환경
- 특정 도구만 사용하는 스킬
- 샌드박스 환경 구성

---

## 체크리스트

### SKILL.md 생성 전

- [ ] 폴더명이 kebab-case인가?
- [ ] SKILL.md 파일명이 정확한가? (대소문자 확인)
- [ ] README.md를 스킬 폴더에 넣지 않았는가?

### YAML Frontmatter 작성 시

- [ ] `---` 구분자가 있는가?
- [ ] `name` 필드가 kebab-case인가?
- [ ] `description`에 WHAT과 WHEN이 모두 있는가?
- [ ] XML 태그 (`<` `>`)를 사용하지 않았는가?
- [ ] 구체적인 트리거 문구를 포함했는가?

### 선택 필드 (필요시)

- [ ] `license` 추가 (오픈소스인 경우)
- [ ] `compatibility` 추가 (환경 의존성이 있는 경우)
- [ ] `metadata` 추가 (버전, 작성자 등)
- [ ] `allowed-tools` 추가 (도구 제한이 필요한 경우)

---

## 다음 단계

YAML frontmatter 작성이 끝났다면:

1. SKILL.md 본문에 명령어 작성
2. 예시와 트러블슈팅 추가
3. 필요시 references/ 디렉토리 생성
4. 테스트 케이스 정의

---

*다음 글에서는 효과적인 명령어 작성법을 다룹니다.*
