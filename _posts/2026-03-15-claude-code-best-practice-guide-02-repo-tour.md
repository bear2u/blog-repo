---
layout: post
title: "claude-code-best-practice 완벽 가이드 (02) - 레포 투어(.claude 중심)"
date: 2026-03-15
permalink: /claude-code-best-practice-guide-02-repo-tour/
author: shanraisshan
categories: [AI 코딩 에이전트, claude-code-best-practice]
tags: [Trending, GitHub, claude-code-best-practice, RepoTour, ClaudeCode, DotClaude]
original_url: "https://github.com/shanraisshan/claude-code-best-practice"
excerpt: "`.claude/`(commands/agents/skills/hooks/rules/settings)와 best-practice/implementation/reports를 축으로, 이 레포를 ‘실전 템플릿’처럼 읽는 방법을 파일 경로 근거로 정리합니다."
---

## 이 문서의 목적

- 이 레포를 “지식베이스 + 구현 템플릿”으로 활용하기 위해, **어디에 무엇이 있는지**를 `.claude/` 중심으로 안내합니다. (`README.md`, `.claude/*`)
- 이후 챕터에서 다룰 핵심 축(Commands/Agents/Skills/MCP/Settings/Memory/Hooks)을 파일 경로로 고정합니다.

---

## 빠른 요약(레포 구조 기반)

README가 표로 정리한 Claude Code 개념(Commands/Subagents/Skills/Hooks/MCP/Settings/Memory 등)은 실제로 아래 위치에 분포합니다. (`README.md`)

- `.claude/commands/` : 커스텀 슬래시 커맨드 프롬프트 템플릿 (예: `.claude/commands/weather-orchestrator.md`)
- `.claude/agents/` : 서브에이전트 정의 (예: `.claude/agents/weather-agent.md`)
- `.claude/skills/` : 스킬 폴더 (예: `.claude/skills/weather-fetcher/SKILL.md`)
- `.claude/hooks/` : 훅 설명/스크립트 (예: `.claude/hooks/HOOKS-README.md`, `.claude/hooks/scripts/*`)
- `.claude/rules/` : 룰/스타일 가이드 (예: `.claude/rules/markdown-docs.md`)
- `.claude/settings.json` : 프로젝트 설정/권한/훅/출력 스타일 등 (예: permissions, hooks)
- `.mcp.json` : MCP 서버 연결 예시(Playwright/Tavily/Context7/DeepWiki)
- `best-practice/` : 베스트 프랙티스 문서 묶음
- `implementation/` : 위 프랙티스를 “이 레포에서 어떻게 구현했는지” 예시
- `reports/` : 관찰/리서치 리포트

근거: 디렉토리/파일 존재 및 `README.md`

---

## `.claude/`만 봐도 “무엇을 하려는지” 감이 온다

예를 들어, `.claude/settings.json`은 다음 성격의 설정을 포함합니다. (`.claude/settings.json`)

- permissions allow/ask/deny 규칙
- 프로젝트 MCP 서버 활성화(`enableAllProjectMcpServers`)
- hooks(PreToolUse/PostToolUse/SessionStart 등)에서 `python3 .../.claude/hooks/scripts/hooks.py` 호출
- outputStyle/statusLine/spinner 팁 등 UI/출력 관련 설정

또한 `.claude/commands/weather-orchestrator.md`는 “Command → Agent(Task) → Skill” 형태의 오케스트레이션 워크플로우 문서를 포함합니다. (`.claude/commands/weather-orchestrator.md`)

---

## 근거(파일/경로)

- 개념 표/링크: `README.md`
- `.claude/` 리소스:
  - `.claude/settings.json`
  - `.claude/commands/weather-orchestrator.md`, `.claude/commands/time-command.md`
  - `.claude/agents/weather-agent.md`, `.claude/agents/time-agent.md`
  - `.claude/skills/*/SKILL.md`
  - `.claude/hooks/HOOKS-README.md`, `.claude/hooks/scripts/*`
  - `.claude/rules/*`
- MCP 예시: `.mcp.json`

---

## 주의사항/함정

- `.claude/settings.json`은 강력한 자동화(훅/권한 등)를 포함할 수 있으므로, 내 프로젝트에 복사할 때 “허용/요청(ask) 규칙”을 반드시 검토해야 합니다. (`.claude/settings.json`)

---

## TODO/확인 필요

- `best-practice/*` ↔ `implementation/*` 간 “대응 관계(링크/번호)”를 표로 만들기
- `.claude/hooks/scripts/hooks.py`가 실제로 어떤 이벤트에서 무엇을 하는지 정리 (`.claude/settings.json`, `.claude/hooks/scripts/*`)

---

## 위키 링크

- `[[claude-code-best-practice Guide - Index]]` → [가이드 목차](/blog-repo/claude-code-best-practice-guide/)
- `[[claude-code-best-practice Guide - Commands/Agents/Skills]]` → [03. Commands/Agents/Skills](/blog-repo/claude-code-best-practice-guide-03-commands-agents-skills/)

---

*다음 글에서는 `.claude/commands/weather-orchestrator.md`와 `.claude/agents/weather-agent.md`를 근거로, “커맨드→에이전트→스킬” 오케스트레이션 패턴을 해부합니다.*

