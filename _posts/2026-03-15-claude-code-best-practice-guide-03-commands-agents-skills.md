---
layout: post
title: "claude-code-best-practice 완벽 가이드 (03) - Commands/Agents/Skills 설계"
date: 2026-03-15
permalink: /claude-code-best-practice-guide-03-commands-agents-skills/
author: shanraisshan
categories: [AI 코딩 에이전트, claude-code-best-practice]
tags: [Trending, GitHub, claude-code-best-practice, Commands, Agents, Skills]
original_url: "https://github.com/shanraisshan/claude-code-best-practice"
excerpt: "weather 예시를 근거로, `.claude/commands/*`가 ‘오케스트레이션 템플릿’, `.claude/agents/*`가 ‘격리된 실행자’, `.claude/skills/*`가 ‘재사용 가능한 지식/절차’ 역할을 하는 패턴을 정리합니다."
---

## 이 문서의 목적

- “Command vs Agent vs Skill”을 추상적으로 설명하는 대신, 레포에 포함된 weather 예시를 근거로 역할 분담을 고정합니다. (`.claude/commands/weather-orchestrator.md`, `.claude/agents/weather-agent.md`, `.claude/skills/weather-fetcher/SKILL.md`)
- 내 프로젝트에 복사할 때 “무엇을 어디에 넣어야 하는지” 체크리스트를 만듭니다.

---

## 빠른 요약(근거)

- Command: `.claude/commands/weather-orchestrator.md`
  - 단계(ask user → agent 호출 → skill 호출)와 “Critical Requirements”를 문서로 강제
- Agent: `.claude/agents/weather-agent.md`
  - allowedTools/model/maxTurns/memory/skills/hooks 등 실행자(서브에이전트) 정의
- Skill: `.claude/skills/weather-fetcher/SKILL.md`, `.claude/skills/weather-svg-creator/SKILL.md`
  - 재사용 가능한 절차/지식 문서(스킬 폴더)

---

## 1) Command = ‘워크플로우 템플릿’

`.claude/commands/weather-orchestrator.md`는 “Workflow / Step 1~3 / Critical Requirements / Output Summary”처럼, 실행 흐름을 문서로 고정합니다. (`.claude/commands/weather-orchestrator.md`)

핵심 포인트(근거):
- Step 1에서 사용자 선호(섭씨/화씨)를 먼저 묻는다
- Step 2에서 weather-agent를 Task tool로 호출한다(“DO NOT use bash…” 요구 포함)
- Step 3에서 SVG 생성 스킬을 Skill tool로 호출한다

---

## 2) Agent = ‘격리된 실행자’

`.claude/agents/weather-agent.md`는 agent 메타데이터와 실행 요구사항을 포함합니다. (`.claude/agents/weather-agent.md`)

관찰 가능한 구성(근거):
- `allowedTools` 목록
- `model`, `maxTurns`, `permissionMode`, `memory`
- `skills:`에 `weather-fetcher` 지정
- hooks(PreToolUse/PostToolUse 등)에서 `hooks.py` 실행

---

## 3) Skill = ‘재사용 가능한 절차/지식’

스킬은 `.claude/skills/<name>/SKILL.md` 형태로 존재합니다. (근거: 파일 존재)

예:
- `.claude/skills/weather-fetcher/SKILL.md`
- `.claude/skills/time-skill/SKILL.md`
- `.claude/skills/agent-browser/SKILL.md`

---

## 4) 설계 체크리스트(이 레포 패턴 기반)

내 프로젝트에 적용 시 최소 체크:

- Command: “단계/요구사항/산출물”이 문서로 고정되어 있는가? (`.claude/commands/*`)
- Agent: 도구 권한/모델/메모리/훅/스킬이 명시되어 있는가? (`.claude/agents/*`)
- Skill: 한 가지 일을 끝까지 수행할 수 있을 만큼 self-contained인가? (`.claude/skills/*/SKILL.md`)

---

## 근거(파일/경로)

- 오케스트레이션 커맨드 예시: `.claude/commands/weather-orchestrator.md`
- 에이전트 예시: `.claude/agents/weather-agent.md`
- 스킬 존재 근거: `.claude/skills/*/SKILL.md`

---

## 주의사항/함정

- Command 문서가 강제하는 도구 호출 방식(Task/Skill tool 등)은 “특정 런타임/도구 체계”를 전제로 합니다. 내 환경의 실행 모델과 맞지 않으면 그대로 복사하지 말고 “의도(단계/검증)”만 가져오는 것이 안전합니다. (`.claude/commands/weather-orchestrator.md`)

---

## TODO/확인 필요

- `.claude/skills/weather-fetcher/SKILL.md`가 요구하는 외부 API/키(예: Open-Meteo 등)를 실제로 어떤 방식으로 호출하는지 확인
- `.claude/skills/agent-browser/SKILL.md`가 기대하는 브라우저 자동화 도구와 런타임 요구사항 문서화

---

## 위키 링크

- `[[claude-code-best-practice Guide - Index]]` → [가이드 목차](/blog-repo/claude-code-best-practice-guide/)
- `[[claude-code-best-practice Guide - MCP/Settings/Memory]]` → [04. MCP/Settings/Memory](/blog-repo/claude-code-best-practice-guide-04-mcp-settings-memory/)

---

*다음 글에서는 `.mcp.json`, `.claude/settings.json`, `CLAUDE.md`, `.claude/rules/*`를 근거로 “설정/권한/메모리” 운영 패턴을 정리합니다.*

