---
layout: post
title: "claude-code-best-practice 완벽 가이드 (04) - MCP/Settings/Memory 운영"
date: 2026-03-15
permalink: /claude-code-best-practice-guide-04-mcp-settings-memory/
author: shanraisshan
categories: [AI 코딩 에이전트, claude-code-best-practice]
tags: [Trending, GitHub, claude-code-best-practice, MCP, Settings, Memory, Hooks]
original_url: "https://github.com/shanraisshan/claude-code-best-practice"
excerpt: "`.mcp.json`과 `.claude/settings.json`을 근거로 MCP 서버 연결/권한 정책/훅 구성의 핵심을 정리하고, `CLAUDE.md` 및 `.claude/rules/*` 중심의 메모리/규칙 운영 포인트를 연결합니다."
---

## 이 문서의 목적

- “프로젝트에 붙는 설정”을 세 갈래로 나눠 정리합니다: MCP 연결, 프로젝트 설정(권한/훅), 메모리/규칙. (`.mcp.json`, `.claude/settings.json`, `CLAUDE.md`, `.claude/rules/*`)
- 내 프로젝트로 가져갈 때 “위험한 부분(권한/훅)”을 먼저 체크하도록 가이드합니다. (`.claude/settings.json`)

---

## 빠른 요약(근거)

- MCP 서버 연결 예시: `.mcp.json` (playwright/tavily/context7/deepwiki)
- 프로젝트 설정: `.claude/settings.json`
  - permissions allow/ask
  - hooks(PreToolUse/PostToolUse 등)
  - enableAllProjectMcpServers 등
- 규칙/메모리:
  - `CLAUDE.md`
  - `.claude/rules/*` (예: `markdown-docs.md`)

---

## 1) MCP: `.mcp.json`

레포에는 다음 MCP 서버 예시가 포함되어 있습니다. (`.mcp.json`)

- `playwright`: `npx -y @playwright/mcp`
- `tavily-web-search`: HTTP 타입 URL 템플릿(`tavilyApiKey=${TAVILY_API_KEY}`)
- `context7`: `npx -y @upstash/context7-mcp`
- `deepwiki`: `npx -y deepwiki-mcp`

---

## 2) Settings: `.claude/settings.json`

`.claude/settings.json`은 크게 다음 종류의 설정을 포함합니다. (근거: 파일 내용)

- permissions: allow/ask/deny 규칙
- hooks: PreToolUse/PostToolUse/SessionStart 등 이벤트 훅에서 `hooks.py` 실행
- plansDirectory/outputStyle/statusLine 등 출력/UX 설정
- `enableAllProjectMcpServers`, `disableAllHooks` 같은 토글

근거: `.claude/settings.json`

---

## 3) Memory/Rules: `CLAUDE.md`, `.claude/rules/*`

README의 Concepts 표는 Memory를 `CLAUDE.md`, `.claude/rules/`, `~/.claude/...` 등 경로로 설명합니다. (`README.md`)

이 레포에도:
- `CLAUDE.md`
- `.claude/rules/markdown-docs.md`, `.claude/rules/presentation.md`

가 존재합니다. (근거: 파일 존재)

---

## 근거(파일/경로)

- MCP 연결 예시: `.mcp.json`
- 프로젝트 설정: `.claude/settings.json`
- 메모리/규칙: `CLAUDE.md`, `.claude/rules/*`
- Concepts 표: `README.md`

---

## 주의사항/함정

- `.claude/settings.json`의 permissions allow/ask는 환경에 따라 과도하게 넓을 수 있습니다. 내 프로젝트에 복사할 때는 allow/ask 목록을 최소화하는 것이 안전합니다. (`.claude/settings.json`)
- 훅은 외부 스크립트를 실행하므로(예: `python3 ...hooks.py`), 성능/보안/재현성 영향을 반드시 평가해야 합니다. (`.claude/settings.json`)

---

## TODO/확인 필요

- `.claude/hooks/scripts/hooks.py`의 실제 동작(로그/사운드/메모리 기록 등)을 읽고 이벤트별 효과 정리 (`.claude/settings.json`, `.claude/hooks/scripts/*`)
- `CLAUDE.md`가 어떤 규칙/메모리를 담고 있는지 “프로젝트 적용 템플릿”으로 추출

---

## 위키 링크

- `[[claude-code-best-practice Guide - Index]]` → [가이드 목차](/blog-repo/claude-code-best-practice-guide/)
- `[[claude-code-best-practice Guide - Adoption]]` → [05. 내 프로젝트 적용 플레이북](/blog-repo/claude-code-best-practice-guide-05-adoption-playbook/)

---

*다음 글에서는 이 레포의 패턴을 내 프로젝트로 옮기는 “최소 복사 세트”와 점검 자동화(설정 JSON 검증 등)를 플레이북 형태로 정리합니다.*

