---
layout: post
title: "claude-code-best-practice 완벽 가이드 (05) - 내 프로젝트 적용 플레이북"
date: 2026-03-15
permalink: /claude-code-best-practice-guide-05-adoption-playbook/
author: shanraisshan
categories: [AI 코딩 에이전트, claude-code-best-practice]
tags: [Trending, GitHub, claude-code-best-practice, Playbook, BestPractices, Automation]
original_url: "https://github.com/shanraisshan/claude-code-best-practice"
excerpt: "`.claude/` 리소스(Commands/Agents/Skills/Rules/Settings)와 `.mcp.json`을 근거로, 내 프로젝트에 안전하게 적용하는 최소 복사 세트/단계/점검 자동화를 체크리스트로 제공합니다."
---

## 이 문서의 목적

- 이 레포를 “복사 가능한 템플릿”으로 삼아, 내 프로젝트에 적용하는 최소 절차를 제공합니다. (`README.md`, `.claude/*`, `.mcp.json`)
- 특히 권한/훅/외부 MCP 연결은 리스크가 있으므로, “안전한 최소 적용”부터 시작하도록 설계합니다. (`.claude/settings.json`)

---

## 빠른 요약(추천 최소 세트)

1) `.claude/commands/` : 자주 쓰는 워크플로우 커맨드부터 (예: `.claude/commands/weather-orchestrator.md`)
2) `.claude/skills/` : 반복 작업을 스킬로 분리 (예: `.claude/skills/weather-fetcher/SKILL.md`)
3) `CLAUDE.md` + `.claude/rules/` : 프로젝트 규칙/컨텍스트를 고정
4) `.mcp.json` : 필요한 MCP 서버만 선택적으로 연결
5) `.claude/settings.json` : 마지막에, 최소 권한으로 단계적 도입(allow/ask/hook 검토)

근거: 파일 존재 및 `.claude/settings.json` 구조

---

## 단계별 적용 체크리스트(근거 기반)

### Step 1) 스킬/커맨드만 먼저 가져오기(낮은 리스크)

- 가져올 후보:
  - `.claude/commands/weather-orchestrator.md`
  - `.claude/skills/weather-fetcher/SKILL.md`
  - `.claude/skills/time-skill/SKILL.md`

근거: `.claude/commands/*`, `.claude/skills/*/SKILL.md`

### Step 2) 규칙/메모리 추가

- `CLAUDE.md`
- `.claude/rules/*`

근거: `CLAUDE.md`, `.claude/rules/*`

### Step 3) MCP 서버를 선택적으로 연결

- `.mcp.json`의 서버들 중 실제로 필요한 것만 포함

근거: `.mcp.json`

### Step 4) settings.json은 “최소 권한”으로

- `.claude/settings.json`의 permissions allow/ask를 환경에 맞게 축소
- hooks는 일단 비활성(또는 제거)하고, 필요할 때만 단계적으로 도입

근거: `.claude/settings.json`

---

## 문서 점검 자동화(예시)

프로젝트에 `.claude/settings.json` / `.mcp.json`을 넣었다면, 최소한 JSON 파싱이 되는지 점검하는 스크립트를 두는 것이 안전합니다.

```bash
python3 - <<'PY'\nimport json\nfor p in [\".claude/settings.json\", \".mcp.json\"]:\n  with open(p, \"r\", encoding=\"utf-8\") as f:\n    json.load(f)\n  print(\"ok:\", p)\nPY\n```

---

## 근거(파일/경로)

- `.claude/` 리소스: `.claude/commands/*`, `.claude/agents/*`, `.claude/skills/*/SKILL.md`, `.claude/rules/*`, `.claude/settings.json`
- MCP 예시: `.mcp.json`
- 메모리 파일: `CLAUDE.md`

---

## 주의사항/함정

- 훅/권한/네트워크 호출은 개발팀의 보안 정책과 충돌할 수 있습니다. “일단 전부 복사”가 아니라, 필요한 것부터 최소 도입 후 확대하는 접근이 안전합니다. (`.claude/settings.json`, `.mcp.json`)

---

## TODO/확인 필요

- `.claude/settings.json`의 allow/ask 패턴을 “팀 정책 템플릿(안전형/공격형)”으로 분리해 문서화
- `.claude/hooks/scripts/hooks.py`를 내 프로젝트에 가져갈 때 필요한 런타임(파이썬/의존성/경로)을 명시

---

## 위키 링크

- `[[claude-code-best-practice Guide - Index]]` → [가이드 목차](/blog-repo/claude-code-best-practice-guide/)
- `[[claude-code-best-practice Guide - Settings]]` → [04. MCP/Settings/Memory](/blog-repo/claude-code-best-practice-guide-04-mcp-settings-memory/)

---

*이 플레이북은 “안전한 최소 도입”을 전제로 합니다. 팀의 보안/권한 정책에 맞춰 단계적으로 확장하세요.*

