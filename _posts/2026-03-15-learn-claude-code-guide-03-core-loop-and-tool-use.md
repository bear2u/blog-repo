---
layout: post
title: "Learn Claude Code 완벽 가이드 (03) - 핵심 루프와 Tool Use"
date: 2026-03-15
permalink: /learn-claude-code-guide-03-core-loop-and-tool-use/
author: shareAI-lab
categories: [GitHub Trending, learn-claude-code]
tags: [Trending, GitHub, learn-claude-code, ToolUse, AgentLoop, stop_reason]
original_url: "https://github.com/shareAI-lab/learn-claude-code"
excerpt: "`agents/s01_agent_loop.py`의 while 루프와 tool_result 피드백 패턴을 근거로, ‘코딩 에이전트의 최소 구조’가 어떻게 성립하는지 정리합니다."
---

## 이 문서의 목적

- s01이 구현한 “최소 에이전트 루프”를 코드 근거로 분해해 이해합니다. (`agents/s01_agent_loop.py`)
- s02가 말하는 “툴을 추가한다 = 핸들러를 추가한다”를 구현 관점으로 연결합니다. (`README.md`, `agents/s02_tool_use.py`)

---

## 빠른 요약(코드 기반)

- 루프의 종료 조건은 `response.stop_reason != "tool_use"` 입니다. (`agents/s01_agent_loop.py`)
- tool_use가 발생하면, 각 `tool_use` 블록을 실행한 결과를 `tool_result`로 다시 `messages[]`에 append하여 다음 LLM 호출에 포함합니다. (`agents/s01_agent_loop.py`)

---

## s01: 최소 루프의 핵심 라인들

`agents/s01_agent_loop.py`에서 핵심은 아래 순서입니다.

1) `client.messages.create(... messages=messages, tools=TOOLS ...)`
2) assistant 응답을 `messages.append(...)`
3) `if response.stop_reason != "tool_use": return`
4) `for block in response.content: if block.type == "tool_use": ...`
5) `results.append({ type: "tool_result", tool_use_id: block.id, content: output })`
6) `messages.append({ role: "user", content: results })`

근거: `agents/s01_agent_loop.py`

---

## s01: 툴(예: bash) 정의와 실행

- 툴 스키마는 `TOOLS = [{ name: "bash", input_schema: ... }]`로 정의됩니다. (`agents/s01_agent_loop.py`)
- 실행은 `run_bash(command)`에서 `subprocess.run(..., shell=True, timeout=120)`로 수행됩니다. (`agents/s01_agent_loop.py`)
- 위험 명령 일부는 문자열 매칭으로 차단합니다: `dangerous = ["rm -rf /", "sudo", ...]` (`agents/s01_agent_loop.py`)

---

## (연결) s02: “툴 추가 = 핸들러 추가”

README는 s02의 모토를 “Adding a tool means adding one handler”로 설명합니다. (`README.md`)

코드 근거:
- s02 구현 파일: `agents/s02_tool_use.py`

이 챕터에서는 s02의 내부를 전부 해설하지 않고, s01의 루프를 유지한 채 “툴 종류가 늘면 dispatch map/handler가 늘어난다”는 방향만 고정합니다(자세한 구조는 다음 챕터에서 필요한 범위로 확대).

---

## 근거(파일/경로)

- 최소 루프: `agents/s01_agent_loop.py`
- tool use 확장(세션 02): `agents/s02_tool_use.py`
- 개념 설명: `README.md`

---

## 주의사항/함정

- s01의 `SYSTEM = "Use bash to solve tasks"`는 학습 목적의 단순화된 시스템 프롬프트이며, 실제 제품에서는 권한/정책/샌드박싱이 별도 레이어로 필요합니다(README의 Scope 참고). (`README.md`, `agents/s01_agent_loop.py`)

---

## TODO/확인 필요

- s02에서 툴 dispatch map이 실제로 어떤 형태(딕셔너리/레지스트리)로 구현되는지 확인하고, “툴 추가 절차”를 체크리스트로 만들기 (`agents/s02_tool_use.py`)
- s01의 위험 명령 차단이 어떤 허점이 있는지(문자열 매칭 한계) 정리하고, 개선 포인트 제시

---

## 위키 링크

- `[[Learn Claude Code Guide - Index]]` → [가이드 목차](/blog-repo/learn-claude-code-guide/)
- `[[Learn Claude Code Guide - Planning/Skills]]` → [04. 계획·스킬·컨텍스트·팀](/blog-repo/learn-claude-code-guide-04-planning-skills-context-teams/)

---

*다음 글에서는 s03~s12의 메커니즘을 “무엇이 추가되며 어디에 구현되어 있는지”를 중심으로 묶어, 큰 그림을 완성합니다.*

