---
layout: post
title: "Learn Claude Code 완벽 가이드 (04) - 계획·스킬·컨텍스트·팀"
date: 2026-03-15
permalink: /learn-claude-code-guide-04-planning-skills-context-teams/
author: shareAI-lab
categories: [GitHub Trending, learn-claude-code]
tags: [Trending, GitHub, learn-claude-code, Planning, Skills, Context, Teams]
original_url: "https://github.com/shareAI-lab/learn-claude-code"
excerpt: "s03~s12가 추가하는 메커니즘(계획, 서브에이전트, 스킬 로딩, 컨텍스트 압축, 태스크/팀/프로토콜/워크트리 격리)을 파일 경로로 연결해 큰 그림을 완성합니다."
---

## 이 문서의 목적

- s03~s12가 “하나의 루프” 위에 무엇을 얹는지, 세션 파일을 근거로 목록화합니다. (`agents/`)
- 실제로 코드를 읽을 때 **어느 파일을 먼저 볼지**를 정합니다.

---

## 빠른 요약(README/agents 기반)

세션 파일(근거): `agents/s03_todo_write.py` ~ `agents/s12_worktree_task_isolation.py`

- s03: 계획(할 일 작성/관리)
- s04: 서브에이전트(자식이 독립 messages[] 사용)
- s05: 스킬 로딩(필요할 때만 지식 주입)
- s06: 컨텍스트 압축(장기 세션 대비)
- s07: 태스크 시스템(디스크에 영속)
- s08: 백그라운드 작업(느린 작업 비동기화)
- s09~s11: 팀/프로토콜/자율 에이전트
- s12: 워크트리 격리(디렉토리 단위 간섭 최소화)

근거: `README.md`, `agents/`

---

## 파일 기반 “읽는 순서” 제안

1) 루프 기반을 고정: `agents/s01_agent_loop.py`, `agents/s02_tool_use.py`
2) 실행을 안정화: `agents/s03_todo_write.py`, `agents/s04_subagent.py`
3) 컨텍스트 전략: `agents/s05_skill_loading.py`, `agents/s06_context_compact.py`
4) 장기/팀 운영: `agents/s07_task_system.py` ~ `agents/s12_worktree_task_isolation.py`
5) 전체 통합: `agents/s_full.py`

근거: `agents/`, `README.md`

---

## 스킬/문서 리소스 위치

- 스킬 파일: `skills/` (예: `skills/agent-builder/`, `skills/code-review/` 등)
- 문서(세션별): `docs/en/` (README가 각 s01~s12 문서 링크 제공)
- Web 플랫폼: `web/` (Next.js) (`web/package.json`)

---

## 근거(파일/경로)

- 세션 개요/모토/구조: `README.md`
- 구현 파일: `agents/`
- 문서: `docs/en/`
- 스킬: `skills/`
- Web: `web/package.json`

---

## 주의사항/함정

- README는 “프로덕션용 메커니즘(권한/신뢰/라이프사이클 등)”이 의도적으로 생략되어 있음을 강조합니다. (`README.md`)

---

## TODO/확인 필요

- s07~s12가 생성/수정하는 파일(태스크 그래프/메일박스/워크트리)을 실제 코드에서 추출해 “산출물 목록” 만들기
- 스킬 로딩(s05)이 어떤 포맷으로 SKILL.md를 주입하는지 확인하고 “스킬 작성 규칙”을 요약하기 (`skills/`, `agents/s05_skill_loading.py`)

---

## 위키 링크

- `[[Learn Claude Code Guide - Index]]` → [가이드 목차](/blog-repo/learn-claude-code-guide/)
- `[[Learn Claude Code Guide - Extension]]` → [05. 확장·문제해결·점검 자동화](/blog-repo/learn-claude-code-guide-05-extension-troubleshooting-doc-automation/)

---

*다음 글에서는 Web 플랫폼과 확장 포인트를 짚고, 레포의 “학습 산출물”이 깨지지 않았는지 빠르게 점검하는 자동화 예시를 제공합니다.*

