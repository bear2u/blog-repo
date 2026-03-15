---
layout: page
title: Learn Claude Code 가이드
permalink: /learn-claude-code-guide/
icon: fas fa-robot
---

# Learn Claude Code 완벽 가이드

> **A nano Claude Code-like agent, built from 0 to 1** (`README.md`)

**shareAI-lab/learn-claude-code**는 “Claude Code 같은 코딩 에이전트”의 최소 루프부터 시작해, 계획/서브에이전트/스킬 로딩/컨텍스트 압축/팀/워크트리 격리까지 단계별로 구현해 보는 학습 레포입니다. (`README.md`, `agents/`)

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 학습 로드맵](/blog-repo/learn-claude-code-guide-01-intro/) | 12 세션 구조, 레포 구성, 학습 순서 |
| 02 | [설치 및 첫 실행](/blog-repo/learn-claude-code-guide-02-setup-and-first-run/) | `requirements.txt`, `.env` 설정, s01 실행 |
| 03 | [핵심 루프와 Tool Use](/blog-repo/learn-claude-code-guide-03-core-loop-and-tool-use/) | `stop_reason == tool_use` 루프, 디스패치 맵 |
| 04 | [계획·스킬·컨텍스트·팀](/blog-repo/learn-claude-code-guide-04-planning-skills-context-teams/) | s03~s12 핵심 메커니즘 요약과 연결 |
| 05 | [확장·문제해결·점검 자동화](/blog-repo/learn-claude-code-guide-05-extension-troubleshooting-doc-automation/) | Web 플랫폼, 안전장치, 문서 점검 스크립트 |

---

## 빠른 시작(README 기반)

```bash
git clone https://github.com/shareAI-lab/learn-claude-code
cd learn-claude-code
pip install -r requirements.txt
cp .env.example .env   # ANTHROPIC_API_KEY, MODEL_ID 설정

python agents/s01_agent_loop.py
```

---

## 관련 링크

- GitHub 저장소: https://github.com/shareAI-lab/learn-claude-code

