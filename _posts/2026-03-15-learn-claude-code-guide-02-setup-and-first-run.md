---
layout: post
title: "Learn Claude Code 완벽 가이드 (02) - 설치 및 첫 실행"
date: 2026-03-15
permalink: /learn-claude-code-guide-02-setup-and-first-run/
author: shareAI-lab
categories: [GitHub Trending, learn-claude-code]
tags: [Trending, GitHub, learn-claude-code, Setup, dotenv, Anthropic]
original_url: "https://github.com/shareAI-lab/learn-claude-code"
excerpt: "`requirements.txt` + `.env.example`를 근거로, API 키/모델 설정 후 `agents/s01_agent_loop.py`를 실행해 ‘tool_use 루프’가 도는지 확인합니다."
---

## 이 문서의 목적

- “에이전트 루프”를 직접 실행해 볼 수 있도록, 레포가 제공하는 공식 Quick Start를 재현합니다. (`README.md`)
- 설정 파일(.env)에서 실제로 무엇이 필요한지(키/모델/옵션)를 근거로 정리합니다. (`.env.example`)

---

## 빠른 요약(README/.env.example 기반)

- 의존성: `requirements.txt` (`anthropic`, `python-dotenv`)  
- 설정: `.env.example` → `.env`
  - 필수: `ANTHROPIC_API_KEY`, `MODEL_ID`
  - 옵션: `ANTHROPIC_BASE_URL` (Anthropic-compatible provider용)
- 실행(시작점): `python agents/s01_agent_loop.py` (`README.md`, `agents/s01_agent_loop.py`)

---

## 1) 설치

```bash
git clone https://github.com/shareAI-lab/learn-claude-code
cd learn-claude-code
pip install -r requirements.txt
```

근거: `README.md`, `requirements.txt`

---

## 2) 환경 변수(.env)

```bash
cp .env.example .env
${EDITOR:-vi} .env
```

`.env.example`가 요구하는 핵심 키(근거):
- `ANTHROPIC_API_KEY=...`
- `MODEL_ID=...`

옵션(근거):
- `ANTHROPIC_BASE_URL=...` (Anthropic-compatible provider) (`.env.example`)

---

## 3) s01 실행: 가장 작은 루프

```bash
python agents/s01_agent_loop.py
```

코드 근거:
- 에이전트 루프: `agents/s01_agent_loop.py`의 `agent_loop(messages)`
- 툴: `TOOLS = [{ name: "bash", ... }]`
- 위험 명령 차단 예시: `run_bash()`의 `dangerous = [...]`

---

## 4) Web 플랫폼(옵션)

README는 학습용 Web 플랫폼을 `web/`에서 실행할 수 있다고 안내합니다. (`README.md`, `web/README.md`, `web/package.json`)

```bash
cd web
npm install
npm run dev
```

---

## 근거(파일/경로)

- Quick Start: `README.md`
- 의존성: `requirements.txt`
- 환경 변수 템플릿: `.env.example`
- 첫 실행 엔트리: `agents/s01_agent_loop.py`
- Web 플랫폼: `web/package.json`, `web/README.md`

---

## 주의사항/함정

- `.env`의 키/모델 설정이 없으면 `agents/s01_agent_loop.py`에서 `MODEL = os.environ[\"MODEL_ID\"]` 등이 실패할 수 있습니다. (`agents/s01_agent_loop.py`, `.env.example`)
- `agents/s01_agent_loop.py`는 `subprocess.run(..., shell=True)`로 bash 실행을 구현하므로, 학습 목적 외의 환경에서는 권한/격리 정책을 별도로 고려해야 합니다. (`agents/s01_agent_loop.py`)

---

## TODO/확인 필요

- 각 세션 파일(s02~s12)이 `.env`에서 추가로 요구하는 키가 있는지 점검(있으면 표로 정리)
- Web 플랫폼이 읽는 설정/환경 변수가 무엇인지 확인 (`web/`)

---

## 위키 링크

- `[[Learn Claude Code Guide - Index]]` → [가이드 목차](/blog-repo/learn-claude-code-guide/)
- `[[Learn Claude Code Guide - Core Loop]]` → [03. 핵심 루프와 Tool Use](/blog-repo/learn-claude-code-guide-03-core-loop-and-tool-use/)

---

*다음 글에서는 s01의 “tool_use 루프”를 분해하고, s02에서 ‘툴을 추가한다 = 핸들러를 추가한다’는 디자인을 코드로 확인합니다.*

