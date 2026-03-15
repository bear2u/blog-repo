---
layout: post
title: "Learn Claude Code 완벽 가이드 (05) - 확장·문제해결·점검 자동화"
date: 2026-03-15
permalink: /learn-claude-code-guide-05-extension-troubleshooting-doc-automation/
author: shareAI-lab
categories: [GitHub Trending, learn-claude-code]
tags: [Trending, GitHub, learn-claude-code, Nextjs, Troubleshooting, Automation]
original_url: "https://github.com/shareAI-lab/learn-claude-code"
excerpt: "Web 플랫폼(`web/`) 실행 경로와 학습 레포 특성상 자주 부딪히는 설정/실행 이슈를 정리하고, 세션 파일/문서 링크의 무결성을 빠르게 점검하는 자동화 예시를 제공합니다."
---

## 이 문서의 목적

- “학습 레포”를 내 환경에서 반복 실행할 때 생길 수 있는 대표 이슈(키/모델/실행)를 정리합니다. (`.env.example`, `agents/`)
- 문서 점검 자동화(링크/파이썬 파일 구문 체크 등)로, 학습 산출물이 깨졌는지 빠르게 확인합니다.

---

## 빠른 요약(파일 기반)

- Web 플랫폼: `web/`에서 `npm run dev` (`README.md`, `web/package.json`)
- 핵심 실행은 `agents/`의 스크립트들(특히 `agents/s01_agent_loop.py`, `agents/s12_worktree_task_isolation.py`) (`README.md`)

---

## Web 플랫폼 실행(README 기반)

```bash
cd web
npm install
npm run dev
```

근거: `README.md`, `web/package.json`

---

## 흔한 문제해결 포인트(근거 중심)

- 환경 변수 미설정:
  - `.env.example`가 요구하는 `ANTHROPIC_API_KEY`, `MODEL_ID`를 `.env`에 반영했는지 확인 (`.env.example`)
- 모델/베이스 URL:
  - `agents/s01_agent_loop.py`는 `Anthropic(base_url=os.getenv(\"ANTHROPIC_BASE_URL\"))` 형태로 base_url을 받습니다. (`agents/s01_agent_loop.py`)

---

## 문서 점검 자동화(예시)

아래는 “학습 산출물이 최소한 깨지지 않았는지”를 확인하는 로컬 점검 예시입니다.

1) 파이썬 구문 체크(컴파일):

```bash
python -m py_compile agents/*.py
```

2) 문서 링크 존재 여부(단순):

```bash
ls docs/en | head
```

> 주의: 이 스크립트들은 “실제 LLM 호출 성공”을 보장하지 않습니다. 네트워크/키/요금/모델 availability는 별도 이슈입니다. (`.env.example`)

---

## 근거(파일/경로)

- Web 플랫폼: `web/package.json`
- 환경 변수: `.env.example`
- 최소 루프: `agents/s01_agent_loop.py`
- 세션 스크립트: `agents/`
- 세션 문서: `docs/en/`

---

## 주의사항/함정

- 학습 목적의 코드이므로 “권한/샌드박싱/정책”이 단순화되어 있습니다. 실제 사용 환경에선 별도 격리/권한 제어를 고려해야 합니다. (`README.md`)

---

## TODO/확인 필요

- `web/`이 참조하는 세션 코드/문서 경로가 무엇인지 확인하고, “문서→코드” 교차 링크 점검 스크립트 확장
- s08~s12의 백그라운드/팀/워크트리 실행이 어떤 OS 제약을 갖는지(Windows 등) 문서화

---

## 위키 링크

- `[[Learn Claude Code Guide - Index]]` → [가이드 목차](/blog-repo/learn-claude-code-guide/)
- `[[Learn Claude Code Guide - Core Loop]]` → [03. 핵심 루프와 Tool Use](/blog-repo/learn-claude-code-guide-03-core-loop-and-tool-use/)

---

*이 레포는 “한 번에 완성된 프레임워크”가 아니라 “메커니즘을 단계적으로 붙여보는 교재”라는 점을 전제로 읽는 것이 가장 좋습니다.*

