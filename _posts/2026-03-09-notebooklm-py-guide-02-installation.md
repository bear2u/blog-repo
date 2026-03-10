---
layout: post
title: "notebooklm-py 완벽 가이드 (02) - 설치 및 빠른 시작"
date: 2026-03-09
permalink: /notebooklm-py-guide-02-installation/
author: teng-lin
categories: [개발 도구, notebooklm-py]
tags: [notebooklm-py, Python, NotebookLM, Playwright, CLI]
original_url: "https://github.com/teng-lin/notebooklm-py"
excerpt: "notebooklm-py 설치와 빠른 시작"
---

마지막 업데이트: 2026-03-10

## 이 문서의 목적

로컬에서 `notebooklm` CLI / Python API를 실행하기 위한 **설치·인증·환경 변수·상태 파일 경로**를 “근거 기반”으로 정리합니다.

## 빠른 요약

- 패키지 설치는 `pip install notebooklm-py`가 기본이며, 브라우저 로그인 기능은 extra `browser`(Playwright)에 들어 있습니다. (근거: `README.md`, `pyproject.toml`)
- 인증은 `notebooklm login`으로 Playwright 브라우저를 열어 로그인한 뒤, `storage_state.json`에 쿠키를 저장합니다. (근거: `src/notebooklm/cli/session.py`, `src/notebooklm/paths.py`)
- CI/자동화에서는 파일 대신 `NOTEBOOKLM_AUTH_JSON`로 인증 JSON을 주입하는 경로가 문서화돼 있습니다. (근거: `docs/cli-reference.md`, `src/notebooklm/data/SKILL.md`, `.env.example`)

## 근거(파일/경로)

- 설치/퀵스타트: `README.md`
- 의존성/extra: `pyproject.toml` (`[project.optional-dependencies] browser = ["playwright>=..."]`)
- 로그인 구현: `src/notebooklm/cli/session.py` (`login` 커맨드)
- 경로 규칙: `src/notebooklm/paths.py` (`NOTEBOOKLM_HOME`, `storage_state.json`, `context.json`)
- 환경 변수 예시: `.env.example`, `docs/configuration.md`, `docs/cli-reference.md`

## 설치

```bash
# 기본 설치
pip install notebooklm-py

# 브라우저 로그인 지원(Playwright) 포함
pip install "notebooklm-py[browser]"

# Playwright 브라우저 설치(Chromium)
playwright install chromium
```

위 커맨드는 README와 로그인 구현이 Playwright를 사용한다는 점에서 일치합니다. (근거: `README.md`, `src/notebooklm/cli/session.py`)

## 인증(로그인) — 반드시 1회 필요

```bash
notebooklm login
```

- `notebooklm login`은 Playwright를 import하고, Chromium 설치 여부를 사전 점검한 뒤 `https://notebooklm.google.com/`으로 이동합니다. (근거: `src/notebooklm/cli/session.py`)
- 로그인 성공 후 storage state를 파일로 저장하며, 파일 권한을 `0o600`으로 제한합니다. (근거: `src/notebooklm/cli/session.py`)
- **주의:** `NOTEBOOKLM_AUTH_JSON`이 설정된 상태에서는 `login`을 실행할 수 없도록 막아둡니다. (근거: `src/notebooklm/cli/session.py`)

## 빠른 시작(CLI)

아래는 “노트북 선택 → 소스 추가 → 질문”의 최소 플로우입니다.

```bash
# 1) 인증 확인
notebooklm status

# 2) 노트북 리스트/생성
notebooklm list
notebooklm create "My Research"

# 3) 현재 노트북 컨텍스트 설정(단일 에이전트/단일 터미널에서만 권장)
notebooklm use <notebook_id>

# 4) 소스 추가 후 질문
notebooklm source add "https://example.com"
notebooklm ask "이 소스를 요약해줘"
```

명령어 구조 자체는 CLI 레퍼런스에도 정리돼 있습니다. (근거: `docs/cli-reference.md`, `src/notebooklm/notebooklm_cli.py`)

## 환경 변수/상태 파일(운영·자동화에서 중요)

| 항목 | 의미 | 근거 |
|---|---|---|
| `NOTEBOOKLM_HOME` | 설정/상태 파일 베이스 디렉토리(기본 `~/.notebooklm`) | `src/notebooklm/paths.py`, `.env.example`, `docs/cli-reference.md` |
| `NOTEBOOKLM_AUTH_JSON` | 파일 없이 인증 JSON을 env로 주입(CI/CD 용) | `.env.example`, `docs/cli-reference.md`, `src/notebooklm/data/SKILL.md` |
| `storage_state.json` | Playwright storage state(쿠키) 저장 파일 | `src/notebooklm/paths.py`, `src/notebooklm/cli/session.py` |
| `context.json` | CLI 컨텍스트(현재 노트북/대화) | `src/notebooklm/paths.py`, `src/notebooklm/data/SKILL.md` |
| `NOTEBOOKLM_DEBUG_RPC` | RPC 디버그 로깅(문제 분석용) | `.env.example`, `docs/cli-reference.md` |

## 주의사항/함정

- **병렬 실행**: CLI가 `context.json`에 현재 노트북/대화 상태를 저장하므로, 여러 에이전트/여러 프로세스가 동시에 `use`를 쓰면 덮어쓸 수 있습니다. (근거: `src/notebooklm/paths.py`, `src/notebooklm/data/SKILL.md`)
  - 대응: 에이전트별 `NOTEBOOKLM_HOME` 분리, 또는 가능한 커맨드에서는 `--notebook/-n`로 명시적 ID 사용 (근거: `docs/cli-reference.md`, `src/notebooklm/cli/chat.py`)

## TODO / 확인 필요

- 조직/팀 환경에서의 “권장 secret 관리 방식(예: Vault/KMS)”은 이 저장소에 특정 솔루션으로 고정된 근거가 없습니다 → 팀 표준에 맞춰 결정 필요.

---

다음 글에서는 코드 구조와 주요 컴포넌트를 Mermaid 다이어그램으로 정리합니다.

- Python 3.10+
- 첫 인증(로그인)을 위해 브라우저 자동화가 필요할 수 있음

---

## 설치

```bash
# 기본 설치
pip install notebooklm-py

# 브라우저 로그인 지원 포함(초기 설정에 필요할 수 있음)
pip install "notebooklm-py[browser]"
playwright install chromium
```

---

## CLI 빠른 시작(README/Docs 기준)

```bash
# 1) 인증(브라우저 열림)
notebooklm login

# 2) 노트북 생성/선택
notebooklm create "My Research"
notebooklm use <notebook_id>

# 3) 소스 추가 → 질의
notebooklm source add "https://example.com"
notebooklm ask "이 소스를 요약해줘"
```

---

## 팁

- 저장 위치를 바꾸고 싶으면 `--storage PATH` 또는 `NOTEBOOKLM_HOME`을 확인하세요.
- 인증/세션 문제는 문서의 Troubleshooting을 먼저 보세요.

---

*다음 글에서는 핵심 개념과 아키텍처를 정리합니다.*
