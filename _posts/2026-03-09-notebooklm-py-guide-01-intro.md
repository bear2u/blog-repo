---
layout: post
title: "notebooklm-py 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-09
permalink: /notebooklm-py-guide-01-intro/
author: teng-lin
categories: [개발 도구, notebooklm-py]
tags: [notebooklm-py, Python, NotebookLM, CLI]
original_url: "https://github.com/teng-lin/notebooklm-py"
excerpt: "notebooklm-py 프로젝트 소개와 핵심 포인트"
---

마지막 업데이트: 2026-03-10

## 이 문서의 목적

`notebooklm-py`가 **무엇을 제공하는지(범위/인터페이스/리스크)** 를 빠르게 잡고, 다음 챕터(설치·구조·사용·운영)에서 무엇을 확인해야 하는지 기준을 세웁니다.

## 빠른 요약

- `notebooklm-py`는 **Google NotebookLM을 코드로 제어**하기 위한 비공식(unofficial) 라이브러리/CLI입니다. (근거: `README.md`)
- **3가지 사용 경로**가 명시돼 있습니다: Python API / CLI / 에이전트 스킬. (근거: `README.md`, `pyproject.toml`, `src/notebooklm/data/SKILL.md`)
- 인증은 **브라우저 로그인으로 얻은 쿠키(Playwright storage_state)** + NotebookLM 페이지에서 추출하는 토큰으로 구성됩니다. (근거: `src/notebooklm/auth.py`, `src/notebooklm/cli/session.py`, `src/notebooklm/client.py`)

## 근거(파일/경로)

- 프로젝트 개요/기능 범위/리스크: `README.md`, `docs/stability.md`, `docs/troubleshooting.md`
- 패키징/CLI 엔트리포인트/의존성: `pyproject.toml` (`[project.scripts] notebooklm = "notebooklm.notebooklm_cli:main"`, optional deps `browser = ["playwright>=..."]`)
- Python API 엔트리포인트: `src/notebooklm/client.py` (`NotebookLMClient`)
- CLI 엔트리포인트: `src/notebooklm/notebooklm_cli.py`
- 인증/토큰/쿠키 처리: `src/notebooklm/auth.py`, `src/notebooklm/paths.py`, `src/notebooklm/cli/session.py`
- 에이전트 스킬(자동화용 문서): `src/notebooklm/data/SKILL.md`

## 무엇을 만들 수 있나(README 기준)

`README.md`가 제시하는 사용 사례를 “코드/CLI로 할 수 있는 일” 관점으로 다시 정리하면 다음과 같습니다.

- **노트북/소스/대화 자동화**: 노트북 생성/선택, URL·파일·텍스트·YouTube·Drive 소스 추가, Q&A 및 히스토리 관리 (근거: `README.md`, `docs/cli-reference.md`)
- **스튜디오 아티팩트 생성/다운로드**: 오디오/비디오/슬라이드/퀴즈/플래시카드/리포트/마인드맵/데이터 테이블 등을 생성하고 파일로 내려받기 (근거: `README.md`, `src/notebooklm/cli/generate.py`, `src/notebooklm/cli/download.py`)
- **에이전트 통합**: Claude Code/Codex/OpenClaw 같은 “에이전트 런타임”에서 NotebookLM 작업을 스킬로 호출 (근거: `README.md`, `src/notebooklm/data/SKILL.md`)

## 주의사항/함정

- **비공식/문서화되지 않은 Google API**를 사용합니다. 구글 내부 변경으로 언제든 깨질 수 있다는 경고가 명시돼 있습니다. (근거: `README.md`, `docs/stability.md`)
- **인증 상태 파일은 민감 정보**입니다. `notebooklm login`은 Playwright storage state를 저장하고 권한을 `0o600`으로 제한합니다. (근거: `src/notebooklm/cli/session.py`, `src/notebooklm/auth.py`)
- 내부 구현(`notebooklm._*`, `notebooklm.rpc.*`)은 **깨질 수 있는 내부 API**로 분류돼 있습니다. (근거: `docs/stability.md`)

## TODO / 확인 필요

- NotebookLM/Google 측 정책(약관/자동화 허용 범위)은 이 저장소에 “정책 문서”로 근거가 있지 않습니다 → 실제 사용 환경에서 **법무/정책 확인 필요**.
- 구체적인 레이트 리밋/쿼터 수치(“몇 번 호출하면 제한”)는 코드/문서에서 숫자로 고정돼 있지 않습니다 → 운영 시 관측/백오프 설계 필요. (근거: `README.md`, `src/notebooklm/_core.py`, `src/notebooklm/cli/generate.py`)

---

다음 글에서는 설치·인증·환경 변수/스토리지 구조를 **코드 근거 중심**으로 정리합니다.

GitHub Trending 기준으로 주목받는 **teng-lin/notebooklm-py**를 한국어로 정리합니다.

- **한 줄 요약**: Unofficial Python API for Google NotebookLM
- **언어**: Python
- **오늘 스타**: +457
- **원본**: https://github.com/teng-lin/notebooklm-py

---

## 왜 주목받나(README 기반)

- NotebookLM 기능을 **Python API / CLI**로 자동화할 수 있습니다.
- 웹 UI에 없는 기능(예: 배치 다운로드, 퀴즈/플래시카드 JSON 내보내기 등)을 다룹니다.

---

## 주의사항(중요)

README에서 명시하듯, **비공식(undocumented) Google API**를 사용합니다.

- Google 내부 API 변경으로 갑자기 동작이 깨질 수 있습니다.
- 과도한 호출은 레이트 리밋/차단 리스크가 있습니다.

---

## 이 가이드에서 다룰 것

- 설치/인증(로그인) 빠른 시작
- 구성요소(Notebook/Sources/Chat/Artifacts) 관점의 구조
- CLI/Python 실전 사용 패턴
- 운영/보안/트러블슈팅 체크리스트

---

*다음 글에서는 설치 및 빠른 시작을 정리합니다.*
