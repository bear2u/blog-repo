---
layout: post
title: "notebooklm-py 완벽 가이드 (05) - 운영/확장/베스트 프랙티스"
date: 2026-03-09
permalink: /notebooklm-py-guide-05-best-practices/
author: teng-lin
categories: [개발 도구, notebooklm-py]
tags: [notebooklm-py, Python, Operations, Security, CI]
original_url: "https://github.com/teng-lin/notebooklm-py"
excerpt: "notebooklm-py 운영/확장 시 체크리스트"
---

마지막 업데이트: 2026-03-10

## 이 문서의 목적

`notebooklm-py`를 자동화/팀 환경에서 쓸 때 필요한 **안정성·보안·테스트·CI 관점의 체크리스트**를 “근거 기반”으로 정리합니다.

## 빠른 요약

- 비공식/undocumented API이므로 “깨져도 회복 가능한 구조(재시도/백오프/관측/격리)”가 전제입니다. (근거: `README.md`, `docs/stability.md`)
- 인증 데이터는 민감하며, 코드상에서도 파일 권한 제한/도메인 allowlist 같은 방어가 들어가 있습니다. (근거: `src/notebooklm/cli/session.py`, `src/notebooklm/auth.py`)
- CI는 ruff/mypy/pytest를 돌리고, OS/파이썬 매트릭스로 테스트합니다. (근거: `.github/workflows/test.yml`, `pyproject.toml`)

## 근거(파일/경로)

- 안정성/버저닝/공개 API 경계: `docs/stability.md`, `CHANGELOG.md`
- 에러/재시도/타임아웃: `src/notebooklm/_core.py`, `src/notebooklm/cli/generate.py`
- 보안(쿠키/도메인/권한): `src/notebooklm/auth.py`, `src/notebooklm/cli/session.py`
- 경로/상태 파일: `src/notebooklm/paths.py`, `.env.example`
- 테스트/CI: `tests/`, `pyproject.toml` (`[tool.pytest.ini_options]`, `[tool.coverage.*]`, `[tool.ruff]`, `[tool.mypy]`), `.github/workflows/test.yml`

## 안정성/운영 체크리스트

- **변경 가능성(리스크) 인지**
  구글이 내부 RPC/API를 변경하면 깨질 수 있으며, 이 프로젝트는 그 상황을 전제로 버전 정책을 설명합니다. (근거: `docs/stability.md`)
- **레이트 리밋/쿼터 대응**
  생성(generation) 커맨드에는 레이트 리밋 시 재시도/백오프 로직이 구현돼 있습니다. 운영 워크플로우에서도 동일한 원칙(백오프/재시도/알림)을 적용하세요. (근거: `src/notebooklm/cli/generate.py`)
- **타임아웃/네트워크**
  코어는 connect/read/write/pool 타임아웃을 분리해 설정합니다. (근거: `src/notebooklm/_core.py`의 `httpx.Timeout(connect=..., read=..., ...)`)
- **로깅/관측성**
  CLI는 `-v/-vv`로 로깅 레벨을 조절하도록 되어 있습니다. (근거: `src/notebooklm/notebooklm_cli.py`)
- **병렬 실행 격리**
  `context.json`은 공유 상태이므로, 병렬 에이전트는 `NOTEBOOKLM_HOME`을 분리하거나 ID를 명시하는 패턴으로 충돌을 피하세요. (근거: `src/notebooklm/paths.py`, `src/notebooklm/data/SKILL.md`)

## 보안 체크리스트

- **쿠키/스토리지 파일 보호**
  `notebooklm login`은 storage_state 파일을 저장하고 `chmod(0o600)`을 적용합니다. (근거: `src/notebooklm/cli/session.py`)
- **도메인 allowlist**
  인증 쿠키는 특정 Google 도메인 allowlist 및 지역 도메인 whitelist를 사용합니다. (근거: `src/notebooklm/auth.py`의 `ALLOWED_COOKIE_DOMAINS`, `GOOGLE_REGIONAL_CCTLDS`)
- **CI/CD secret 주입**
  파일을 만들기 어려운 환경에서는 `NOTEBOOKLM_AUTH_JSON`로 인증을 주입할 수 있도록 안내돼 있습니다. (근거: `.env.example`, `docs/cli-reference.md`, `src/notebooklm/data/SKILL.md`)

## 테스트/CI 체크리스트

- `pytest` 설정은 `pyproject.toml`의 `[tool.pytest.ini_options]`에 있으며, 기본적으로 `tests/e2e`는 ignore하고 글로벌 timeout(60s)을 둡니다. (근거: `pyproject.toml`)
- GitHub Actions는 ruff/mypy/pytest를 돌리고, Windows/macOS/Linux + Python 3.10~3.14 매트릭스로 테스트합니다. (근거: `.github/workflows/test.yml`)
- e2e는 별도 설정이 필요하며, `.env.example`에 테스트용 노트북 ID가 요구되는 것이 명시돼 있습니다. (근거: `.env.example`)

## 주의사항/함정

- “정상 동작”의 상당 부분이 **구글 계정 상태/보안 챌린지/쿼터/국가별 도메인**에 영향을 받습니다. 이 저장소는 지역 도메인을 whitelist로 다루고 있지만, 모든 케이스를 보장하진 않습니다. (근거: `src/notebooklm/auth.py`)
- Playwright 기반 로그인은 자동화 탐지로 막힐 수 있다는 트러블슈팅이 문서에 포함돼 있습니다. (근거: `docs/troubleshooting.md`)

## TODO / 확인 필요

- 팀에서 “공식 지원이 없는 API”를 사용하는 리스크를 어떻게 수용할지(운영 SLA, 장애 대응, 대체 경로)는 코드로 결정할 수 없습니다 → 제품/운영 정책으로 합의 필요.

---

시리즈를 마칩니다. 필요하면 다음 확장 주제로 추가 챕터를 만들 수 있습니다.

- RPC 개발/디버깅: `docs/rpc-development.md`, `docs/rpc-reference.md`
- 릴리즈 프로세스: `docs/releasing.md`
