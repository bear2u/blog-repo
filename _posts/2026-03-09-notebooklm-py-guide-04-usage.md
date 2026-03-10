---
layout: post
title: "notebooklm-py 완벽 가이드 (04) - 실전 사용 패턴"
date: 2026-03-09
permalink: /notebooklm-py-guide-04-usage/
author: teng-lin
categories: [개발 도구, notebooklm-py]
tags: [notebooklm-py, Python, CLI, Automation, CI]
original_url: "https://github.com/teng-lin/notebooklm-py"
excerpt: "notebooklm-py CLI/Python 실전 사용 패턴"
---

마지막 업데이트: 2026-03-10

## 이 문서의 목적

CLI/Python에서 **재현 가능한 작업 흐름(소스→질의→생성→다운로드)** 을 만들고, 자동화/CI에서 자주 부딪히는 함정을 피하는 사용 패턴을 정리합니다.

## 빠른 요약

- CLI는 `notebooklm` 단일 엔트리포인트 아래에 `source`, `generate`, `download` 같은 하위 그룹을 제공합니다. (근거: `src/notebooklm/notebooklm_cli.py`, `docs/cli-reference.md`)
- 생성은 `notebooklm generate <type> ... --wait/--no-wait` 패턴을 기본으로 하며, 레이트 리밋 대응을 위한 `--retry` 옵션과 백오프 로직이 구현돼 있습니다. (근거: `src/notebooklm/cli/generate.py`)
- 다운로드는 아티팩트 타입별 커맨드와 `--all/--latest/--earliest` 같은 선택 로직을 가집니다. (근거: `src/notebooklm/cli/download.py`)

## 근거(파일/경로)

- CLI 레퍼런스: `docs/cli-reference.md`
- CLI 구현: `src/notebooklm/notebooklm_cli.py`, `src/notebooklm/cli/source.py`, `src/notebooklm/cli/chat.py`, `src/notebooklm/cli/generate.py`, `src/notebooklm/cli/download.py`
- Python API 예시/클라이언트: `src/notebooklm/client.py`
- 자동화/병렬 실행 주의: `src/notebooklm/data/SKILL.md`, `src/notebooklm/paths.py`

## CLI 패턴: 소스 → 질의 → 생성 → 다운로드

아래 예시는 실제 커맨드들이 제공되는지(옵션 포함)를 코드로 확인한 뒤 구성했습니다.

```bash
# 0) 인증(최초 1회)
notebooklm login

# 1) 노트북 선택(단일 프로세스 기준)
notebooklm use <notebook_id>

# 2) 소스 추가 (URL / 파일 / 텍스트)
notebooklm source add "https://en.wikipedia.org/wiki/Artificial_intelligence"
notebooklm source add "./paper.pdf"

# 3) 질문
notebooklm ask "핵심 테마를 요약해줘"

# 4) 생성(예: 오디오) + 완료까지 대기
notebooklm generate audio "더 재미있게" --wait

# 5) 결과물 다운로드
notebooklm download audio ./podcast.mp3
```

- `source add`, `ask`, `generate audio --wait`는 각각 `src/notebooklm/cli/source.py`, `src/notebooklm/cli/chat.py`, `src/notebooklm/cli/generate.py`에 구현돼 있습니다.
- `download audio`는 `src/notebooklm/cli/download.py`에 구현돼 있습니다.

## 생성(Generate)에서 중요한 옵션: `--wait` / `--retry`

`generate` 계열은 “바로 task id를 받고 끝내기” 또는 “완료까지 대기”로 나뉘며, 이 선택이 자동화 품질을 크게 좌우합니다.

- `--wait/--no-wait`(기본 `no-wait`) 옵션이 여러 generate 서브커맨드에 선언돼 있습니다. (근거: `src/notebooklm/cli/generate.py`)
- 레이트 리밋 시 재시도를 위해 지수 백오프(초기 60s, 최대 300s)가 구현돼 있습니다. (근거: `src/notebooklm/cli/generate.py`의 `RETRY_INITIAL_DELAY`, `RETRY_MAX_DELAY`, `calculate_backoff_delay`)

## 다운로드(Download)에서 중요한 옵션: `--all` / `--latest` / `--earliest`

다운로드는 “특정 아티팩트 1개”뿐 아니라 “완료된 결과 전부”를 내려받는 플로우가 가능합니다.

- 아티팩트 타입별 기본 디렉토리 및 확장자가 `ARTIFACT_CONFIGS`로 관리됩니다. (근거: `src/notebooklm/cli/download.py`)
- `--all` 플로우는 “해당 타입의 완료된 아티팩트 목록을 가져와 반복 다운로드”로 구현돼 있습니다. (근거: `src/notebooklm/cli/download.py`)

## Python 패턴: 비동기 워크플로우(최소 골격)

Python API의 메인 엔트리포인트는 `NotebookLMClient`입니다.

```python
from notebooklm.client import NotebookLMClient

async def run():
    async with await NotebookLMClient.from_storage() as client:
        notebooks = await client.notebooks.list()
        # 이후 sources/artifacts/chat 등을 조합
```

위 패턴은 `NotebookLMClient.from_storage()`가 storage state를 기반으로 인증을 구성한다는 점에서 핵심입니다. (근거: `src/notebooklm/client.py`, `src/notebooklm/auth.py`)

## CI/자동화 팁: `NOTEBOOKLM_AUTH_JSON`과 홈 디렉토리 분리

- CI에서는 `NOTEBOOKLM_AUTH_JSON`로 인증 JSON을 주입하는 가이드가 있습니다. (근거: `docs/cli-reference.md`, `.env.example`, `src/notebooklm/data/SKILL.md`)
- 여러 에이전트가 동시에 돌아가면 `context.json` 덮어쓰기 위험이 있으므로, 에이전트별 `NOTEBOOKLM_HOME` 분리가 권장됩니다. (근거: `src/notebooklm/paths.py`, `src/notebooklm/data/SKILL.md`)

## 주의사항/함정

- `notebooklm use`는 로컬 컨텍스트에 의존하므로 **병렬 자동화에서는 충돌**할 수 있습니다. 가능한 경우 `--notebook/-n`로 노트북 ID를 명시하세요. (근거: `docs/cli-reference.md`, `src/notebooklm/cli/chat.py`)
- 생성/다운로드는 네트워크·쿼터·서버 상태에 영향을 받습니다. `--wait`, `--retry`의 타임아웃/백오프를 워크플로우에 맞게 설계하세요. (근거: `src/notebooklm/cli/generate.py`)

## TODO / 확인 필요

- 팀의 실행 환경(컨테이너/CI 런너)에서 Playwright 설치 방식(브라우저 캐시/의존 패키지)은 표준이 정해져 있지 않다면 별도 운영 가이드가 필요합니다. (근거: `README.md`, `.github/workflows/test.yml`은 Playwright 브라우저 캐시/설치를 수행)

---

다음 글에서는 운영/보안/테스트/CI 관점의 체크리스트를 정리합니다.
