---
layout: post
title: "notebooklm-py 위키형 가이드 (04) - CLI와 Python API"
date: 2026-03-10
permalink: /notebooklm-py-guide-04-cli-python-api/
author: Teng Lin
categories: [개발 도구, notebooklm-py]
tags: [notebooklm-py, CLI, Python API, Async, Click]
series: notebooklm-py-guide
part: 4
original_url: "https://github.com/teng-lin/notebooklm-py"
excerpt: "CLI 명령 그룹과 async Python API 사용 패턴을 대응시켜 설명합니다."
---

마지막 업데이트: 2026-03-10

## 이 문서의 목적

실제 사용자가 가장 많이 접하는 두 인터페이스인 CLI와 Python API를 한눈에 비교하고, 어떤 상황에서 무엇을 쓰는 게 맞는지 정리합니다.

## 빠른 요약

- CLI는 Click 기반 명령 그룹 구조로, `notebooklm` 아래에 `source`, `artifact`, `generate`, `download`, `note`, `research`, `skill`을 붙입니다. 근거: `src/notebooklm/notebooklm_cli.py`
- Python API는 `NotebookLMClient` 한 개 아래에 `client.notebooks`, `client.sources`, `client.artifacts`, `client.chat` 식으로 접근합니다. 근거: `src/notebooklm/client.py`
- CLI는 사람과 스크립트 모두를 겨냥하고, Python API는 async workflow와 애플리케이션 통합에 적합합니다. 근거: `README.md`, `docs/python-api.md`

## 근거(파일/경로)

- CLI 등록: `src/notebooklm/notebooklm_cli.py`
- CLI 상세: `docs/cli-reference.md`
- Python API 진입점: `src/notebooklm/client.py`
- 예제: `README.md`, `docs/python-api.md`

## 대응 관계

| 작업 | CLI | Python API |
|------|-----|------------|
| 클라이언트 시작 | `notebooklm login`, `notebooklm list` | `async with await NotebookLMClient.from_storage()` |
| 노트북 생성 | `notebooklm create "Title"` | `await client.notebooks.create("Title")` |
| URL 소스 추가 | `notebooklm source add "https://..."` | `await client.sources.add_url(...)` |
| 질문 | `notebooklm ask "..."` | `await client.chat.ask(...)` |
| 오디오 생성 | `notebooklm generate audio --wait` | `await client.artifacts.generate_audio(...)` |
| 다운로드 | `notebooklm download audio out.mp3` | `await client.artifacts.download_audio(...)` |

## CLI 사용 흐름

```bash
notebooklm login
notebooklm create "My Research"
notebooklm use <notebook_id>
notebooklm source add "https://example.com"
notebooklm ask "핵심 요약은?"
notebooklm generate report --format study-guide --wait
```

## Python API 사용 흐름

```python
from notebooklm import NotebookLMClient

async with await NotebookLMClient.from_storage() as client:
    nb = await client.notebooks.create("Research")
    await client.sources.add_url(nb.id, "https://example.com", wait=True)
    result = await client.chat.ask(nb.id, "Summarize this")
    print(result.answer)
```

## 왜 두 인터페이스가 같이 유지되는가

- CLI는 shell automation, quick task, CI에서 유리합니다. 근거: `README.md`
- Python API는 하나의 프로세스 안에서 여러 호출을 조합하기 쉽습니다.
- 두 인터페이스 모두 결국 동일 라이브러리를 사용하므로 기능 편차가 적습니다.

## 컨텍스트 사용 vs 명시적 ID

CLI는 `use` 명령으로 현재 notebook을 저장할 수 있지만, 병렬 자동화에서는 명시적 ID가 더 안전합니다. upstream skill 문서도 parallel agents에서는 `NOTEBOOKLM_HOME` 분리 또는 `--notebook/-n` 명시를 권합니다. 근거: `src/notebooklm/data/SKILL.md`, `docs/cli-reference.md`

## 주의사항/함정

- CLI는 편하지만 `context.json`에 의존하는 부분이 있어 병렬 에이전트에서 충돌할 수 있습니다.
- Python API는 async 기반이라 동기 코드베이스에 붙일 때 event loop 처리를 명확히 해야 합니다.
- Windows는 별도 event loop policy와 UTF-8 보정이 들어갑니다. 근거: `src/notebooklm/notebooklm_cli.py`

## TODO/확인 필요

- `docs/python-api.md` 전체를 기반으로 한 세부 method catalog는 별도 확장 챕터로 분리할 수 있습니다.
- framework integration 예시는 저장소에 FastAPI/Django 샘플로 제공되지는 않습니다.

## 위키 링크

- `[[notebooklm-py Guide - 아키텍처와 호출 계층]]` [이전 문서](/blog-repo/notebooklm-py-guide-03-architecture/)
- `[[notebooklm-py Guide - 소스, 리서치, 채팅]]` [다음 문서](/blog-repo/notebooklm-py-guide-05-sources-research-chat/)
- [시리즈 허브](/blog-repo/notebooklm-py-guide/)

