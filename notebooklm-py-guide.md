---
layout: page
title: notebooklm-py 가이드
permalink: /notebooklm-py-guide/
icon: fas fa-book
---

# notebooklm-py 위키형 가이드

> **Unofficial Python API and agentic skill for Google NotebookLM**

`notebooklm-py`를 코드 근거 중심으로 다시 정리한 위키형 시리즈입니다. 기존 5챕터 요약본이 아니라, 설치, 인증, 구조, 사용 흐름, 운영, 문서 자동화까지 **재개 가능한 v2 시리즈**로 재구성했습니다.

마지막 업데이트: 2026-03-10

## 빠른 요약

- 프로젝트는 `Python API`, `CLI`, `AI Agent Skill` 3개 진입점을 제공합니다. 근거: `README.md`, `pyproject.toml`, `src/notebooklm/data/SKILL.md`
- 인증은 브라우저 기반 `storage_state.json` 또는 `NOTEBOOKLM_AUTH_JSON` 경로를 사용합니다. 근거: `src/notebooklm/auth.py`, `src/notebooklm/paths.py`, `docs/configuration.md`
- 코어 구조는 `NotebookLMClient -> *_API -> ClientCore -> rpc/*` 계층입니다. 근거: `src/notebooklm/client.py`, `src/notebooklm/_core.py`, `docs/development.md`
- 생성물 자동화와 다운로드 범위가 넓고, 일부 기능은 웹 UI보다 더 많습니다. 근거: `README.md`, `docs/cli-reference.md`, `src/notebooklm/_artifacts.py`

## 위키 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 범위](/blog-repo/notebooklm-py-guide-01-intro/) | 프로젝트가 제공하는 범위, 리스크, 핵심 인터페이스 |
| 02 | [설치와 인증](/blog-repo/notebooklm-py-guide-02-install-auth/) | 패키지 설치, 로그인, 상태 파일, CI 인증 |
| 03 | [아키텍처와 호출 계층](/blog-repo/notebooklm-py-guide-03-architecture/) | Client, Core, RPC, CLI 구조 |
| 04 | [CLI와 Python API](/blog-repo/notebooklm-py-guide-04-cli-python-api/) | 명령형 사용 패턴과 async API 대응 관계 |
| 05 | [소스, 리서치, 채팅](/blog-repo/notebooklm-py-guide-05-sources-research-chat/) | source/research/chat 흐름과 병렬 사용 주의점 |
| 06 | [생성물과 다운로드](/blog-repo/notebooklm-py-guide-06-artifacts-downloads/) | generate/download/export, 백오프, 포맷 |
| 07 | [테스트, CI, 보안, 운영](/blog-repo/notebooklm-py-guide-07-testing-ci-security/) | 테스트 레이어, 워크플로우, 시크릿 관리 |
| 08 | [문서 점검 자동화와 트러블슈팅](/blog-repo/notebooklm-py-guide-08-doc-automation-troubleshooting/) | 문서 자동화, RPC 헬스체크, 장애 대응 |

## 위키 링크

- `[[notebooklm-py Guide - 소개 및 범위]]` [01 소개](/blog-repo/notebooklm-py-guide-01-intro/)
- `[[notebooklm-py Guide - 설치와 인증]]` [02 설치와 인증](/blog-repo/notebooklm-py-guide-02-install-auth/)
- `[[notebooklm-py Guide - 아키텍처와 호출 계층]]` [03 아키텍처](/blog-repo/notebooklm-py-guide-03-architecture/)
- `[[notebooklm-py Guide - CLI와 Python API]]` [04 CLI와 Python API](/blog-repo/notebooklm-py-guide-04-cli-python-api/)
- `[[notebooklm-py Guide - 소스, 리서치, 채팅]]` [05 소스, 리서치, 채팅](/blog-repo/notebooklm-py-guide-05-sources-research-chat/)
- `[[notebooklm-py Guide - 생성물과 다운로드]]` [06 생성물과 다운로드](/blog-repo/notebooklm-py-guide-06-artifacts-downloads/)
- `[[notebooklm-py Guide - 테스트, CI, 보안, 운영]]` [07 테스트, CI, 보안, 운영](/blog-repo/notebooklm-py-guide-07-testing-ci-security/)
- `[[notebooklm-py Guide - 문서 점검 자동화와 트러블슈팅]]` [08 문서 점검 자동화와 트러블슈팅](/blog-repo/notebooklm-py-guide-08-doc-automation-troubleshooting/)

## 관련 링크

- GitHub 저장소: `https://github.com/teng-lin/notebooklm-py`
- CLI 레퍼런스: `docs/cli-reference.md`
- Python API 문서: `docs/python-api.md`
- 설정 문서: `docs/configuration.md`

