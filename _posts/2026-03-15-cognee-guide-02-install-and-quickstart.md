---
layout: post
title: "Cognee 완벽 가이드 (02) - 설치 및 빠른 시작"
date: 2026-03-15
permalink: /cognee-guide-02-install-and-quickstart/
author: topoteretes
categories: [GitHub Trending, cognee]
tags: [Trending, GitHub, cognee, Quickstart, uv, Python]
original_url: "https://github.com/topoteretes/cognee"
excerpt: "README의 최소 예제(add→cognify→search)와 CLI 명령(cognee-cli add/cognify/search)을 근거로, Cognee를 가장 빠르게 실행하는 경로를 정리합니다."
---

## 이 문서의 목적

- README의 최소 파이프라인을 “그대로” 실행할 수 있는 형태로 정리합니다. (`README.md`)
- CLI와 라이브러리 중, 어떤 상황에 무엇을 쓰는지 기준을 잡습니다. (`README.md`, `pyproject.toml`)

---

## 빠른 요약(README/pyproject.toml 기반)

- 설치(README): `uv pip install cognee`
- LLM 키(README): `LLM_API_KEY` 환경 변수 또는 `.env` (템플릿: `.env.template`)
- 라이브러리 파이프라인(README): `await cognee.add(...)` → `await cognee.cognify()` → `await cognee.search(...)`
- CLI 엔트리(근거): `cognee-cli` (`pyproject.toml`의 `[project.scripts]`)

---

## 1) 설치(README)

```bash
uv pip install cognee
```

근거: `README.md`

---

## 2) LLM API 키 설정(README)

README는 다음처럼 환경 변수로 키를 설정하는 예시를 제공합니다. (`README.md`)

```python
import os
os.environ[\"LLM_API_KEY\"] = \"YOUR OPENAI_API_KEY\"
```

또는 `.env.template`을 기반으로 `.env`를 작성하는 흐름을 안내합니다. (`README.md`, `.env.template`)

---

## 3) 최소 파이프라인 실행(README)

README 예시의 핵심 호출:

- `cognee.add(...)`
- `cognee.cognify()`
- `cognee.search(...)`

근거: `README.md`

---

## 4) CLI로 실행(README + scripts 근거)

README는 아래 CLI 흐름을 예시로 듭니다. (`README.md`)

```bash
cognee-cli add \"Cognee turns documents into AI memory.\"
cognee-cli cognify
cognee-cli search \"What does Cognee do?\"\n+cognee-cli delete --all
```

CLI 엔트리 근거: `pyproject.toml`의 `cognee-cli = \"cognee.cli._cognee:main\"`

---

## 근거(파일/경로)

- 설치/퀵스타트/CLI 명령: `README.md`
- CLI 엔트리: `pyproject.toml` (`[project.scripts]`)
- 환경 변수 템플릿: `.env.template`

---

## 주의사항/함정

- `LLM_API_KEY`/프로바이더 설정은 환경마다 달라 “복붙”으로 해결되지 않을 수 있습니다. README의 “LLM Provider Documentation” 링크를 참고해 자신의 프로바이더 설정으로 맞추는 것이 필요합니다. (`README.md`)

---

## TODO/확인 필요

- `cognee-cli` 명령이 실제로 매핑되는 코드 위치 확인: `cognee/cli/_cognee.py` (scripts 근거)
- 기본 저장소(로컬 DB/파일)의 위치와 초기화/삭제 동작(`delete --all`)의 범위를 코드로 확인

---

## 위키 링크

- `[[Cognee Guide - Index]]` → [가이드 목차](/blog-repo/cognee-guide/)
- `[[Cognee Guide - Architecture]]` → [03. 아키텍처](/blog-repo/cognee-guide-03-architecture/)

---

*다음 글에서는 레포 구조(`cognee/`, `cognee/cli/`, `cognee/api/`, `cognee-mcp/`)를 기준으로 파이프라인/스토리지 관점을 정리합니다.*

