---
layout: post
title: "fish-speech 완벽 가이드 (02) - 설치(로컬)"
date: 2026-03-13
permalink: /fish-speech-guide-02-local-install/
author: fishaudio
categories: [AI, fish-speech]
tags: [Trending, GitHub, fish-speech, TTS, Python, Install, GitHub Trending]
original_url: "https://github.com/fishaudio/fish-speech"
excerpt: "pyproject.toml(요구 Python/의존성)과 tools/*(실행 스크립트)를 근거로 로컬 설치 체크리스트를 정리합니다."
---

## 이 문서의 목적

- fish-speech를 **로컬 환경(비-Docker)** 에서 실행하기 위한 “확인해야 할 것”을 정리합니다.
- 레포에 포함된 근거 파일만으로 확정할 수 없는 부분(정확한 설치 커맨드 등)은 “공식 설치 문서 확인 필요”로 분리합니다.

---

## 빠른 요약 (근거 기반)

- Python 요구사항: `pyproject.toml`에 `requires-python = ">=3.10"`가 명시되어 있습니다.
- 의존성은 `pyproject.toml`의 `[project].dependencies`(torch/torchaudio/gradio/uvicorn/kui 등)에 정의되어 있습니다.
- `uv.lock`와 `[tool.uv]` 섹션이 존재하며, extra로 `cpu`, `cu126`, `cu128`, `cu129` 등의 선택지가 정의되어 있습니다. (`pyproject.toml`)

---

## 1) 로컬 설치 체크리스트

### Python 버전

- 최소 요구: `>=3.10` (`pyproject.toml`)

### 런타임에서 중요한 Python 패키지(발췌)

`tools/run_webui.py`, `tools/api_server.py` 관점에서 자주 등장하는 패키지:

- `torch`, `torchaudio` (모델 실행)
- `gradio` (WebUI)
- `uvicorn` (API 서버 런타임)
- `kui` + `ormsgpack` (API 서버 스택/직렬화)
- `pyrootutils` (프로젝트 루트 셋업: `.project-root` 지표 사용)

근거:
- `pyproject.toml`
- `tools/run_webui.py`
- `tools/api_server.py`

### 체크포인트 경로(기본값)

`tools/run_webui.py`와 `tools/server/api_utils.py`는 기본 체크포인트 경로를 다음과 같이 둡니다.

- `--llama-checkpoint-path`: 기본 `checkpoints/s2-pro`
- `--decoder-checkpoint-path`: 기본 `checkpoints/s2-pro/codec.pth`

근거:
- `tools/run_webui.py`
- `tools/server/api_utils.py`

---

## 2) 로컬 실행 엔트리(“커맨드 자체는 파일 기준”)

> 아래 커맨드는 “이 레포에 실제로 존재하는 엔트리 파일”을 실행하는 형태입니다.
> 단, Python 의존성 설치 방식(uv/pip 등)은 공식 문서에서 확정하는 것을 권장합니다.

### WebUI

`tools/run_webui.py`는 Gradio 앱을 구성하고 `app.launch()`를 호출합니다. (`tools/run_webui.py`, `tools/webui/*`)

```bash
python tools/run_webui.py --device cuda
```

CPU 강제는 `--device cpu`로 가능합니다. (`tools/run_webui.py`)

### API 서버

`tools/api_server.py`는 `uvicorn.run(...)`으로 ASGI 앱을 실행합니다. (`tools/api_server.py`)

```bash
python tools/api_server.py --listen 127.0.0.1:8080
```

---

## 주의사항/함정

- **체크포인트는 레포에 포함되지 않습니다.** Dockerfile 주석에도 이미지에 체크포인트가 없다고 명시되어 있습니다. (`docker/Dockerfile`)
- 로컬 실행 시에도 `checkpoints/` 디렉토리 구성(예: `s2-pro/codec.pth`)을 맞춰야 합니다. (기본값: `tools/run_webui.py`)
- GPU/CPU 설치는 torch/torchaudio 빌드(또는 wheel) 선택에 영향을 받으므로, 본문은 “파일로 확정 가능한 범위”까지만 다룹니다.

---

## TODO / 확인 필요

- 로컬 설치의 “권장 단일 명령(uv/pip/conda)”은 레포 내부 문서가 아니라 공식 문서 사이트로 안내됩니다. 최신/정확 커맨드는 다음 링크에서 확인하세요.
  - 설치: `docs/README.ko.md`가 `https://speech.fish.audio/ko/install/`로 안내

---

## 위키 링크

- `[[fish-speech Guide - Index]]` → [가이드 목차](/blog-repo/fish-speech-guide/)
- `[[fish-speech Guide - Docker]]` → [03. Docker로 실행](/blog-repo/fish-speech-guide-03-docker/)
- `[[fish-speech Guide - Inference]]` → [04. 추론(WebUI/서버)](/blog-repo/fish-speech-guide-04-inference/)

---

*다음 글에서는 `compose.yml`/`docker/Dockerfile`을 근거로 Docker 실행 루트를 정리합니다.*

