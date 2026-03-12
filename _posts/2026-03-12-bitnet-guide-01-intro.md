---
layout: post
title: "BitNet 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-12
permalink: /bitnet-guide-01-intro/
author: microsoft
categories: [AI, bitnet]
tags: [Trending, GitHub, bitnet, LLM, Inference, GitHub Trending]
original_url: "https://github.com/microsoft/BitNet"
excerpt: "1-bit LLM(BitNet b1.58)용 공식 추론 프레임워크 bitnet.cpp를 개요부터 정리합니다."
---

## BitNet(bitnet.cpp)이란?

GitHub Trending(daily, 2026-03-12 기준) 상위에 오른 **microsoft/BitNet**을 한국어로 정리합니다.

- **한 줄 요약(Trending 표시)**: Official inference framework for 1-bit LLMs
- **언어(Trending 표시)**: Python
- **오늘 스타(Trending 표시)**: +2,149
- **원본**: https://github.com/microsoft/BitNet

---

## 이 문서의 목적

- BitNet 레포(README/코드 구조)에서 드러나는 “무엇을 위한 프로젝트인지”와 “어떤 구성으로 되어 있는지”를 빠르게 잡습니다.
- 다음 챕터에서 설치/빌드/실행(예: CPU/GPU)로 넘어가기 위한 기준점(링크/경로)을 마련합니다.

---

## 빠른 요약 (README 기반)

- BitNet 레포의 핵심은 **`bitnet.cpp`(1-bit LLM 추론 프레임워크)** 입니다. (`README.md`)
- CPU/GPU용 커널/구현과 관련 문서가 분리되어 있습니다. (`src/`, `gpu/`, `preset_kernels/`)
- “직접 실행”을 위한 엔트리 스크립트가 함께 제공됩니다. (`run_inference.py`, `run_inference_server.py`)

---

## 근거(파일/경로)

- 개요/목표/링크: `README.md`
- 빌드 단서: `CMakeLists.txt`, `requirements.txt`
- CPU 구현/가이드: `src/`
- GPU 가이드/구현: `gpu/`
- 실행 스크립트: `run_inference.py`, `run_inference_server.py`, `setup_env.py`

---

## 레포 구조(상위)

```text
BitNet/
  src/
  include/
  gpu/
  preset_kernels/
  utils/
  docs/
  run_inference.py
  run_inference_server.py
  CMakeLists.txt
  requirements.txt
```

---

## (다음 챕터 예고) 설치/빌드/실행에서 확인할 것

- README가 안내하는 CPU/GPU 빌드 경로(문서 링크)와 실제 소스/스크립트가 어떻게 매칭되는지
- 로컬에서 재현 가능한 “최소 실행 루트” (예: `run_inference.py` 기준)

---

## 위키 링크

- `[[BitNet Guide - Index]]` → [가이드 목차](/blog-repo/bitnet-guide/)

---

*다음 글에서는 설치/빌드(환경 준비 포함) 흐름을 정리합니다.*

