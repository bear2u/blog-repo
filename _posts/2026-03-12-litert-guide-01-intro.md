---
layout: post
title: "LiteRT 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-12
permalink: /litert-guide-01-intro/
author: google-ai-edge
categories: [개발 도구, litert]
tags: [Trending, GitHub, litert, Edge, On-device, GitHub Trending]
original_url: "https://github.com/google-ai-edge/LiteRT"
excerpt: "TensorFlow Lite의 후속 런타임 LiteRT의 목표/특징/빌드 경로를 빠르게 정리합니다."
---

## LiteRT란?

GitHub Trending(daily, 2026-03-12 기준) 상위에 오른 **google-ai-edge/LiteRT**를 한국어로 정리합니다.

- **한 줄 요약(Trending 표시)**: LiteRT, successor to TensorFlow Lite. is Google's On-device framework for high-performance ML & GenAI deployment on edge platforms, via efficient conversion, runtime, and optimization
- **언어(Trending 표시)**: C++
- **오늘 스타(Trending 표시)**: +6
- **원본**: https://github.com/google-ai-edge/LiteRT

---

## 이 문서의 목적

- LiteRT가 README에서 강조하는 목표(온디바이스 ML/GenAI 런타임)와 “무엇이 새로워졌는지”를 개요로 잡습니다.
- 소스 빌드(도커) 경로와 주요 디렉토리(런타임/문서/빌드 시스템)를 다음 챕터로 연결합니다.

---

## 빠른 요약 (README 기반)

- LiteRT는 TensorFlow Lite의 레거시를 이어가는 온디바이스 런타임을 지향합니다. (`README.md`)
- “New Compiled Model API”, GPU/NPU 가속, 플랫폼 지원 범위를 README에서 강조합니다. (`README.md`)
- 소스 빌드는 `docker_build/` 아래의 스크립트 실행으로 안내됩니다. (`README.md`, `docker_build/`)

---

## 바로 시작하기(README 링크)

- Get Started guide: https://ai.google.dev/edge/litert

---

## 근거(파일/경로)

- 개요/특징/설치 가이드: `README.md`
- 빌드/워크스페이스: `BUILD`, `WORKSPACE`, `configure.py`
- 런타임/구현: `litert/`, `tflite/`
- 도커 빌드: `docker_build/`
- 문서: `g3doc/`
- 의존성: `third_party/`

---

## 레포 구조(상위)

```text
LiteRT/
  litert/
  tflite/
  g3doc/
  docker_build/
  third_party/
  BUILD
  WORKSPACE
```

---

## 위키 링크

- `[[LiteRT Guide - Index]]` → [가이드 목차](/blog-repo/litert-guide/)

---

*다음 글에서는 도커 빌드 스크립트를 중심으로 “가장 짧은 소스 빌드 루트”를 정리합니다.*

