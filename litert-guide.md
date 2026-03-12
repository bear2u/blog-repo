---
layout: page
title: litert 가이드
permalink: /litert-guide/
icon: fas fa-book
---

# LiteRT 완벽 가이드

> Google's on-device framework for high-performance ML & GenAI deployment

GitHub Trending에 오른 **google-ai-edge/LiteRT**를 기반으로, 설치/빌드(도커)와 핵심 구성(런타임/가속/플랫폼 지원) 흐름을 단계적으로 정리합니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/blog-repo/litert-guide-01-intro/) | 목표/특징/레포 구조 |
| 02 | [소스 빌드(도커)](/blog-repo/litert-guide-02-build-with-docker/) | `docker_build/` 기준 빌드 |
| 03 | [빌드 시스템 개요](/blog-repo/litert-guide-03-build-systems/) | Bazel/CMake 경로 정리 |
| 04 | [런타임 구성요소](/blog-repo/litert-guide-04-runtime-architecture/) | `litert/`, `tflite/` 중심 |
| 05 | [운영/최적화/트러블슈팅](/blog-repo/litert-guide-05-ops-and-troubleshooting/) | 플랫폼/가속 이슈 체크리스트 |

## 관련 링크

- GitHub 저장소: https://github.com/google-ai-edge/LiteRT
- Get Started: https://ai.google.dev/edge/litert

