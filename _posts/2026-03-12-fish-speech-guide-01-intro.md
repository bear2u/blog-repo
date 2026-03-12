---
layout: post
title: "fish-speech 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-12
permalink: /fish-speech-guide-01-intro/
author: fishaudio
categories: [AI, fish-speech]
tags: [Trending, GitHub, fish-speech, TTS, Speech, GitHub Trending]
original_url: "https://github.com/fishaudio/fish-speech"
excerpt: "SOTA 오픈소스 TTS(Fish Audio S2) 프로젝트 fish-speech의 문서/구성과 시작 지점을 정리합니다."
---

## fish-speech(Fish Audio S2)란?

GitHub Trending(daily, 2026-03-12 기준) 상위에 오른 **fishaudio/fish-speech**를 한국어로 정리합니다.

- **한 줄 요약(Trending 표시)**: SOTA Open Source TTS
- **언어(Trending 표시)**: Python
- **오늘 스타(Trending 표시)**: +630
- **원본**: https://github.com/fishaudio/fish-speech

---

## 이 문서의 목적

- “어디서부터 시작해야 하는지”를 빠르게 찾을 수 있도록 공식 문서/가이드 링크를 한 곳에 모읍니다.
- 레포에서 제공하는 실행/배포 단서(예: Docker, WebUI/서버 추론)를 다음 챕터 작업 범위로 정의합니다.

---

## 빠른 요약 (docs/README.ko 기반)

- 프로젝트는 **Fish Audio S2**(텍스트→음성 변환) 모델/툴링을 다룹니다. (`docs/README.ko.md`)
- 설치/추론/서버/WebUI/Docker 가이드는 **공식 문서 사이트**로 안내됩니다. (`docs/README.ko.md`)
- 법적/라이선스 고지사항이 명시되어 있으므로, 활용 전 반드시 확인이 필요합니다. (`LICENSE`, `docs/README.ko.md`)

---

## 바로 시작하기(공식 문서)

- 설치: https://speech.fish.audio/ko/install/
- 커맨드라인 추론: https://speech.fish.audio/ko/inference/
- 서버 추론: https://speech.fish.audio/ko/server/
- Docker 설정: https://speech.fish.audio/ko/install/

---

## 근거(파일/경로)

- 한국어 개요/빠른 시작/고지: `docs/README.ko.md`
- 패키징/의존성 단서: `pyproject.toml`, `uv.lock`
- 주요 코드 위치: `fish_speech/`
- Docker 관련: `docker/`, `compose.yml`, `Dockerfile*`

---

## 레포 구조(상위)

```text
fish-speech/
  docs/
  fish_speech/
  docker/
  pyproject.toml
  compose.yml
  mkdocs.yml
```

---

## 위키 링크

- `[[fish-speech Guide - Index]]` → [가이드 목차](/blog-repo/fish-speech-guide/)

---

*다음 글에서는 “로컬 설치 vs Docker” 관점에서 가장 짧은 실행 루트를 정리합니다.*

