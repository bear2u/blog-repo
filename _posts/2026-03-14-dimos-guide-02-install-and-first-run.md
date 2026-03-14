---
layout: post
title: "dimos 완벽 가이드 (02) - 설치 및 첫 실행"
date: 2026-03-14
permalink: /dimos-guide-02-install-and-first-run/
author: dimensionalOS
categories: [로보틱스, dimos]
tags: [Trending, GitHub, dimos, Installation, Blueprints, Simulation]
original_url: "https://github.com/dimensionalOS/dimos"
excerpt: "공식 설치 스크립트와 OS별 설치 문서를 기준으로, Python 패키지 설치와 첫 runfile(리플레이/시뮬레이션/실기기) 실행 포인트를 정리합니다."
---

## 이 문서의 목적

- 설치가 “시스템 의존성(드라이버/FFmpeg/그래픽)”과 “파이썬 패키지”로 나뉘는 지점을 정리합니다.
- 하드웨어가 없어도 동작 가능한 `--replay` / `--simulation` 루트를 우선 안내합니다.

---

## 빠른 요약 (README/Docs 기반)

- 설치(대화형): `scripts/install.sh`를 curl 파이프로 실행하는 경로가 README에 있습니다. (`README.md`, `scripts/install.sh`)
- OS별 가이드: `docs/installation/ubuntu.md`, `docs/installation/nix.md`, `docs/installation/osx.md`
- 첫 실행 예: `dimos --replay run unitree-go2`, `dimos --simulation run unitree-go2` 등(“Featured Runfiles”) (`README.md`)

---

## 1) 시스템 설치(권장 경로)

문서가 제공하는 “시스템 의존성 설치” 가이드를 1차 기준으로 삼는 편이 안전합니다.

- Ubuntu: `docs/installation/ubuntu.md`
- NixOS/일반 Linux: `docs/installation/nix.md`
- macOS: `docs/installation/osx.md`
- 요구사항/티어: `docs/requirements.md`

---

## 2) Python 설치(패키지/extra)

`pyproject.toml`의 `project.optional-dependencies`를 보면 하드웨어/에이전트/웹/시뮬레이션 등 기능이 extra로 분리되어 있습니다. (`pyproject.toml`)

예(README에 등장하는 패턴):

- 기본 + 플랫폼: `dimos[base,unitree]`
- 시뮬레이션: `dimos[base,unitree,sim]`

> 실제 extra 이름/조합은 `pyproject.toml`을 기준으로 확인하는 것이 안전합니다.

---

## 3) 첫 실행: 리플레이/시뮬레이션/실기기

README의 Featured Runfiles 표는 “하드웨어 없이도 재현 가능한 루트”를 제공합니다. (`README.md`)

추천 시작 순서:

1. **Replay**: 녹화된 세션 재생(하드웨어 없이 가능) → `--replay`
2. **Simulation**: MuJoCo 기반 시뮬레이션 → `--simulation`
3. **Real Robot**: 네트워크/SDK/권한 설정 포함 → 실기기용 환경 변수/접속 필요

CLI 근거:

- `dimos` 스크립트 엔트리: `pyproject.toml` (`dimos = "dimos.robot.cli.dimos:main"`)
- CLI 구현: `dimos/robot/cli/dimos.py`

---

## 근거(파일/경로)

- 설치/예제: `README.md`
- 설치 스크립트: `scripts/install.sh`
- 설치 문서: `docs/installation/*`, `docs/requirements.md`
- 사용법: `docs/usage/cli.md`, `docs/usage/blueprints.md`
- 패키지/extra: `pyproject.toml`
- CLI: `dimos/robot/cli/dimos.py`

---

## 주의사항/함정

- 리플레이/데모는 Git LFS 다운로드가 걸릴 수 있어 “첫 실행이 느린” 문제가 생길 수 있습니다. (`README.md`, `docs/development/large_file_management.md`)
- 로봇 연결은 환경 변수(예: IP)나 드라이버/권한이 추가로 필요할 수 있습니다. 해당 플랫폼 문서/코드를 먼저 확인하세요. (`README.md`, `docs/platforms/*` 링크)

---

## TODO/확인 필요

- `docs/usage/configuration.md`와 `dimos/core/global_config.py`(글로벌 설정) 연결 정리
- “blueprint 이름 → 코드 위치”를 `dimos/robot/get_all_blueprints.py` 기준으로 색인화

---

## 위키 링크

- `[[dimos Guide - Index]]` → [가이드 목차](/blog-repo/dimos-guide/)
- `[[dimos Guide - Architecture]]` → [03. 아키텍처](/blog-repo/dimos-guide-03-architecture/)

---

*다음 글에서는 패키지 디렉토리(`dimos/*`)를 기준으로 모듈 지도를 만들고, CLI가 어떤 런타임을 조립하는지 정리합니다.*

