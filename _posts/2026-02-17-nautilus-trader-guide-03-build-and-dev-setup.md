---
layout: post
title: "NautilusTrader 가이드 (03) - 소스 빌드/개발 환경: rustup + clang + uv + Makefile"
date: 2026-02-17
permalink: /nautilus-trader-guide-03-build-and-dev-setup/
author: Nautech Systems
categories: [개발 도구, 퀀트 트레이딩]
tags: [NautilusTrader, Rust, Cython, PyO3, Makefile, Development]
original_url: "https://github.com/nautechsystems/nautilus_trader"
excerpt: "소스 설치는 Rust/C toolchain과 Python 환경을 함께 맞추는 작업입니다. 설치 문서와 Makefile을 기준으로 재현 가능한 개발 셋업 경로를 정리합니다."
---

## 왜 빌드가 복잡한가

NautilusTrader는 Python 패키지이지만 내부에:

- Rust 크레이트
- Cython/PyO3 바인딩

이 함께 들어있는 하이브리드 구조입니다. 그래서 단순 pip 설치보다 툴체인 정렬이 중요합니다.

---

## 소스 설치 흐름(요약)

설치 문서 기준 준비:

1. `rustup` 설치
2. `clang` 준비
3. `uv` 설치
4. 소스 클론 후 `uv sync --all-extras`

Linux/macOS에서는 PyO3 관련 환경 변수(`PYO3_PYTHON`, `PYTHONHOME`, 경우에 따라 `LD_LIBRARY_PATH`)를 맞춰야 Rust 테스트가 안정적으로 도는 케이스가 있습니다.

---

## Makefile 기반 개발 루프

README/Makefile에서 유용한 타깃:

- `make build` / `make build-debug`
- `make install` / `make install-debug`
- `make cargo-test`
- `make pytest`
- `make docs`
- `make pre-commit`

실무 팁:

- 코어 변경(Rust/Cython)이 잦으면 `build-debug` 루프가 체감이 좋습니다.
- CI와 동일한 검사(ruff/pre-commit/test)를 로컬에서 먼저 맞추는 편이 PR 속도를 높입니다.

---

## 기여 시 주의

`CONTRIBUTING.md` 기준:

- 먼저 이슈로 제안 정렬
- PR은 `develop` 브랜치 대상
- CLA 서명 필요

다음 장부터는 실제 사용자 관점 핵심인 백테스트 API 선택(고수준/저수준)을 다룹니다.

