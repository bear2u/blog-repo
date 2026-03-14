---
layout: post
title: "Heretic 가이드 (01) - 소개 및 안전 고지"
date: 2026-03-14
permalink: /heretic-guide-01-intro-and-safety/
author: p-e-w
categories: [AI, LLM 안전]
tags: [Trending, GitHub, heretic, Safety, Research]
original_url: "https://github.com/p-e-w/heretic"
excerpt: "Heretic 프로젝트의 목표/기술적 개요를 소개하되, 오남용을 방지하기 위해 연구·보안 관점에서 다루는 범위와 주의사항을 먼저 명확히 합니다."
---

## 안전 고지(필독)

**Heretic**은 README에서 “언어 모델의 safety alignment(안전 정렬) 제거”를 목적으로 한다고 설명합니다. 이런 도구는 악용될 경우 유해 콘텐츠 생성, 정책/약관 위반, 사용자 피해로 이어질 수 있습니다. (`README.md`)

따라서 이 시리즈는:

- 코드/아키텍처 이해, 재현성 있는 실험 설계, 방어/평가 관점의 체크리스트를 제공합니다.
- 반대로, 안전장치 우회/무력화를 실제로 수행하는 데 필요한 단계별 실행 절차, 파라미터 튜닝 가이드, 배포 방법 등은 다루지 않습니다.

---

## 이 문서의 목적

- 프로젝트가 “무엇을 한다고 주장하는지”를 README/코드 근거로 요약합니다.
- 이후 챕터에서 다룰 “코드베이스 지도/아키텍처”의 범위를 정합니다.

---

## 빠른 요약 (README/pyproject 기반)

- 프로젝트 성격: 콘솔 도구(스크립트 엔트리)로 제공됩니다. (`pyproject.toml`의 `[project.scripts]`)
- 주요 기술 키워드: directional ablation(abliteration), Optuna 기반 파라미터 최적화(TPE) 등을 언급합니다. (`README.md`)
- 코드 위치: `src/heretic/*`에 구현이 모여 있습니다. (`src/heretic/main.py` 등)

---

## “무엇을 분석할 것인가” (코드 관점)

이 레포를 연구/보안 관점으로 볼 때, 최소한 아래 축을 분리해서 읽는 편이 안전합니다.

- 설정/입력 스키마: `src/heretic/config.py`, `config.default.toml`
- 파이프라인(진입점): `src/heretic/main.py`
- 모델 래퍼/변환 로직: `src/heretic/model.py`
- 분석/지표: `src/heretic/analyzer.py`, `src/heretic/evaluator.py`

---

## 근거(파일/경로)

- 프로젝트 설명/의도/한계: `README.md`
- 패키징/엔트리포인트: `pyproject.toml`
- 설정 템플릿: `config.default.toml`, `config.noslop.toml`
- 구현: `src/heretic/*` (예: `main.py`, `config.py`, `model.py`, `analyzer.py`, `evaluator.py`)

---

## 주의사항/함정

- 이 프로젝트는 본질적으로 안전 정책을 약화시키는 방향으로 악용될 수 있습니다. 조직/팀/개인의 윤리/컴플라이언스를 우선 확인하세요.
- 라이선스(AGPL) 조건을 위반하면 법적 리스크가 생길 수 있습니다. 배포/서비스 적용은 특히 주의가 필요합니다. (`LICENSE`, `pyproject.toml`)

---

## TODO/확인 필요

- “평가 데이터/프롬프트 로딩”이 어떤 파일/경로를 사용하는지(로컬 파일 vs 내장) `src/heretic/utils.py`에서 확인하기
- “결과물 저장/병합 전략”이 어떤 형식으로 산출되는지 `src/heretic/main.py`의 저장 로직 기준으로 확인하기

---

## 위키 링크

- `[[Heretic Guide - Index]]` → [가이드 목차](/blog-repo/heretic-guide/)
- `[[Heretic Guide - Repo Structure & Config]]` → [02. 레포 구조 & 설정 스키마](/blog-repo/heretic-guide-02-repo-structure-and-config/)

---

*다음 글에서는 `src/heretic/*`와 `config.default.toml`을 “설정 스키마/구성요소 지도” 관점으로 정리합니다.*

