---
layout: post
title: "Heretic 가이드 (02) - 레포 구조 & 설정 스키마(config)"
date: 2026-03-14
permalink: /heretic-guide-02-repo-structure-and-config/
author: p-e-w
categories: [AI, LLM 안전]
tags: [Trending, GitHub, heretic, Architecture, Config]
original_url: "https://github.com/p-e-w/heretic"
excerpt: "Heretic의 코드가 모여 있는 src/heretic/* 구조와, Settings(config) 스키마/입력 경로를 ‘읽기용 지도’로 정리합니다."
---

## 안전 고지(요약)

이 시리즈는 연구/보안 관점의 코드 분석을 목적으로 하며, 안전장치 우회/무력화를 실제로 수행하는 실행 가이드는 제공하지 않습니다. (`README.md`)

---

## 이 문서의 목적

- `src/heretic/*`에서 어떤 파일을 먼저 봐야 하는지 정리합니다.
- 설정이 어디에서 로딩되고(경로/우선순위), 어떤 “축”으로 구성되는지 스키마 수준에서 파악합니다.

---

## 레포 구조(상위)

```text
heretic/
  README.md
  pyproject.toml
  src/heretic/           # 구현
  config.default.toml    # 설정 템플릿(예시)
  config.noslop.toml     # 대체 템플릿(예시)
```

구현 파일(1차 탐색 추천):

- 진입점/오케스트레이션: `src/heretic/main.py`
- 설정 스키마: `src/heretic/config.py`
- 모델 래퍼/파라미터 정의: `src/heretic/model.py`
- 분석: `src/heretic/analyzer.py`
- 평가: `src/heretic/evaluator.py`
- 유틸: `src/heretic/utils.py`

---

## 설정 스키마: `Settings`(Pydantic Settings)

`src/heretic/config.py`는 `Settings(BaseSettings)`로 설정 스키마를 정의합니다. 요지는 다음입니다.

- 모델 식별자/경로 입력 필드가 존재합니다. (`Settings.model`)
- 성능/자원/로딩(예: dtype, quantization, device_map 등) 관련 필드가 존재합니다.
- 실험 파이프라인(예: trial 수, 체크포인트 저장, 평가 관련) 필드가 존재합니다.
- 설정은 “파일(TOML) + 환경 변수 + CLI” 형태로 유입될 수 있도록 `pydantic_settings` 소스를 구성합니다. (`TomlConfigSettingsSource`, `EnvSettingsSource`, `CliSettingsSource`)

> 구체적인 값/데이터셋/마커 목록은 오남용 소지가 있어 이 문서에 재현하지 않습니다. 필요하다면 원본 파일을 직접 확인하세요. (`config.default.toml`)

---

## (개략) 설정 → 실행 파이프라인(읽기용)

```mermaid
flowchart LR
  In[Config Sources\nTOML/ENV/CLI] --> S[Settings\n(src/heretic/config.py)]
  S --> Main[run()\n(src/heretic/main.py)]
  Main --> Model[Model\n(src/heretic/model.py)]
  Main --> Opt[Optuna Study\n(TPE/JournalStorage)]
  Opt --> Trial[Trials]
  Trial --> Eval[Evaluator\n(src/heretic/evaluator.py)]
  Trial --> Ana[Analyzer\n(src/heretic/analyzer.py)]
```

---

## 근거(파일/경로)

- 프로젝트 의도/주의: `README.md`
- 패키징/엔트리포인트: `pyproject.toml` (`heretic = "heretic.main:main"`)
- 설정 스키마: `src/heretic/config.py`
- 실행 오케스트레이션: `src/heretic/main.py`
- 설정 템플릿: `config.default.toml`, `config.noslop.toml`

---

## 주의사항/함정

- 설정은 “실험 재현성”과 “오남용 리스크”를 동시에 좌우합니다. 조직 정책(연구윤리/데이터/배포)과 분리해서 생각하면 사고가 납니다.
- 계산 비용/장비 의존성이 크므로, 실험 설계(로그/체크포인트/시드 관리)가 핵심입니다. (`src/heretic/main.py`에서 Optuna/Journaling 사용)

---

## TODO/확인 필요

- `Settings.settings_customise_sources(...)`(또는 동등 로직)가 실제로 어떤 경로에서 TOML을 찾는지 확인
- `src/heretic/utils.py`의 프롬프트/데이터 로딩 동작(로컬 파일 vs HF dataset)을 코드 근거로 정리

---

## 위키 링크

- `[[Heretic Guide - Index]]` → [가이드 목차](/blog-repo/heretic-guide/)
- `[[Heretic Guide - Architecture]]` → [03. 아키텍처](/blog-repo/heretic-guide-03-architecture/)

---

*다음 글에서는 Analyzer/Model/Evaluator의 역할 분리를 기준으로 전체 아키텍처를 더 구체적으로 그립니다.*

