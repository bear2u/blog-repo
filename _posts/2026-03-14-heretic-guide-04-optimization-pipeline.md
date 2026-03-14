---
layout: post
title: "Heretic 가이드 (04) - 최적화 파이프라인(개념): Optuna TPE/Journaling"
date: 2026-03-14
permalink: /heretic-guide-04-optimization-pipeline/
author: p-e-w
categories: [AI, LLM 안전]
tags: [Trending, GitHub, heretic, Optuna, Reproducibility]
original_url: "https://github.com/p-e-w/heretic"
excerpt: "Heretic가 사용하는 Optuna 기반 최적화(TPE, trial pruning, journal checkpoint)를 ‘실험 파이프라인’ 관점에서만 정리합니다."
---

## 안전 고지(요약)

이 문서는 오남용을 유발할 수 있는 파라미터 튜닝/실행 절차를 제공하지 않습니다. 코드를 읽고 실험을 “추적/재현”하는 구조를 이해하는 것이 목적입니다. (`README.md`)

---

## 이 문서의 목적

- Optuna 기반 최적화가 코드에서 어떤 구성요소(Study/Trial/Storage/Sampler)로 표현되는지 파악합니다.
- 실험 추적(체크포인트/로그) 관점에서 “재현성 레일”을 만들 포인트를 제시합니다.

---

## 코드에서 보이는 최적화 구성요소(근거)

`src/heretic/main.py`는 Optuna 관련 구성요소를 import/use 합니다.

- Sampler: `TPESampler`
- Storage: `JournalStorage`, `JournalFileBackend`, `JournalFileOpenLock`
- Trial 상태/프루닝: `Trial`, `TrialPruned`, `TrialState`

> 어떤 하이퍼파라미터가 최적화 대상인지(예: AbliterationParameters 등)는 `src/heretic/model.py` 쪽 정의를 확인하는 것이 안전합니다.

---

## (개략) Study/Trial 루프 흐름

```mermaid
flowchart LR
  Start[run()\n(main.py)] --> Settings[Settings\n(config.py)]
  Settings --> Study[Optuna Study\n(main.py)]
  Study --> Trial[Trial]
  Trial --> Params[Sample parameters\n(model.py)]
  Params --> Eval[Evaluate metrics\n(evaluator.py)]
  Eval --> Obj[Objective value]
  Obj --> Study
  Study --> Checkpoint[Journal checkpoint\n(main.py)]
```

---

## 재현성 관점 체크리스트

- **설정 스냅샷**: 실행에 사용한 설정(파일/환경/CLI)을 그대로 저장할 것 (`src/heretic/config.py`)
- **의존성 스냅샷**: `pyproject.toml`/lockfile(`uv.lock`)를 함께 남길 것
- **스토리지/체크포인트**: journaling storage 경로/잠금 방식이 포함되어 있으니, 실험을 중단/재개할 수 있는지 확인할 것 (`src/heretic/main.py`)
- **장비/드라이버 로그**: 실행 초기에 디바이스 정보 출력이 존재하므로, 로그에 보존할 것 (`src/heretic/main.py`)

---

## 근거(파일/경로)

- 최적화/저장/프루닝 단서: `src/heretic/main.py`
- 설정 스키마: `src/heretic/config.py`
- 파라미터 구조/모델 래퍼: `src/heretic/model.py`
- 평가/지표: `src/heretic/evaluator.py`, `src/heretic/analyzer.py`
- 의존성/락: `pyproject.toml`, `uv.lock`

---

## 주의사항/함정

- 실험 파이프라인은 자원/시간 비용이 커서, “작게 시작 → 로그/체크포인트 검증 → 확장” 순서가 안전합니다.
- journaling 기반 스토리지는 파일 락/파일 시스템 특성(예: NFS) 영향을 받을 수 있습니다. (`JournalFileOpenLock` import 근거)

---

## TODO/확인 필요

- `src/heretic/main.py`에서 objective를 구성하는 지표(예: 거부/거리 기반 스코어)의 정확한 정의 추적
- `src/heretic/utils.py`의 캐시/메모리 정리(`empty_cache`)가 어떤 상황에서 호출되는지 확인

---

## 위키 링크

- `[[Heretic Guide - Index]]` → [가이드 목차](/blog-repo/heretic-guide/)
- `[[Heretic Guide - Defensive Use]]` → [05. 방어적 활용 & 재현성](/blog-repo/heretic-guide-05-defensive-use-and-reproducibility/)

---

*다음 글에서는 이 레포를 “안전 연구/방어” 관점에서만 활용하려면 어떤 가드레일이 필요한지 정리합니다.*

