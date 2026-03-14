---
layout: post
title: "Heretic 가이드 (03) - 아키텍처: Analyzer/Model/Evaluator/Settings"
date: 2026-03-14
permalink: /heretic-guide-03-architecture/
author: p-e-w
categories: [AI, LLM 안전]
tags: [Trending, GitHub, heretic, Architecture, Optuna, Transformers]
original_url: "https://github.com/p-e-w/heretic"
excerpt: "Heretic의 핵심 구성요소(설정 스키마, 모델 래퍼, 분석/평가, Optuna 기반 최적화)를 코드 파일 기준으로 분해합니다."
---

## 안전 고지(요약)

오남용을 조장하는 실행/튜닝 절차는 다루지 않습니다. 이 문서는 “코드 구조 이해”가 목적입니다. (`README.md`)

---

## 이 문서의 목적

- 주요 클래스를 파일 단위로 나누고, 서로 어떤 데이터(메트릭/파라미터)를 주고받는지 개략도를 만듭니다.
- 이후(04~05) 챕터에서 “최적화/재현성/방어적 활용”을 다루기 위한 공통 용어를 고정합니다.

---

## 핵심 구성요소(코드 근거)

- 설정: `src/heretic/config.py` (`Settings`, `DatasetSpecification`, Enum들)
- 메인 오케스트레이션: `src/heretic/main.py` (`run()`, Optuna 스터디/저널링/디바이스 점검)
- 모델 계층: `src/heretic/model.py` (모델 래퍼/파라미터 구조/모델 클래스 선택)
- 분석/지표: `src/heretic/analyzer.py`
- 평가: `src/heretic/evaluator.py`
- 유틸: `src/heretic/utils.py`

---

## (개략) 데이터 흐름

```mermaid
flowchart TB
  Settings[Settings\n(config.py)] --> Main[run()\n(main.py)]
  Main --> Load[Model load\n(model.py)]
  Main --> Study[Optuna Study\n(main.py)]
  Study --> Trial[Trial loop]
  Trial --> Ana[Analyzer\n(analyzer.py)]
  Trial --> Eval[Evaluator\n(evaluator.py)]
  Ana --> Metrics[Metrics]
  Eval --> Metrics
  Metrics --> Study
```

---

## 읽을 때의 포인트(보안/재현성)

- **입력 고정**: 어떤 프롬프트/데이터를 쓰는지가 결과를 크게 좌우합니다. “데이터셋 스펙”이 `DatasetSpecification`으로 모델링되어 있는 점을 먼저 확인하세요. (`src/heretic/config.py`)
- **자원/장비 의존**: 디바이스/메모리 출력과 로딩 전략(dtypes/quantization 등)이 실행 초기에 크게 관여합니다. (`src/heretic/main.py`, `src/heretic/config.py`)
- **실험 추적**: Optuna의 journaling storage를 사용해 진행 상황을 체크포인트로 남기는 구조가 보입니다. (`src/heretic/main.py`)

---

## 근거(파일/경로)

- 메인: `src/heretic/main.py`
- 설정: `src/heretic/config.py`, `config.default.toml`
- 모델/평가/분석: `src/heretic/model.py`, `src/heretic/evaluator.py`, `src/heretic/analyzer.py`

---

## 주의사항/함정

- “동일한 실험”이 하드웨어/드라이버/라이브러리 버전에 따라 재현이 어려울 수 있습니다. 문서화할 때는 최소한 `pyproject.toml`(의존성)과 장비 정보(로그)를 함께 남기는 편이 안전합니다.
- 라이선스(AGPL)는 네트워크 제공 형태의 사용(서비스 제공)에서 특히 의무가 생길 수 있어, 연구 외 적용은 사전에 검토가 필요합니다. (`LICENSE`)

---

## TODO/확인 필요

- `src/heretic/model.py`에서 “지원하는 모델 타입/아키텍처”를 어떤 방식으로 분기하는지(예: `get_model_class`) 확인
- `src/heretic/evaluator.py`가 생성하는 지표(예: KL divergence/거부 카운트 등)의 정의를 코드 근거로 표준화

---

## 위키 링크

- `[[Heretic Guide - Index]]` → [가이드 목차](/blog-repo/heretic-guide/)
- `[[Heretic Guide - Optimization Pipeline]]` → [04. 최적화 파이프라인(개념)](/blog-repo/heretic-guide-04-optimization-pipeline/)

---

*다음 글에서는 Optuna(TPE) 기반 최적화가 어떤 단계로 진행되는지, 오남용 없이 “실험 파이프라인” 관점에서만 정리합니다.*

