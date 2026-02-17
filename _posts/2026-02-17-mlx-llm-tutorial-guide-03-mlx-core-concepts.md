---
layout: post
title: "MLX LLM Tutorial 가이드 (03) - MLX 핵심 개념: Apple Silicon 최적화 포인트 이해하기"
date: 2026-02-17
permalink: /mlx-llm-tutorial-guide-03-mlx-core-concepts/
author: ddttom
categories: ['LLM 학습', 'MLX']
tags: [MLX, Apple Silicon, Unified Memory, Metal, JIT]
original_url: "https://github.com/ddttom/mlx-llm-tutorial/blob/main/public/docs/intro-to-mlx.md"
excerpt: "MLX 소개 문서를 바탕으로 Apple Silicon에서 MLX가 유리한 이유와 학습할 때 꼭 잡아야 할 개념을 정리합니다."
---

## MLX를 쓰는 이유

이 프로젝트가 MLX를 선택한 이유는 명확합니다.

- Apple Silicon 최적화
- NumPy 유사 API
- 자동미분/함수 변환(`grad`, `vmap`, `jit`)
- 로컬 환경에서 LLM 실습 가능

특히 Apple 하드웨어에서 CPU/GPU 메모리 복사 비용을 줄일 수 있다는 점을 핵심 장점으로 설명합니다.

---

## 학습자가 체감하는 장점

튜토리얼 관점에서 중요한 포인트는 다음입니다.

1. Python 인터페이스로 빠르게 실험 가능
2. 작은 모델 실습에서 성능/전력 효율 균형이 좋음
3. 로컬에서 바로 디버깅 가능

초기 학습 단계에서는 "대규모 분산 학습"보다 "작은 모델을 빠르게 돌려보며 구조를 이해"하는 쪽이 효율적입니다.

---

## 실습에서 바로 연결되는 개념

이후 코드에서 자주 보게 되는 MLX 요소:

- `mlx.core as mx`
- `mlx.nn as nn`
- `mlx.optimizers as optim`
- `mx.value_and_grad(...)`
- `mx.compile` (추론 최적화)

문서의 추상 개념이 `simple_llm.py`에서 그대로 구현되는 구조입니다.

---

## 프레임워크 선택 기준

이 저장소는 범용 프레임워크와의 비교도 제공합니다.

- PyTorch/TensorFlow/JAX와 비교하면
- MLX는 Apple Silicon 로컬 실습/제품화에 초점을 둔 선택지
- 대신 생태계 성숙도는 상대적으로 작은 편

다음 장에서는 LLM 아키텍처 문서로 넘어가 Transformer 기본기를 정리합니다.

