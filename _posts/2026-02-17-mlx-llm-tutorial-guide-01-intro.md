---
layout: post
title: "MLX LLM Tutorial 가이드 (01) - 소개/학습 경로: Apple Silicon에서 LLM 실습 시작하기"
date: 2026-02-17
permalink: /mlx-llm-tutorial-guide-01-intro/
author: ddttom
categories: ['LLM 학습', 'Apple Silicon']
tags: [MLX, LLM, Tutorial, Apple Silicon, Python]
original_url: "https://github.com/ddttom/mlx-llm-tutorial"
excerpt: "mlx-llm-tutorial 저장소의 목표, 구성, 학습 순서를 정리하고 어떤 범위까지 구현돼 있는지 먼저 잡아봅니다."
---

## 이 저장소가 다루는 범위

이 프로젝트는 "MLX로 LLM을 이해하고 직접 만져보는 입문~중급 튜토리얼"에 가깝습니다.

- `public/docs/`: 설치, MLX 개요, LLM 아키텍처 문서
- `tutorials/`: 단계별 튜토리얼 문서
- `code/`: 실제 실행 스크립트(`simple_llm.py`, `finetune_llm.py`)
- `code/web_interface/`: Flask + JS 데모 UI

핵심은 Apple Silicon(M1/M2/M3) 환경에서 로컬 학습/추론 흐름을 직접 체험하는 것입니다.

---

## 먼저 알아둘 점

문서 전체에서 반복되는 운영 원칙이 있습니다.

1. 시스템 기본 폴더(Downloads/Documents)에서 직접 작업하지 않기
2. 전용 작업 폴더(예: `~/ai-training`)를 만들어 관리하기
3. Miniconda 기반 가상환경(`mlx-env`)으로 의존성 고정하기

튜토리얼용 레포지만, 환경 분리 원칙을 강조하는 이유는 대용량 파일/패키지 충돌 문제를 줄이기 위해서입니다.

---

## 추천 학습 순서

이 시리즈는 아래 순서로 보는 것이 효율적입니다.

1. 설치/환경 구성
2. MLX 핵심 개념
3. Transformer 기본 구조
4. `simple_llm.py` 데이터 준비 + 모델 학습
5. 텍스트 생성/최적화
6. `finetune_llm.py` 파이프라인
7. 웹 인터페이스 연동

이 순서를 따르면 "이론 문서 -> 단일 스크립트 구현 -> 확장(파인튜닝/웹)"으로 자연스럽게 이어집니다.

---

## 현재 구현 수준 한 줄 요약

- `simple_llm.py`: 동작 가능한 교육용 문자 단위 LM 예제
- `finetune_llm.py`: 구조는 갖췄지만 일부 로딩/호환은 추가 구현 필요
- 웹 UI: 데모 중심이며 시각화 데이터 일부는 더미 기반

다음 장부터 설치와 환경 구성부터 바로 시작합니다.

