---
layout: page
title: MLX LLM Tutorial 가이드
permalink: /mlx-llm-tutorial-guide/
icon: fas fa-microchip
---

# MLX LLM Tutorial 완벽 가이드

> **Apple Silicon에서 MLX로 LLM을 직접 만들고, 미세조정하고, 웹 인터페이스까지 연결하는 학습형 저장소 해설**

**mlx-llm-tutorial**은 Apple의 MLX 프레임워크를 기반으로, LLM의 핵심 개념부터 간단한 구현, 파인튜닝 스크립트, Flask 웹 데모까지 단계적으로 보여주는 튜토리얼 레포입니다.

- 원문 저장소: https://github.com/ddttom/mlx-llm-tutorial
- 설치 문서: https://github.com/ddttom/mlx-llm-tutorial/blob/main/public/docs/installation.md
- 아키텍처 문서: https://github.com/ddttom/mlx-llm-tutorial/blob/main/public/docs/llm-architecture.md

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개/학습 경로](/blog-repo/mlx-llm-tutorial-guide-01-intro/) | 저장소 목적, 폴더 구조, 추천 학습 순서 |
| 02 | [설치/환경 구성](/blog-repo/mlx-llm-tutorial-guide-02-installation-and-environment/) | Miniconda, `mlx-env`, 의존성 설치 |
| 03 | [MLX 핵심 개념](/blog-repo/mlx-llm-tutorial-guide-03-mlx-core-concepts/) | Apple Silicon 최적화 포인트와 MLX 기본기 |
| 04 | [LLM 아키텍처 기초](/blog-repo/mlx-llm-tutorial-guide-04-llm-architecture-basics/) | Transformer 핵심 구성과 디코더 관점 |
| 05 | [데이터/토크나이저](/blog-repo/mlx-llm-tutorial-guide-05-simple-llm-data-and-tokenizer/) | `simple_llm.py`의 데이터 준비 파이프라인 |
| 06 | [모델/학습 루프](/blog-repo/mlx-llm-tutorial-guide-06-simple-llm-model-and-training/) | 어텐션/블록 구조, 손실 계산, Adam 학습 |
| 07 | [생성/추론 최적화](/blog-repo/mlx-llm-tutorial-guide-07-generation-and-inference-optimization/) | temperature 샘플링, `mx.compile` 기반 최적화 |
| 08 | [파인튜닝 파이프라인](/blog-repo/mlx-llm-tutorial-guide-08-finetuning-pipeline/) | `finetune_llm.py` CLI, 데이터 포맷, 제약 사항 |
| 09 | [웹 인터페이스](/blog-repo/mlx-llm-tutorial-guide-09-web-interface/) | Flask API, 프런트엔드 연동, 데모 동작 |
| 10 | [한계/트러블슈팅](/blog-repo/mlx-llm-tutorial-guide-10-limitations-and-troubleshooting/) | 현재 구현 한계와 실전 개선 체크리스트 |
