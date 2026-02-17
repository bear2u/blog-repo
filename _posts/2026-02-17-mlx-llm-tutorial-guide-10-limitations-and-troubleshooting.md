---
layout: post
title: "MLX LLM Tutorial 가이드 (10) - 한계/트러블슈팅: 실전 적용 전 체크리스트"
date: 2026-02-17
permalink: /mlx-llm-tutorial-guide-10-limitations-and-troubleshooting/
author: ddttom
categories: ['LLM 학습', '트러블슈팅']
tags: [Troubleshooting, Limitations, MLX, Fine Tuning, Web Interface]
original_url: "https://github.com/ddttom/mlx-llm-tutorial/blob/main/projectstate.md"
excerpt: "이 저장소를 학습용에서 실전용으로 확장할 때 부딪히는 제약과 우선 보강 포인트를 체크리스트로 정리합니다."
---

## 학습용 레포로서 강점

- 설치/이론/코드/웹 데모가 한 레포에 정리됨
- `simple_llm.py`는 end-to-end 흐름이 명확함
- Apple Silicon 사용자에게 적합한 실습 출발점

---

## 바로 실전에 쓰기 어려운 지점

1. 파인튜닝 가중치 로딩 경로 일부 미구현
2. 웹 시각화 데이터가 더미 중심
3. 문서에서 언급된 일부 디렉토리(예: notebooks)가 현재 트리에 없음
4. 대규모 모델/데이터에 대한 운영 가이드 부족

따라서 "개념 학습 + 프로토타이핑"에는 적합하지만, 배포형 서비스 용도로는 추가 개발이 필요합니다.

---

## 자주 만나는 문제와 점검 순서

### 1) `mlx` import 오류

- `conda activate mlx-env`
- `python -c "import mlx; print(mlx.__version__)"`

### 2) 모델이 웹에서 안 보임

- `simple_llm_model.npz`, `tokenizer.json` 위치 확인
- `python server.py --model-dir ...` 경로 재확인

### 3) 파인튜닝 결과가 불안정함

- 데이터 품질 점검(지시문/응답 형식 일관성)
- 배치 크기/러닝레이트 축소
- 모델 호환성(토크나이저/가중치 포맷) 확인

---

## 실전 확장을 위한 우선순위

1. HF 가중치 로딩/변환 로직 완성
2. 실제 attention 추출 기반 시각화로 교체
3. 학습/평가 스크립트 분리 및 실험 로그 체계화
4. 배포 시나리오(서비스 API, 모델 버전 관리) 문서화

이 체크리스트를 기준으로 확장하면, 학습용 레포에서 실무형 베이스라인으로 전환하기 쉽습니다.

