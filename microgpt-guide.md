---
layout: page
title: microgpt.py 코드 해설 가이드
permalink: /microgpt-guide/
icon: fas fa-code
---

# microgpt.py 코드 해설 가이드

> **의존성 없이 순수 Python으로 GPT 학습/추론을 구현한 199줄짜리 알고리즘을 줄 단위로 해부하는 시리즈**

이 시리즈는 Andrej Karpathy의 `microgpt.py`를 기준으로, 데이터 준비부터 오토그라드, 어텐션, 학습 루프, 샘플링까지 코드 흐름을 그대로 따라가며 설명합니다.

- 원문 코드: https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개/전체 구조 지도](/blog-repo/microgpt-guide-01-intro-and-code-map/) | 199줄 코드가 어떻게 GPT 전체 파이프라인을 담는지 |
| 02 | [데이터셋/문자 토크나이저](/blog-repo/microgpt-guide-02-dataset-and-character-tokenizer/) | `input.txt`, `docs`, `uchars`, BOS 토큰 설계 |
| 03 | [Value 클래스 오토그라드](/blog-repo/microgpt-guide-03-value-class-and-autograd/) | 계산 그래프와 backward 전파 핵심 |
| 04 | [파라미터 초기화/state_dict](/blog-repo/microgpt-guide-04-parameter-init-and-state-dict/) | tiny GPT 파라미터를 순수 Python 객체로 보관 |
| 05 | [linear/softmax/rmsnorm 유틸](/blog-repo/microgpt-guide-05-linear-softmax-and-rmsnorm/) | forward를 구성하는 최소 수학 함수 |
| 06 | [gpt 함수 (어텐션 블록)](/blog-repo/microgpt-guide-06-gpt-function-attention-block/) | QKV 계산, head 분할, causal 누적 |
| 07 | [gpt 함수 (MLP/Residual/Logits)](/blog-repo/microgpt-guide-07-gpt-function-mlp-residual-and-logits/) | MLP 블록과 최종 로짓 계산 |
| 08 | [학습 루프/손실 계산](/blog-repo/microgpt-guide-08-training-loop-and-loss/) | next-token loss를 문서 단위로 평균하는 방식 |
| 09 | [Adam 업데이트/스케줄](/blog-repo/microgpt-guide-09-adam-update-and-schedule/) | 1st/2nd moment, bias correction, lr decay |
| 10 | [추론/샘플링/확장 포인트](/blog-repo/microgpt-guide-10-inference-sampling-and-extensions/) | temperature 샘플링과 실전 개선 방향 |

---

## 빠른 실행

```bash
curl -L "https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py" -o microgpt.py
python microgpt.py
```

---

## 읽는 순서

1. 03장(`Value.backward`)과 06장(Attention)을 먼저 보면 전체 작동 원리가 빨리 잡힙니다.
2. 08~09장에서 학습 안정성과 수치 업데이트를 연결해 보면 코드의 의도를 정확히 이해할 수 있습니다.
