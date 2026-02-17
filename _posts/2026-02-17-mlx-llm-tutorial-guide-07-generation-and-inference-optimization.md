---
layout: post
title: "MLX LLM Tutorial 가이드 (07) - 생성/추론 최적화: 샘플링과 mx.compile 적용"
date: 2026-02-17
permalink: /mlx-llm-tutorial-guide-07-generation-and-inference-optimization/
author: ddttom
categories: ['LLM 학습', '추론 최적화']
tags: [Generation, Temperature, Sampling, Inference, mx.compile]
original_url: "https://github.com/ddttom/mlx-llm-tutorial/blob/main/code/simple_llm.py"
excerpt: "텍스트 생성 루프, temperature 샘플링, JIT 컴파일 기반 추론 최적화 흐름을 정리합니다."
---

## 생성 루프 핵심

`generate_text(...)`는 매우 단순한 autoregressive 루프입니다.

1. seed 텍스트를 인덱스로 변환
2. 마지막 토큰 logits 추출
3. temperature 적용
4. `mx.random.categorical`로 다음 토큰 샘플링
5. 최대 길이 또는 줄바꿈에서 종료

이 흐름만 정확히 이해해도 대부분의 LLM 생성 원리를 설명할 수 있습니다.

---

## temperature 해석

스크립트는 temperature를 직접 logits에 나눠 적용합니다.

- 값이 낮을수록 보수적(반복/안정)
- 값이 높을수록 다양성 증가

실습할 때는 `0.7~1.0` 범위를 먼저 시험해보는 것이 안전합니다.

---

## 추론 최적화

`optimize_model_for_inference(...)`에서 `@mx.compile`을 사용해 forward를 래핑합니다.

```python
@mx.compile
def forward(x):
    return model(x)
```

그리고 추론 시 dropout을 끄기 위해 `model.eval()`을 호출합니다.

---

## 생성 모드 실행

```bash
python code/simple_llm.py generate
```

코드에 미리 정의된 seed 문장을 기반으로 결과를 출력하며,
학습 결과가 충분하지 않으면 문장 품질이 낮을 수 있습니다.

