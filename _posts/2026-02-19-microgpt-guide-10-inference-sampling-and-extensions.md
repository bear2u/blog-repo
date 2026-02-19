---
layout: post
title: "microgpt.py 가이드 (10) - 추론/샘플링/확장 포인트: 장난감에서 연구 코드로"
date: 2026-02-19
permalink: /microgpt-guide-10-inference-sampling-and-extensions/
author: Andrej Karpathy
categories: ['LLM 학습', '추론']
tags: [Inference, Sampling, Temperature, Limitations, microgpt]
original_url: "https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py"
excerpt: "temperature 샘플링 루프를 해설하고, microgpt.py를 더 실전적인 실험 코드로 확장하는 방향을 제시합니다."
---

## 추론 루프 (186~200행)

샘플마다 `token_id = BOS`로 시작해 최대 `block_size`까지 생성합니다.

```python
logits = gpt(token_id, pos_id, keys, values)
probs = softmax([l / temperature for l in logits])
token_id = random.choices(...)
```

`token_id == BOS`가 나오면 문장 종료로 간주합니다.

---

## temperature 의미

- 낮을수록 분포가 뾰족해져 보수적 생성
- 높을수록 다양성 증가(노이즈 증가)

코드 기본값 `0.5`는 짧은 문자열 생성에서 과도한 랜덤성을 줄이기 위한 선택입니다.

---

## microgpt.py의 한계

1. 완전 스칼라 연산이라 매우 느림
2. 미니배치 없음
3. dropout/regularization 없음
4. 체크포인트 저장/복원 없음
5. tokenizer가 문자 단위로 제한적

---

## 실전 확장 아이디어

- NumPy/PyTorch 텐서화로 속도 개선
- BPE tokenizer 도입
- Layer 수/hidden 확장 + dropout 추가
- train/val split과 perplexity 모니터링
- checkpoint + sample evaluation 자동화

---

## 최종 정리

`microgpt.py`의 가치는 "작고 완전한 알고리즘"입니다.

이 한 파일을 정확히 이해하면,
프레임워크 기반 대규모 GPT 코드에서도 핵심 경로를 잃지 않게 됩니다.

시리즈를 마칩니다.
