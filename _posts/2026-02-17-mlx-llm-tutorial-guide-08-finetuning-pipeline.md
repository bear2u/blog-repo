---
layout: post
title: "MLX LLM Tutorial 가이드 (08) - 파인튜닝 파이프라인: finetune_llm.py 구조와 사용법"
date: 2026-02-17
permalink: /mlx-llm-tutorial-guide-08-finetuning-pipeline/
author: ddttom
categories: ['LLM 학습', '파인튜닝']
tags: [Fine Tuning, Hugging Face, TinyLlama, Instruction Data, AdamW]
original_url: "https://github.com/ddttom/mlx-llm-tutorial/blob/main/code/finetune_llm.py"
excerpt: "finetune_llm.py의 CLI 인자, 데이터 포맷, 학습 루프, 현재 구현 제약까지 실무 관점에서 정리합니다."
---

## 실행 인터페이스

`finetune_llm.py`는 CLI 기반으로 동작합니다.

```bash
python code/finetune_llm.py \
  --data code/sample_dataset.json \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 3 \
  --batch-size 4 \
  --lr 2e-5 \
  --output fine_tuned_model
```

선택적으로 `--generate --prompt "..."`를 붙여 생성 테스트를 할 수 있습니다.

---

## 데이터 포맷

샘플 데이터(`code/sample_dataset.json`)는 instruction-response 구조입니다.

```json
{
  "instruction": "...",
  "response": "..."
}
```

스크립트는 이를 `<s>[INST] ... [/INST] ...</s>` 형태로 합쳐 토크나이즈합니다.

---

## 학습 루프

- 옵티마이저: `optim.AdamW`
- 손실: cross entropy + label mask(`-100` 무시)
- 저장: `fine_tuned_model/model.npz`

instruction 구간을 loss에서 제외하는 로직이 포함되어 있어, 응답 생성 쪽 학습에 집중하도록 구성돼 있습니다.

---

## 현재 제약 사항(중요)

코드 주석과 로그 기준으로 다음 한계가 분명합니다.

1. Hugging Face `pytorch_model.bin` 로딩 경로는 안내되어 있지만 실제 변환/적용 코드는 미구현
2. 모델별 가중치 포맷 차이를 자동 해결하지 않음
3. 실제 사용 전 모델 호환성 점검/변환 로직 보강이 필요

즉, 구조 학습에는 좋지만 "바로 프로덕션 파인튜닝" 단계는 아닙니다.

