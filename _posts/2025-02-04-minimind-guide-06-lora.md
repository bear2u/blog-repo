---
layout: post
title: "MiniMind 완벽 가이드 (6) - LoRA 미세조정"
date: 2025-02-04
permalink: /minimind-guide-06-lora/
author: jingyaogong
categories: [LLM 학습, MiniMind]
tags: [MiniMind, LoRA, Low-Rank Adaptation, Fine-tuning, Efficient]
original_url: "https://github.com/jingyaogong/minimind"
excerpt: "MiniMind의 LoRA(Low-Rank Adaptation) 구현을 분석합니다."
---

## LoRA 개요

**LoRA(Low-Rank Adaptation)**는 사전 훈련된 모델 가중치를 동결하고, **저랭크 행렬**만 훈련하는 효율적인 미세조정 기법입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    LoRA Architecture                             │
│                                                                  │
│   Input x ────────────────────────────────────▶ W₀x             │
│      │                                          +               │
│      └──▶ [A] ──▶ [B] ──▶ ΔWx ─────────────────┘               │
│           r×d    d×r                                            │
│                                                                  │
│   Output = W₀x + ΔWx = W₀x + BAx                                │
│                                                                  │
│   W₀: 동결된 원본 가중치 (d×d)                                  │
│   A, B: 훈련 가능한 저랭크 행렬 (r << d)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## LoRA의 장점

| 장점 | 설명 |
|------|------|
| **메모리 효율** | 전체 파라미터의 0.1~1%만 훈련 |
| **빠른 훈련** | 그래디언트 계산 대상 감소 |
| **모듈화** | 여러 LoRA 어댑터 교체 가능 |
| **원본 보존** | 원본 모델 가중치 변경 없음 |

---

## LoRA 구현

MiniMind는 peft 라이브러리에 의존하지 않고 **직접 구현**합니다.

```python
# model/model_lora.py

import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """LoRA가 적용된 Linear 레이어"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,              # 랭크
        alpha: int = 16,         # 스케일링 팩터
        dropout: float = 0.0,
        original_layer: nn.Linear = None,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 원본 가중치 (동결)
        if original_layer is not None:
            self.weight = original_layer.weight
            self.bias = original_layer.bias
        else:
            self.weight = nn.Parameter(torch.zeros(out_features, in_features))
            self.bias = None

        # LoRA 어댑터
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # 드롭아웃
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # 초기화
        self._init_weights()

    def _init_weights(self):
        # A: 정규분포 초기화
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B: 0 초기화 (훈련 시작 시 ΔW = 0)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 원본 출력
        result = F.linear(x, self.weight, self.bias)

        # LoRA 출력
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        result = result + lora_output * self.scaling

        return result

    def merge(self):
        """LoRA 가중치를 원본에 병합"""
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling

    def unmerge(self):
        """병합 해제"""
        self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
```

---

## 모델에 LoRA 적용

```python
# model/model_lora.py

def apply_lora_to_model(model, r=8, alpha=16, target_modules=['wq', 'wv']):
    """모델에 LoRA 적용"""

    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Linear → LoRALinear 교체
                parent = get_parent_module(model, name)
                attr_name = name.split('.')[-1]

                lora_layer = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=r,
                    alpha=alpha,
                    original_layer=module,
                )

                setattr(parent, attr_name, lora_layer)

    # 원본 가중치 동결
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    return model


def get_lora_params(model):
    """LoRA 파라미터만 반환"""
    return [p for n, p in model.named_parameters() if 'lora_' in n]


def count_trainable_params(model):
    """훈련 가능 파라미터 수 계산"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
```

---

## LoRA 훈련

```python
# trainer/train_lora.py

def train_lora(
    model: nn.Module,
    train_dataset: Dataset,
    r: int = 8,
    alpha: int = 16,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
):
    # LoRA 적용
    model = apply_lora_to_model(model, r=r, alpha=alpha)
    count_trainable_params(model)

    model = model.to('cuda')
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # LoRA 파라미터만 옵티마이저에 전달
    lora_params = get_lora_params(model)
    optimizer = AdamW(lora_params, lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()

            logits, loss = model(input_ids, targets=labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # LoRA 가중치만 저장
    lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
    torch.save(lora_state_dict, "lora_adapter.pt")
```

---

## LoRA 어댑터 로드

```python
def load_lora_adapter(model, adapter_path):
    """저장된 LoRA 어댑터 로드"""
    # 먼저 LoRA 구조 적용
    model = apply_lora_to_model(model)

    # LoRA 가중치 로드
    lora_state = torch.load(adapter_path)
    model.load_state_dict(lora_state, strict=False)

    return model


def merge_lora_and_save(model, output_path):
    """LoRA를 원본에 병합하여 단일 모델로 저장"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()

    torch.save(model.state_dict(), output_path)
```

---

## LoRA 하이퍼파라미터

| 파라미터 | 권장값 | 설명 |
|----------|--------|------|
| **r (랭크)** | 4-64 | 낮을수록 효율적, 높을수록 표현력 |
| **alpha** | r의 2배 | 스케일링 팩터 |
| **dropout** | 0.05-0.1 | 과적합 방지 |
| **target_modules** | wq, wv | 적용할 레이어 |

---

## 실험: 전체 미세조정 vs LoRA

| 방법 | 훈련 파라미터 | 메모리 | 성능 |
|------|--------------|--------|------|
| Full Fine-tuning | 100% | 높음 | 기준 |
| LoRA (r=8) | 0.5% | 낮음 | ~97% |
| LoRA (r=16) | 1% | 낮음 | ~99% |

---

*다음 글에서는 RLHF 강화 학습을 살펴봅니다.*
