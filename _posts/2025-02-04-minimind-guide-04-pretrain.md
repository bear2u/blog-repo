---
layout: post
title: "MiniMind 완벽 가이드 (4) - Pretrain 사전 훈련"
date: 2025-02-04
permalink: /minimind-guide-04-pretrain/
author: jingyaogong
category: AI
tags: [MiniMind, Pretrain, Causal LM, Training, PyTorch]
series: minimind-guide
part: 4
original_url: "https://github.com/jingyaogong/minimind"
excerpt: "MiniMind의 사전 훈련(Pretrain) 구현과 데이터 처리를 분석합니다."
---

## Pretrain 개요

사전 훈련은 **대량의 텍스트 데이터**로 언어 모델의 기본 능력을 학습시키는 단계입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pretrain Pipeline                             │
│                                                                  │
│   Raw Text ──▶ Tokenize ──▶ Batch ──▶ Forward ──▶ Loss ──▶ Update │
│                                                                  │
│   목표: 다음 토큰 예측 (Causal Language Modeling)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 데이터셋

MiniMind는 **JSONL 형식**의 데이터셋을 사용합니다.

### 데이터 형식

```jsonl
{"text": "인공지능은 컴퓨터 과학의 한 분야로..."}
{"text": "Python은 1991년에 만들어진 프로그래밍 언어입니다..."}
{"text": "大语言模型是一种基于深度学习的自然语言处理技术..."}
```

### 데이터 로더

```python
# trainer/train_pretrain.py

import json
from torch.utils.data import Dataset, DataLoader

class PretrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # JSONL 파일 로드
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item['text'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        # 토큰화
        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # 입력과 타겟 생성 (shift by 1)
        input_ids = tokens[:-1]
        labels = tokens[1:]

        return {
            'input_ids': input_ids,
            'labels': labels,
        }
```

---

## 훈련 루프

```python
# trainer/train_pretrain.py

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_pretrain(
    model: nn.Module,
    train_dataset: Dataset,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    warmup_steps: int = 1000,
    device: str = 'cuda',
):
    model = model.to(device)
    model.train()

    # 데이터 로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 옵티마이저
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # 스케줄러
    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # 손실 함수
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    global_step = 0

    for epoch in range(epochs):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward
            logits, loss = model(input_ids, targets=labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % 100 == 0:
                print(f"Step {global_step}, Loss: {loss.item():.4f}")

        # 에폭 종료 시 체크포인트 저장
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")
```

---

## 분산 훈련 (DDP)

```python
# trainer/train_pretrain.py

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def train_pretrain_ddp(
    rank: int,
    world_size: int,
    model: nn.Module,
    train_dataset: Dataset,
    **kwargs
):
    # DDP 초기화
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
    )

    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # 분산 샘플러
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=kwargs['batch_size'],
        sampler=sampler,
        num_workers=4,
    )

    # 훈련 루프 (위와 동일)
    ...

    dist.destroy_process_group()


# 실행
if __name__ == '__main__':
    import torch.multiprocessing as mp

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_pretrain_ddp,
        args=(world_size, model, train_dataset),
        nprocs=world_size,
    )
```

---

## 학습률 스케줄링

### Warmup + Cosine Decay

```python
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr
```

---

## 체크포인트 관리

```python
def save_checkpoint(model, optimizer, scheduler, step, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['step']
```

---

## 훈련 설정 예시

```yaml
# config/pretrain.yaml

# 모델
model:
  dim: 512
  n_layers: 8
  n_heads: 8
  vocab_size: 6400

# 데이터
data:
  train_path: dataset/pretrain.jsonl
  max_length: 512

# 훈련
training:
  epochs: 3
  batch_size: 32
  learning_rate: 3e-4
  warmup_steps: 1000
  weight_decay: 0.1
  grad_clip: 1.0

# 하드웨어
hardware:
  device: cuda
  mixed_precision: true
  num_workers: 4
```

---

## 혼합 정밀도 훈련

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        logits, loss = model(input_ids, targets=labels)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 훈련 모니터링 (WandB/SwanLab)

```python
import wandb

wandb.init(project="minimind-pretrain")

for step, batch in enumerate(train_loader):
    # 훈련 코드...

    wandb.log({
        'loss': loss.item(),
        'learning_rate': scheduler.get_last_lr()[0],
        'step': step,
    })
```

---

*다음 글에서는 SFT(지도 학습 미세조정)를 살펴봅니다.*
