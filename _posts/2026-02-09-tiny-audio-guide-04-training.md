---
layout: post
title: "Tiny Audio 완벽 가이드 (04) - 모델 훈련"
date: 2026-02-09
permalink: /tiny-audio-guide-04-training/
author: Alex Kroman
categories: [머신러닝, 음성인식]
tags: [ASR, Speech Recognition, GLM-ASR, Qwen3, PyTorch, HuggingFace, Audio ML]
original_url: "https://github.com/alexkroman/tiny-audio"
excerpt: "24시간에 $12로 커스텀 ASR 모델을 훈련하는 방법을 상세히 알아봅니다."
---

## 훈련 개요

Tiny Audio의 훈련 프로세스는 놀라울 정도로 효율적입니다. 단일 A40 GPU에서 24시간 동안 약 $12의 비용으로 전체 모델을 훈련할 수 있습니다.

### 핵심 수치

- **훈련 시간**: 24시간 (단일 A40 GPU)
- **훈련 비용**: $12 (클라우드 GPU 기준)
- **훈련 파라미터**: 12M (전체 1.2B 중)
- **데이터셋**: Multi-ASR 데이터셋 (다양한 소스)
- **최대 스텝**: 50,000 steps

### 효율성의 비밀

```
전체 모델: 1.2B 파라미터
├── GLM-ASR Encoder: 600M (Frozen ❄️)
├── Projector: 12M (Trainable ✓)
└── Qwen3 Decoder: 600M (Frozen ❄️)

훈련 대상: 12M만! (전체의 1%)
```

사전 훈련된 인코더와 디코더를 고정하고 중간의 Projector만 훈련하기 때문에:

- **메모리 효율**: 그래디언트 계산이 12M에만 필요
- **속도**: 역전파가 빠름
- **안정성**: 사전 학습된 지식 유지
- **비용 절감**: GPU 시간과 전력 소비 최소화

## 빠른 테스트 실행

본격적인 훈련 전에 설정을 검증하는 것이 중요합니다.

### Quick Test (약 5분)

전체 파이프라인이 정상 동작하는지 빠르게 확인:

```bash
# 100개 샘플로 10 스텝만 실행
ta train \
    experiment=transcription \
    data.max_samples=100 \
    training.max_steps=10 \
    training.eval_steps=5
```

출력 예시:

```
[INFO] Loading GLM-ASR encoder...
[INFO] Loading Qwen3 decoder...
[INFO] Initializing MLP projector...
[INFO] Trainable parameters: 12.3M
[INFO] Frozen parameters: 1.2B
[INFO] Loading dataset (max 100 samples)...
[INFO] Starting training...

Step 1/10: loss=0.234, lr=5e-4
Step 2/10: loss=0.198, lr=5e-4
Step 3/10: loss=0.167, lr=5e-4
Step 4/10: loss=0.145, lr=5e-4
Step 5/10: loss=0.128, lr=5e-4, eval_loss=0.132
Step 6/10: loss=0.115, lr=5e-4
Step 7/10: loss=0.104, lr=5e-4
Step 8/10: loss=0.095, lr=5e-4
Step 9/10: loss=0.088, lr=5e-4
Step 10/10: loss=0.083, lr=5e-4, eval_loss=0.087

[INFO] Quick test completed successfully!
[INFO] Time: 4m 32s
```

### 체크포인트 확인

테스트가 완료되면 체크포인트가 저장됩니다:

```bash
# 저장된 체크포인트 확인
ls outputs/transcription/checkpoints/

# 출력:
# checkpoint-step-10.pt
# config.yaml
# training_state.json
```

체크포인트 로드 테스트:

```bash
# 저장된 모델로 추론 테스트
ta infer \
    checkpoint=outputs/transcription/checkpoints/checkpoint-step-10.pt \
    audio=test.wav
```

### 트러블슈팅

빠른 테스트에서 문제가 발생하면:

```bash
# CUDA 메모리 부족
ta train experiment=transcription \
    data.max_samples=50 \
    training.batch_size=1  # 배치 크기 줄이기

# 데이터셋 로딩 오류
ta train experiment=transcription \
    data.max_samples=10 \
    data.num_workers=0  # 멀티프로세싱 비활성화

# 모델 로딩 실패
export HF_HOME=/path/to/cache  # 캐시 디렉토리 명시
ta train experiment=transcription data.max_samples=100
```

## 전체 훈련

빠른 테스트가 성공했다면 전체 훈련을 시작합니다.

### 기본 훈련 명령

```bash
# MLP Projector로 50K 스텝 훈련
ta train experiment=transcription
```

이 명령은 기본 설정으로 훈련을 시작합니다:

- **Projector**: MLP (Simple)
- **데이터셋**: Multi-ASR (모든 샘플)
- **최대 스텝**: 50,000
- **배치 크기**: 16
- **학습률**: 5e-4

### Multi-ASR 데이터셋

Tiny Audio는 다양한 소스의 ASR 데이터를 결합합니다:

```yaml
# config/data/multi_asr.yaml
datasets:
  - name: librispeech
    split: train-clean-360
    weight: 0.3

  - name: common_voice
    language: en
    split: train
    weight: 0.2

  - name: gigaspeech
    subset: L  # Large subset
    split: train
    weight: 0.3

  - name: tedlium
    release: 3
    split: train
    weight: 0.1

  - name: voxpopuli
    language: en
    split: train
    weight: 0.1

total_samples: ~400,000
total_hours: ~1,200 hours
```

### 데이터셋 혼합 전략

```python
# 가중치 기반 샘플링
def sample_from_datasets(datasets, weights):
    """각 데이터셋에서 가중치에 비례하여 샘플링"""

    # 정규화
    weights = np.array(weights) / sum(weights)

    # 각 배치마다 무작위로 데이터셋 선택
    dataset_idx = np.random.choice(
        len(datasets),
        p=weights
    )

    return datasets[dataset_idx].get_batch()
```

이점:

- **다양성**: 다양한 화자, 악센트, 녹음 환경
- **일반화**: 특정 데이터셋에 오버피팅 방지
- **균형**: 고품질 데이터와 대용량 데이터 균형

### 훈련 모니터링

훈련 중 실시간 로그:

```
Epoch 1/3 [================================] 16,667/16,667 steps
Step 1000: loss=0.245, wer=28.3%, lr=5e-4, time=12.3s/step
Step 2000: loss=0.198, wer=22.1%, lr=5e-4, time=12.1s/step
Step 3000: loss=0.167, wer=18.5%, lr=5e-4, time=12.0s/step
...
Step 16667: loss=0.082, wer=7.2%, lr=5e-4, time=11.8s/step

Epoch 2/3 [================================] 33,334/33,334 steps
Step 17000: loss=0.078, wer=6.9%, lr=4e-4, time=11.7s/step
Step 18000: loss=0.075, wer=6.5%, lr=4e-4, time=11.6s/step
...
Step 33334: loss=0.061, wer=5.3%, lr=4e-4, time=11.5s/step

Epoch 3/3 [================================] 50,000/50,000 steps
Step 34000: loss=0.059, wer=5.1%, lr=3e-4, time=11.4s/step
...
Step 50000: loss=0.048, wer=4.2%, lr=1e-4, time=11.3s/step

[INFO] Training completed!
[INFO] Total time: 23h 45m
[INFO] Final WER: 4.2%
[INFO] Best checkpoint: step-47500 (WER: 4.1%)
```

## Hydra 설정 시스템

Tiny Audio는 Facebook의 Hydra를 사용하여 설정을 관리합니다.

### 설정 구조

```
configs/
├── config.yaml                 # 메인 진입점
├── experiments/                # 실험 프리셋
│   ├── transcription.yaml      # MLP Projector
│   ├── mosa.yaml              # Dense MoE
│   ├── moe.yaml               # Sparse MoE
│   └── qformer.yaml           # Transformer
├── data/                      # 데이터셋 설정
│   ├── librispeech.yaml
│   ├── common_voice.yaml
│   ├── multi_asr.yaml
│   └── custom.yaml
└── training/                  # 훈련 하이퍼파라미터
    ├── default.yaml
    ├── fast.yaml
    └── high_quality.yaml
```

### config.yaml (메인)

```yaml
# Main configuration entry point
defaults:
  - _self_
  - data: multi_asr           # 데이터셋 설정
  - training: default         # 훈련 설정
  - override hydra/launcher: submitit_slurm  # Optional: Slurm 지원

# Model
model:
  encoder: "THUDM/glm-asr-large"
  decoder: "Qwen/Qwen3-0.6B"
  projector_type: "mlp"

# Training
training:
  max_steps: 50000
  batch_size: 16
  gradient_accumulation_steps: 1
  learning_rate: 5e-4
  warmup_steps: 1000
  eval_steps: 500
  save_steps: 5000

# Data
data:
  sample_rate: 16000
  max_audio_length: 30  # seconds
  num_workers: 4
  prefetch_factor: 2

# Logging
logging:
  use_wandb: true
  project_name: "tiny-audio"
  log_steps: 100

# Hardware
hardware:
  device: "cuda"
  mixed_precision: "fp16"
  compile: false  # PyTorch 2.0 compile

# Paths
paths:
  output_dir: "./outputs"
  cache_dir: "./cache"
  data_dir: "./data"
```

### Hydra 명령줄 오버라이드

설정의 모든 값을 명령줄에서 오버라이드할 수 있습니다:

```bash
# 배치 크기 변경
ta train training.batch_size=32

# 학습률 조정
ta train training.learning_rate=1e-3

# 여러 값 동시 변경
ta train \
    training.batch_size=32 \
    training.learning_rate=1e-3 \
    training.max_steps=100000

# 다른 데이터셋 사용
ta train data=librispeech

# 다른 훈련 프리셋 사용
ta train training=fast

# 조합
ta train \
    experiment=mosa \
    data=common_voice \
    training=high_quality \
    training.batch_size=8
```

### 설정 확인

실제 훈련 전에 최종 설정 확인:

```bash
# 설정 출력 (훈련 안 함)
ta train --cfg job

# 출력:
# model:
#   encoder: THUDM/glm-asr-large
#   decoder: Qwen/Qwen3-0.6B
#   projector_type: mlp
# training:
#   max_steps: 50000
#   batch_size: 16
#   ...
```

## 4가지 Projector 실험

Tiny Audio는 4가지 Projector 아키텍처를 제공합니다.

### 1. transcription.yaml (Simple MLP)

가장 간단하고 빠른 옵션:

```yaml
# configs/experiments/transcription.yaml
defaults:
  - override /data: multi_asr
  - override /training: default

model:
  projector_type: "mlp"
  projector_config:
    input_dim: 1024
    output_dim: 896
    hidden_dim: 2048
    frame_stack: 4
    dropout: 0.1

training:
  learning_rate: 5e-4
  weight_decay: 0.01
  max_steps: 50000
```

실행:

```bash
ta train experiment=transcription
```

예상 결과:

- **WER**: ~5.5%
- **훈련 시간**: 20-24시간
- **메모리**: 16GB VRAM

### 2. mosa.yaml (Dense MoE)

여러 전문가를 항상 활성화:

```yaml
# configs/experiments/mosa.yaml
defaults:
  - override /data: multi_asr
  - override /training: default

model:
  projector_type: "mosa"
  projector_config:
    input_dim: 1024
    output_dim: 896
    num_experts: 4
    expert_dim: 512
    frame_stack: 4
    dropout: 0.1

training:
  learning_rate: 3e-4  # 더 낮은 학습률
  weight_decay: 0.01
  max_steps: 60000     # 더 긴 훈련
```

실행:

```bash
ta train experiment=mosa
```

예상 결과:

- **WER**: ~5.0%
- **훈련 시간**: 26-30시간
- **메모리**: 18GB VRAM

### 3. moe.yaml (Sparse MoE)

선택적 전문가 활성화:

```yaml
# configs/experiments/moe.yaml
defaults:
  - override /data: multi_asr
  - override /training: default

model:
  projector_type: "moe"
  projector_config:
    input_dim: 1024
    output_dim: 896
    num_experts: 8
    num_experts_per_token: 2  # Top-2 routing
    expert_dim: 512
    frame_stack: 4
    dropout: 0.1
    load_balancing_loss_weight: 0.01

training:
  learning_rate: 3e-4
  weight_decay: 0.01
  max_steps: 70000  # 더 긴 훈련 필요
  gradient_clip_norm: 1.0
```

실행:

```bash
ta train experiment=moe
```

예상 결과:

- **WER**: ~4.5%
- **훈련 시간**: 30-35시간
- **메모리**: 20GB VRAM

Load Balancing Loss:

```python
# MoE는 전문가 간 균형을 위한 추가 loss 사용
def load_balancing_loss(router_probs, num_experts):
    """
    각 전문가가 고르게 선택되도록 유도

    Args:
        router_probs: [batch, seq_len, num_experts]

    Returns:
        load_balance_loss: scalar
    """
    # 각 전문가의 평균 확률
    expert_usage = router_probs.mean(dim=[0, 1])  # [num_experts]

    # 이상적으로는 1/num_experts
    ideal_usage = 1.0 / num_experts

    # Variance를 최소화
    load_loss = torch.var(expert_usage) / (ideal_usage ** 2)

    return load_loss
```

### 4. qformer.yaml (Transformer)

Learnable query 기반:

```yaml
# configs/experiments/qformer.yaml
defaults:
  - override /data: multi_asr
  - override /training: high_quality

model:
  projector_type: "qformer"
  projector_config:
    input_dim: 1024
    output_dim: 896
    num_queries: 32  # 출력 시퀀스 길이
    num_layers: 2
    num_heads: 8
    hidden_dim: 2048
    dropout: 0.1

training:
  learning_rate: 2e-4  # 더 낮은 학습률
  weight_decay: 0.05   # 더 높은 정규화
  max_steps: 80000     # 가장 긴 훈련
  warmup_steps: 2000
```

실행:

```bash
ta train experiment=qformer
```

예상 결과:

- **WER**: ~4.2%
- **훈련 시간**: 35-40시간
- **메모리**: 22GB VRAM

### Projector 비교 요약

| Projector | WER | 훈련 시간 | VRAM | 추론 속도 | 사용 사례 |
|-----------|-----|-----------|------|-----------|-----------|
| MLP | 5.5% | 20-24h | 16GB | 가장 빠름 | 프로토타입, 빠른 실험 |
| MOSA | 5.0% | 26-30h | 18GB | 빠름 | 균형잡힌 성능 |
| MoE | 4.5% | 30-35h | 20GB | 중간 | 최고 성능 추구 |
| QFormer | 4.2% | 35-40h | 22GB | 중간 | 연구, 최고 품질 |

## 3-Stage LoRA 훈련

성능을 더욱 향상시키기 위해 LoRA 어댑터를 추가할 수 있습니다.

### Stage 1: Projector만 훈련

기본 훈련 (이미 수행):

```bash
ta train experiment=transcription
```

결과:

- **체크포인트**: `outputs/transcription/checkpoints/final.pt`
- **WER**: ~5.5%
- **훈련된 부분**: Projector만

### Stage 2: LoRA 어댑터만 훈련

Projector를 고정하고 LLM에 LoRA 추가:

```yaml
# configs/experiments/mlp_lora.yaml
defaults:
  - override /data: multi_asr
  - override /training: default

model:
  projector_type: "mlp"

  # Stage 1 체크포인트 로드
  projector_checkpoint: "outputs/transcription/checkpoints/final.pt"
  freeze_projector: true  # Projector 고정

  # LoRA 설정
  use_lora: true
  lora_config:
    r: 16                # LoRA rank
    lora_alpha: 32       # Scaling factor
    lora_dropout: 0.05
    target_modules:      # LoRA를 적용할 레이어
      - "q_proj"
      - "v_proj"
      - "o_proj"
    trainable_params: 4.2M  # 추가 훈련 파라미터

training:
  learning_rate: 1e-4  # 더 낮은 학습률
  max_steps: 20000     # 더 짧은 훈련
  warmup_steps: 500
```

실행:

```bash
ta train experiment=mlp_lora
```

LoRA의 동작 원리:

```python
class LoRALinear(nn.Module):
    """LoRA를 적용한 Linear 레이어"""

    def __init__(self, original_layer, r=16, lora_alpha=32):
        super().__init__()

        # 원본 가중치 (고정)
        self.weight = original_layer.weight
        self.weight.requires_grad = False

        in_features = self.weight.shape[1]
        out_features = self.weight.shape[0]

        # LoRA 저랭크 분해: W = W₀ + BA
        self.lora_A = nn.Parameter(torch.randn(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))

        self.scaling = lora_alpha / r

    def forward(self, x):
        # 원본 출력
        output = F.linear(x, self.weight)

        # LoRA 추가
        lora_output = (x @ self.lora_A) @ self.lora_B
        output = output + lora_output * self.scaling

        return output
```

예상 결과:

- **WER**: ~4.8% (0.7%p 향상)
- **훈련 시간**: 8-10시간
- **추가 파라미터**: 4.2M

### Stage 3: Projector + LoRA 동시 Fine-tune

두 부분을 함께 미세 조정:

```yaml
# configs/experiments/mlp_fine_tune.yaml
defaults:
  - override /data: multi_asr
  - override /training: default

model:
  projector_type: "mlp"

  # Stage 2 체크포인트 로드
  checkpoint: "outputs/mlp_lora/checkpoints/final.pt"

  freeze_projector: false  # Projector 훈련
  freeze_lora: false       # LoRA 훈련

training:
  learning_rate: 5e-5  # 매우 낮은 학습률
  max_steps: 10000     # 짧은 fine-tuning
  warmup_steps: 200
  weight_decay: 0.01
```

실행:

```bash
ta train experiment=mlp_fine_tune
```

예상 결과:

- **WER**: ~4.5% (0.3%p 추가 향상)
- **훈련 시간**: 4-5시간
- **총 훈련 파라미터**: 16.2M (Projector + LoRA)

### 3-Stage 훈련 요약

```
Stage 1: Projector만
├── 훈련: 12M params
├── 시간: 20-24h
└── WER: 5.5%

Stage 2: LoRA만 (Projector 고정)
├── 훈련: 4.2M params
├── 시간: 8-10h
└── WER: 4.8% (↓0.7%p)

Stage 3: 둘 다 Fine-tune
├── 훈련: 16.2M params
├── 시간: 4-5h
└── WER: 4.5% (↓0.3%p)

총 시간: 32-39h
총 향상: 1.0%p
```

### 언제 3-Stage 훈련을 사용할까?

**사용하는 경우**:

- 최고 성능이 필요할 때
- 충분한 GPU 시간이 있을 때
- 도메인 특화 모델 구축 시

**생략하는 경우**:

- 빠른 프로토타이핑
- 제한된 리소스
- Stage 1 성능으로 충분할 때

## 체크포인트 Resume

훈련이 중단되었을 때 이어서 시작할 수 있습니다.

### 자동 Resume

Tiny Audio는 자동으로 체크포인트를 저장합니다:

```bash
# 훈련 시작
ta train experiment=transcription

# 중단됨... (Ctrl+C 또는 시스템 오류)

# 동일 명령으로 재시작 - 자동으로 마지막 체크포인트에서 시작
ta train experiment=transcription
```

출력:

```
[INFO] Found checkpoint at outputs/transcription/checkpoints/checkpoint-step-15000.pt
[INFO] Resuming from step 15000...
[INFO] Loading model state...
[INFO] Loading optimizer state...
[INFO] Loading scheduler state...
[INFO] Loading RNG state...
[INFO] Resume successful!

Step 15001/50000: loss=0.123, ...
```

### 수동 Resume

특정 체크포인트에서 시작:

```bash
ta train \
    experiment=transcription \
    training.resume_from_checkpoint=outputs/transcription/checkpoints/checkpoint-step-10000.pt
```

### Best Checkpoint 선택

WER이 가장 낮은 체크포인트 사용:

```bash
# 모든 체크포인트 평가
ta eval --checkpoints-dir outputs/transcription/checkpoints/

# 출력:
# checkpoint-step-10000.pt: WER=6.2%
# checkpoint-step-20000.pt: WER=5.1%
# checkpoint-step-30000.pt: WER=4.8%
# checkpoint-step-40000.pt: WER=4.5%
# checkpoint-step-45000.pt: WER=4.3%  ← Best!
# checkpoint-step-50000.pt: WER=4.4%  (약간 오버피팅)

# Best 체크포인트로 추론
ta infer \
    checkpoint=outputs/transcription/checkpoints/checkpoint-step-45000.pt \
    audio=test.wav
```

### 체크포인트 관리

디스크 공간 절약:

```yaml
# config.yaml
training:
  save_steps: 5000
  save_total_limit: 5  # 최근 5개만 유지
  save_best_only: false  # True면 best만 유지
  metric_for_best: "wer"  # Best 판단 기준
  greater_is_better: false  # WER은 낮을수록 좋음
```

수동 정리:

```bash
# 오래된 체크포인트 삭제
ta clean-checkpoints \
    --keep-best 3 \
    --keep-last 2 \
    --output-dir outputs/transcription/
```

## WandB 모니터링

Weights & Biases로 실시간 훈련 추적:

### 설정

```yaml
# config.yaml
logging:
  use_wandb: true
  project_name: "tiny-audio"
  entity: "your-username"  # WandB 사용자명
  run_name: null  # Auto-generate
  tags:
    - "transcription"
    - "mlp"
  notes: "Baseline training with MLP projector"
```

### 로그인

```bash
# WandB 로그인 (최초 1회)
wandb login

# API 키 입력
# xxx...xxx
```

### 훈련 시작

```bash
ta train experiment=transcription
```

WandB 대시보드가 자동으로 열립니다:

```
[INFO] WandB run initialized: https://wandb.ai/your-username/tiny-audio/runs/abc123
```

### 모니터링 메트릭

자동으로 로깅되는 메트릭:

**손실 값**:
- `train/loss`: 훈련 손실
- `train/perplexity`: 퍼플렉시티
- `eval/loss`: 검증 손실
- `eval/wer`: Word Error Rate
- `eval/cer`: Character Error Rate

**학습 과정**:
- `train/learning_rate`: 학습률
- `train/grad_norm`: 그래디언트 노름
- `train/epoch`: 에폭
- `train/samples_per_second`: 처리 속도

**시스템**:
- `system/gpu_memory_allocated`: GPU 메모리 사용량
- `system/gpu_memory_reserved`: GPU 메모리 예약량
- `system/cpu_percent`: CPU 사용률

### 커스텀 메트릭 추가

```python
import wandb

# 훈련 루프 내에서
def training_step(batch):
    loss = compute_loss(batch)

    # 커스텀 메트릭 로깅
    wandb.log({
        "custom/batch_wer": compute_batch_wer(batch),
        "custom/audio_length": batch["audio"].shape[-1],
        "custom/text_length": len(batch["text"])
    })

    return loss
```

### 실험 비교

여러 실험을 동시에 비교:

```bash
# Terminal 1: MLP
ta train experiment=transcription

# Terminal 2: MOSA
ta train experiment=mosa

# Terminal 3: MoE
ta train experiment=moe
```

WandB에서 모든 실험을 한 화면에서 비교할 수 있습니다.

### Hyperparameter Sweep

최적의 하이퍼파라미터 탐색:

```yaml
# sweep.yaml
program: train.py
method: bayes  # random, grid, bayes
metric:
  name: eval/wer
  goal: minimize

parameters:
  training.learning_rate:
    min: 1e-5
    max: 1e-3
  training.batch_size:
    values: [8, 16, 32]
  training.weight_decay:
    min: 0.0
    max: 0.1
  model.projector_config.hidden_dim:
    values: [1024, 2048, 4096]
```

실행:

```bash
# Sweep 생성
wandb sweep sweep.yaml

# 출력:
# wandb: Created sweep with ID: xyz789
# wandb: View sweep at: https://wandb.ai/...

# Agent 실행 (여러 개 동시 가능)
wandb agent your-username/tiny-audio/xyz789
```

## 다음 단계

모델 훈련을 마쳤다면 다음 단계로 진행하세요:

- **[챕터 5: 평가 및 분석](/tiny-audio-guide-05-evaluation/)** - 모델 성능 평가와 오류 분석
- **[챕터 6: 배포 및 확장](/tiny-audio-guide-06-deployment/)** - HuggingFace Hub 배포와 프로덕션 사용

## 참고 자료

- Hydra 문서: [https://hydra.cc](https://hydra.cc)
- WandB 문서: [https://docs.wandb.ai](https://docs.wandb.ai)
- LoRA 논문: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- MoE 논문: [https://arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538)
