---
layout: post
title: "Tiny Audio 완벽 가이드 (03) - 아키텍처 상세"
date: 2026-02-09
permalink: /tiny-audio-guide-03-architecture/
author: Alex Kroman
categories: [머신러닝, 음성인식]
tags: [ASR, Speech Recognition, GLM-ASR, Qwen3, PyTorch, HuggingFace, Audio ML]
original_url: "https://github.com/alexkroman/tiny-audio"
excerpt: "Tiny Audio의 3계층 아키텍처와 4가지 Projector 타입을 상세히 알아봅니다."
---

## 전체 아키텍처 다이어그램

Tiny Audio는 세 가지 핵심 컴포넌트로 구성된 모듈러 아키텍처를 채택하고 있습니다.

```
┌──────────────────────────────────────────────────────────────┐
│                     Tiny Audio System                        │
│                                                              │
│  Input: Raw Audio (16kHz PCM)                               │
│  Output: Transcribed Text                                   │
└──────────────────────────────────────────────────────────────┘

                            │
                            ▼
        ┌───────────────────────────────────┐
        │   Audio Preprocessing             │
        │   - Resampling to 16kHz           │
        │   - Normalization                 │
        │   - Chunking (30s segments)       │
        └───────────────┬───────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────┐
│                  Component 1: Audio Encoder                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              GLM-ASR Encoder (Frozen)                   │ │
│  │  - Model: GLM-ASR (600M parameters)                     │ │
│  │  - Status: Frozen (not trained)                         │ │
│  │  - Input: Raw audio waveform [batch, time]              │ │
│  │  - Output: Audio features [batch, seq_len, 1024]        │ │
│  │  - Purpose: Extract high-level audio representations    │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                Component 2: Modality Projector                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │        Projector (Trainable - 12M parameters)           │ │
│  │  - Status: Trained (only trainable component)           │ │
│  │  - Input: Audio features [batch, seq_len, 1024]         │ │
│  │  - Output: Text embeddings [batch, seq_len, 896]        │ │
│  │  - Purpose: Bridge audio and language modalities        │ │
│  │                                                          │ │
│  │  Types (choose one):                                    │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │ 1. MLP: Simple 2-layer network                   │  │ │
│  │  │ 2. MOSA: Dense Mixture of Experts                │  │ │
│  │  │ 3. MoE: Sparse routed experts                    │  │ │
│  │  │ 4. QFormer: Transformer with queries             │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│              Component 3: Language Model                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           Qwen3-0.6B (Frozen)                           │ │
│  │  - Model: Qwen3-0.6B (600M parameters)                  │ │
│  │  - Status: Frozen (not trained)                         │ │
│  │  - Input: Text embeddings [batch, seq_len, 896]         │ │
│  │  - Output: Generated text tokens                        │ │
│  │  - Purpose: Decode embeddings to natural language       │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │   Post-processing                 │
        │   - Token decoding                │
        │   - Timestamp alignment           │
        │   - Text formatting               │
        └───────────────┬───────────────────┘
                        │
                        ▼
                Final Transcription
```

## 3가지 핵심 컴포넌트

### Component 1: GLM-ASR Encoder

#### 개요

GLM-ASR은 강력한 사전 훈련된 오디오 인코더로, 원시 오디오 신호를 고차원 특징 벡터로 변환합니다.

#### 기술 사양

- **파라미터 수**: 약 600M
- **아키텍처**: Transformer 기반 인코더
- **입력**: 16kHz PCM 오디오
- **출력**: 1024차원 특징 벡터 시퀀스
- **훈련 상태**: Frozen (가중치 고정)

#### 동작 원리

```python
class GLMASREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 사전 훈련된 GLM-ASR 로드
        self.encoder = AutoModel.from_pretrained(
            "THUDM/glm-asr-large"
        )
        # 가중치 고정
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, audio_input):
        """
        Args:
            audio_input: [batch_size, audio_length]
                - 16kHz raw audio waveform

        Returns:
            audio_features: [batch_size, seq_len, 1024]
                - High-level audio representations
        """
        with torch.no_grad():  # 그래디언트 계산 안 함
            outputs = self.encoder(audio_input)
            audio_features = outputs.last_hidden_state

        return audio_features
```

#### 특징 추출 과정

1. **입력 처리**: 원시 오디오를 16kHz로 리샘플링
2. **스펙트로그램 변환**: 내부적으로 mel-spectrogram 생성
3. **Transformer 인코딩**: 다층 self-attention으로 특징 추출
4. **시퀀스 출력**: 각 타임스텝마다 1024차원 벡터

#### 왜 Frozen인가?

- **전이 학습**: 방대한 오디오 데이터로 사전 훈련됨
- **안정성**: 잘 학습된 특징 추출기 유지
- **효율성**: 훈련 시 메모리와 시간 절약
- **일반화**: 다양한 도메인에 적용 가능

### Component 2: Modality Projector (핵심!)

#### 개요

Projector는 **오디오 모달리티와 텍스트 모달리티를 연결하는 핵심 브리지**입니다. Tiny Audio에서 유일하게 훈련되는 컴포넌트로, 전체 시스템의 성능을 결정합니다.

#### 기술 사양

- **파라미터 수**: 약 12M (타입에 따라 다름)
- **입력 차원**: 1024 (GLM-ASR 출력)
- **출력 차원**: 896 (Qwen3 입력)
- **훈련 상태**: Trainable (유일한 학습 대상)

#### 역할

1. **차원 변환**: 1024 → 896 차원 매핑
2. **모달리티 정렬**: 오디오 특징을 텍스트 임베딩 공간으로 변환
3. **시퀀스 압축**: Frame stacking으로 시퀀스 길이 감소
4. **의미 보존**: 오디오의 언어적 정보 유지

#### Frame Stacking

Projector는 인접한 프레임을 묶어 시퀀스를 압축합니다:

```python
def frame_stacking(features, stack_size=4):
    """
    인접한 프레임을 스택하여 시퀀스 길이 감소

    Args:
        features: [batch, seq_len, feature_dim]
        stack_size: 묶을 프레임 수

    Returns:
        stacked: [batch, seq_len//stack_size, feature_dim*stack_size]
    """
    batch_size, seq_len, feature_dim = features.shape

    # 시퀀스 길이를 stack_size의 배수로 맞춤
    new_seq_len = (seq_len // stack_size) * stack_size
    features = features[:, :new_seq_len, :]

    # Reshape and stack
    features = features.reshape(
        batch_size,
        new_seq_len // stack_size,
        feature_dim * stack_size
    )

    return features
```

이점:
- **계산 효율**: LLM 입력 길이 감소 → 추론 속도 향상
- **문맥 통합**: 여러 프레임의 정보를 하나로 압축
- **메모리 절약**: Attention 메모리 사용량 감소 (O(n²) → O((n/4)²))

### Component 3: Qwen3-0.6B LLM

#### 개요

Qwen3는 Alibaba의 소형 언어 모델로, 텍스트 임베딩을 자연어로 디코딩합니다.

#### 기술 사양

- **파라미터 수**: 약 600M
- **아키텍처**: Causal Transformer (GPT-style)
- **어휘 크기**: 151,936 tokens
- **최대 컨텍스트**: 32,768 tokens
- **훈련 상태**: Frozen (가중치 고정)

#### 동작 원리

```python
class QwenDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Qwen3 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B"
        )
        # 가중치 고정
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text_embeddings):
        """
        Args:
            text_embeddings: [batch_size, seq_len, 896]
                - Projector 출력

        Returns:
            generated_text: List[str]
                - 생성된 텍스트
        """
        with torch.no_grad():
            # Embeddings를 직접 입력 (토큰화 생략)
            outputs = self.model.generate(
                inputs_embeds=text_embeddings,
                max_new_tokens=256,
                num_beams=5,
                temperature=0.0  # Greedy decoding
            )

            # Token IDs를 텍스트로 디코딩
            generated_text = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )

        return generated_text
```

#### 왜 소형 모델을 사용하는가?

- **효율성**: 600M 파라미터로 충분한 언어 능력
- **속도**: 빠른 추론 (RTF < 0.2)
- **배포**: Edge 디바이스에 배포 가능
- **비용**: GPU 메모리 요구량 낮음

## Projector 동작 원리

Projector는 Tiny Audio의 핵심 혁신입니다. 작은 모듈(12M)로 두 거대 모델(각 600M)을 연결합니다.

### Modality Bridge

```
Audio Space                           Text Space
(1024-dim)                            (896-dim)

   [a₁]                                [t₁]
   [a₂]                                [t₂]
   [a₃]        ┌─────────┐            [t₃]
   [a₄]  ─────>│Projector│────>       [t₄]
   [a₅]        └─────────┘            [t₅]
   [...]                               [...]

Audio features            Text embeddings
from GLM-ASR              for Qwen3
```

### 학습 과정

Projector는 다음을 학습합니다:

1. **특징 변환**: 오디오 특징의 선형/비선형 변환
2. **공간 정렬**: 오디오와 텍스트의 의미 공간 매핑
3. **시간 정렬**: 오디오 프레임과 텍스트 토큰의 시간 대응
4. **노이즈 필터링**: 불필요한 오디오 정보 제거

### 학습 목표

```python
def training_step(audio, text):
    """Projector 훈련 스텝"""

    # 1. Frozen 인코더로 오디오 특징 추출
    with torch.no_grad():
        audio_features = encoder(audio)  # [B, T, 1024]

    # 2. Projector로 텍스트 임베딩 생성
    text_embeddings = projector(audio_features)  # [B, T', 896]

    # 3. Frozen LLM으로 텍스트 생성
    with torch.no_grad():
        # Ground truth 텍스트의 임베딩
        target_embeddings = llm.get_input_embeddings()(text_tokens)

    # 4. Loss 계산: 생성된 임베딩과 타겟 임베딩 비교
    loss = F.mse_loss(text_embeddings, target_embeddings)

    return loss
```

## 4가지 Projector 타입

Tiny Audio는 4가지 Projector 변형을 제공합니다. 각각 다른 트레이드오프를 가집니다.

### 1. MLP (Multi-Layer Perceptron)

#### 구조

가장 간단한 구조로, 2개의 선형 레이어와 활성화 함수로 구성됩니다.

```python
class MLPProjector(nn.Module):
    def __init__(self, input_dim=1024, output_dim=896, hidden_dim=2048):
        super().__init__()

        self.layers = nn.Sequential(
            # Frame stacking
            nn.Linear(input_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, audio_features):
        """
        Args:
            audio_features: [B, T, 1024]
        Returns:
            text_embeddings: [B, T//4, 896]
        """
        # Frame stacking (4 frames -> 1)
        B, T, D = audio_features.shape
        audio_features = audio_features.reshape(B, T//4, D*4)

        # MLP projection
        text_embeddings = self.layers(audio_features)

        return text_embeddings
```

#### 특징

- **파라미터**: ~12M
- **장점**:
  - 구현이 간단
  - 훈련이 빠름
  - 메모리 효율적
- **단점**:
  - 표현력 제한적
  - 복잡한 패턴 학습 어려움
- **적합한 경우**:
  - 빠른 프로토타이핑
  - 리소스 제약이 큰 환경

### 2. MOSA (Mixture of Soft Attention)

#### 구조

Dense Mixture of Experts 변형으로, 여러 전문가 네트워크를 결합합니다.

```python
class MOSAProjector(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        output_dim=896,
        num_experts=4,
        expert_dim=512
    ):
        super().__init__()

        # 여러 전문가 네트워크
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim * 4, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, output_dim)
            )
            for _ in range(num_experts)
        ])

        # Soft attention for expert weighting
        self.attention = nn.Sequential(
            nn.Linear(input_dim * 4, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, audio_features):
        """
        Args:
            audio_features: [B, T, 1024]
        Returns:
            text_embeddings: [B, T//4, 896]
        """
        # Frame stacking
        B, T, D = audio_features.shape
        audio_features = audio_features.reshape(B, T//4, D*4)

        # 각 전문가의 출력 계산
        expert_outputs = [
            expert(audio_features) for expert in self.experts
        ]  # num_experts x [B, T//4, 896]

        expert_outputs = torch.stack(expert_outputs, dim=-1)
        # [B, T//4, 896, num_experts]

        # Attention weights 계산
        attention_weights = self.attention(audio_features)
        # [B, T//4, num_experts]

        # Weighted combination
        attention_weights = attention_weights.unsqueeze(2)
        # [B, T//4, 1, num_experts]

        text_embeddings = (expert_outputs * attention_weights).sum(dim=-1)
        # [B, T//4, 896]

        return text_embeddings
```

#### 특징

- **파라미터**: ~16M
- **장점**:
  - MLP보다 표현력 향상
  - 다양한 오디오 패턴 처리
  - 안정적인 훈련
- **단점**:
  - 모든 전문가를 항상 사용 (계산 비용)
- **적합한 경우**:
  - 다양한 음향 환경
  - 중간 수준의 성능 요구

### 3. MoE (Mixture of Experts)

#### 구조

Sparse routed MoE로, 각 입력마다 일부 전문가만 선택적으로 활성화합니다.

```python
class MoEProjector(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        output_dim=896,
        num_experts=8,
        num_experts_per_token=2,
        expert_dim=512
    ):
        super().__init__()

        self.num_experts_per_token = num_experts_per_token

        # 전문가 네트워크들
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim * 4, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, output_dim)
            )
            for _ in range(num_experts)
        ])

        # Router network
        self.router = nn.Linear(input_dim * 4, num_experts)

    def forward(self, audio_features):
        """
        Args:
            audio_features: [B, T, 1024]
        Returns:
            text_embeddings: [B, T//4, 896]
        """
        # Frame stacking
        B, T, D = audio_features.shape
        audio_features = audio_features.reshape(B, T//4, D*4)

        # Router scores
        router_logits = self.router(audio_features)
        # [B, T//4, num_experts]

        # Top-k routing (각 토큰마다 2개 전문가만 선택)
        topk_logits, topk_indices = torch.topk(
            router_logits,
            k=self.num_experts_per_token,
            dim=-1
        )  # [B, T//4, 2]

        topk_weights = F.softmax(topk_logits, dim=-1)

        # 선택된 전문가만 실행
        text_embeddings = torch.zeros(
            B, T//4, output_dim,
            device=audio_features.device
        )

        for i in range(self.num_experts_per_token):
            expert_idx = topk_indices[:, :, i]  # [B, T//4]
            expert_weight = topk_weights[:, :, i:i+1]  # [B, T//4, 1]

            # 해당 전문가 실행
            for b in range(B):
                for t in range(T//4):
                    idx = expert_idx[b, t].item()
                    output = self.experts[idx](
                        audio_features[b:b+1, t:t+1]
                    )
                    text_embeddings[b, t] += (
                        expert_weight[b, t] * output[0, 0]
                    )

        return text_embeddings
```

#### 특징

- **파라미터**: ~24M
- **장점**:
  - 높은 모델 용량 (8개 전문가)
  - 계산 효율적 (2개만 활성화)
  - 전문화된 학습 가능
- **단점**:
  - 훈련이 불안정할 수 있음
  - Router 학습이 중요
- **적합한 경우**:
  - 최고 성능 추구
  - 다양한 언어/악센트

### 4. QFormer (Query Transformer)

#### 구조

Transformer 기반 Projector로, learnable query를 사용합니다.

```python
class QFormerProjector(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        output_dim=896,
        num_queries=32,
        num_layers=2,
        num_heads=8
    ):
        super().__init__()

        # Learnable queries
        self.queries = nn.Parameter(
            torch.randn(1, num_queries, output_dim)
        )

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=num_heads,
                kdim=input_dim,
                vdim=input_dim,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Feed-forward layers
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, output_dim * 4),
                nn.GELU(),
                nn.Linear(output_dim * 4, output_dim)
            )
            for _ in range(num_layers)
        ])

        # Layer norms
        self.ln_layers = nn.ModuleList([
            nn.LayerNorm(output_dim)
            for _ in range(num_layers * 2)
        ])

    def forward(self, audio_features):
        """
        Args:
            audio_features: [B, T, 1024]
        Returns:
            text_embeddings: [B, num_queries, 896]
        """
        B = audio_features.shape[0]

        # Queries 확장
        queries = self.queries.expand(B, -1, -1)  # [B, 32, 896]

        # Transformer layers
        for i in range(len(self.cross_attention_layers)):
            # Cross-attention
            ln_queries = self.ln_layers[i*2](queries)
            attn_output, _ = self.cross_attention_layers[i](
                query=ln_queries,
                key=audio_features,
                value=audio_features
            )
            queries = queries + attn_output

            # Feed-forward
            ln_queries = self.ln_layers[i*2+1](queries)
            ffn_output = self.ffn_layers[i](ln_queries)
            queries = queries + ffn_output

        return queries
```

#### 특징

- **파라미터**: ~18M
- **장점**:
  - 유연한 시퀀스 길이 처리
  - Query 수로 출력 길이 조절
  - 강력한 문맥 모델링
- **단점**:
  - 훈련 시간이 가장 김
  - 메모리 사용량 높음
- **적합한 경우**:
  - 연구 목적
  - 최고 품질 필요

## Projector 비교

| 타입 | 파라미터 | 훈련 속도 | 추론 속도 | 성능 | 복잡도 |
|------|----------|-----------|-----------|------|--------|
| MLP | 12M | 가장 빠름 | 가장 빠름 | 기본 | 낮음 |
| MOSA | 16M | 빠름 | 빠름 | 좋음 | 중간 |
| MoE | 24M | 중간 | 중간 | 최고 | 높음 |
| QFormer | 18M | 느림 | 중간 | 매우 좋음 | 높음 |

### 선택 가이드

```python
# 프로토타이핑 / 빠른 실험
projector = MLPProjector()

# 균형잡힌 성능
projector = MOSAProjector()

# 최고 성능 추구
projector = MoEProjector()

# 연구 / 고급 실험
projector = QFormerProjector()
```

## 프로젝트 구조

```
tiny-audio/
├── tiny_audio/                 # 메인 패키지
│   ├── __init__.py
│   ├── models/                 # 모델 정의
│   │   ├── encoder.py          # GLM-ASR wrapper
│   │   ├── projector/          # Projector 구현
│   │   │   ├── mlp.py          # MLP Projector
│   │   │   ├── mosa.py         # MOSA Projector
│   │   │   ├── moe.py          # MoE Projector
│   │   │   └── qformer.py      # QFormer Projector
│   │   ├── decoder.py          # Qwen3 wrapper
│   │   └── tiny_audio.py       # 전체 모델 통합
│   ├── inference/              # 추론 엔진
│   │   ├── pipeline.py         # HF Pipeline
│   │   ├── streaming.py        # 스트리밍 추론
│   │   └── timestamps.py       # Word-level timestamps
│   ├── training/               # 훈련 코드
│   │   ├── trainer.py          # 훈련 루프
│   │   ├── dataset.py          # 데이터 로더
│   │   └── loss.py             # Loss 함수
│   └── utils/                  # 유틸리티
│       ├── audio.py            # 오디오 처리
│       └── text.py             # 텍스트 처리
├── scripts/                    # 실행 스크립트
│   ├── train.py                # 훈련 스크립트
│   ├── inference.py            # 추론 스크립트
│   ├── demo.py                 # Gradio 데모
│   └── benchmark.py            # 벤치마크
├── configs/                    # 설정 파일
│   ├── train_config.yaml       # 훈련 설정
│   └── model_config.yaml       # 모델 설정
├── tests/                      # 테스트
│   ├── test_models.py
│   ├── test_inference.py
│   └── test_training.py
├── pyproject.toml              # Poetry 설정
├── README.md
└── LICENSE
```

## 핵심 파일 설명

### 1. models/tiny_audio.py

전체 시스템을 통합하는 메인 모델:

```python
class TinyAudioModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Components
        self.encoder = GLMASREncoder()
        self.projector = self._create_projector(config.projector_type)
        self.decoder = Qwen3Decoder()

        # Freeze encoder and decoder
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, audio_input):
        # Encode audio
        audio_features = self.encoder(audio_input)

        # Project to text space
        text_embeddings = self.projector(audio_features)

        # Decode to text
        transcription = self.decoder(text_embeddings)

        return transcription
```

### 2. training/trainer.py

Projector 훈련 로직:

```python
class TinyAudioTrainer:
    def __init__(self, model, train_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.config = config

        # Only projector parameters are trainable
        self.optimizer = torch.optim.AdamW(
            model.projector.parameters(),
            lr=config.learning_rate
        )

    def train(self):
        for epoch in range(self.config.num_epochs):
            for batch in self.train_dataset:
                loss = self.training_step(batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
```

### 3. inference/streaming.py

스트리밍 추론 구현:

```python
class StreamingASRInference:
    def __init__(self, model_name, chunk_size=1600):
        self.model = TinyAudioModel.from_pretrained(model_name)
        self.chunk_size = chunk_size
        self.buffer = []

    def process_chunk(self, audio_chunk):
        self.buffer.append(audio_chunk)

        if len(self.buffer) >= self.min_buffer_size:
            audio = np.concatenate(self.buffer)
            result = self.model(audio)
            return result

        return None
```

## 다음 단계

아키텍처를 이해했다면, 다음 단계로 진행하세요:

- **챕터 4: 훈련** - 커스텀 데이터로 Projector 훈련하기
- **챕터 5: 최적화** - 성능 튜닝 및 배포 최적화

## 참고 자료

- GLM-ASR: [https://github.com/THUDM/GLM-ASR](https://github.com/THUDM/GLM-ASR)
- Qwen3: [https://github.com/QwenLM/Qwen](https://github.com/QwenLM/Qwen)
- Mixture of Experts 논문: [Arxiv](https://arxiv.org)
- QFormer 논문: [BLIP-2](https://arxiv.org)
