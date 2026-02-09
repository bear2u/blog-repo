---
layout: page
title: Tiny Audio 가이드
permalink: /tiny-audio-guide/
icon: fas fa-microphone
---

# Tiny Audio 완벽 가이드

> **24시간에 $12로 음성 인식 모델 훈련하기**

**Tiny Audio**는 최소한의 코드로 ASR(자동 음성 인식) 모델을 구축하고 훈련할 수 있는 미니멀한 프레임워크입니다. Frozen audio encoder와 Frozen LLM을 소형 Projector로 연결하여 효율적으로 학습합니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/blog-repo/tiny-audio-guide-01-intro/) | Tiny Audio란?, 핵심 아이디어, 아키텍처 개요 |
| 02 | [설치 및 빠른 시작](/blog-repo/tiny-audio-guide-02-quick-start/) | 설치, 기본 추론, 스트리밍, Timestamps |
| 03 | [아키텍처 상세](/blog-repo/tiny-audio-guide-03-architecture/) | 3가지 컴포넌트, 4가지 Projector 타입 |
| 04 | [모델 훈련](/blog-repo/tiny-audio-guide-04-training/) | Hydra 설정, 4가지 실험, 3-Stage LoRA 훈련 |
| 05 | [평가 및 분석](/blog-repo/tiny-audio-guide-05-evaluation/) | CLI 평가, WER 분석, 모델 비교 |
| 06 | [배포 및 확장](/blog-repo/tiny-audio-guide-06-deployment/) | HuggingFace 배포, Voice Agent, 커스텀 확장 |

---

## 주요 특징

- **💰 저비용 훈련** - A40 GPU로 24시간, $12에 훈련 가능
- **🔧 미니멀 & 해킹 가능** - 핵심 코드만 포함, 쉽게 수정 가능
- **❄️ Frozen Architecture** - Encoder와 LLM은 동결, Projector만 훈련 (~12M params)
- **🚀 빠른 추론** - HuggingFace Pipeline, 스트리밍, Word-level timestamps
- **🎓 교육 친화적** - 무료 3.5시간 ASR 코스 포함
- **🔌 확장 가능** - 4가지 Projector 타입, 커스텀 확장 지원

---

## 빠른 시작

### 설치

```bash
# Poetry로 설치 (권장)
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio
poetry install

# PyPI에서 설치 (추론만)
pip install tiny-audio
```

### 기본 사용

```python
from transformers import pipeline

# 모델 로드
pipe = pipeline(
    "automatic-speech-recognition",
    model="mazesmazes/tiny-audio",
    trust_remote_code=True
)

# 오디오 파일 변환
result = pipe("audio.wav")
print(result["text"])

# Word-level timestamps
result = pipe("audio.wav", return_timestamps="word")
print(result["chunks"])  # [{"text": "hello", "start": 0.0, "end": 0.5}, ...]
```

### 모델 훈련

```bash
# 빠른 테스트 (~5분)
poetry run python scripts/train.py \
    +experiments=transcription \
    data.max_train_samples=100 \
    training.max_steps=10

# 전체 훈련 (~24시간, $12)
poetry run python scripts/train.py +experiments=transcription
```

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                       Tiny Audio Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Audio (16kHz)                                                   │
│       ↓                                                          │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  GLM-ASR Encoder (~600M params)      [FROZEN ❄️]   │        │
│  │  - Frame-level audio embeddings                      │        │
│  └─────────────────────────────────────────────────────┘        │
│       ↓                                                          │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  MLP Projector (~12M params)         [TRAINED 🔥]   │        │
│  │  - Modality bridge: audio → text space               │        │
│  │  - Frame stacking for sequence reduction             │        │
│  └─────────────────────────────────────────────────────┘        │
│       ↓                                                          │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Qwen3-0.6B LLM (~600M params)       [FROZEN ❄️]   │        │
│  │  - Text generation from audio features               │        │
│  └─────────────────────────────────────────────────────┘        │
│       ↓                                                          │
│  Text Output                                                     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 핵심 아이디어

**Only the Projector trains!** 🎯

- **Encoder (Frozen)**: GLM-ASR가 오디오를 임베딩으로 변환
- **Projector (Trained)**: 오디오 공간 → 텍스트 공간 매핑 (~12M params만 학습)
- **LLM (Frozen)**: Qwen3가 변환된 임베딩으로 텍스트 생성

→ **전체 1.2B params 중 1% (12M)만 훈련!** ⚡

---

## Projector 타입 비교

| Projector | Params | 구조 | 장점 | 단점 |
|-----------|--------|------|------|------|
| **MLP** | ~12M | Simple 2-layer MLP | 빠른 훈련, 낮은 메모리 | 기본 성능 |
| **MOSA** | ~16M | Dense MoE | 파라미터 공유 효율적 | 약간 느림 |
| **MoE** | ~24M | Sparse Experts | 높은 표현력 | 더 많은 파라미터 |
| **QFormer** | ~18M | Transformer Queries | 유연한 매핑 | 훈련 복잡도 높음 |

---

## 3-Stage LoRA 훈련

```bash
# Stage 1: Projector만 훈련 (기본)
poetry run python scripts/train.py +experiments=transcription
# → WER: 5.5%, Params: 12M

# Stage 2: LoRA 어댑터 추가
poetry run python scripts/train.py +experiments=mlp_lora
# → WER: 4.8%, Params: +4.2M

# Stage 3: Projector + LoRA Fine-tune
poetry run python scripts/train.py +experiments=mlp_fine_tune
# → WER: 4.5%, Params: 16.2M
```

---

## CLI 도구

```bash
# 평가
poetry run ta eval -m mazesmazes/tiny-audio -n 100

# WER 분석
poetry run ta analysis high-wer mazesmazes/tiny-audio --threshold 30
poetry run ta analysis compare model1 model2

# 배포
poetry run ta push my-model             # HuggingFace Hub에 푸시
poetry run ta deploy my-space           # Space 배포
poetry run ta demo                      # 로컬 Gradio 데모

# 개발
poetry run ta dev lint                  # 코드 Lint
poetry run ta dev format                # 코드 포맷
poetry run ta dev test                  # 테스트 실행
```

---

## 데이터셋 & 성능

### 훈련 데이터

- **Multi-ASR Dataset**: 다양한 ASR 데이터셋 조합
- **LoquaciousSet**: 고품질 음성 데이터
- **커스텀 데이터셋** 추가 가능

### 벤치마크 (WER %)

| 모델 | LibriSpeech-test-clean | 훈련 비용 | 훈련 시간 |
|------|----------------------|---------|---------|
| Whisper-tiny | 5.4% | N/A | N/A |
| **Tiny Audio (MLP)** | 5.5% | **$12** | **24h** |
| Tiny Audio (MoE) | 5.1% | $15 | 28h |

---

## Voice Agent 통합

```python
from tiny_audio.integrations import PipecatASRService

# Pipecat-AI와 통합
asr = PipecatASRService(model="mazesmazes/tiny-audio")

# WebRTC 스트리밍
async for transcript in asr.stream(audio_chunks):
    print(transcript.text)

# OpenAI Realtime API 대체
```

---

## 프로젝트 구조

```
tiny-audio/
├── tiny_audio/              # 핵심 라이브러리
│   ├── asr_modeling.py      # ASRModel: encoder + projector + decoder
│   ├── asr_config.py        # ASRConfig: 설정
│   ├── asr_pipeline.py      # HuggingFace Pipeline
│   ├── asr_processing.py    # Processor: 전처리
│   ├── projectors.py        # 4가지 Projector 구현
│   └── integrations/        # Voice agent 통합
├── scripts/
│   ├── train.py             # 훈련 스크립트 (Hydra)
│   ├── cli.py               # CLI 진입점 (ta)
│   ├── eval/                # 평가 프레임워크
│   ├── analysis.py          # WER 분석
│   ├── deploy/              # HF Space 배포
│   └── debug/               # 디버그 도구
├── configs/                 # Hydra 설정
│   ├── config.yaml          # 메인 설정
│   ├── experiments/         # Projector 프리셋
│   ├── data/                # 데이터셋 설정
│   └── training/            # 훈련 하이퍼파라미터
└── docs/                    # 문서 및 코스
    ├── QUICKSTART.md        # 빠른 시작 가이드
    └── course/              # 무료 3.5시간 ASR 코스
```

---

## 커스텀 확장

### Projector 추가

```python
# tiny_audio/projectors.py
class MyProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config):
        super().__init__()
        # 커스텀 아키텍처 구현

    def forward(self, x, attention_mask=None):
        return projected_features

    def get_output_length(self, input_length: int) -> int:
        return output_length

# PROJECTOR_CLASSES에 등록
PROJECTOR_CLASSES["my_projector"] = MyProjector
```

### 데이터셋 추가

```yaml
# configs/data/my_dataset.yaml
dataset_name: "your-org/your-dataset"
dataset_split: "train"
audio_column: "audio"
text_column: "text"
```

```bash
# 훈련
poetry run python scripts/train.py data=my_dataset
```

---

## 학습 리소스

### 무료 3.5시간 ASR 코스

Tiny Audio는 ASR을 처음부터 구축하는 무료 코스를 제공합니다:

```
docs/course/
├── 0-course-overview.md
├── 1-audio-preprocessing.md
├── 2-feature-extraction.md
├── 3-encoder-models.md
├── 4-projector-design.md
├── 5-decoder-integration.md
└── 6-training-evaluation.md
```

**코스 링크**: [docs/course/0-course-overview.md](https://github.com/alexkroman/tiny-audio/tree/main/docs/course)

---

## 프로덕션 배포

### Docker 컨테이너

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime
RUN pip install tiny-audio
COPY app.py /app/
CMD ["python", "/app/app.py"]
```

### FastAPI 서버

```python
from fastapi import FastAPI, File, UploadFile
from tiny_audio import ASRModel, ASRProcessor

app = FastAPI()
model = ASRModel.from_pretrained("mazesmazes/tiny-audio")
processor = ASRProcessor.from_pretrained("mazesmazes/tiny-audio")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio = await file.read()
    result = model.transcribe(audio)
    return {"text": result}
```

### Kubernetes 배포

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tiny-audio
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: asr
        image: your-registry/tiny-audio:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

---

## 성능 최적화

### 양자화

```python
# INT8 양자화
model = ASRModel.from_pretrained(
    "mazesmazes/tiny-audio",
    quantization_config={"load_in_8bit": True}
)
```

### ONNX 변환

```bash
# ONNX로 변환 (추론 2-3배 속도 향상)
poetry run python scripts/export_onnx.py --model mazesmazes/tiny-audio
```

### TensorRT

```bash
# TensorRT 최적화 (NVIDIA GPU)
poetry run python scripts/export_tensorrt.py --model mazesmazes/tiny-audio
```

---

## 환경 변수

| 변수 | 설명 |
|------|------|
| `HF_TOKEN` | HuggingFace API 토큰 (private 모델/푸시용) |
| `WANDB_API_KEY` | Weights & Biases API 키 |
| `WANDB_RUN_ID` | 특정 W&B 실행 재개 |
| `ASSEMBLYAI_API_KEY` | AssemblyAI 평가 비교용 |

---

## 라이선스 및 인용

**라이선스**: MIT License

**감사의 말**:
- [GLM-ASR](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) - 오디오 인코더
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) - 언어 모델
- [LoquaciousSet](https://huggingface.co/datasets/speechbrain/LoquaciousSet) - 훈련 데이터

---

## 관련 링크

- [GitHub 저장소](https://github.com/alexkroman/tiny-audio)
- [HuggingFace 모델](https://huggingface.co/mazesmazes/tiny-audio)
- [Live Demo](https://huggingface.co/spaces/mazesmazes/tiny-audio)
- [빠른 시작 가이드](https://github.com/alexkroman/tiny-audio/blob/main/docs/QUICKSTART.md)
- [무료 ASR 코스](https://github.com/alexkroman/tiny-audio/tree/main/docs/course)
- [모델 카드](https://github.com/alexkroman/tiny-audio/blob/main/MODEL_CARD.md)

---

*작성일: 2026년 2월 9일*
*저자: Alex Kroman*
