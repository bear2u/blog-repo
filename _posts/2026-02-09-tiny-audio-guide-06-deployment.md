---
layout: post
title: "Tiny Audio 완벽 가이드 (06) - 배포 및 확장"
date: 2026-02-09
permalink: /tiny-audio-guide-06-deployment/
author: Alex Kroman
categories: [머신러닝, 음성인식]
tags: [ASR, Speech Recognition, GLM-ASR, Qwen3, PyTorch, HuggingFace, Audio ML]
original_url: "https://github.com/alexkroman/tiny-audio"
excerpt: "훈련된 모델을 HuggingFace Hub에 배포하고 프로덕션 환경에서 사용하는 방법을 알아봅니다."
---

## HuggingFace Hub에 푸시

훈련된 모델을 HuggingFace Hub에 업로드하여 공유할 수 있습니다.

### 기본 푸시

```bash
# HuggingFace 로그인
huggingface-cli login

# 모델 푸시
ta push \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --repo-id your-username/tiny-audio-custom \
    --private
```

실행 과정:

```
[INFO] Preparing model for upload...
[INFO] Converting checkpoint to HuggingFace format...
[INFO] Creating model card...
[INFO] Uploading to your-username/tiny-audio-custom...

Uploading files:
  config.json               ━━━━━━━━━━━━━━━━ 100% 2.3 KB
  pytorch_model.bin         ━━━━━━━━━━━━━━━━ 100% 48.7 MB
  tokenizer_config.json     ━━━━━━━━━━━━━━━━ 100% 1.2 KB
  README.md                 ━━━━━━━━━━━━━━━━ 100% 4.5 KB
  training_args.json        ━━━━━━━━━━━━━━━━ 100% 3.1 KB

[INFO] Upload successful!
[INFO] Model available at: https://huggingface.co/your-username/tiny-audio-custom
```

### 모델 카드 자동 생성

`README.md`가 자동으로 생성됩니다:

```markdown
---
language: en
license: mit
tags:
  - audio
  - automatic-speech-recognition
  - tiny-audio
datasets:
  - librispeech
  - common_voice
metrics:
  - wer
model-index:
  - name: tiny-audio-custom
    results:
      - task:
          type: automatic-speech-recognition
          name: Speech Recognition
        dataset:
          name: LibriSpeech test-clean
          type: librispeech_asr
        metrics:
          - type: wer
            value: 5.2
            name: Word Error Rate
---

# Tiny Audio Custom

This is a custom ASR model trained with Tiny Audio framework.

## Model Description

- **Base Model**: GLM-ASR + Qwen3-0.6B
- **Projector Type**: MLP
- **Training Data**: Multi-ASR dataset
- **Training Steps**: 50,000
- **Training Time**: 24 hours on A40 GPU

## Performance

| Dataset | WER | CER |
|---------|-----|-----|
| LibriSpeech test-clean | 5.2% | 2.1% |
| LibriSpeech test-other | 12.3% | 5.8% |
| Common Voice test | 8.7% | 3.9% |

## Usage

```python
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="your-username/tiny-audio-custom",
    trust_remote_code=True
)

result = pipe("audio.wav")
print(result["text"])
```

## Training Details

- Learning rate: 5e-4
- Batch size: 16
- Optimizer: AdamW
- Mixed precision: FP16

## Citation

```bibtex
@misc{tiny-audio-custom,
  author = {Your Name},
  title = {Tiny Audio Custom Model},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/your-username/tiny-audio-custom}
}
```
```

### 커스텀 모델 카드

자체 모델 카드를 제공할 수 있습니다:

```bash
ta push \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --repo-id your-username/tiny-audio-medical \
    --model-card custom_model_card.md \
    --tags "medical,healthcare,asr"
```

### 비공개 모델

```bash
# Private repository로 업로드
ta push \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --repo-id your-username/tiny-audio-private \
    --private

# 사용 시 토큰 필요
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="your-username/tiny-audio-private",
    use_auth_token="hf_xxxxxxxxxxxxx",
    trust_remote_code=True
)
```

### 조직 계정에 푸시

```bash
ta push \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --repo-id your-org/tiny-audio-enterprise \
    --organization your-org
```

## HuggingFace Space 배포

Gradio 데모를 HuggingFace Space에 배포:

### 기본 배포

```bash
ta deploy \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --space-id your-username/tiny-audio-demo
```

실행 과정:

```
[INFO] Creating HuggingFace Space...
[INFO] Preparing demo application...
[INFO] Uploading files...

Files uploaded:
  app.py                    ━━━━━━━━━━━━━━━━ 100% 8.2 KB
  requirements.txt          ━━━━━━━━━━━━━━━━ 100% 0.5 KB
  README.md                 ━━━━━━━━━━━━━━━━ 100% 3.1 KB
  model/                    ━━━━━━━━━━━━━━━━ 100% 48.7 MB

[INFO] Building Space... (this may take 5-10 minutes)
[INFO] Space is ready!
[INFO] URL: https://huggingface.co/spaces/your-username/tiny-audio-demo
```

생성된 `app.py`:

```python
import gradio as gr
from transformers import pipeline

# 모델 로드
pipe = pipeline(
    "automatic-speech-recognition",
    model="your-username/tiny-audio-custom",
    trust_remote_code=True
)

def transcribe(audio):
    """오디오 전사"""
    if audio is None:
        return "Please upload an audio file."

    result = pipe(audio, return_timestamps="word")

    # 전체 텍스트
    full_text = result["text"]

    # Word-level timestamps
    timestamps = "\n".join([
        f"{chunk['text']}: {chunk['timestamp'][0]:.2f}s - {chunk['timestamp'][1]:.2f}s"
        for chunk in result["chunks"]
    ])

    return full_text, timestamps

# Gradio 인터페이스
demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Upload Audio"),
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Word Timestamps", lines=10)
    ],
    title="Tiny Audio Demo",
    description="Upload an audio file to get transcription with word-level timestamps.",
    examples=[
        ["examples/sample1.wav"],
        ["examples/sample2.wav"],
    ],
    cache_examples=True
)

if __name__ == "__main__":
    demo.launch()
```

### 커스텀 UI

고급 데모 UI:

```python
import gradio as gr
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np

pipe = pipeline(
    "automatic-speech-recognition",
    model="your-username/tiny-audio-custom",
    trust_remote_code=True
)

def transcribe_and_visualize(audio, show_confidence=True):
    """오디오 전사 및 시각화"""
    if audio is None:
        return None, None, None

    # 전사
    result = pipe(audio, return_timestamps="word")

    # 텍스트
    text = result["text"]

    # 타임라인 시각화
    fig, ax = plt.subplots(figsize=(12, 4))

    for i, chunk in enumerate(result["chunks"]):
        start, end = chunk["timestamp"]
        word = chunk["text"]

        # 막대 그래프
        ax.barh(0, end - start, left=start, height=0.5, alpha=0.7)

        # 텍스트 레이블
        ax.text(
            (start + end) / 2, 0,
            word,
            ha='center', va='center',
            fontsize=8,
            rotation=45
        )

    ax.set_ylim(-1, 1)
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Word Timeline')
    ax.grid(axis='x', alpha=0.3)

    # 포맷팅된 타임스탬프
    timestamps_html = "<div style='font-family: monospace;'>"
    for chunk in result["chunks"]:
        start, end = chunk["timestamp"]
        word = chunk["text"]
        timestamps_html += f"<div>{word:20s} {start:6.2f}s - {end:6.2f}s</div>"
    timestamps_html += "</div>"

    return text, fig, timestamps_html

# Gradio 블록 API
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Tiny Audio Transcription Demo")
    gr.Markdown("Upload an audio file to get high-quality transcription with word-level timestamps.")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Audio")
            submit_btn = gr.Button("Transcribe", variant="primary")

            gr.Markdown("### Examples")
            gr.Examples(
                examples=[
                    ["examples/sample1.wav"],
                    ["examples/sample2.wav"],
                    ["examples/sample3.wav"],
                ],
                inputs=audio_input
            )

        with gr.Column():
            text_output = gr.Textbox(label="Transcription", lines=3)
            timeline_plot = gr.Plot(label="Word Timeline")
            timestamps_output = gr.HTML(label="Detailed Timestamps")

    submit_btn.click(
        transcribe_and_visualize,
        inputs=[audio_input],
        outputs=[text_output, timeline_plot, timestamps_output]
    )

demo.launch()
```

### GPU Space

GPU가 필요한 경우:

```yaml
# README.md에 추가
---
title: Tiny Audio Demo
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
duplicated_from: your-username/tiny-audio-demo
hardware: t4-small  # or t4-medium, a10g-small, a10g-large
---
```

## Gradio 데모 실행

로컬에서 데모 실행:

### 기본 데모

```bash
ta demo \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --port 7860
```

출력:

```
[INFO] Loading model...
[INFO] Creating Gradio interface...
[INFO] Starting demo server...

Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live  (expires in 72 hours)

To create a permanent demo, deploy to HuggingFace Spaces.
```

### 공유 링크 생성

```bash
ta demo \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --share  # 공개 URL 생성
```

### 인증 추가

```bash
ta demo \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --auth username:password
```

### 커스텀 예제

```bash
ta demo \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --examples examples/ \
    --cache-examples
```

## Voice Agent 통합

실시간 음성 대화 시스템에 통합:

### Pipecat-AI 통합

[Pipecat](https://github.com/pipecat-ai/pipecat)은 실시간 음성 AI 파이프라인 프레임워크입니다.

#### 설치

```bash
pip install pipecat-ai[tiny-audio]
```

#### 기본 사용

```python
from pipecat.pipeline import Pipeline
from pipecat.transports import WebRTCTransport
from pipecat.audio import VAD
from pipecat.services import TinyAudioSTT, OpenAILLM, OpenAITTS

# 파이프라인 구성
pipeline = Pipeline([
    # 1. WebRTC로 오디오 수신
    WebRTCTransport(
        room_url="https://your-room.daily.co",
    ),

    # 2. VAD (Voice Activity Detection)
    VAD(
        threshold=0.5,
        min_silence_ms=500
    ),

    # 3. Speech-to-Text (Tiny Audio)
    TinyAudioSTT(
        model="your-username/tiny-audio-custom",
        device="cuda"
    ),

    # 4. LLM 처리
    OpenAILLM(
        model="gpt-4",
        system_prompt="You are a helpful assistant."
    ),

    # 5. Text-to-Speech
    OpenAITTS(
        voice="nova"
    )
])

# 파이프라인 실행
pipeline.run()
```

#### OpenAI Realtime API 통합

OpenAI의 Realtime API와 함께 사용:

```python
from pipecat.services import OpenAIRealtimeSTT, OpenAIRealtimeLLM

# OpenAI Realtime 대신 Tiny Audio 사용
pipeline = Pipeline([
    WebRTCTransport(room_url="https://your-room.daily.co"),
    VAD(threshold=0.5),

    # Tiny Audio STT (OpenAI보다 저렴)
    TinyAudioSTT(
        model="your-username/tiny-audio-custom",
        stream_mode=True,  # 실시간 스트리밍
        device="cuda"
    ),

    # OpenAI LLM
    OpenAIRealtimeLLM(
        model="gpt-4-realtime",
        instructions="You are a helpful assistant."
    )
])
```

#### 커스텀 Tiny Audio 서비스

```python
from pipecat.services.base import STTService
from tiny_audio.inference import StreamingASRInference
import numpy as np

class TinyAudioSTT(STTService):
    """Pipecat용 Tiny Audio STT 서비스"""

    def __init__(self, model, device="cuda", chunk_size=1600):
        super().__init__()
        self.inference = StreamingASRInference(
            model_name=model,
            device=device,
            chunk_size=chunk_size
        )
        self.buffer = []

    async def process_audio(self, audio_chunk: np.ndarray):
        """오디오 청크 처리"""
        # 스트리밍 추론
        partial_text = self.inference.process_chunk(audio_chunk)

        if partial_text:
            # 부분 결과 전송
            await self.emit_partial(partial_text)

        return None

    async def finalize(self):
        """최종 결과 생성"""
        final_text = self.inference.finalize()

        # 최종 결과 전송
        await self.emit_final(final_text)

        # 버퍼 초기화
        self.inference.reset()

        return final_text
```

### WebRTC 스트리밍

실시간 WebRTC 음성 전송:

```python
from aiortc import RTCPeerConnection, RTCSessionDescription
from av import AudioFrame
import asyncio

class TinyAudioWebRTC:
    """WebRTC를 통한 실시간 ASR"""

    def __init__(self, model):
        self.pc = RTCPeerConnection()
        self.inference = StreamingASRInference(model_name=model)
        self.setup_tracks()

    def setup_tracks(self):
        """오디오 트랙 설정"""

        @self.pc.on("track")
        async def on_track(track):
            """오디오 트랙 수신"""

            if track.kind == "audio":
                while True:
                    try:
                        frame = await track.recv()
                        await self.process_frame(frame)
                    except Exception as e:
                        print(f"Error: {e}")
                        break

    async def process_frame(self, frame: AudioFrame):
        """오디오 프레임 처리"""
        # 오디오 데이터 추출
        audio_data = frame.to_ndarray()

        # 16kHz로 리샘플링 (필요시)
        if frame.sample_rate != 16000:
            audio_data = resample(audio_data, frame.sample_rate, 16000)

        # 스트리밍 추론
        result = self.inference.process_chunk(audio_data)

        if result:
            # WebSocket으로 결과 전송
            await self.send_result(result)

    async def send_result(self, text):
        """결과 전송"""
        # 클라이언트에게 전송
        print(f"Transcription: {text}")

# 사용
webrtc = TinyAudioWebRTC(model="your-username/tiny-audio-custom")

# WebRTC 연결 설정
offer = await webrtc.pc.createOffer()
await webrtc.pc.setLocalDescription(offer)
```

### VAD (Voice Activity Detection) 통합

음성 구간만 전사:

```python
import webrtcvad
import numpy as np

class VADWithTinyAudio:
    """VAD + Tiny Audio"""

    def __init__(self, model, vad_aggressiveness=3):
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.inference = StreamingASRInference(model_name=model)
        self.is_speaking = False
        self.speech_buffer = []

    def process_audio(self, audio_chunk, sample_rate=16000):
        """오디오 처리 with VAD"""
        # VAD 체크 (10, 20, 30ms 청크만 지원)
        frame_duration_ms = 30
        frame_size = int(sample_rate * frame_duration_ms / 1000)

        # 청크를 프레임으로 분할
        for i in range(0, len(audio_chunk), frame_size):
            frame = audio_chunk[i:i+frame_size]

            if len(frame) < frame_size:
                break

            # Int16으로 변환
            frame_int16 = (frame * 32768).astype(np.int16)

            # VAD 실행
            is_speech = self.vad.is_speech(
                frame_int16.tobytes(),
                sample_rate
            )

            if is_speech:
                # 음성 시작
                if not self.is_speaking:
                    self.is_speaking = True
                    print("[VAD] Speech started")

                # 버퍼에 추가
                self.speech_buffer.append(frame)

            elif self.is_speaking:
                # 음성 종료
                self.is_speaking = False
                print("[VAD] Speech ended")

                # 전체 음성 구간 전사
                full_audio = np.concatenate(self.speech_buffer)
                result = self.inference.process_chunk(full_audio)

                # 버퍼 초기화
                self.speech_buffer = []

                return result

        return None

# 사용
vad_asr = VADWithTinyAudio(
    model="your-username/tiny-audio-custom",
    vad_aggressiveness=3  # 0-3, 높을수록 엄격
)

# 오디오 스트림 처리
for chunk in audio_stream:
    result = vad_asr.process_audio(chunk)
    if result:
        print(f"Transcription: {result}")
```

## 커스텀 Projector 추가

새로운 Projector 아키텍처를 추가하는 방법:

### 1. Projector 클래스 구현

```python
# tiny_audio/models/projector/custom.py

import torch
import torch.nn as nn

class CustomProjector(nn.Module):
    """커스텀 Projector 구현"""

    def __init__(
        self,
        input_dim=1024,
        output_dim=896,
        # 커스텀 파라미터들
        num_layers=3,
        attention_heads=8
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 커스텀 레이어 구현
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim * 4,
                nhead=attention_heads,
                dim_feedforward=input_dim * 4 * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # 출력 프로젝션
        self.output_proj = nn.Linear(input_dim * 4, output_dim)

    def forward(self, audio_features):
        """
        Args:
            audio_features: [batch, seq_len, input_dim]

        Returns:
            text_embeddings: [batch, seq_len//4, output_dim]
        """
        # Frame stacking
        B, T, D = audio_features.shape
        audio_features = audio_features.reshape(B, T//4, D*4)

        # Transformer layers
        x = audio_features
        for layer in self.layers:
            x = layer(x)

        # Output projection
        text_embeddings = self.output_proj(x)

        return text_embeddings
```

### 2. 설정 파일 추가

```yaml
# configs/experiments/custom.yaml
defaults:
  - override /data: multi_asr
  - override /training: default

model:
  projector_type: "custom"  # 새 타입 이름
  projector_config:
    input_dim: 1024
    output_dim: 896
    num_layers: 3
    attention_heads: 8

training:
  learning_rate: 3e-4
  max_steps: 60000
```

### 3. Factory에 등록

```python
# tiny_audio/models/projector/__init__.py

from .mlp import MLPProjector
from .mosa import MOSAProjector
from .moe import MoEProjector
from .qformer import QFormerProjector
from .custom import CustomProjector  # 추가

PROJECTOR_REGISTRY = {
    "mlp": MLPProjector,
    "mosa": MOSAProjector,
    "moe": MoEProjector,
    "qformer": QFormerProjector,
    "custom": CustomProjector,  # 등록
}

def create_projector(projector_type, config):
    """Projector 생성 factory"""
    if projector_type not in PROJECTOR_REGISTRY:
        raise ValueError(
            f"Unknown projector type: {projector_type}. "
            f"Available: {list(PROJECTOR_REGISTRY.keys())}"
        )

    projector_class = PROJECTOR_REGISTRY[projector_type]
    return projector_class(**config)
```

### 4. 훈련 실행

```bash
ta train experiment=custom
```

## 커스텀 데이터셋 추가

자체 데이터셋으로 훈련:

### 1. 데이터 준비

```
my_dataset/
├── audio/
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
└── transcripts.json
```

`transcripts.json` 형식:

```json
[
  {
    "audio_path": "audio/sample_001.wav",
    "text": "This is the transcription",
    "duration": 4.2,
    "speaker_id": "speaker_01",
    "metadata": {
      "domain": "medical",
      "quality": "clean"
    }
  },
  {
    "audio_path": "audio/sample_002.wav",
    "text": "Another transcription here",
    "duration": 6.8,
    "speaker_id": "speaker_02"
  }
]
```

### 2. Dataset 클래스 구현

```python
# tiny_audio/data/custom_dataset.py

import torch
from torch.utils.data import Dataset
import torchaudio
import json

class CustomDataset(Dataset):
    """커스텀 데이터셋"""

    def __init__(
        self,
        data_dir,
        transcript_file,
        sample_rate=16000,
        max_audio_length=30,  # seconds
        split="train"
    ):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length

        # 전사 파일 로드
        with open(transcript_file) as f:
            self.samples = json.load(f)

        # Train/val 분할 (예: 90/10)
        split_idx = int(len(self.samples) * 0.9)
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 오디오 로드
        audio_path = f"{self.data_dir}/{sample['audio_path']}"
        waveform, sr = torchaudio.load(audio_path)

        # 리샘플링
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # 모노로 변환
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 길이 제한
        max_length = self.max_audio_length * self.sample_rate
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]

        return {
            "audio": waveform.squeeze(0),
            "text": sample["text"],
            "audio_path": sample["audio_path"],
            "duration": sample.get("duration", 0),
        }
```

### 3. 설정 파일 추가

```yaml
# configs/data/custom.yaml
name: custom_dataset
class_path: tiny_audio.data.CustomDataset

# 초기화 인자
init_args:
  data_dir: /path/to/my_dataset
  transcript_file: /path/to/my_dataset/transcripts.json
  sample_rate: 16000
  max_audio_length: 30

# DataLoader 설정
dataloader:
  batch_size: 16
  num_workers: 4
  shuffle: true
  pin_memory: true
  drop_last: true

# 데이터 증강 (선택)
augmentation:
  speed_perturbation: true
  noise_injection: true
  spec_augment: true
```

### 4. 훈련 실행

```bash
ta train \
    experiment=transcription \
    data=custom
```

## RunPod 원격 훈련

클라우드 GPU로 훈련:

### 1. RunPod 설정

```bash
# RunPod API 키 설정
export RUNPOD_API_KEY=your_api_key_here

# RunPod CLI 설치
pip install runpod
```

### 2. 원격 훈련 실행

```bash
ta runpod train \
    --experiment transcription \
    --gpu-type "NVIDIA A40" \
    --max-bid 0.50  # $/hour
```

실행 과정:

```
[INFO] Connecting to RunPod...
[INFO] Finding available GPU...
[INFO] Found: NVIDIA A40 (48GB) at $0.48/hour
[INFO] Starting pod...
[INFO] Pod ID: abc123def456
[INFO] Uploading code and data...
[INFO] Starting training...

[REMOTE] Step 1000/50000: loss=0.245, wer=28.3%
[REMOTE] Step 2000/50000: loss=0.198, wer=22.1%
...

[INFO] Training complete!
[INFO] Downloading checkpoints...
[INFO] Stopping pod...
[INFO] Total cost: $11.52
```

### 3. 커스텀 RunPod 스크립트

```python
# scripts/runpod_train.py

import runpod
import os

def setup_and_train():
    """RunPod에서 훈련 실행"""

    # 환경 설정
    os.system("pip install -e .")

    # 데이터 다운로드
    os.system("ta download-data multi_asr")

    # 훈련 실행
    os.system("ta train experiment=transcription")

    # 체크포인트 업로드
    os.system("ta upload-checkpoints s3://my-bucket/checkpoints/")

if __name__ == "__main__":
    setup_and_train()
```

실행:

```bash
runpod create pod \
    --name "tiny-audio-training" \
    --gpu-type "NVIDIA A40" \
    --image pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime \
    --script scripts/runpod_train.py \
    --volume /workspace:/data
```

## 개발 도구

코드 품질 도구:

### Lint

```bash
# Ruff로 린팅
ta lint

# 자동 수정
ta lint --fix
```

### Format

```bash
# Black으로 포맷팅
ta format

# 체크만
ta format --check
```

### Test

```bash
# 전체 테스트
ta test

# 특정 파일
ta test tests/test_models.py

# 커버리지
ta test --coverage
```

### Pre-commit

```bash
# Pre-commit 설치
ta precommit install

# 수동 실행
ta precommit run --all-files
```

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.270
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
```

## 프로덕션 배포 가이드

프로덕션 환경에서 안정적으로 배포:

### Docker 컨테이너

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# 의존성 설치
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev

# 코드 복사
COPY . .

# 모델 다운로드
RUN python -c "from transformers import pipeline; \
    pipeline('automatic-speech-recognition', \
    model='your-username/tiny-audio-custom', \
    trust_remote_code=True)"

# 서버 실행
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

빌드 및 실행:

```bash
# 빌드
docker build -t tiny-audio:latest .

# 실행
docker run --gpus all -p 8000:8000 tiny-audio:latest
```

### FastAPI 서버

```python
# app.py
from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
import tempfile
import os

app = FastAPI()

# 모델 로드 (시작 시 1회)
pipe = pipeline(
    "automatic-speech-recognition",
    model="your-username/tiny-audio-custom",
    trust_remote_code=True,
    device="cuda"
)

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """오디오 파일 전사"""

    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 전사
        result = pipe(tmp_path, return_timestamps="word")

        return {
            "text": result["text"],
            "chunks": result["chunks"]
        }

    finally:
        # 임시 파일 삭제
        os.unlink(tmp_path)

@app.get("/health")
async def health():
    """헬스 체크"""
    return {"status": "healthy"}
```

실행:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Kubernetes 배포

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tiny-audio
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tiny-audio
  template:
    metadata:
      labels:
        app: tiny-audio
    spec:
      containers:
      - name: tiny-audio
        image: your-registry/tiny-audio:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: tiny-audio-service
spec:
  selector:
    app: tiny-audio
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

배포:

```bash
kubectl apply -f deployment.yaml
```

## 성능 최적화 팁

프로덕션 성능 향상:

### 1. 모델 양자화

```python
from torch.quantization import quantize_dynamic

# 동적 양자화
model_int8 = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# 저장
torch.save(model_int8.state_dict(), "model_int8.pt")
```

### 2. ONNX 변환

```python
import torch.onnx

# 더미 입력
dummy_input = torch.randn(1, 16000 * 10)  # 10초

# ONNX 내보내기
torch.onnx.export(
    model,
    dummy_input,
    "tiny_audio.onnx",
    input_names=["audio"],
    output_names=["transcription"],
    dynamic_axes={
        "audio": {0: "batch", 1: "time"},
        "transcription": {0: "batch"}
    }
)
```

### 3. TensorRT 최적화

```bash
# ONNX → TensorRT
trtexec \
    --onnx=tiny_audio.onnx \
    --saveEngine=tiny_audio.trt \
    --fp16
```

### 4. 배치 처리

```python
# 여러 파일을 배치로 처리
audio_files = ["file1.wav", "file2.wav", ..., "file16.wav"]
results = pipe(audio_files, batch_size=16)
```

## 향후 로드맵

Tiny Audio의 개발 계획:

- [ ] 다국어 지원 (100+ 언어)
- [ ] 실시간 diarization
- [ ] Emotion recognition
- [ ] Code-switching 지원
- [ ] Edge 디바이스 최적화
- [ ] Streaming fine-tuning
- [ ] Active learning

## 기여 방법

프로젝트에 기여하기:

```bash
# 1. Fork & Clone
git clone https://github.com/your-username/tiny-audio.git
cd tiny-audio

# 2. 브랜치 생성
git checkout -b feature/new-projector

# 3. 개발
# ... 코드 작성 ...

# 4. 테스트
ta test
ta lint
ta format

# 5. Commit & Push
git add .
git commit -m "Add new projector architecture"
git push origin feature/new-projector

# 6. Pull Request 생성
# GitHub에서 PR 생성
```

## 무료 3.5시간 ASR 코스

더 깊이 배우고 싶다면:

### 코스 내용

1. **ASR 기초** (30분)
   - 음성 인식 역사
   - 주요 접근법
   - 평가 메트릭

2. **오디오 처리** (45분)
   - 신호 처리
   - 특징 추출
   - 데이터 증강

3. **모델 아키텍처** (60분)
   - Encoder-Decoder
   - Attention 메커니즘
   - Transformer

4. **훈련 기법** (45분)
   - CTC Loss
   - Sequence-to-Sequence
   - Self-supervised learning

5. **프로덕션 배포** (30분)
   - 최적화
   - 서빙
   - 모니터링

### 등록

무료 코스 링크: [https://tiny-audio-course.com](https://github.com/alexkroman/tiny-audio)

## 참고 자료

- GitHub: [https://github.com/alexkroman/tiny-audio](https://github.com/alexkroman/tiny-audio)
- HuggingFace: [https://huggingface.co/alexkroman](https://huggingface.co/alexkroman)
- Pipecat: [https://github.com/pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat)
- RunPod: [https://www.runpod.io](https://www.runpod.io)
- WebRTC: [https://webrtc.org](https://webrtc.org)

## 마치며

축하합니다! Tiny Audio의 모든 기능을 살펴보았습니다.

이제 여러분은:
- 커스텀 ASR 모델을 훈련하고
- 다양한 방법으로 평가하며
- HuggingFace Hub에 배포하고
- 실시간 음성 애플리케이션에 통합할 수 있습니다

질문이나 피드백은 GitHub Issues에 남겨주세요!
