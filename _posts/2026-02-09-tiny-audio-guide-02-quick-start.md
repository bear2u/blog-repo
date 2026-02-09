---
layout: post
title: "Tiny Audio 완벽 가이드 (02) - 설치 및 빠른 시작"
date: 2026-02-09
permalink: /tiny-audio-guide-02-quick-start/
author: Alex Kroman
categories: [머신러닝, 음성인식]
tags: [ASR, Speech Recognition, GLM-ASR, Qwen3, PyTorch, HuggingFace, Audio ML]
original_url: "https://github.com/alexkroman/tiny-audio"
excerpt: "Tiny Audio의 설치부터 기본 사용법까지 단계별로 알아봅니다."
---

## 요구사항

Tiny Audio를 사용하기 위해서는 다음 환경이 필요합니다:

### 필수 요구사항

- **Python**: 3.10 이상
- **PyTorch**: 2.0 이상
- **CUDA**: 11.8 이상 (GPU 사용 시)
- **메모리**: 최소 8GB RAM, 권장 16GB
- **GPU**: 추론 시 4GB VRAM, 훈련 시 16GB VRAM

### 운영체제

- Linux (권장)
- macOS (Apple Silicon 지원)
- Windows (WSL2 권장)

### 의존성 관리자

- **Poetry** (권장): 의존성 관리 및 가상환경
- **pip**: 표준 패키지 관리자

## 설치 방법

Tiny Audio는 두 가지 방법으로 설치할 수 있습니다.

### 방법 1: Poetry로 설치 (권장)

Poetry를 사용하면 의존성 관리가 자동화되어 가장 안정적입니다.

#### 1. Poetry 설치

```bash
# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

#### 2. 저장소 클론

```bash
git clone https://github.com/alexkroman/tiny-audio.git
cd tiny-audio
```

#### 3. 의존성 설치

```bash
# 기본 설치 (추론만)
poetry install

# 개발 의존성 포함
poetry install --with dev

# 전체 설치 (훈련 포함)
poetry install --with dev,train
```

#### 4. 가상환경 활성화

```bash
poetry shell
```

### 방법 2: PyPI로 설치

pip를 사용한 간단한 설치:

```bash
# 기본 설치
pip install tiny-audio

# PyTorch와 함께 설치
pip install tiny-audio torch torchvision torchaudio

# 개발 버전 설치
pip install git+https://github.com/alexkroman/tiny-audio.git
```

### 설치 확인

설치가 완료되면 다음 명령어로 확인할 수 있습니다:

```bash
# CLI 도구 확인
tiny-audio --version

# Python 모듈 확인
python -c "import tiny_audio; print(tiny_audio.__version__)"
```

## 기본 추론

가장 간단한 사용법은 HuggingFace Transformers의 pipeline API를 활용하는 것입니다.

### Pipeline 기본 사용

```python
from transformers import pipeline

# 모델 로드
pipe = pipeline(
    "automatic-speech-recognition",
    model="alexkroman/tiny-audio",
    trust_remote_code=True,
    device="cuda:0"  # CPU의 경우 "cpu"
)

# 오디오 파일 전사
result = pipe("path/to/audio.wav")
print(result["text"])
```

### 배치 처리

여러 파일을 한 번에 처리:

```python
audio_files = [
    "audio1.wav",
    "audio2.wav",
    "audio3.wav"
]

# 배치 처리
results = pipe(audio_files, batch_size=4)

for audio_file, result in zip(audio_files, results):
    print(f"{audio_file}: {result['text']}")
```

### 옵션 설정

다양한 옵션으로 추론을 커스터마이징:

```python
result = pipe(
    "audio.wav",
    return_timestamps="word",     # word-level timestamps
    chunk_length_s=30,             # 청크 길이 (초)
    stride_length_s=5,             # 오버랩 길이 (초)
    generate_kwargs={
        "max_new_tokens": 256,
        "num_beams": 5,            # beam search
        "temperature": 0.0         # greedy decoding
    }
)
```

## 스트리밍 추론

실시간 음성 인식을 위한 스트리밍 API를 제공합니다.

### 기본 스트리밍

```python
from tiny_audio.inference import StreamingASRInference
import numpy as np

# 스트리밍 추론 객체 생성
inference = StreamingASRInference(
    model_name="alexkroman/tiny-audio",
    device="cuda:0",
    chunk_size=1600  # 100ms at 16kHz
)

# 오디오 스트림 시뮬레이션
audio_stream = load_audio_stream("audio.wav")

# 청크 단위 처리
for chunk in audio_stream:
    # chunk: numpy array, shape (chunk_size,)
    partial_result = inference.process_chunk(chunk)

    if partial_result:
        print(f"Partial: {partial_result}", end="\r")

# 최종 결과
final_result = inference.finalize()
print(f"\nFinal: {final_result}")
```

### 마이크 입력 스트리밍

```python
import sounddevice as sd

def audio_callback(indata, frames, time, status):
    """마이크 입력 콜백"""
    if status:
        print(f"Status: {status}")

    # 오디오 데이터 처리
    audio_data = indata[:, 0]  # 모노 채널
    result = inference.process_chunk(audio_data)

    if result:
        print(f"Transcription: {result}", end="\r")

# 마이크 스트림 시작
with sd.InputStream(
    channels=1,
    samplerate=16000,
    callback=audio_callback,
    blocksize=1600
):
    print("Recording... Press Ctrl+C to stop.")
    input()
```

### 파일 기반 스트리밍

```python
import wave

def stream_from_file(file_path, chunk_size=1600):
    """WAV 파일을 청크 단위로 스트리밍"""
    with wave.open(file_path, 'rb') as wf:
        while True:
            data = wf.readframes(chunk_size)
            if not data:
                break

            # bytes to numpy array
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0

            yield audio_chunk

# 사용 예시
for chunk in stream_from_file("audio.wav"):
    result = inference.process_chunk(chunk)
    if result:
        print(f"Partial: {result}")

final_result = inference.finalize()
print(f"Final: {final_result}")
```

## Word-level Timestamps

각 단어의 시작과 끝 시간을 추출할 수 있습니다.

### 기본 사용

```python
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="alexkroman/tiny-audio",
    trust_remote_code=True
)

# Word-level timestamps 활성화
result = pipe("audio.wav", return_timestamps="word")

# 결과 출력
print(f"Full text: {result['text']}\n")
print("Word-level timestamps:")

for chunk in result["chunks"]:
    word = chunk["text"]
    start, end = chunk["timestamp"]
    print(f"{word:15s} {start:6.2f}s - {end:6.2f}s")
```

### 출력 예시

```
Full text: Hello world this is Tiny Audio

Word-level timestamps:
Hello           0.00s - 0.50s
world           0.60s - 1.20s
this            1.30s - 1.60s
is              1.70s - 1.90s
Tiny            2.00s - 2.40s
Audio           2.50s - 3.00s
```

### 자막 파일 생성

SRT 형식의 자막 파일을 생성:

```python
def generate_srt(result, output_file, words_per_line=10):
    """Word-level timestamps를 SRT 자막으로 변환"""
    chunks = result["chunks"]

    with open(output_file, 'w', encoding='utf-8') as f:
        subtitle_id = 1

        for i in range(0, len(chunks), words_per_line):
            segment = chunks[i:i+words_per_line]

            # 시작/끝 시간
            start_time = segment[0]["timestamp"][0]
            end_time = segment[-1]["timestamp"][1]

            # 텍스트
            text = " ".join([c["text"] for c in segment])

            # SRT 형식 출력
            f.write(f"{subtitle_id}\n")
            f.write(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n")
            f.write(f"{text}\n\n")

            subtitle_id += 1

def format_srt_time(seconds):
    """초를 SRT 시간 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# 사용 예시
result = pipe("audio.wav", return_timestamps="word")
generate_srt(result, "subtitles.srt")
```

## HuggingFace Demo

온라인에서 바로 시험해볼 수 있는 데모가 제공됩니다.

### Gradio 데모 실행

```bash
# 로컬에서 데모 실행
python scripts/demo.py

# 포트 지정
python scripts/demo.py --port 7860

# 공개 URL 생성
python scripts/demo.py --share
```

### 데모 코드

직접 Gradio 인터페이스를 만들 수 있습니다:

```python
import gradio as gr
from transformers import pipeline

# 모델 로드
pipe = pipeline(
    "automatic-speech-recognition",
    model="alexkroman/tiny-audio",
    trust_remote_code=True
)

def transcribe(audio_file):
    """오디오 파일을 전사"""
    if audio_file is None:
        return "오디오 파일을 업로드하세요."

    result = pipe(audio_file, return_timestamps="word")

    # 전체 텍스트
    full_text = result["text"]

    # Word-level 정보
    word_details = "\n".join([
        f"{c['text']}: {c['timestamp'][0]:.2f}s - {c['timestamp'][1]:.2f}s"
        for c in result["chunks"]
    ])

    return full_text, word_details

# Gradio 인터페이스
demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="오디오 업로드"),
    outputs=[
        gr.Textbox(label="전사 결과"),
        gr.Textbox(label="Word-level Timestamps", lines=10)
    ],
    title="Tiny Audio Demo",
    description="음성 파일을 업로드하면 자동으로 전사합니다."
)

demo.launch()
```

## 빠른 테스트 실행

프로젝트에는 빠른 테스트를 위한 스크립트가 포함되어 있습니다.

### 기본 추론 테스트

```bash
# 단일 파일 테스트
python scripts/test_inference.py \
    --audio audio.wav \
    --model alexkroman/tiny-audio

# 배치 테스트
python scripts/test_inference.py \
    --audio-dir ./test_audio/ \
    --batch-size 4
```

### 스트리밍 테스트

```bash
# 파일 기반 스트리밍
python scripts/test_streaming.py \
    --audio audio.wav \
    --chunk-size 1600

# 마이크 입력
python scripts/test_streaming.py \
    --input microphone \
    --duration 10
```

### 벤치마크

모델 성능을 측정:

```bash
python scripts/benchmark.py \
    --dataset librispeech_test \
    --model alexkroman/tiny-audio \
    --output results.json
```

출력 예시:
```json
{
  "wer": 5.2,
  "cer": 2.1,
  "rtf": 0.15,
  "latency_ms": 45,
  "throughput_hours_per_hour": 6.67
}
```

## 환경 변수 설정

Tiny Audio의 동작을 환경 변수로 제어할 수 있습니다.

### 주요 환경 변수

```bash
# 모델 캐시 디렉토리
export TINY_AUDIO_CACHE_DIR=/path/to/cache

# 디바이스 설정
export TINY_AUDIO_DEVICE=cuda:0

# 로그 레벨
export TINY_AUDIO_LOG_LEVEL=INFO

# HuggingFace 토큰 (private 모델 사용 시)
export HF_TOKEN=your_token_here

# 오프라인 모드
export TRANSFORMERS_OFFLINE=1
```

### .env 파일 사용

프로젝트 루트에 `.env` 파일을 생성:

```bash
# .env
TINY_AUDIO_CACHE_DIR=/data/tiny-audio-cache
TINY_AUDIO_DEVICE=cuda:0
TINY_AUDIO_LOG_LEVEL=INFO
HF_TOKEN=hf_xxxxxxxxxxxxx
```

Python에서 로드:

```python
from dotenv import load_dotenv
load_dotenv()

from tiny_audio import TinyAudioModel

# 환경 변수가 자동으로 적용됨
model = TinyAudioModel.from_pretrained("alexkroman/tiny-audio")
```

## CLI 도구

명령줄에서 바로 사용할 수 있는 도구들입니다.

### 단일 파일 전사

```bash
tiny-audio transcribe audio.wav
```

### 배치 전사

```bash
# 디렉토리 내 모든 오디오 파일 처리
tiny-audio batch-transcribe ./audio_dir/ \
    --output ./transcriptions/ \
    --format json

# 특정 확장자만
tiny-audio batch-transcribe ./audio_dir/ \
    --pattern "*.mp3" \
    --workers 4
```

### 스트리밍 모드

```bash
# 마이크 입력
tiny-audio stream --input-device microphone

# 파일 스트리밍
tiny-audio stream --input-file audio.wav --chunk-size 1600
```

### 모델 정보

```bash
# 모델 상세 정보
tiny-audio model-info alexkroman/tiny-audio

# 로컬 캐시 확인
tiny-audio cache-info
```

## 문제 해결

### CUDA 메모리 부족

GPU 메모리가 부족한 경우:

```python
# 배치 크기 줄이기
result = pipe(audio, batch_size=1)

# FP16 사용
pipe.model.half()

# CPU 사용
pipe = pipeline(..., device="cpu")
```

### 느린 추론 속도

추론 속도 개선:

```python
# Flash Attention 활성화 (GPU 메모리 효율)
pipe = pipeline(..., use_flash_attention=True)

# Compiled 모드 (PyTorch 2.0+)
pipe.model = torch.compile(pipe.model)
```

### 오디오 형식 오류

지원되는 형식으로 변환:

```bash
# ffmpeg로 변환
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## 다음 단계

기본 사용법을 익혔다면, 다음 챕터에서 아키텍처를 깊이 있게 살펴보겠습니다:

- **[챕터 3: 아키텍처 상세](/tiny-audio-guide-03-architecture/)** - 각 컴포넌트의 상세 구조와 동작 원리

## 참고 자료

- GitHub 저장소: [https://github.com/alexkroman/tiny-audio](https://github.com/alexkroman/tiny-audio)
- HuggingFace 데모: [https://huggingface.co/spaces/alexkroman/tiny-audio-demo](https://huggingface.co/spaces/alexkroman/tiny-audio-demo)
- PyTorch 문서: [https://pytorch.org/docs](https://pytorch.org/docs)
- Transformers 문서: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
