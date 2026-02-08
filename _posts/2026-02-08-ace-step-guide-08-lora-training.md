---
layout: post
title: "ACE-Step 1.5 완벽 가이드 (08) - LoRA 훈련"
date: 2026-02-08
permalink: /ace-step-guide-08-lora-training/
author: ACE Studio & StepFun
categories: [AI 음악, 오픈소스]
tags: [ACE-Step, AI Music, LoRA, Fine-tuning, PEFT, Model Training]
original_url: "https://github.com/ace-step/ACE-Step-1.5"
excerpt: "8곡, 1시간으로 나만의 음악 스타일 모델 만들기 - LoRA 훈련 완벽 가이드"
---

## 개요

**LoRA (Low-Rank Adaptation)**는 대규모 모델을 효율적으로 미세 조정하는 기술입니다. ACE-Step 1.5에서는 단 **8곡의 데이터**와 **RTX 3090 기준 1시간**의 훈련으로 나만의 음악 스타일 모델을 만들 수 있습니다.

---

## LoRA란?

### 개념

**LoRA (Low-Rank Adaptation of Large Language Models)**:
- 원본 모델 가중치는 고정(freeze)
- 작은 크기의 어댑터(adapter) 행렬만 학습
- 파라미터 효율적 미세 조정 (PEFT: Parameter-Efficient Fine-Tuning)

### 작동 원리

```
원본 가중치 (W)              LoRA 어댑터
    ↓                          ↓
  고정됨              A (rank × input_dim)
                    × B (output_dim × rank)
                          ↓
                    ΔW = B × A
                          ↓
              최종 가중치 = W + α × ΔW
```

- `rank` (r): LoRA의 용량, 보통 64
- `alpha` (α): 스케일링 팩터, 보통 128 (2×rank)
- **전체 모델 대비 훈련 파라미터: < 1%**

### 장점

| 장점 | 설명 |
|------|------|
| **효율성** | 전체 모델 대비 1% 미만 파라미터만 훈련 |
| **속도** | RTX 3090에서 8곡 기준 1시간 |
| **저메모리** | 12GB VRAM으로 충분 (vs 풀 파인튜닝 80GB+) |
| **모듈성** | 여러 LoRA 어댑터 교체하여 사용 가능 |
| **안정성** | 원본 모델 보존, 언제든 되돌릴 수 있음 |

### 활용 사례

- **특정 장르 특화**: 재즈, K-pop, 로파이 등
- **아티스트 스타일**: 특정 아티스트 음악 스타일 학습
- **음색 커스터마이징**: 특정 보컬/악기 음색 강화
- **테마 특화**: 크리스마스, 뉴이어, 게임 OST 등

---

## Gradio UI의 LoRA Training 탭

ACE-Step 1.5는 **원클릭 LoRA 훈련**을 지원합니다.

### 전체 워크플로우

```
1. Dataset Builder → 데이터셋 준비
   ├─ 오디오 폴더 스캔
   ├─ 자동 라벨링 (Caption, Lyrics, BPM, Key)
   └─ 데이터셋 JSON 저장

2. Preprocess → 전처리
   ├─ VAE latent 인코딩
   ├─ Text embedding 생성
   └─ Tensor 파일 저장

3. Train LoRA → 훈련
   ├─ LoRA 파라미터 설정
   ├─ 훈련 실행
   └─ 체크포인트 저장

4. Export → LoRA 어댑터 내보내기
```

---

## 데이터셋 준비

### 권장 사항

| 항목 | 권장 값 | 최소 | 최적 |
|------|---------|------|------|
| **곡 수** | 8+ | 5 | 20+ |
| **총 길이** | 30분+ | 15분 | 60분+ |
| **오디오 형식** | WAV, MP3, FLAC | - | WAV (무손실) |
| **샘플레이트** | 48kHz | 44.1kHz | 48kHz |
| **비트 뎁스** | 16-bit | 16-bit | 24-bit |
| **길이 분포** | 2-5분/곡 | - | 다양한 길이 |
| **스타일 일관성** | 높음 | - | 단일 스타일 |

### 데이터 품질 팁

**좋은 데이터**:
- 명확한 스타일 정체성
- 고품질 프로덕션
- 일관된 음색/믹싱
- 다양한 곡 구조

**피해야 할 것**:
- 라이브 녹음 (관중 소음)
- 압축 아티팩트 심한 MP3
- 스타일 혼재 (재즈 + 메탈 혼합)
- 너무 짧은 클립 (< 1분)

---

## Dataset Builder 탭: 단계별 가이드

### Step 1: 오디오 스캔

**새 데이터셋 시작**:
```
Audio Folder Path: /path/to/your/music/folder
→ Scan 버튼 클릭
```

지원 형식: `.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`

**기존 데이터셋 로드**:
```
Dataset JSON Path: /path/to/dataset.json
→ Load 버튼 클릭
```

### Step 2: 데이터셋 설정

| 설정 | 설명 | 예시 |
|------|------|------|
| **Dataset Name** | 데이터셋 이름 | "my-jazz-dataset" |
| **All Instrumental** | 모든 트랙이 연주곡인 경우 체크 | ☐ |
| **Custom Activation Tag** | LoRA 활성화 태그 (고유해야 함) | "myjazz", "xmas2024" |
| **Tag Position** | 태그 위치 선택 | Prepend / Append / Replace |

**Activation Tag 설명**:
- 생성 시 Caption에 이 태그를 포함하면 LoRA 스타일 활성화
- 예: Caption에 "myjazz" 포함 → 재즈 스타일 LoRA 적용

**Tag Position**:
- **Prepend**: Caption 앞에 추가 (`"myjazz, piano ballad"`)
- **Append**: Caption 뒤에 추가 (`"piano ballad, myjazz"`)
- **Replace**: Caption 전체를 태그로 대체 (`"myjazz"`)

### Step 3: 자동 라벨링

```
Auto-Label All 버튼 클릭
```

**자동 생성되는 항목**:
- **Caption**: 음악 스타일, 악기, 분위기 설명
- **BPM**: 템포 추론
- **Key**: 조성 추론 (C Major, Am 등)
- **Time Signature**: 박자 추론 (4/4, 3/4 등)

**Skip Metas 옵션**:
- 체크 시 LLM 라벨링 건너뛰고 N/A 값 사용
- 메타데이터 불필요한 경우 시간 절약

### Step 4: 수동 편집 (선택사항)

슬라이더로 샘플 선택 후 수동 편집:

```
Caption: [자동 생성된 caption 수정 가능]
Lyrics: [가사 입력 또는 수정]
BPM: [템포 조정]
Key: [조성 선택]
Time Signature: [박자 선택]
Language: [보컬 언어]
Instrumental: [연주곡 여부]

→ Save Changes 클릭
```

**편집 팁**:
- Caption 구체화: "jazz" → "smooth jazz with saxophone and piano"
- Lyrics 추가: 자동 추출되지 않은 가사 수동 입력
- 메타데이터 정확도 검증

### Step 5: 데이터셋 저장

```
Save Path: /path/to/save/dataset.json
→ Save Dataset 버튼 클릭
```

**저장되는 내용**:
```json
{
  "dataset_name": "my-jazz-dataset",
  "activation_tag": "myjazz",
  "tag_position": "prepend",
  "samples": [
    {
      "audio_path": "/path/to/song1.wav",
      "caption": "myjazz, smooth jazz with saxophone",
      "lyrics": "[Instrumental]",
      "bpm": 90,
      "keyscale": "Bb Major",
      "timesignature": 4,
      "language": "unknown",
      "instrumental": true
    },
    ...
  ]
}
```

---

## Preprocess: 전처리

### 목적

훈련 속도 향상을 위해 사전 계산:
1. **VAE Latents**: 오디오 → 잠재 표현 인코딩
2. **Text Embeddings**: Caption/Lyrics → 임베딩
3. **Condition Encoder**: 조건 인코더 실행

### 사용 방법

```
Dataset JSON Path: /path/to/dataset.json
Preprocessed Tensors Output Directory: /path/to/tensors/
→ Preprocess 버튼 클릭
```

### 처리 시간

| GPU | 8곡 (30분) 전처리 시간 |
|-----|----------------------|
| RTX 3090 | ~5-10분 |
| RTX 4090 | ~3-5분 |
| A100 | ~2-3분 |

### 출력 구조

```
/path/to/tensors/
├── sample_0.pt
├── sample_1.pt
├── sample_2.pt
...
└── sample_7.pt
```

각 `.pt` 파일 포함 내용:
```python
{
    "target_latents": tensor,      # VAE 인코딩 오디오
    "encoder_hidden_states": tensor,  # Text embedding
    "context_latents": tensor,     # 조건 컨텍스트
    "metadata": {...}              # BPM, Key 등
}
```

---

## Train LoRA 탭: 훈련 실행

### Step 1: 데이터셋 로드

```
Preprocessed Tensors Directory: /path/to/tensors/
→ Load Dataset 버튼 클릭
```

### Step 2: LoRA 설정

| 파라미터 | 기본값 | 범위 | 설명 |
|----------|--------|------|------|
| **LoRA Rank (r)** | 64 | 8-256 | LoRA 용량. 높을수록 표현력 증가, 메모리 증가 |
| **LoRA Alpha** | 128 | r-4r | 스케일링 팩터. 보통 2×rank |
| **LoRA Dropout** | 0.1 | 0.0-0.5 | 과적합 방지. 0.1 권장 |

**Rank 선택 가이드**:
- **rank=32**: 미세한 스타일 조정, 빠른 훈련
- **rank=64**: 균형잡힌 선택 (권장)
- **rank=128**: 복잡한 스타일, 더 많은 메모리

### Step 3: 훈련 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|----------|--------|------|------|
| **Learning Rate** | 1e-4 | 1e-5 ~ 1e-3 | 학습률. 너무 높으면 불안정 |
| **Max Epochs** | 500 | 100-2000 | 최대 에포크. 8곡 기준 500 적절 |
| **Batch Size** | 1 | 1-4 | 배치 크기. GPU 메모리에 따라 조정 |
| **Gradient Accumulation** | 1 | 1-8 | 유효 배치 = batch_size × accumulation |
| **Save Every N Epochs** | 200 | 50-500 | 체크포인트 저장 주기 |
| **Shift** | 3.0 | 1.0-5.0 | Turbo 모델용 타임스텝 시프트 |
| **Seed** | 42 | - | 재현성을 위한 랜덤 시드 |

### Step 4: 훈련 시작

```
→ Start Training 버튼 클릭
```

**훈련 중 모니터링**:
- **Training Progress**: 현재 에포크 및 손실 표시
- **Training Log**: 상세 로그 출력
- **Training Loss Plot**: 손실 그래프 시각화

**예상 훈련 시간**:

| GPU | 8곡, 500 에포크 |
|-----|----------------|
| RTX 3090 (24GB) | ~1시간 |
| RTX 4090 (24GB) | ~40분 |
| A100 (40GB) | ~30분 |

### Step 5: 훈련 중단 (선택사항)

```
→ Stop Training 버튼 클릭
```

마지막 체크포인트가 저장됩니다.

### Step 6: LoRA 내보내기

```
Export Path: /path/to/save/my_lora/
→ Export LoRA 버튼 클릭
```

**내보내진 파일**:
```
/path/to/save/my_lora/
├── adapter_config.json   # LoRA 설정
└── adapter_model.bin     # LoRA 가중치
```

---

## LoRA 모델 사용 방법

### 1. Gradio UI에서 사용

**Service Configuration 탭**:
```
LoRA Path: /path/to/my_lora/
→ Load LoRA 버튼 클릭
→ Use LoRA 체크박스 활성화
```

**생성 시**:
```
Caption: "myjazz, smooth piano and saxophone duet"
# → LoRA 스타일 자동 적용
```

### 2. Python API에서 사용

```python
from acestep import AceStepPipeline

# LoRA 로드
pipeline = AceStepPipeline(
    model_path="acestep-v15-turbo",
    lora_path="/path/to/my_lora/"
)

# 생성
result = pipeline.generate(
    caption="myjazz, late night jazz cafe atmosphere",
    lyrics="[Instrumental]"
)
```

### 3. LoRA 활성화/비활성화

```python
# LoRA 활성화
pipeline.enable_lora()

# LoRA 비활성화 (원본 모델 사용)
pipeline.disable_lora()

# LoRA 언로드
pipeline.unload_lora()
```

### 4. 여러 LoRA 교체 사용

```python
# 재즈 LoRA
pipeline.load_lora("/path/to/jazz_lora/")
jazz_song = pipeline.generate(caption="myjazz, piano trio")

# 로파이 LoRA로 교체
pipeline.unload_lora()
pipeline.load_lora("/path/to/lofi_lora/")
lofi_song = pipeline.generate(caption="mylofi, chill beats")
```

---

## 훈련 파라미터 최적화

### Learning Rate 조정

**증상별 조정**:

| 증상 | 원인 | 해결책 |
|------|------|--------|
| Loss가 떨어지지 않음 | LR 너무 낮음 | LR 증가 (1e-4 → 3e-4) |
| Loss가 폭발함 (NaN) | LR 너무 높음 | LR 감소 (1e-4 → 5e-5) |
| Loss가 진동함 | LR 불안정 | LR 감소 + Warmup 사용 |

### Epoch 수 조정

**데이터셋 크기별**:

| 데이터셋 | 권장 Epochs | 이유 |
|----------|-------------|------|
| 5곡 미만 | 800-1000 | 더 많은 반복 필요 |
| 8-15곡 | 500-800 | 균형 (권장) |
| 20곡 이상 | 300-500 | 과적합 위험 감소 |

**Early Stopping 판단**:
- Loss가 더 이상 감소하지 않음
- 생성 결과가 훈련 데이터 그대로 복제 (과적합)
- Validation loss 증가 시작

### Batch Size & Gradient Accumulation

**메모리 제약 시**:

| VRAM | Batch Size | Accumulation | 유효 Batch |
|------|-----------|--------------|-----------|
| 12GB | 1 | 4 | 4 |
| 16GB | 1 | 8 | 8 |
| 24GB | 2 | 4 | 8 |
| 40GB+ | 4 | 4 | 16 |

```python
# 예: 12GB GPU
batch_size = 1
gradient_accumulation = 4
# → 유효 배치 크기 = 4
```

### LoRA Rank 최적화

**스타일 복잡도별**:

| 스타일 복잡도 | Rank | 예시 |
|--------------|------|------|
| 단순 (음색 조정) | 32 | 특정 보컬 음색 |
| 중간 (장르 스타일) | 64 | 재즈, 로파이 |
| 복잡 (다중 요소) | 128 | 오케스트라, 퓨전 |

---

## Fine-tuning 모범 사례

### 1. 데이터 큐레이션

**DO**:
- 일관된 스타일 선택
- 고품질 프로덕션만 포함
- 다양한 곡 구조 포함
- 명확한 스타일 정체성

**DON'T**:
- 여러 장르 혼합
- 저품질 녹음 포함
- 모든 곡이 비슷한 구조
- 스타일이 모호한 곡

### 2. Activation Tag 전략

**좋은 태그**:
- 짧고 기억하기 쉬움: `"myjazz"`, `"xmas24"`
- 고유함: 기존 단어와 충돌 없음
- 소문자: `"myjazz"` (O), `"MyJazz"` (X)

**나쁜 태그**:
- 일반 단어: `"jazz"` (모델이 이미 알고 있음)
- 너무 길거나 복잡: `"my-custom-jazz-style-2024"`
- 특수문자: `"my_jazz!"` (파싱 오류 가능)

### 3. 훈련 모니터링

**Loss 체크포인트**:
```
Epoch 100: Loss 0.25  ← 초기, 빠르게 감소
Epoch 200: Loss 0.12  ← 중간, 감소 속도 둔화
Epoch 300: Loss 0.08  ← 수렴 시작
Epoch 400: Loss 0.06  ← 안정적
Epoch 500: Loss 0.05  ← 완료
```

**과적합 신호**:
- Loss가 0에 너무 가까움 (< 0.01)
- 생성 결과가 훈련 데이터와 거의 동일
- 새로운 Caption에 대한 일반화 실패

### 4. 테스트 전략

**훈련 중 테스트**:
```python
# 200 에포크마다 체크포인트 저장
save_every_n_epochs = 200

# 각 체크포인트로 테스트 생성
for checkpoint in ["epoch_200", "epoch_400", "epoch_600"]:
    pipeline.load_lora(f"/path/to/{checkpoint}/")
    test = pipeline.generate(caption="myjazz, test prompt")
    save(test, f"test_{checkpoint}.mp3")
```

**최적 체크포인트 선택**:
1. 여러 에포크 체크포인트 생성
2. 각 체크포인트로 테스트 생성
3. 품질 + 일반화 능력 균형 평가
4. 최적 체크포인트 선택

### 5. 데이터 증강

**부족한 데이터 보완**:
```python
# 원본 8곡 → Repaint로 변형 생성 → 16곡

for song in original_8_songs:
    # 중간 구간 재생성 (변형)
    variant = generate(
        task_type="repaint",
        src_audio=song,
        repainting_start=60,
        repainting_end=90,
        caption="slight variation in style"
    )
    augmented_dataset.append(variant)

# 총 16곡으로 훈련
```

---

## 실전 예제

### 예제 1: 크리스마스 캐롤 LoRA

**데이터셋**:
- 20곡 크리스마스 캐롤
- 스타일: 전통적, 오케스트라, 합창

**설정**:
```
Dataset Name: christmas-carols
Activation Tag: xmas
Tag Position: Prepend

LoRA Rank: 64
Learning Rate: 1e-4
Max Epochs: 400
```

**사용**:
```python
pipeline.load_lora("/path/to/xmas_lora/")

result = pipeline.generate(
    caption="xmas, traditional carol with choir and bells",
    lyrics="""
[Chorus]
Jingle bells, jingle bells
Jingle all the way
...
"""
)
```

### 예제 2: K-pop 스타일 LoRA

**데이터셋**:
- 15곡 현대 K-pop
- 특징: 강한 비트, 신스, 다이나믹한 구조

**설정**:
```
Dataset Name: kpop-style
Activation Tag: mykpop
Tag Position: Prepend

LoRA Rank: 128  # 복잡한 스타일
Learning Rate: 8e-5
Max Epochs: 600
```

**사용**:
```python
result = pipeline.generate(
    caption="mykpop, energetic dance pop with heavy bass and synths",
    lyrics="""
[Intro - building]

[Verse 1]
시작되는 이 순간
...

[Chorus - explosive]
WE ARE THE CHAMPIONS
...
"""
)
```

### 예제 3: 로파이 비트 LoRA

**데이터셋**:
- 10곡 로파이 힙합 비트
- 특징: 크래클 노이즈, 재즈 샘플, 느린 템포

**설정**:
```
Dataset Name: lofi-beats
Activation Tag: mylofi
Tag Position: Prepend
All Instrumental: ✓

LoRA Rank: 64
Learning Rate: 1e-4
Max Epochs: 500
```

**사용**:
```python
result = pipeline.generate(
    caption="mylofi, chill lofi beat with vinyl crackle and jazz samples",
    lyrics="[Instrumental]",
    bpm=85
)
```

---

## 문제 해결

### 훈련 중 메모리 부족

**Windows / 낮은 VRAM 시스템**:

**증상**:
- 전처리 중 멈춤
- Epoch 사이에 긴 정지
- Out of Memory 오류

**해결책**:

1. **미사용 모델 오프로드**:
```
Service Configuration:
  Offload to CPU: ✓
  Offload DiT to CPU: ✓
```

2. **타일 인코딩 사용**:
```python
# 전처리 시 타일 인코딩으로 피크 메모리 감소
use_tiled_encode = True
```

3. **배치 크기 감소**:
```
Batch Size: 1
Gradient Accumulation: 4  # 유효 배치 유지
```

4. **Persistent Workers 개선**:
- 최신 버전은 Windows epoch 경계 정지 자동 개선
- 여전히 문제 시 `num_workers=0` 설정

### Loss가 NaN이 됨

**원인**: Learning rate 너무 높음

**해결책**:
```
Learning Rate: 1e-4 → 5e-5
```

### 과적합 (생성 결과가 훈련 데이터 복제)

**해결책**:
1. 더 많은 데이터 추가 (8곡 → 15곡)
2. Dropout 증가 (`0.1` → `0.2`)
3. Epoch 감소 (`500` → `300`)
4. LoRA Rank 감소 (`128` → `64`)

### 생성 결과가 스타일 반영 안 됨

**체크리스트**:
1. Activation tag 포함 확인: Caption에 `"myjazz"` 포함?
2. LoRA 로드 확인: `Use LoRA` 체크박스 활성화?
3. 충분한 훈련: 최소 300+ epoch?
4. Caption과 훈련 데이터 일치: 스타일 설명이 데이터와 일치?

---

## 성능 최적화

### GPU별 최적 설정

**RTX 3090 (24GB)**:
```
Batch Size: 1
Gradient Accumulation: 4
LoRA Rank: 64
Offload to CPU: ✗  (충분한 VRAM)
```

**RTX 4080 (16GB)**:
```
Batch Size: 1
Gradient Accumulation: 4
LoRA Rank: 64
Offload to CPU: ✓  (안전)
```

**RTX 3060 (12GB)**:
```
Batch Size: 1
Gradient Accumulation: 2
LoRA Rank: 32
Offload to CPU: ✓
Use Tiled Encode: ✓
```

### 훈련 속도 향상

1. **전처리 캐싱**: 전처리는 한 번만, 여러 번 재사용
2. **체크포인트 주기 조정**: `Save Every N Epochs: 200 → 500` (I/O 감소)
3. **Mixed Precision**: 자동 활성화 (bfloat16)

---

## 다음 단계

LoRA 훈련을 마스터했다면:

1. **여러 LoRA 조합**: 다양한 스타일 LoRA 만들어 교체 사용
2. **데이터셋 확장**: 더 많은 곡으로 품질 향상
3. **하이퍼파라미터 실험**: Rank, LR 등 최적값 탐색
4. **커뮤니티 공유**: 훌륭한 LoRA는 커뮤니티와 공유

---

## 참고 자료

- [ACE-Step 1.5 LoRA Training Code](https://github.com/ace-step/ACE-Step-1.5/tree/main/acestep/training)
- [PEFT Library Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [ACE-Step 1.5 Gradio Guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GRADIO_GUIDE.md)

---

## 마무리

ACE-Step 1.5 완벽 가이드 시리즈를 완료했습니다! 이제 여러분은:

- ✅ ACE-Step 설치 및 설정
- ✅ 모델 아키텍처 이해
- ✅ 효과적인 프롬프트 작성
- ✅ 고급 기능 활용 (Cover, Repaint, Multi-Track)
- ✅ LoRA로 커스텀 모델 훈련

ACE-Step을 활용하여 멋진 음악을 만들어보세요! 🎵
