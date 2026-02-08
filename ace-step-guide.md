---
layout: page
title: ACE-Step 1.5 가이드
permalink: /ace-step-guide/
icon: fas fa-music
---

# ACE-Step 1.5 완벽 가이드

> **오픈소스 AI 음악 생성의 새로운 지평**

**ACE-Step 1.5**는 상업급 품질의 음악을 로컬 하드웨어에서 생성하는 오픈소스 AI 음악 생성 모델입니다. A100에서 풀송을 2초 이내에, RTX 3090에서 10초 이내에 생성할 수 있습니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/ace-step-guide-01-intro/) | 프로젝트 소개, 주요 특징, 디자인 철학 |
| 02 | [설치 및 시작](/ace-step-guide-02-installation/) | 플랫폼별 설치 방법, GPU 환경 설정 |
| 03 | [아키텍처 분석](/ace-step-guide-03-architecture/) | DiT + LM 하이브리드 구조, Model Zoo |
| 04 | [Gradio UI 사용법](/ace-step-guide-04-gradio-ui/) | 웹 인터페이스로 음악 생성 |
| 05 | [REST API 가이드](/ace-step-guide-05-rest-api/) | API 서버 구축 및 프로그래밍 |
| 06 | [음악 생성 전략](/ace-step-guide-06-generation-strategy/) | 프롬프트 작성, 메타데이터 제어 |
| 07 | [고급 기능](/ace-step-guide-07-advanced-features/) | Cover, Repaint, Add Layer, Vocal2BGM |
| 08 | [LoRA 훈련](/ace-step-guide-08-lora-training/) | 자신만의 스타일 학습 |
| 09 | [GPU 최적화](/ace-step-guide-09-gpu-optimization/) | VRAM 관리, 모델 선택, 성능 튜닝 |
| 10 | [결론 및 활용](/ace-step-guide-10-conclusion/) | 요약, 활용 시나리오, 다음 단계 |

---

## 주요 특징

### ⚡ 성능

- **초고속 생성** - A100에서 풀송 2초 이내, RTX 3090에서 10초 이내
- **유연한 길이** - 10초부터 10분(600초)까지 오디오 생성 지원
- **배치 생성** - 최대 8개 노래 동시 생성

### 🎵 생성 품질

- **상업급 출력** - 대부분의 상업용 음악 모델을 능가 (Suno v4.5와 v5 사이)
- **풍부한 스타일 지원** - 1000개 이상의 악기와 스타일
- **다국어 가사** - 50개 이상 언어 지원

### 🎛️ 다양성 & 제어

| 기능 | 설명 |
|------|------|
| ✅ 참조 오디오 | 참조 오디오로 스타일 가이드 |
| ✅ Cover 생성 | 기존 오디오로 커버 생성 |
| ✅ Repaint & 편집 | 선택적 로컬 편집 및 재생성 |
| ✅ 트랙 분리 | 개별 스템으로 분리 |
| ✅ Multi-Track | Suno Studio의 "Add Layer"처럼 레이어 추가 |
| ✅ Vocal2BGM | 보컬 트랙에 자동 반주 생성 |
| ✅ 메타데이터 제어 | Duration, BPM, Key/Scale, Time Signature 제어 |
| ✅ LoRA 훈련 | 8곡, RTX 3090에서 1시간 (12GB VRAM) |

---

## 빠른 시작

### 설치 (3단계)

```bash
# 1. uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 클론 & 설치
git clone https://github.com/ACE-Step/ACE-Step-1.5.git
cd ACE-Step-1.5
uv sync

# 3. 실행
uv run acestep  # Gradio UI (http://localhost:7860)
```

### Windows Portable 패키지

```batch
# 1. 다운로드 & 압축 해제
https://files.acemusic.ai/acemusic/win/ACE-Step-1.5.7z

# 2. 실행
start_gradio_ui.bat
```

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    ACE-Step 1.5                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Input (Text/Audio)                                     │
│         ↓                                                    │
│    Language Model (LM) - Omni-capable Planner                │
│         ↓                                                    │
│    Blueprint (Metadata + Lyrics + Caption)                   │
│         ↓                                                    │
│    Diffusion Transformer (DiT)                               │
│         ↓                                                    │
│    Generated Music (10s ~ 10min)                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**하이브리드 아키텍처:**

1. **Language Model (LM)** - 사용자 쿼리를 포괄적인 음악 블루프린트로 변환
2. **Diffusion Transformer (DiT)** - LM의 블루프린트를 기반으로 음악 생성
3. **Intrinsic Reinforcement Learning** - 외부 보상 모델 없이 내부 메커니즘으로 정렬

---

## 기술 스택

| 기술 | 용도 |
|------|------|
| Python 3.11 | 핵심 언어 |
| PyTorch 2.7+ | 딥러닝 프레임워크 |
| Diffusers | Diffusion Transformer |
| Transformers | Language Model (Qwen3 기반) |
| Gradio | 웹 UI |
| FastAPI | REST API 서버 |
| vLLM / MLX | 추론 가속 (CUDA / Apple Silicon) |

---

## GPU 요구사항

| VRAM | 모델 설정 | 성능 |
|------|-----------|------|
| **≤6GB** | DiT only (LLM 비활성화) | 빠름, 기본 품질 |
| **6-12GB** | LM-0.6B + offload | 중간 품질 |
| **12-16GB** | LM-1.7B | 좋은 품질 |
| **16GB+** | LM-4B + 배치 생성 | 최고 품질 |

**지원 GPU:**
- NVIDIA CUDA (권장)
- AMD ROCm (RX 6000/7000/9000 시리즈)
- Intel GPU (Arc, Integrated)
- Apple Silicon (M1/M2/M3 - MPS/MLX)

---

## 주요 워크플로우

### 1. Simple Mode (영감 모드)

```
자연어 프롬프트 입력
    ↓
LM이 자동으로 Blueprint 생성
    ↓
DiT가 음악 생성
    ↓
결과 청취 및 선택
```

### 2. Advanced Mode (고급 제어)

```
프롬프트 + 메타데이터 (BPM, Key, Duration)
    ↓
참조 오디오 업로드 (선택)
    ↓
생성 파라미터 조정 (Shift, CFG)
    ↓
배치 생성 (최대 8개)
    ↓
AutoGen + AutoScore로 최적화
```

### 3. Cover & Repaint

```
기존 오디오 업로드
    ↓
Cover: 스타일 변환 (Strength 조절)
Repaint: 특정 부분 편집
    ↓
반복적 개선
```

### 4. LoRA 훈련

```
8-20곡 데이터셋 준비
    ↓
자동 라벨링 (Dataset Builder)
    ↓
LoRA 훈련 (RTX 3090에서 1시간)
    ↓
자신만의 스타일 모델 생성
```

---

## 디자인 철학

### 원클릭 생성 vs 인간 중심 생성

ACE-Step은 **"원클릭 생성"**이 아닌 **"인간 중심 생성"**을 위해 설계되었습니다.

```
원클릭 생성 (Finite Game):
프롬프트 → 생성 → 선택 → 완료

인간 중심 생성 (Infinite Game):
영감 → 생성 → 탐색 → 조정 (Cover/Repaint/Layer)
        ↑                           ↓
        └───────── 반복 ─────────────┘
```

**핵심 가치:**

1. **오픈소스 & 로컬** - 생성한 음악은 영원히 여러분의 것
2. **빠른 생성** - 몰입(Flow) 상태 유지
3. **세밀한 제어** - 창의적 탐색 가능

### 코끼리와 기수 비유

```
        기수 (Human)
          │
          │ 방향 제시
          ↓
    ┌─────────────┐
    │  코끼리 (AI) │  ← 자신만의 관성, 기질, 의지
    └─────────────┘
```

AI는 여러분의 하인이 아니라 **"영감 제공자(inspirer)"**입니다.

---

## 활용 시나리오

### 1. 콘텐츠 제작자

```python
# 유튜브 배경음악 자동 생성
prompt = "Upbeat corporate background music, energetic"
duration = 180  # 3분
result = generate(prompt, duration=duration)
```

### 2. 음악 프로듀서

```python
# 데모 트랙 빠른 프로토타이핑
sketch = generate("Lo-fi hip hop beat, jazzy chords")
refined = cover(sketch, strength=0.3, prompt="더 풍부한 베이스")
final = add_layer(refined, "smooth saxophone solo")
```

### 3. 게임 개발자

```python
# 동적 BGM 생성
for scene in ["peaceful", "tense", "combat"]:
    bgm = generate(f"{scene} game background music")
    save_bgm(scene, bgm)
```

### 4. 교육 & 실험

```python
# 음악 이론 실험
scales = ["C major", "A minor", "D dorian"]
for scale in scales:
    example = generate(f"Piano melody in {scale}")
    analyze(example)
```

---

## 관련 링크

- **GitHub**: [https://github.com/ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5)
- **HuggingFace**: [https://huggingface.co/ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5)
- **ModelScope**: [https://modelscope.cn/organization/ACE-Step](https://modelscope.cn/organization/ACE-Step)
- **Space Demo**: [https://huggingface.co/spaces/ACE-Step/Ace-Step-v1.5](https://huggingface.co/spaces/ACE-Step/Ace-Step-v1.5)
- **Discord**: [https://discord.gg/PeWDxrkdj7](https://discord.gg/PeWDxrkdj7)
- **Technical Report**: [https://arxiv.org/abs/2602.00744](https://arxiv.org/abs/2602.00744)

---

## 라이선스

이 프로젝트는 [MIT 라이선스](https://github.com/ace-step/ACE-Step-1.5/blob/main/LICENSE)로 배포됩니다.

---

## 인용

```bibtex
@misc{gong2026acestep,
	title={ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation},
	author={Junmin Gong, Yulin Song, Wenxiao Zhao, Sen Wang, Shengyuan Xu, Jing Guo},
	howpublished={\url{https://github.com/ace-step/ACE-Step-1.5}},
	year={2026},
	note={GitHub repository}
}
```

---

*ACE-Step 1.5로 음악을 "Play(연주/놀이)"하세요 - 단순히 재생하는 것이 아니라 창의적으로 노는 것입니다.* 🎵
