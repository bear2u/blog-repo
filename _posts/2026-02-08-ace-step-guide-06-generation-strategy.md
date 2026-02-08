---
layout: post
title: "ACE-Step 1.5 완벽 가이드 (06) - 음악 생성 전략"
date: 2026-02-08
permalink: /ace-step-guide-06-generation-strategy/
author: ACE Studio & StepFun
categories: [AI 음악, 오픈소스]
tags: [ACE-Step, AI Music, Prompt Engineering, Music Generation, CoT, Query Rewriting]
original_url: "https://github.com/ace-step/ACE-Step-1.5"
excerpt: "프롬프트 작성부터 고급 생성 전략까지 - ACE-Step 1.5 활용법"
---

## 개요

ACE-Step 1.5는 단순한 원클릭 생성이 아닌 **인간 중심의 창작 도구**입니다. 이 챕터에서는 효과적인 음악 생성을 위한 프롬프트 작성 원칙과 전략을 다룹니다.

---

## 프롬프트 작성 원칙

### Caption과 Lyrics의 역할

**Caption**은 음악의 "전체 초상화"입니다:
- 스타일, 장르, 악기
- 감정, 분위기, 질감
- 프로덕션 스타일, 음색 특징

**Lyrics**는 음악의 "시간적 시나리오"입니다:
- 가사 텍스트
- 구조 태그 (`[Verse]`, `[Chorus]`, `[Bridge]`)
- 보컬 스타일 힌트 (`[raspy vocal]`, `[whispered]`)
- 에너지 변화 (`[building energy]`, `[explosive drop]`)

### 핵심 원칙

| 원칙 | 설명 | 예시 |
|------|------|------|
| **구체성** | 구체적일수록 좋음 | ❌ "슬픈 노래" → ✅ "피아노 발라드, 허스키한 여성 보컬, 내밀한 분위기" |
| **다차원 조합** | 여러 차원 결합 | 스타일 + 감정 + 악기 + 질감 |
| **일관성** | Caption과 Lyrics 일치 | Caption: "피아노 발라드" → Lyrics: `[Piano Solo]` ✅ |
| **충돌 회피** | 모순된 요소 피하기 | ❌ "클래식 현악 + 하드코어 메탈" |
| **메타데이터 분리** | BPM/Key는 파라미터로 | Caption에 "120 BPM" 쓰지 말고 `bpm=120` 사용 |

---

## Simple Mode: 빠른 풀송 생성

**Simple Mode**는 자연어 설명만으로 완전한 노래를 생성합니다.

### 사용 방법

1. Gradio UI에서 "Simple" 모드 선택
2. "Song Description" 필드에 간단한 설명 입력
3. (선택) Instrumental 체크박스로 연주곡 지정
4. (선택) 보컬 언어 선택
5. **Create Sample** 클릭 → LM이 Caption, Lyrics, Metadata 자동 생성
6. **Generate Music** 클릭

### 예시 설명

```
a soft Bengali love song for a quiet evening
```

```
upbeat electronic dance music with heavy bass drops
```

```
melancholic indie folk with acoustic guitar and male vocals
```

```
jazz trio playing in a smoky bar, saxophone lead
```

### 작동 원리

Simple Mode는 5Hz LM이:
1. 자연어 설명 분석
2. 적절한 Caption 생성 (스타일, 악기, 분위기)
3. 구조화된 Lyrics 생성 (Verse, Chorus, Bridge 등)
4. Metadata 추론 (BPM, Key, Duration)

**장점**: 빠른 프로토타이핑, 초보자 친화적
**단점**: 세밀한 제어 어려움

---

## Advanced Mode: 메타데이터 제어

Custom Mode에서는 모든 파라미터를 직접 제어할 수 있습니다.

### Caption 작성 차원

| 차원 | 예시 |
|------|------|
| **스타일/장르** | pop, rock, jazz, electronic, hip-hop, R&B, folk, classical, lo-fi, synthwave |
| **감정/분위기** | melancholic, uplifting, energetic, dreamy, dark, nostalgic, euphoric, intimate |
| **악기** | acoustic guitar, piano, synth pads, 808 drums, strings, brass, electric bass |
| **음색 질감** | warm, bright, crisp, muddy, airy, punchy, lush, raw, polished |
| **시대 참조** | 80s synth-pop, 90s grunge, 2010s EDM, vintage soul, modern trap |
| **프로덕션 스타일** | lo-fi, high-fidelity, live recording, studio-polished, bedroom pop |
| **보컬 특징** | female vocal, male vocal, breathy, powerful, falsetto, raspy, choir |
| **속도/리듬** | slow tempo, mid-tempo, fast-paced, groovy, driving, laid-back |

### Caption 예시

```
female vocal, piano ballad, emotional, intimate atmosphere,
strings, building to powerful chorus, warm production,
reminiscent of Bon Iver
```

### Lyrics 구조 태그

#### 기본 구조

```
[Intro]         # 오프닝, 분위기 설정
[Verse 1]       # 1절, 내러티브 전개
[Pre-Chorus]    # 프리코러스, 에너지 축적
[Chorus]        # 후렴구, 감정 클라이맥스
[Bridge]        # 브릿지, 전환 또는 고조
[Outro]         # 엔딩, 마무리
```

#### 다이나믹 섹션

```
[Build]         # 에너지 점진적 상승
[Drop]          # 전자음악 에너지 해방
[Breakdown]     # 악기 축소, 공간감
```

#### 악기 섹션

```
[Instrumental]  # 순수 연주, 보컬 없음
[Guitar Solo]   # 기타 솔로
[Piano Interlude] # 피아노 간주
```

### 태그 조합 예시

```
[Chorus - anthemic]
Dreams are burning
We rise together

[Bridge - whispered]
Whisper those words softly
In the silence we find truth
```

⚠️ **주의**: 너무 많은 태그 중첩 피하기
```
❌ [Chorus - anthemic - stacked harmonies - high energy - powerful - epic]
✅ [Chorus - anthemic]
```

### 메타데이터 설정

| 파라미터 | 범위 | 설명 |
|----------|------|------|
| `bpm` | 30-300 | 템포. 느린 곡 60-80, 중간 90-120, 빠른 곡 130-180 |
| `keyscale` | C Major, Am 등 | 조성. C, G, D, Am, Em이 가장 안정적 |
| `timesignature` | 2/4, 3/4, 4/4, 6/8 | 박자. 4/4가 가장 신뢰도 높음 |
| `vocal_language` | en, zh, ja, ko 등 | 보컬 언어. auto-detect 가능 |
| `duration` | 10-600초 | 목표 길이. -1이면 자동 |

**참고**: 이들은 "엄격한 명령"이 아닌 "가이드라인"입니다. 모델은 이 값 근처에서 샘플링합니다.

---

## 참조 오디오 활용 전략

### 1. Reference Audio: 전역 음향 특성 제어

**제어 대상**:
- 음색 질감 (보컬 음색, 악기 음색)
- 믹싱 스타일 (공간감, 다이나믹 레인지, 주파수 분포)
- 연주 스타일 (보컬 기법, 연주 기법)
- 전체 분위기

**처리 과정**:
1. 오디오를 스테레오 48kHz로 정규화
2. 침묵 감지 및 제거
3. 30초 미만이면 반복하여 채움
4. 앞/중간/뒤에서 10초씩 랜덤 추출하여 30초 세그먼트 생성
5. VAE로 latent representation 인코딩
6. **시간 차원 평균화**하여 전역적으로 작용

**사용 예시**:
```python
reference_audio = "style_reference.wav"  # 원하는 음색/믹싱 스타일
caption = "rock song with electric guitars"
# → Reference 음색 + Caption 스타일 조합
```

### 2. Source Audio (Cover): 멜로디 구조 제어

**제어 대상**:
- 멜로디 (음정 방향과 높낮이)
- 리듬 (비트, 악센트, 그루브)
- 화음 (화성 진행과 변화)
- 편곡 (악기 배치와 레이어)
- 일부 음색

**Cover 활용법**:

```python
task_type = "cover"
src_audio = "original_song.mp3"
caption = "convert to jazz style with saxophone"
audio_cover_strength = 0.8  # 0.0-1.0
```

**Strength 조절**:
- `1.0`: 원곡 구조 최대한 유지
- `0.5`: 구조와 자유도 균형
- `0.0`: 구조 무시하고 새로 생성

**Remix 창작**:
```python
src_audio = "pop_song.mp3"
caption = "heavy metal rock with distorted guitars"
lyrics = """
[Verse 1 - aggressive]
새로운 가사로 변경
...
"""
# → 원곡 멜로디 유지 + 스타일/가사 변경
```

### 3. Repaint: 컨텍스트 기반 로컬 제어

**작업 범위**: 3초 ~ 90초

**사용 예시**:

```python
task_type = "repaint"
src_audio = "song.mp3"
repainting_start = 30  # 30초부터
repainting_end = 45    # 45초까지 재생성
caption = "change to piano solo section"
```

**활용 사례**:
1. **로컬 수정**: 특정 구간 가사/구조 변경
2. **이어쓰기**: 앞/뒤 연장
3. **음색 복제**: 컨텍스트 기반 음색 유지
4. **무한 길이 생성**: 여러 Repaint로 계속 연장
5. **지능형 오디오 연결**: 두 오디오 자연스럽게 연결

---

## Query Rewriting: LM 자동 확장

### CoT (Chain-of-Thought) 재작성

ACE-Step은 5Hz LM을 사용하여 입력을 자동으로 최적화합니다.

**활성화 방법**:
```python
thinking = True          # LM 활성화
use_cot_caption = True   # Caption 재작성
use_cot_metas = True     # Metadata 추론
use_cot_language = True  # 언어 자동 감지
```

### 재작성 예시

**입력**:
```
Caption: "sad piano song"
Lyrics: ""
BPM: Auto
```

**LM 재작성 후**:
```
Caption: "melancholic piano ballad, intimate atmosphere,
         female breathy vocal, warm strings, building to
         emotional chorus, reminiscent of Norah Jones"

Lyrics: """
[Intro - piano]

[Verse 1]
월광 같은 너의 목소리
내 마음을 적셔와
...

[Chorus - emotional]
우리의 시간은 흘러가도
이 순간만큼은 영원해
...
"""

BPM: 72
Key: Am
Duration: 180s
```

### Format 기능

**Format 버튼** 클릭 시 LM이:
1. Caption 확장 및 구체화
2. Lyrics 구조 개선
3. 태그 추가

```
입력: "electronic music"
→ 출력: "upbeat electronic dance music, heavy bass drops,
        synth leads, energetic atmosphere, club-ready production"
```

### Audio to Caption

오디오 파일로부터 Caption 추출:

```python
src_audio = "reference.mp3"
# LM이 스타일, 악기, 분위기 분석하여 Caption 생성
```

정확도는 제한적이지만 시작점으로 충분합니다.

---

## 50+ 언어 지원

ACE-Step은 다국어 보컬을 지원합니다.

### 지원 언어 (일부)

| 언어 | 코드 | 언어 | 코드 |
|------|------|------|------|
| English | en | 中文 | zh |
| 한국어 | ko | 日本語 | ja |
| Español | es | Français | fr |
| Deutsch | de | Português | pt |
| हिन्दी | hi | বাংলা | bn |
| العربية | ar | Русский | ru |

### 언어 설정

**자동 감지**:
```python
vocal_language = "unknown"  # LM이 가사에서 자동 감지
use_cot_language = True
```

**수동 지정**:
```python
vocal_language = "ko"  # 한국어로 강제
```

**다국어 노래**:
```
Lyrics: """
[Verse 1]
Hello world, this is my song

[Verse 2]
안녕하세요, 이것은 제 노래입니다

[Chorus]
We are one, 우리는 하나
"""
vocal_language = "unknown"  # 혼합 언어는 auto
```

---

## 랜덤 요소 및 최적화 전략

### 랜덤 요소의 양면성

**장점**:
- 창작 공간 탐험
- 예상치 못한 놀라움 발견
- 반복 패턴 회피

**단점**:
- 결과 예측 어려움
- 재현 어려움
- 파라미터 튜닝 혼란
- 선별 비용 증가

### 랜덤 제어: Seed

```python
seed = 42  # 고정 seed로 재현 가능
random_seed = False
```

**사용 전략**:
- **파라미터 테스트**: Seed 고정으로 파라미터 효과만 관찰
- **창작 탐험**: Seed 랜덤으로 다양한 변형 생성

### Large Batch + AutoGen + Auto Scoring

ACE-Step의 초고속 생성을 활용한 최적화:

```python
batch_size = 8      # 8개 동시 생성
auto_gen = True     # 자동으로 다음 배치 생성
auto_score = True   # 자동 품질 평가
```

**워크플로우**:
1. **배치 생성**: 8개 버전 동시 생성
2. **AutoGen**: 현재 배치 듣는 동안 백그라운드에서 다음 배치 생성
3. **자동 채점**: DiT Lyrics Alignment Score로 초기 스크리닝
4. **수동 선택**: 점수 높은 버전 중 최종 선택

**DiT Lyrics Alignment Score**:
- 가사와 오디오 정렬 정도 평가
- 높을수록 가사 정확도 높음
- 특히 가사 있는 음악에 중요

---

## Think Mode & Diffusion Steps

### Think Mode (LM 활성화)

```python
thinking = True
lm_temperature = 0.85     # 0.0-2.0, 높을수록 창의적
lm_cfg_scale = 2.0        # 1.0-3.0, 높을수록 프롬프트 준수
lm_top_p = 0.9            # Nucleus sampling
```

**Think Mode 효과**:
- Metadata 자동 추론 (BPM, Key, Duration)
- Caption 최적화 및 확장
- Semantic codes 생성 (멜로디, 편곡 정보)

**언제 끄나요?**:
- Cover mode (참조 오디오가 구조 제공)
- 명확한 계획이 이미 있을 때
- 저사양 GPU (< 8GB VRAM)

### Diffusion Steps

| 모델 | 기본 Steps | 범위 | 특징 |
|------|-----------|------|------|
| Turbo | 8 | 1-20 | 8 steps 최적, 더 늘려도 개선 미미 |
| SFT | 50 | 1-200 | Steps 많을수록 세밀하지만 느림 |
| Base | 32-100 | 1-200 | 특수 작업용 |

**Steps 조절**:
```python
inference_steps = 8   # Turbo 기본값
# 더 많은 steps = 더 세밀하지만 느림
# Turbo는 8 이상으로 늘려도 품질 개선 거의 없음
```

### Shift 파라미터

**Shift**는 DiT 디노이징의 "주의 집중 배분"을 결정합니다.

```python
shift = 3.0  # Turbo 권장
```

| Shift | 효과 |
|-------|------|
| 높음 (3-5) | 초기 구조 집중 → 명확한 골격, 강한 시맨틱 |
| 낮음 (1-2) | 균등 분배 → 풍부한 디테일, 시맨틱 약함 |

**Turbo 변형 모델**:
- `turbo` (기본): Shift 1, 2, 3 공동 학습 → 균형
- `turbo-shift1`: Shift=1 전용 → 디테일 풍부
- `turbo-shift3`: Shift=3 전용 → 명확한 음색
- `turbo-continuous`: Shift 1-5 연속 조절 가능

### CFG (Classifier-Free Guidance)

**Base/SFT 모델만 지원**:

```python
guidance_scale = 7.0  # 1.0-15.0
# 높을수록 프롬프트 엄격 준수, 너무 높으면 과적합
```

**ADG (Adaptive Dual Guidance)**:
```python
use_adg = True  # Base 모델에서 동적 CFG 조절
cfg_interval_start = 0.0
cfg_interval_end = 1.0
```

---

## 실전 예시

### 예시 1: 감성적인 피아노 발라드

```python
# Simple Mode
song_description = "emotional piano ballad with female vocals"

# → LM이 자동 생성:
caption = """
female vocal, piano ballad, emotional, intimate atmosphere,
strings, building to powerful chorus, warm production,
breathy vocal style, melancholic yet hopeful
"""

lyrics = """
[Intro - piano]

[Verse 1]
월광 속에 홀로 서서
그대 생각에 잠겨요
시간은 흘러가지만
우리의 추억은 남아

[Pre-Chorus]
조용한 이 밤에
당신의 목소리가 들려요

[Chorus - powerful]
영원히 함께할 거라 믿었죠
우리의 사랑은 변치 않을 거라고
하지만 지금 이 순간만큼은
당신을 느낄 수 있어요

[Verse 2]
별빛 아래 우리 둘이
나눴던 약속들이
아직도 내 마음속에
살아 숨 쉬고 있어요

[Final Chorus]
영원히 함께할 거라 믿었죠
우리의 사랑은 변치 않을 거라고
THIS IS OUR MOMENT

[Outro - fade out]
"""

bpm = 72
keyscale = "Am"
duration = 200
```

### 예시 2: 업비트 일렉트로닉 댄스

```python
caption = """
upbeat electronic dance music, heavy bass drops,
synth leads, energetic atmosphere, club-ready production,
punchy drums, modern EDM style, festival anthem
"""

lyrics = """
[Intro - building]

[Build]
Feel the energy rising
We're about to take off

[Drop]
[Instrumental]

[Verse 1]
밤이 깊어질수록
우리의 열기는 뜨거워져
리듬에 몸을 맡겨
이 순간을 느껴봐

[Build - high energy]
HANDS UP IN THE AIR
LET ME SEE YOU MOVE

[Drop]
[Instrumental]

[Outro - fade out]
"""

bpm = 128
timesignature = 4
vocal_language = "ko"
```

### 예시 3: 재즈 트리오 연주곡

```python
caption = """
jazz trio, smoky bar atmosphere, saxophone lead,
walking bass, brush drums, intimate setting,
late night vibe, acoustic, warm tones, sophisticated
"""

lyrics = """
[Instrumental]
"""

# 또는 구조적 가이드:
lyrics = """
[Intro - piano establishing]

[Main Theme - saxophone lead]

[Piano Solo]

[Saxophone Returns - expressive]

[Outro - fade out]
"""

bpm = 90
keyscale = "Bb Major"
```

---

## 다음 챕터 예고

다음 챕터에서는 **고급 기능**을 다룹니다:
- Cover Generation
- Repaint & Edit
- Multi-Track Generation (Add Layer)
- Vocal2BGM
- Track Separation
- Audio Understanding
- LRC Generation
- Quality Scoring

---

## 참고 자료

- [ACE-Step 1.5 Tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md)
- [Gradio UI Guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GRADIO_GUIDE.md)
- [Suno 마스터 가이드](https://www.notion.so/The-Complete-Guide-to-Mastering-Suno-Advanced-Strategies-for-Professional-Music-Generation-2d6ae744ebdf8024be42f6645f884221)
