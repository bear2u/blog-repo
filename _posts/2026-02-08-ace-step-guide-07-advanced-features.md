---
layout: post
title: "ACE-Step 1.5 완벽 가이드 (07) - 고급 기능"
date: 2026-02-08
permalink: /ace-step-guide-07-advanced-features/
author: ACE Studio & StepFun
categories: [AI 음악, 오픈소스]
tags: [ACE-Step, AI Music, Cover Generation, Repaint, Multi-Track, Vocal2BGM, Track Separation]
original_url: "https://github.com/ace-step/ACE-Step-1.5"
excerpt: "커버 생성부터 트랙 분리까지 - ACE-Step 1.5의 강력한 고급 기능"
---

## 개요

ACE-Step 1.5는 단순한 text-to-music을 넘어 다양한 고급 음악 제작 기능을 제공합니다. 이 챕터에서는 프로페셔널한 음악 제작을 위한 8가지 핵심 기능을 상세히 다룹니다.

---

## 1. Cover Generation (커버 생성)

### 개념

**Cover Mode**는 원곡의 멜로디 구조를 유지하면서 스타일, 음색, 장르를 변경하는 기능입니다.

### 작동 원리

1. Source Audio를 **5Hz semantic codes**로 변환
2. Codes에 멜로디, 리듬, 화음, 편곡 정보 인코딩
3. 새로운 Caption과 함께 DiT에 전달
4. 구조는 유지하되 스타일/음색 변경

### 사용 방법

**Gradio UI**:
```
Task Type: cover
Source Audio: [원곡 업로드]
Caption: "jazz version with saxophone and upright bass"
Audio Cover Strength: 0.8
```

**Python API**:
```python
from acestep import generate

result = generate(
    task_type="cover",
    src_audio="original_song.mp3",
    caption="heavy metal rock with distorted guitars and aggressive drums",
    audio_cover_strength=0.8,
    thinking=False  # Cover는 구조 참조하므로 LM 선택사항
)
```

### Cover Strength 조절

| Strength | 효과 | 사용 사례 |
|----------|------|----------|
| `1.0` | 원곡 구조 최대한 충실 | 정확한 커버 버전 |
| `0.8` | 구조 유지, 약간의 자유도 | 일반적인 커버 (권장) |
| `0.5` | 구조와 창의성 균형 | 리믹스 스타일 |
| `0.3` | 구조 약하게 참조 | 영감만 가져오기 |
| `0.0` | 구조 무시 | 사실상 text2music |

### 실전 예시

**예시 1: 팝송을 재즈로**
```python
src_audio = "pop_song.mp3"
caption = """
jazz arrangement, smooth saxophone lead, walking bass,
brush drums, smoky bar atmosphere, sophisticated,
warm acoustic tones, intimate setting
"""
audio_cover_strength = 0.75
```

**예시 2: 어쿠스틱을 일렉트로닉으로**
```python
src_audio = "acoustic_ballad.mp3"
caption = """
electronic version, heavy synth pads, pulsing bass,
electronic drums, ambient atmosphere, modern production,
ethereal synth leads, reverb-heavy
"""
audio_cover_strength = 0.7
```

**예시 3: 가사 변경 Remix**
```python
src_audio = "original.mp3"
caption = "rock version with electric guitars"
lyrics = """
[Verse 1]
새로운 가사 내용
원곡 멜로디는 유지하되 가사만 변경

[Chorus - powerful]
REWRITTEN CHORUS LYRICS
...
"""
audio_cover_strength = 0.8
```

### 고급 활용: 복잡한 음악 구조 구축

Cover를 여러 번 체인하여 복잡한 구조 만들기:

```python
# 1단계: 간단한 멜로디 생성
base = generate(
    caption="simple piano melody",
    lyrics="[Instrumental]"
)

# 2단계: 오케스트라 버전으로 확장
orchestral = generate(
    task_type="cover",
    src_audio=base,
    caption="full orchestral arrangement with strings, brass, timpani",
    audio_cover_strength=0.6
)

# 3단계: 일렉트로닉 요소 추가
final = generate(
    task_type="cover",
    src_audio=orchestral,
    caption="hybrid orchestral electronic with synth bass and electronic drums",
    audio_cover_strength=0.5
)
```

---

## 2. Repaint & Edit (선택적 로컬 편집)

### 개념

**Repaint**는 기존 오디오의 특정 시간 구간만 재생성하여 로컬 수정하는 기능입니다.

### 작동 원리

- 컨텍스트 기반 완성 (Inpainting)
- 지정 구간 외부는 그대로 유지
- 주변 컨텍스트를 참조하여 자연스럽게 연결

### 작업 범위

- **최소**: 3초
- **최대**: 90초
- **위치**: 시작, 중간, 끝, 어디든 가능

### 사용 방법

**Gradio UI**:
```
Task Type: repaint
Source Audio: [원본 업로드]
Repainting Start: 30
Repainting End: 45
Caption: "change to piano solo section"
```

**Python API**:
```python
result = generate(
    task_type="repaint",
    src_audio="song.mp3",
    repainting_start=30,  # 30초부터
    repainting_end=45,     # 45초까지 재생성
    caption="energetic guitar solo",
    lyrics="[Guitar Solo - expressive]"
)
```

### 활용 사례

#### 1. 가사 변경

```python
# 30-50초 구간 가사만 변경, 멜로디 유지
result = generate(
    task_type="repaint",
    src_audio="song.mp3",
    repainting_start=30,
    repainting_end=50,
    caption="same style, maintain melody and instrumentation",
    lyrics="""
[Verse 2]
새로운 가사 내용
다른 이야기로 변경
"""
)
```

#### 2. 구조 변경

```python
# Verse를 Chorus로 변경
result = generate(
    task_type="repaint",
    src_audio="song.mp3",
    repainting_start=60,
    repainting_end=80,
    caption="energetic, powerful, anthemic chorus",
    lyrics="""
[Chorus - anthemic]
WE RISE TOGETHER
INTO THE LIGHT
"""
)
```

#### 3. 이어쓰기 (Continuation)

```python
# 끝부분 연장 (end=-1은 파일 끝을 의미)
result = generate(
    task_type="repaint",
    src_audio="song.mp3",
    repainting_start=180,  # 3분부터 끝까지
    repainting_end=-1,
    caption="extended outro with fade out",
    lyrics="[Outro - fade out]"
)
```

#### 4. 인트로 추가

```python
# 시작 부분 추가
result = generate(
    task_type="repaint",
    src_audio="song.mp3",
    repainting_start=0,
    repainting_end=10,
    caption="atmospheric intro with ambient pads",
    lyrics="[Intro - atmospheric]"
)
```

### 고급 활용: 무한 길이 생성

Repaint를 반복하여 긴 곡 만들기:

```python
audio = initial_generation  # 초기 2분 곡

# 30초씩 계속 연장
for i in range(10):  # 5분 추가 (10 × 30초)
    audio = generate(
        task_type="repaint",
        src_audio=audio,
        repainting_start=len(audio) - 10,  # 마지막 10초 오버랩
        repainting_end=-1,
        caption="continue in same style",
        duration=30
    )
```

### 고급 활용: 지능형 오디오 연결

두 오디오를 자연스럽게 연결:

```python
# audio1과 audio2를 연결
transition = generate(
    task_type="repaint",
    src_audio=concat(audio1, audio2),  # 단순 연결
    repainting_start=len(audio1) - 5,  # 연결 지점 전 5초
    repainting_end=len(audio1) + 5,    # 연결 지점 후 5초
    caption="smooth transition, blend both styles"
)
```

---

## 3. Multi-Track Generation (Add Layer)

### 개념 (Base Model 전용)

**Lego Task**는 기존 트랙에 새로운 악기 레이어를 지능적으로 추가하는 기능입니다.

### 사용 방법

**Gradio UI**:
```
Task Type: lego
Source Audio: [베이스 트랙 업로드]
Track Name: drums
Caption: "energetic rock drums, driving rhythm"
```

**Python API**:
```python
result = generate(
    task_type="lego",
    src_audio="vocal_track.wav",
    track_name="drums",
    caption="heavy rock drums with double kick"
)
```

### 지원 트랙

| 트랙 이름 | 설명 |
|----------|------|
| `vocals` | 리드 보컬 |
| `backing_vocals` | 백킹 보컬, 하모니 |
| `drums` | 드럼, 퍼커션 |
| `bass` | 베이스 라인 |
| `guitar` | 기타 (어쿠스틱/일렉트릭) |
| `keyboard` | 키보드, 피아노 |
| `percussion` | 추가 퍼커션 |
| `strings` | 현악기 (바이올린, 첼로 등) |
| `synth` | 신시사이저 |
| `fx` | 효과음, 앰비언트 |
| `brass` | 금관악기 (트럼펫, 색소폰 등) |
| `woodwinds` | 목관악기 (플룻, 클라리넷 등) |

### 실전 예시

**예시 1: 어쿠스틱 기타에 드럼 추가**
```python
# Step 1: 어쿠스틱 기타 생성
guitar = generate(
    caption="acoustic guitar fingerpicking, folk style",
    lyrics="[Instrumental]"
)

# Step 2: 드럼 레이어 추가
with_drums = generate(
    task_type="lego",
    src_audio=guitar,
    track_name="drums",
    caption="light brush drums, jazz style, subtle rhythm"
)
```

**예시 2: 보컬에 풀 밴드 추가**
```python
# 보컬만 있는 트랙
vocal = "acapella.wav"

# 베이스 추가
with_bass = generate(
    task_type="lego",
    src_audio=vocal,
    track_name="bass",
    caption="groovy bass line, funk style"
)

# 드럼 추가
with_drums = generate(
    task_type="lego",
    src_audio=with_bass,
    track_name="drums",
    caption="funky drums with syncopation"
)

# 기타 추가
full_band = generate(
    task_type="lego",
    src_audio=with_drums,
    track_name="guitar",
    caption="rhythm guitar, funky strumming"
)
```

**예시 3: 오케스트라 레이어링**
```python
# 피아노 베이스
piano = generate(caption="grand piano, classical", lyrics="[Instrumental]")

# 현악 추가
with_strings = generate(
    task_type="lego",
    src_audio=piano,
    track_name="strings",
    caption="lush string section, romantic, soaring"
)

# 금관 추가
orchestral = generate(
    task_type="lego",
    src_audio=with_strings,
    track_name="brass",
    caption="powerful brass section, cinematic, epic"
)
```

---

## 4. Vocal2BGM (보컬 반주 자동 생성)

### 개념 (Base Model 전용)

**Complete Task**는 단일 트랙(주로 보컬)에 자동으로 완전한 반주를 생성하는 기능입니다.

### 사용 방법

**Gradio UI**:
```
Task Type: complete
Source Audio: [보컬 트랙 업로드]
Track Names: [drums, bass, guitar, keyboard 선택]
Caption: "rock band arrangement"
```

**Python API**:
```python
result = generate(
    task_type="complete",
    src_audio="acapella_vocals.wav",
    track_names=["drums", "bass", "guitar", "keyboard"],
    caption="rock band arrangement, energetic, driving rhythm"
)
```

### 실전 예시

**예시 1: 아카펠라를 팝송으로**
```python
result = generate(
    task_type="complete",
    src_audio="acapella.wav",
    track_names=["drums", "bass", "guitar", "keyboard", "synth"],
    caption="""
modern pop production, electronic drums, synth bass,
electric guitar, pad synths, polished studio sound
"""
)
```

**예시 2: 보컬을 재즈 반주로**
```python
result = generate(
    task_type="complete",
    src_audio="vocal_only.wav",
    track_names=["drums", "bass", "guitar", "keyboard"],
    caption="""
jazz trio accompaniment, brush drums, walking bass,
jazz guitar chords, piano comping, smoky bar atmosphere
"""
)
```

**예시 3: 오케스트라 반주**
```python
result = generate(
    task_type="complete",
    src_audio="opera_vocal.wav",
    track_names=["strings", "brass", "woodwinds", "percussion"],
    caption="""
full orchestral accompaniment, lush strings, powerful brass,
delicate woodwinds, timpani, cinematic, dramatic
"""
)
```

---

## 5. Track Separation (트랙 분리)

### 개념 (Base Model 전용)

**Extract Task**는 믹스된 오디오에서 특정 악기/보컬 트랙을 분리 추출하는 기능입니다.

### 사용 방법

**Gradio UI**:
```
Task Type: extract
Source Audio: [믹스 오디오 업로드]
Track Name: vocals
```

**Python API**:
```python
# 보컬만 추출
vocals = generate(
    task_type="extract",
    src_audio="full_mix.mp3",
    track_name="vocals"
)

# 드럼만 추출
drums = generate(
    task_type="extract",
    src_audio="full_mix.mp3",
    track_name="drums"
)
```

### 실전 예시

**예시 1: 스템 분리**
```python
full_mix = "song.mp3"

# 모든 트랙 분리
vocals = generate(task_type="extract", src_audio=full_mix, track_name="vocals")
drums = generate(task_type="extract", src_audio=full_mix, track_name="drums")
bass = generate(task_type="extract", src_audio=full_mix, track_name="bass")
guitar = generate(task_type="extract", src_audio=full_mix, track_name="guitar")
keyboard = generate(task_type="extract", src_audio=full_mix, track_name="keyboard")
```

**예시 2: 리믹스를 위한 아카펠라 추출**
```python
# 보컬 추출
acapella = generate(
    task_type="extract",
    src_audio="original_song.mp3",
    track_name="vocals"
)

# 추출된 보컬로 새 반주 생성
remix = generate(
    task_type="complete",
    src_audio=acapella,
    track_names=["drums", "bass", "synth"],
    caption="electronic dance remix with heavy bass"
)
```

**예시 3: 커버 제작 워크플로우**
```python
# 1. 원곡에서 보컬 제거 (반주만 추출)
instrumental = generate(
    task_type="extract",
    src_audio="original.mp3",
    track_name="vocals",
    # NOTE: 보컬을 추출한 뒤 반전하거나, 별도로 instrumental 추출
)

# 2. 자신의 보컬 녹음
my_vocal = record_vocal()

# 3. 반주와 합성
cover = mix(my_vocal, instrumental)
```

---

## 6. Audio Understanding (오디오 분석)

### BPM 추출

ACE-Step의 5Hz LM은 오디오에서 BPM을 자동 추출할 수 있습니다.

```python
# Audio to Caption 기능 사용
metadata = analyze_audio("song.mp3")
# → BPM, Key, Time Signature 자동 추출
```

**Gradio UI**:
```
Audio Uploads → Source Audio 업로드
→ Convert to Codes 클릭
→ Transcribe 버튼 클릭
→ BPM, Key 등 메타데이터 표시
```

### Key/Scale 추출

```python
# LM이 오디오 분석하여 조성 추출
# 예: "C Major", "Am", "F# Minor"
```

### 활용 사례

**예시 1: 커버 제작 시 원곡 분석**
```python
# 원곡 메타데이터 추출
metadata = analyze_audio("original.mp3")
bpm = metadata["bpm"]          # 120
key = metadata["keyscale"]     # "C Major"

# 같은 BPM/Key로 커버 생성
cover = generate(
    task_type="cover",
    src_audio="original.mp3",
    caption="jazz version",
    bpm=bpm,
    keyscale=key
)
```

**예시 2: 프로젝트 템포 통일**
```python
# 여러 트랙의 BPM 맞추기
track1_bpm = analyze_audio("track1.mp3")["bpm"]
track2_bpm = analyze_audio("track2.mp3")["bpm"]

# BPM 통일하여 재생성
track2_adjusted = generate(
    task_type="cover",
    src_audio="track2.mp3",
    caption="maintain style",
    bpm=track1_bpm
)
```

---

## 7. LRC Generation (가사 타임스탬프)

### 개념

**LRC (Lyrics)**는 가사와 타임스탬프를 포함한 동기화 가사 파일 형식입니다.

### 생성 방법

**Gradio UI**:
```
Results → 생성된 오디오에서 "LRC" 버튼 클릭
→ LRC 형식 가사 표시
```

**Python API**:
```python
result = generate(
    caption="pop song",
    lyrics="""
[Verse 1]
Walking down the street
...
""",
    auto_lrc=True  # 자동 LRC 생성
)

lrc_content = result["lrc"]
```

### LRC 형식 예시

```
[00:12.50]Walking down the street
[00:15.30]Thinking of the words you used to say
[00:18.70]I'm moving on, I'm staying strong
[00:22.40]This is where I belong
```

### 활용 사례

**예시 1: 카라오케 제작**
```python
song = generate(
    caption="karaoke backing track, energetic pop",
    lyrics="""
[Verse 1]
노래 가사 내용
...
[Chorus]
후렴구 내용
...
""",
    auto_lrc=True
)

# LRC 파일 저장
save_lrc(song["lrc"], "karaoke.lrc")
```

**예시 2: 가사 비디오 제작**
```python
# LRC 타임스탬프 사용하여 자막 비디오 생성
lrc = song["lrc"]
video = create_lyrics_video(
    audio=song["audio"],
    lrc=lrc,
    background="visualizer.mp4"
)
```

---

## 8. Quality Scoring (품질 평가)

### 자동 품질 평가

ACE-Step은 생성된 음악의 품질을 자동으로 평가합니다.

### 주요 메트릭

#### 1. DiT Lyrics Alignment Score

**가장 중요한 메트릭**:
- 가사와 오디오의 정렬 정도 평가
- 높을수록 가사 정확도 높음
- 가사 있는 음악에 특히 중요

```python
result = generate(
    caption="pop song",
    lyrics="...",
    auto_score=True
)

score = result["lyrics_alignment_score"]
# 높은 점수 = 가사 정확하게 싱잉
```

#### 2. Perplexity-based Quality

전반적인 생성 품질 평가:
- 낮을수록 좋음
- 모델의 확신도 측정

### 사용 방법

**Gradio UI**:
```
Advanced Settings → Auto Score 체크
→ 생성 후 각 샘플의 점수 자동 표시
```

**Python API**:
```python
result = generate(
    caption="...",
    batch_size=8,
    auto_score=True
)

# 점수 순으로 정렬
sorted_results = sorted(
    result["batch"],
    key=lambda x: x["lyrics_alignment_score"],
    reverse=True
)

best = sorted_results[0]  # 가장 높은 점수
```

### Large Batch + AutoGen + Scoring 워크플로우

```python
# 최적화된 생성 워크플로우
result = generate(
    caption="emotional ballad",
    lyrics="...",
    batch_size=8,        # 8개 동시 생성
    auto_score=True,     # 자동 채점
    auto_gen=True        # 백그라운드 생성 계속
)

# 생성하는 동안 현재 배치 듣기
# → 마음에 안 들면 "Next" 클릭 (이미 다음 배치 준비됨)
# → 좋은 샘플 발견 시 점수 확인 및 저장
```

### 활용 사례

**예시 1: 최적 버전 자동 선택**
```python
# 20개 생성하여 최고 점수 자동 선택
best_song = None
best_score = 0

for i in range(5):  # 5 배치 × 4개 = 20개
    batch = generate(
        caption="...",
        batch_size=4,
        auto_score=True
    )

    for song in batch:
        if song["lyrics_alignment_score"] > best_score:
            best_score = song["lyrics_alignment_score"]
            best_song = song

save(best_song, "best_version.mp3")
```

**예시 2: 품질 임계값 설정**
```python
# 특정 점수 이상만 수락
threshold = 0.85

while True:
    result = generate(caption="...", auto_score=True)

    if result["lyrics_alignment_score"] >= threshold:
        print("Quality threshold met!")
        save(result, "final.mp3")
        break
    else:
        print(f"Score {result['lyrics_alignment_score']:.2f} below threshold, regenerating...")
```

---

## 고급 워크플로우 조합

### 워크플로우 1: 완벽한 커버 제작

```python
# 1. 원곡 분석
metadata = analyze_audio("original.mp3")

# 2. 커버 생성 (여러 버전)
covers = []
for i in range(8):
    cover = generate(
        task_type="cover",
        src_audio="original.mp3",
        caption="jazz version with saxophone",
        audio_cover_strength=0.75,
        bpm=metadata["bpm"],
        keyscale=metadata["keyscale"],
        auto_score=True
    )
    covers.append(cover)

# 3. 최고 점수 선택
best_cover = max(covers, key=lambda x: x["lyrics_alignment_score"])

# 4. 특정 구간 재생성 (필요시)
final = generate(
    task_type="repaint",
    src_audio=best_cover["audio"],
    repainting_start=60,
    repainting_end=80,
    caption="more energetic saxophone solo"
)
```

### 워크플로우 2: 스템 기반 리믹스

```python
# 1. 원곡 스템 분리
vocals = generate(task_type="extract", src_audio="song.mp3", track_name="vocals")
drums = generate(task_type="extract", src_audio="song.mp3", track_name="drums")

# 2. 새 베이스/신스 생성
new_track = generate(
    task_type="lego",
    src_audio=vocals,
    track_name="synth",
    caption="heavy electronic bass, EDM style"
)

# 3. 드럼 추가
remix = generate(
    task_type="lego",
    src_audio=new_track,
    track_name="drums",
    caption="electronic drums, four-on-the-floor"
)

# 4. 트랜지션 추가
final = generate(
    task_type="repaint",
    src_audio=remix,
    repainting_start=0,
    repainting_end=10,
    caption="epic buildup intro"
)
```

### 워크플로우 3: AI 작곡가 협업

```python
# 1. Simple Mode로 초안 생성
draft = generate(
    mode="simple",
    song_description="emotional piano ballad about lost love"
)

# 2. 특정 섹션 개선
improved_chorus = generate(
    task_type="repaint",
    src_audio=draft,
    repainting_start=60,
    repainting_end=90,
    caption="more powerful, anthemic chorus with strings"
)

# 3. 오케스트라 레이어 추가
with_strings = generate(
    task_type="lego",
    src_audio=improved_chorus,
    track_name="strings",
    caption="lush string section, cinematic"
)

# 4. 아웃트로 연장
final = generate(
    task_type="repaint",
    src_audio=with_strings,
    repainting_start=180,
    repainting_end=-1,
    caption="extended outro with piano solo and fade out",
    duration=30
)
```

---

## 성능 팁

### GPU 메모리 최적화

```python
# 큰 배치 작업 시
batch_size = 8
lm_batch_chunk_size = 4  # LM 배치를 4개씩 나눠 처리
```

### 오프로딩 활용

```python
# Gradio UI Service Configuration
offload_to_cpu = True      # 유휴 시 CPU로 오프로드
offload_dit_to_cpu = True  # DiT 모델 오프로드
```

### 병렬 처리

```python
# 여러 작업 동시 실행
import concurrent.futures

tasks = [
    {"task_type": "cover", "src_audio": "song1.mp3", "caption": "jazz"},
    {"task_type": "cover", "src_audio": "song2.mp3", "caption": "rock"},
    {"task_type": "extract", "src_audio": "song3.mp3", "track_name": "vocals"},
]

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(lambda t: generate(**t), tasks))
```

---

## 다음 챕터 예고

다음 챕터에서는 **LoRA 훈련**을 다룹니다:
- LoRA 개념 및 원리
- 데이터셋 준비 (8곡 권장)
- Gradio UI의 LoRA Training 탭
- 원클릭 어노테이션 및 훈련
- 훈련 파라미터 최적화
- 커스텀 스타일 모델 만들기

---

## 참고 자료

- [ACE-Step 1.5 Gradio Guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GRADIO_GUIDE.md)
- [ACE-Step 1.5 Tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md)
- [ACE-Step 1.5 API Documentation](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md)
