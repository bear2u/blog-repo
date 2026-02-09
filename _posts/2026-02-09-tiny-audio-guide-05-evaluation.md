---
layout: post
title: "Tiny Audio 완벽 가이드 (05) - 평가 및 분석"
date: 2026-02-09
permalink: /tiny-audio-guide-05-evaluation/
author: Alex Kroman
categories: [머신러닝, 음성인식]
tags: [ASR, Speech Recognition, GLM-ASR, Qwen3, PyTorch, HuggingFace, Audio ML]
original_url: "https://github.com/alexkroman/tiny-audio"
excerpt: "훈련된 ASR 모델을 다양한 메트릭으로 평가하고 오류를 분석하는 방법을 알아봅니다."
---

## CLI 평가 도구

Tiny Audio는 강력한 CLI 평가 도구를 제공합니다.

### 기본 평가

```bash
# 기본 테스트셋으로 평가
ta eval --checkpoint outputs/transcription/checkpoints/final.pt
```

출력:

```
[INFO] Loading model from checkpoint...
[INFO] Loading test dataset (default)...
[INFO] Evaluating 2,620 samples...

Progress: [========================================] 2620/2620

Results:
  WER (Word Error Rate): 5.2%
  CER (Character Error Rate): 2.1%
  Samples: 2,620
  Total duration: 4.8 hours
  Average duration: 6.6 seconds
  RTF (Real-Time Factor): 0.15

Detailed metrics:
  Substitutions: 892 (3.1%)
  Deletions: 412 (1.4%)
  Insertions: 198 (0.7%)
  Correct: 27,498 (95.8%)

Time statistics:
  Total inference time: 43m 12s
  Average per sample: 0.99s
  Throughput: 6.67 hours/hour
```

### 평가 옵션

```bash
# 샘플 수 제한
ta eval \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --max-samples 100

# 특정 데이터셋 사용
ta eval \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --dataset librispeech_test_clean

# 배치 크기 조정
ta eval \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --batch-size 8

# 결과 저장
ta eval \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --output results.json \
    --save-predictions predictions.jsonl
```

## 데이터셋 평가

### Default 테스트셋

기본 설정은 혼합 테스트셋을 사용합니다:

```yaml
# configs/data/test_default.yaml
datasets:
  - name: librispeech
    split: test-clean
    samples: 1000

  - name: common_voice
    language: en
    split: test
    samples: 500

  - name: tedlium
    release: 3
    split: test
    samples: 500

  - name: voxpopuli
    language: en
    split: test
    samples: 620

total_samples: 2,620
```

실행:

```bash
ta eval --checkpoint outputs/transcription/checkpoints/final.pt
```

### Loquacious 테스트셋

실제 환경에 가까운 어려운 데이터:

```yaml
# configs/data/test_loquacious.yaml
datasets:
  - name: earnings22  # 실제 기업 실적 발표
    split: test
    note: "Challenging: jargon, numbers, background noise"

  - name: rev16  # Rev.ai의 어려운 벤치마크
    split: test
    note: "Various accents and recording conditions"

  - name: swb  # Switchboard 전화 대화
    split: test
    note: "Conversational, overlapping speech"

total_samples: 1,500
expected_wer: 12-18%  # 더 어려움
```

실행:

```bash
ta eval \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --dataset loquacious
```

예상 결과:

```
Results:
  WER: 14.3%  (기본 테스트의 약 3배)
  CER: 6.8%

Per-dataset breakdown:
  earnings22: WER=16.2%  (숫자와 전문용어)
  rev16: WER=13.8%       (다양한 악센트)
  swb: WER=12.9%         (대화체)
```

### Custom 데이터셋

자체 데이터로 평가:

```yaml
# configs/data/test_custom.yaml
name: my_custom_test
audio_dir: /path/to/audio/
transcript_file: /path/to/transcripts.json
format: json  # or csv, txt

# JSON 형식:
# [
#   {
#     "audio": "audio1.wav",
#     "text": "This is the ground truth transcription"
#   },
#   ...
# ]
```

실행:

```bash
ta eval \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --dataset custom \
    --config configs/data/test_custom.yaml
```

### 도메인별 평가

특정 도메인의 성능 확인:

```bash
# 의료
ta eval \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --dataset medical \
    --output results_medical.json

# 법률
ta eval \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --dataset legal \
    --output results_legal.json

# 기술
ta eval \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --dataset technical \
    --output results_technical.json
```

## 다른 모델과 비교

### 상용 ASR 서비스 비교

AssemblyAI, Deepgram 등과 비교:

```bash
# Tiny Audio 평가
ta eval \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --dataset comparison_benchmark \
    --output tiny_audio_results.json

# AssemblyAI 평가
ta eval \
    --provider assemblyai \
    --api-key $ASSEMBLYAI_API_KEY \
    --dataset comparison_benchmark \
    --output assemblyai_results.json

# Deepgram 평가
ta eval \
    --provider deepgram \
    --api-key $DEEPGRAM_API_KEY \
    --dataset comparison_benchmark \
    --output deepgram_results.json

# 결과 비교
ta compare \
    tiny_audio_results.json \
    assemblyai_results.json \
    deepgram_results.json
```

비교 결과:

```
Model Comparison on comparison_benchmark (1,000 samples)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model           WER    CER    RTF    Cost/hour   Quality
────────────────────────────────────────────────────────
Tiny Audio      5.2%   2.1%   0.15   $0.00       ★★★★☆
AssemblyAI      4.8%   1.9%   0.28   $0.37       ★★★★★
Deepgram        5.1%   2.0%   0.22   $0.43       ★★★★☆
Whisper Large   4.5%   1.8%   0.45   $0.00       ★★★★★
Whisper Medium  6.1%   2.5%   0.18   $0.00       ★★★★☆

Summary:
- Tiny Audio는 무료 모델 중 가장 빠름
- 상용 서비스와 비슷한 정확도
- 프라이버시: 데이터를 외부로 전송하지 않음
- 커스터마이징: 자유롭게 fine-tuning 가능
```

### 벤치마크 스위트

표준 벤치마크에서 평가:

```bash
# LibriSpeech 벤치마크
ta benchmark librispeech \
    --checkpoint outputs/transcription/checkpoints/final.pt

# 출력:
# test-clean: WER=3.2%
# test-other: WER=8.7%

# Common Voice 벤치마크
ta benchmark common_voice \
    --checkpoint outputs/transcription/checkpoints/final.pt

# TEDLIUM 벤치마크
ta benchmark tedlium3 \
    --checkpoint outputs/transcription/checkpoints/final.pt
```

## WER 분석 도구

Word Error Rate를 상세히 분석합니다.

### high-wer: 높은 WER 샘플 분석

WER이 높은 샘플을 찾아 패턴 파악:

```bash
ta analyze high-wer \
    --predictions predictions.jsonl \
    --threshold 20.0 \
    --output high_wer_analysis.json
```

출력:

```
High WER Samples Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Found 127 samples with WER > 20.0%

Top 10 worst samples:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Sample ID: audio_1234.wav
   WER: 67.3%
   Reference: "The quarterly revenue was twenty three point five million dollars"
   Hypothesis: "The quarterly revenue was twenty three point five million"
   Issues: Missing words, numbers
   Duration: 4.2s
   SNR: 12.3 dB (low)

2. Sample ID: audio_5678.wav
   WER: 58.1%
   Reference: "Dr. Smith discussed the implementation of HIPAA regulations"
   Hypothesis: "Doctor Smith discussed the implementation of hippo regulations"
   Issues: Proper nouns, acronyms
   Duration: 5.8s
   SNR: 18.7 dB

3. Sample ID: audio_9012.wav
   WER: 52.4%
   Reference: "She said quote the results are inconclusive unquote"
   Hypothesis: "She said the results are inconclusive"
   Issues: Missing punctuation markers
   Duration: 3.9s
   SNR: 15.2 dB

Common patterns in high WER samples:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Numbers and dates (43% of high-WER samples)
2. Proper nouns and acronyms (35%)
3. Low SNR / background noise (28%)
4. Fast speech rate (22%)
5. Technical jargon (19%)

Recommendations:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Add more number-heavy training data
- Create custom vocabulary for proper nouns
- Apply audio enhancement preprocessing
- Consider speed perturbation augmentation
```

상세 분석 JSON:

```json
{
  "summary": {
    "total_high_wer_samples": 127,
    "threshold": 20.0,
    "average_wer": 31.2,
    "median_wer": 26.8
  },
  "samples": [
    {
      "id": "audio_1234.wav",
      "wer": 67.3,
      "reference": "The quarterly revenue was...",
      "hypothesis": "The quarterly revenue was...",
      "alignment": [
        {"ref": "The", "hyp": "The", "type": "correct"},
        {"ref": "quarterly", "hyp": "quarterly", "type": "correct"},
        {"ref": "dollars", "hyp": null, "type": "deletion"}
      ],
      "metadata": {
        "duration": 4.2,
        "snr": 12.3,
        "speaker_id": "speaker_042"
      }
    }
  ],
  "patterns": {
    "numbers": 0.43,
    "proper_nouns": 0.35,
    "low_snr": 0.28,
    "fast_speech": 0.22,
    "jargon": 0.19
  }
}
```

### entity-errors: 엔티티 오류 분석

특정 타입의 단어에서 발생하는 오류 분석:

```bash
ta analyze entity-errors \
    --predictions predictions.jsonl \
    --output entity_errors.json
```

출력:

```
Entity Error Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Numbers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: 1,234 occurrences
Errors: 187 (15.2%)

Common mistakes:
  "fifteen" → "fifty"          (23 times)
  "thirteen" → "thirty"        (19 times)
  "2023" → "twenty twenty three" (17 times)
  "$3.50" → "three dollars fifty" (15 times)

Dates
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: 456 occurrences
Errors: 89 (19.5%)

Common mistakes:
  "June 15th" → "June 50th"    (8 times)
  "2023" → "twenty twenty three" (7 times)

Proper Nouns
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: 789 occurrences
Errors: 156 (19.8%)

Common mistakes:
  "COVID" → "kovid"            (12 times)
  "LinkedIn" → "linked in"     (10 times)
  "Nvidia" → "in video"        (9 times)

Acronyms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: 345 occurrences
Errors: 98 (28.4%)

Common mistakes:
  "API" → "a p i"              (15 times)
  "CEO" → "c e o"              (12 times)
  "FBI" → "f b i"              (11 times)

Technical Terms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: 567 occurrences
Errors: 134 (23.6%)

Common mistakes:
  "kubernetes" → "cooler natives" (8 times)
  "algorithm" → "algarithm"       (7 times)
  "cache" → "cash"                (6 times)
```

개선 방법:

```python
# 커스텀 vocabulary 추가
custom_vocab = [
    "COVID-19",
    "LinkedIn",
    "Nvidia",
    "API",
    "CEO",
    "kubernetes",
    # ... 도메인 특화 단어들
]

# 훈련 시 적용
ta train \
    experiment=transcription \
    data.custom_vocabulary=custom_vocab.txt
```

### compare: 모델 간 비교

여러 모델의 결과를 직접 비교:

```bash
# 3개 모델 평가
ta eval --checkpoint model_v1.pt --output results_v1.jsonl
ta eval --checkpoint model_v2.pt --output results_v2.jsonl
ta eval --checkpoint model_v3.pt --output results_v3.jsonl

# 비교
ta analyze compare \
    results_v1.jsonl \
    results_v2.jsonl \
    results_v3.jsonl \
    --output comparison.html
```

비교 리포트:

```
Model Comparison Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Overall Metrics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model      WER    CER    Better  Worse   Same
───────────────────────────────────────────────
model_v1   5.2%   2.1%   -       -       -      (baseline)
model_v2   4.8%   1.9%   823     412     1385   (↑0.4%p)
model_v3   4.5%   1.8%   1124    298     1198   (↑0.7%p)

Samples where model_v3 is significantly better
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. audio_1234.wav
   Reference: "The revenue was $23.5 million"
   model_v1: "The revenue was twenty three dollars five million" (WER: 60%)
   model_v2: "The revenue was twenty three point five million" (WER: 20%)
   model_v3: "The revenue was $23.5 million" (WER: 0%)

   Improvement: v3의 숫자 처리가 우수

2. audio_5678.wav
   Reference: "Dr. Smith discussed HIPAA"
   model_v1: "Doctor Smith discussed hippo" (WER: 25%)
   model_v2: "Dr. Smith discussed hippo" (WER: 12.5%)
   model_v3: "Dr. Smith discussed HIPAA" (WER: 0%)

   Improvement: v3의 약어 처리가 우수

Samples where model_v3 is worse
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. audio_9012.wav
   Reference: "It's gonna be fine"
   model_v1: "It's gonna be fine" (WER: 0%)
   model_v2: "It's gonna be fine" (WER: 0%)
   model_v3: "It is going to be fine" (WER: 75%)

   Issue: v3가 구어체를 formal하게 변환

Category-specific performance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Category         v1 WER  v2 WER  v3 WER  Winner
────────────────────────────────────────────────
Numbers          8.9%    6.2%    4.1%    v3
Proper nouns     7.2%    6.8%    5.9%    v3
Technical terms  9.1%    8.3%    7.2%    v3
Conversational   3.8%    3.9%    4.5%    v1
Clean speech     2.1%    2.0%    1.8%    v3
Noisy speech     12.3%   11.8%   10.9%   v3

Conclusion:
- model_v3가 전반적으로 가장 우수
- 단, 구어체 처리에서는 v1이 더 자연스러움
- 숫자/전문용어가 많은 경우 v3 추천
- 일상 대화 전사는 v1 추천
```

## Diarization 평가

화자 분리(Speaker Diarization) 성능 평가:

```bash
ta eval-diarization \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --dataset ami_test \
    --output diarization_results.json
```

### DER 메트릭

Diarization Error Rate:

```
DER = (False Alarm + Missed Speech + Speaker Error) / Total Speech Time
```

결과:

```
Diarization Error Rate (DER) Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Overall DER: 8.3%

Component breakdown:
  False Alarm:    1.2% (non-speech detected as speech)
  Missed Speech:  2.1% (speech detected as non-speech)
  Speaker Error:  5.0% (wrong speaker assigned)

Per-condition results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Condition              DER    FA     Miss   SpkErr
────────────────────────────────────────────────────
2 speakers            5.2%   0.8%   1.4%   3.0%
3-4 speakers          8.3%   1.2%   2.1%   5.0%
5+ speakers          15.7%   2.1%   3.8%   9.8%
Overlapping speech   18.9%   3.2%   4.5%   11.2%

Best samples:
  meeting_001.wav: DER=1.2% (2 speakers, clean)
  meeting_042.wav: DER=2.8% (3 speakers, moderate)

Worst samples:
  meeting_089.wav: DER=28.3% (6 speakers, overlapping)
  meeting_123.wav: DER=31.7% (8 speakers, noisy)
```

## Alignment 분석

시간 정렬(Forced Alignment) 품질 평가:

```bash
ta eval-alignment \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --dataset librispeech_test_clean \
    --output alignment_results.json
```

결과:

```
Alignment Quality Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Word-level alignment errors:
  Mean Absolute Error: 42ms
  Median Absolute Error: 28ms
  95th percentile: 156ms

Phone-level alignment errors:
  Mean Absolute Error: 18ms
  Median Absolute Error: 12ms
  95th percentile: 67ms

Timing accuracy by word type:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type           MAE     Samples
──────────────────────────────
Content words  38ms    12,345
Function words 45ms    8,901
Numbers        52ms    1,234
Proper nouns   48ms    789

Examples:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sample: audio_1234.wav

Word       Ref Start  Pred Start  Error
──────────────────────────────────────────
The        0.00s      0.00s       0ms    ✓
quick      0.25s      0.27s       20ms   ✓
brown      0.52s      0.50s       20ms   ✓
fox        0.89s      0.95s       60ms   ~
jumped     1.15s      1.18s       30ms   ✓

Overall alignment quality: Excellent
```

## 결과 시각화 및 해석

### Confusion Matrix

자주 혼동되는 단어 쌍:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 혼동 행렬 생성
ta visualize confusion-matrix \
    --predictions predictions.jsonl \
    --output confusion_matrix.png \
    --top-n 20
```

출력:

![Confusion Matrix](confusion_matrix.png)

자주 혼동되는 쌍:

```
Top 20 Confused Word Pairs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reference → Prediction          Count   Fix Strategy
──────────────────────────────────────────────────────
"fifty" → "fifteen"              87     숫자 후처리
"their" → "there"                65     LM rescoring
"its" → "it's"                   58     문법 규칙
"your" → "you're"                52     LM rescoring
"two" → "to"                     48     문맥 분석
"than" → "then"                  45     문법 규칙
"effect" → "affect"              42     LM rescoring
```

### WER 분포

WER 히스토그램:

```bash
ta visualize wer-distribution \
    --predictions predictions.jsonl \
    --output wer_distribution.png
```

```
WER Distribution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

      │
 600  │  ████
      │  ████
 500  │  ████
      │  ████ ███
 400  │  ████ ███
      │  ████ ███ ██
 300  │  ████ ███ ██
      │  ████ ███ ██ █
 200  │  ████ ███ ██ █
      │  ████ ███ ██ █ █
 100  │  ████ ███ ██ █ █ █
      │  ████ ███ ██ █ █ █
   0  └──────────────────────────
      0%  5%  10% 15% 20% 25% 30%+

Statistics:
  Mean: 5.2%
  Median: 3.8%
  Std: 4.1%

  < 5%:  1,825 samples (69.7%)
  5-10%:   523 samples (20.0%)
  10-20%:  198 samples (7.6%)
  > 20%:    74 samples (2.8%)
```

### 오류 타입별 분포

```bash
ta visualize error-types \
    --predictions predictions.jsonl \
    --output error_types.png
```

```
Error Type Distribution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Substitutions (60.2%)   ████████████████████████
Deletions (27.8%)       ███████████
Insertions (12.0%)      █████

Common substitution patterns:
  Phonetically similar: 45%
  Homophones: 28%
  Short words: 18%
  Other: 9%

Common deletion patterns:
  Function words: 52%
  Word endings: 28%
  Fast speech: 20%

Common insertion patterns:
  Filler words: 48%
  Disfluencies: 32%
  Background speech: 20%
```

## 성능 벤치마크

### 처리 속도

```bash
ta benchmark speed \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --samples 1000 \
    --output speed_benchmark.json
```

결과:

```
Speed Benchmark
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hardware: NVIDIA A40 (48GB)
Batch size: 16
Samples: 1,000

Throughput:
  Audio processed: 2.78 hours
  Time taken: 25m 12s
  Real-time factor: 0.151
  Throughput: 6.62 hours/hour

Latency (per sample):
  Mean: 1.51s
  Median: 1.48s
  P95: 1.89s
  P99: 2.34s

Memory usage:
  Peak GPU memory: 12.3 GB
  Average GPU memory: 10.8 GB
  CPU memory: 4.2 GB

By audio duration:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Duration    Samples  Avg RTF  Avg Latency
────────────────────────────────────────────
0-5s        234      0.128    0.64s
5-10s       456      0.145    1.09s
10-20s      287      0.167    2.51s
20-30s      23       0.189    5.32s

Comparison with other models:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model              RTF    Throughput  GPU Memory
────────────────────────────────────────────────
Tiny Audio        0.151   6.62h/h     12.3 GB
Whisper Medium    0.183   5.46h/h     8.2 GB
Whisper Large     0.452   2.21h/h     14.7 GB
```

### 배치 크기 최적화

```bash
ta benchmark batch-size \
    --checkpoint outputs/transcription/checkpoints/final.pt \
    --batch-sizes 1,2,4,8,16,32 \
    --output batch_size_benchmark.json
```

결과:

```
Batch Size Optimization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Batch  RTF    Throughput  GPU Mem   Latency
────────────────────────────────────────────
1      0.245  4.08h/h     6.2 GB    2.45s
2      0.198  5.05h/h     7.8 GB    1.98s
4      0.172  5.81h/h     9.1 GB    1.72s
8      0.158  6.33h/h     10.9 GB   1.58s
16     0.151  6.62h/h     12.3 GB   1.51s  ← Optimal
32     0.149  6.71h/h     OOM       N/A

Recommendation: Use batch_size=16
  - Best throughput without OOM
  - Good balance of speed and memory
```

## 다음 단계

모델 평가를 완료했다면 마지막 단계로 진행하세요:

- **[챕터 6: 배포 및 확장](/tiny-audio-guide-06-deployment/)** - HuggingFace Hub 배포, Gradio 데모, Voice Agent 통합

## 참고 자료

- WER 계산: [https://github.com/jitsi/jiwer](https://github.com/jitsi/jiwer)
- Pyannote Audio (Diarization): [https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
- Montreal Forced Aligner: [https://montreal-forced-aligner.readthedocs.io](https://montreal-forced-aligner.readthedocs.io)
