---
layout: post
title: "ACE-Step 1.5 완벽 가이드 (05) - REST API 가이드"
date: 2026-02-08
permalink: /ace-step-guide-05-rest-api/
author: ACE Studio & StepFun
categories: [AI 음악, 오픈소스]
tags: [ACE-Step, REST API, HTTP API, Python Client, cURL, API Server, Studio UI]
original_url: "https://github.com/ace-step/ACE-Step-1.5"
excerpt: "REST API로 프로그래밍 방식 음악 생성"
---

## REST API 개요

ACE-Step의 REST API는 HTTP 기반 비동기 음악 생성 서비스를 제공합니다.

```
┌────────────────────────────────────────────────────┐
│  ACE-Step REST API Architecture                    │
├────────────────────────────────────────────────────┤
│                                                     │
│  Client (Python/cURL/HTTP)                          │
│      ↓                                              │
│  ┌─────────────────────────────────────────────┐  │
│  │  FastAPI Server (Port 8001)                  │  │
│  │  ─────────────────────────────────────────  │  │
│  │  • POST /release_task                        │  │
│  │  • POST /query_result                        │  │
│  │  • POST /format_input                        │  │
│  │  • POST /create_random_sample                │  │
│  │  • GET  /v1/models                           │  │
│  │  • GET  /v1/audio                            │  │
│  │  • GET  /health                              │  │
│  └─────────────────────────────────────────────┘  │
│      ↓                                              │
│  ┌─────────────────────────────────────────────┐  │
│  │  Task Queue (In-Memory)                      │  │
│  │  Max Size: 200 (configurable)                │  │
│  └─────────────────────────────────────────────┘  │
│      ↓                                              │
│  ┌─────────────────────────────────────────────┐  │
│  │  Worker (DiT + LM)                           │  │
│  │  • Music Generation                          │  │
│  │  • Sample Creation                           │  │
│  │  • Format Enhancement                        │  │
│  └─────────────────────────────────────────────┘  │
│      ↓                                              │
│  Generated Audio Files (.cache/acestep/tmp/)       │
│                                                     │
└────────────────────────────────────────────────────┘
```

---

## API 서버 시작 및 설정

### 서버 시작

```bash
# uv 사용
uv run acestep-api

# Python 직접 실행
python acestep/api_server.py

# 서버 주소
http://localhost:8001

# Health check
curl http://localhost:8001/health
```

### 환경변수 설정 (.env)

```bash
# .env 파일 생성
cp .env.example .env

# .env 편집
nano .env
```

```bash
# .env 내용
# ============================================
# Server Configuration
# ============================================
ACESTEP_API_HOST=127.0.0.1
ACESTEP_API_PORT=8001
ACESTEP_API_KEY=sk-your-secret-key  # 비워두면 인증 비활성화
ACESTEP_API_WORKERS=1

# ============================================
# Model Configuration
# ============================================
ACESTEP_CONFIG_PATH=acestep-v15-turbo
ACESTEP_CONFIG_PATH2=acestep-v15-base  # 2번째 모델 (선택)
ACESTEP_CONFIG_PATH3=  # 3번째 모델 (선택)

ACESTEP_DEVICE=auto
ACESTEP_USE_FLASH_ATTENTION=true
ACESTEP_OFFLOAD_TO_CPU=false
ACESTEP_OFFLOAD_DIT_TO_CPU=false

# ============================================
# LM Configuration
# ============================================
ACESTEP_INIT_LLM=auto  # auto, true, false
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
ACESTEP_LM_BACKEND=vllm
ACESTEP_LM_DEVICE=auto
ACESTEP_LM_OFFLOAD_TO_CPU=false

# ============================================
# Queue Configuration
# ============================================
ACESTEP_QUEUE_MAXSIZE=200
ACESTEP_QUEUE_WORKERS=1
ACESTEP_AVG_JOB_SECONDS=5.0
ACESTEP_AVG_WINDOW=50

# ============================================
# Download Source
# ============================================
ACESTEP_DOWNLOAD_SOURCE=auto  # auto, huggingface, modelscope
```

### 커맨드라인 옵션

```bash
# 모든 옵션 확인
python acestep/api_server.py --help

# 예시
python acestep/api_server.py \
  --api-host 0.0.0.0 \
  --api-port 8001 \
  --api-key sk-123456 \
  --config-path acestep-v15-turbo \
  --lm-model-path acestep-5Hz-lm-1.7B \
  --init-llm true
```

---

## API 엔드포인트 목록

### 전체 엔드포인트

```yaml
음악 생성:
  POST /release_task          # 음악 생성 작업 생성
  POST /query_result          # 작업 결과 배치 쿼리

LLM 기능:
  POST /format_input          # Caption/Lyrics 향상
  POST /create_random_sample  # 랜덤 샘플 생성

모델 & 시스템:
  GET  /v1/models             # 사용 가능한 모델 목록
  GET  /v1/audio              # 오디오 파일 다운로드
  GET  /v1/stats              # 서버 통계
  GET  /health                # 헬스 체크
```

### Response Format (통일된 래퍼)

```json
{
  "data": { ... },
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

| Field | Type | 설명 |
|-------|------|------|
| `data` | any | 실제 응답 데이터 |
| `code` | int | 상태 코드 (200=성공) |
| `error` | string | 에러 메시지 (null이면 성공) |
| `timestamp` | int | 응답 타임스탬프 (밀리초) |
| `extra` | any | 추가 정보 (보통 null) |

---

## /v1/audio/generations (음악 생성)

### 작업 생성: POST /release_task

#### 기본 요청 (JSON)

```bash
curl -X POST http://localhost:8001/release_task \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "upbeat electronic dance music with heavy bass",
    "lyrics": "[Instrumental]",
    "thinking": true,
    "inference_steps": 8,
    "batch_size": 2
  }'
```

#### 응답

```json
{
  "data": {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "queued",
    "queue_position": 1
  },
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

#### 핵심 파라미터

```yaml
필수 파라미터:
  prompt (caption):
    설명: 음악 설명
    예시: "upbeat pop song with guitar"

  thinking:
    설명: 5Hz LM 활성화
    기본값: false
    권장: true (품질 향상)

선택 파라미터:
  lyrics:
    설명: 가사
    기본값: ""

  audio_duration (duration):
    설명: 목표 길이 (초)
    범위: 10-600
    기본값: null (자동)

  bpm:
    설명: BPM
    범위: 30-300
    기본값: null (자동 추론)

  key_scale (keyscale):
    설명: 조성 (예: "C Major", "Am")
    기본값: ""

  time_signature (timesignature):
    설명: 박자 (2, 3, 4, 6)
    기본값: ""

  vocal_language:
    설명: 보컬 언어 (en, zh, ko, ja, ...)
    기본값: "en"

  batch_size:
    설명: 배치 크기
    범위: 1-8
    기본값: 2

  inference_steps:
    설명: 추론 스텝 수
    Turbo: 1-20 (권장 8)
    Base: 1-200 (권장 50)

  audio_format:
    설명: 출력 포맷
    옵션: mp3, wav, flac
    기본값: mp3

  seed:
    설명: 랜덤 시드
    기본값: -1 (랜덤)

  use_random_seed:
    설명: 랜덤 시드 사용 여부
    기본값: true
```

### 고급 파라미터

#### LM Parameters

```json
{
  "thinking": true,
  "lm_temperature": 0.85,
  "lm_cfg_scale": 2.5,
  "lm_top_k": 0,
  "lm_top_p": 0.9,
  "lm_negative_prompt": "NO USER INPUT",
  "use_cot_caption": true,
  "use_cot_language": true,
  "constrained_decoding": true
}
```

#### DiT Advanced Parameters

```json
{
  "shift": 3.0,
  "infer_method": "ode",
  "timesteps": null,
  "guidance_scale": 7.0,
  "use_adg": false,
  "cfg_interval_start": 0.0,
  "cfg_interval_end": 1.0
}
```

#### Multi-Model Support

```json
{
  "model": "acestep-v15-turbo",
  "prompt": "jazz piano trio"
}
```

### Task Type 별 요청

#### Text2Music (기본)

```bash
curl -X POST http://localhost:8001/release_task \
  -H 'Content-Type: application/json' \
  -d '{
    "task_type": "text2music",
    "prompt": "energetic rock music with electric guitar",
    "lyrics": "[Instrumental]",
    "bpm": 140,
    "duration": 30,
    "thinking": true
  }'
```

#### Cover Generation

```bash
curl -X POST http://localhost:8001/release_task \
  -H 'Content-Type: application/json' \
  -d '{
    "task_type": "cover",
    "src_audio_path": "/path/to/original.mp3",
    "prompt": "jazz piano version",
    "audio_cover_strength": 0.8,
    "thinking": true
  }'
```

#### Repaint

```bash
curl -X POST http://localhost:8001/release_task \
  -H 'Content-Type: application/json' \
  -d '{
    "task_type": "repaint",
    "src_audio_path": "/path/to/source.mp3",
    "repainting_start": 10.0,
    "repainting_end": 20.0,
    "prompt": "smooth transition with piano solo",
    "thinking": true
  }'
```

### File Upload (multipart/form-data)

```bash
curl -X POST http://localhost:8001/release_task \
  -F "prompt=remix this song" \
  -F "src_audio=@/path/to/local/song.mp3" \
  -F "task_type=cover" \
  -F "thinking=true"
```

---

## 비동기 생성 워크플로우

### Task Status Lifecycle

```
┌─────────────────────────────────────────────────────┐
│  Task Lifecycle                                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. POST /release_task                               │
│     → task_id 반환                                   │
│     → status: "queued" (0)                           │
│                                                      │
│  2. Worker가 작업 처리                                │
│     → status: "running" (0)                          │
│                                                      │
│  3. 생성 완료                                         │
│     → status: "succeeded" (1)                        │
│     또는                                              │
│     → status: "failed" (2)                           │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Status Codes:**

```yaml
0: queued/running (작업 대기 중 또는 진행 중)
1: succeeded (생성 성공)
2: failed (생성 실패)
```

### 결과 쿼리: POST /query_result

```bash
curl -X POST http://localhost:8001/query_result \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id_list": ["550e8400-e29b-41d4-a716-446655440000"]
  }'
```

#### 응답 (진행 중)

```json
{
  "data": [
    {
      "task_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": 0,
      "result": null
    }
  ],
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

#### 응답 (완료)

```json
{
  "data": [
    {
      "task_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": 1,
      "result": "[{\"file\": \"/v1/audio?path=%2Ftmp%2Fapi_audio%2Fabc123.mp3\", \"wave\": \"\", \"status\": 1, \"create_time\": 1700000000, \"env\": \"development\", \"prompt\": \"upbeat pop song\", \"lyrics\": \"[Instrumental]\", \"metas\": {\"bpm\": 120, \"duration\": 30, \"genres\": \"\", \"keyscale\": \"C Major\", \"timesignature\": \"4\"}, \"generation_info\": \"...\", \"seed_value\": \"12345,67890\", \"lm_model\": \"acestep-5Hz-lm-1.7B\", \"dit_model\": \"acestep-v15-turbo\"}]"
    }
  ],
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

### Result 필드 구조

`result` 필드는 JSON string이며, 파싱하면 다음 구조를 가집니다:

```json
[
  {
    "file": "/v1/audio?path=%2Ftmp%2Fapi_audio%2Fabc123.mp3",
    "wave": "",
    "status": 1,
    "create_time": 1700000000,
    "env": "development",
    "prompt": "upbeat pop song",
    "lyrics": "[Instrumental]",
    "metas": {
      "bpm": 120,
      "duration": 30,
      "genres": "",
      "keyscale": "C Major",
      "timesignature": "4"
    },
    "generation_info": "DiT: acestep-v15-turbo, LM: acestep-5Hz-lm-1.7B",
    "seed_value": "12345,67890",
    "lm_model": "acestep-5Hz-lm-1.7B",
    "dit_model": "acestep-v15-turbo"
  }
]
```

### 오디오 다운로드: GET /v1/audio

```bash
# result에서 받은 file URL 사용
curl "http://localhost:8001/v1/audio?path=%2Ftmp%2Fapi_audio%2Fabc123.mp3" -o output.mp3
```

---

## Python 클라이언트 예제

### 기본 생성

```python
import requests
import json
import time

API_BASE = "http://localhost:8001"

def create_music(prompt, lyrics="[Instrumental]", thinking=True):
    """음악 생성 작업 생성"""
    url = f"{API_BASE}/release_task"
    payload = {
        "prompt": prompt,
        "lyrics": lyrics,
        "thinking": thinking,
        "inference_steps": 8,
        "batch_size": 2,
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()

    return data["data"]["task_id"]


def query_task(task_id):
    """작업 상태 쿼리"""
    url = f"{API_BASE}/query_result"
    payload = {"task_id_list": [task_id]}

    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()

    return data["data"][0]


def wait_for_completion(task_id, timeout=600, poll_interval=2):
    """작업 완료 대기"""
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Task {task_id} timed out after {timeout}s")

        task = query_task(task_id)
        status = task["status"]

        if status == 1:  # succeeded
            return json.loads(task["result"])
        elif status == 2:  # failed
            raise RuntimeError(f"Task {task_id} failed")

        # status == 0 (queued/running)
        print(f"Task {task_id} is still running...")
        time.sleep(poll_interval)


def download_audio(file_url, output_path):
    """오디오 파일 다운로드"""
    url = f"{API_BASE}{file_url}"

    response = requests.get(url)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"Downloaded: {output_path}")


# 사용 예시
if __name__ == "__main__":
    # 1. 작업 생성
    task_id = create_music(
        prompt="upbeat pop rock with electric guitars",
        lyrics="[Instrumental]",
        thinking=True
    )
    print(f"Task created: {task_id}")

    # 2. 완료 대기
    print("Waiting for completion...")
    results = wait_for_completion(task_id)

    # 3. 결과 다운로드
    for i, result in enumerate(results, 1):
        file_url = result["file"]
        output_path = f"output_{i}.mp3"
        download_audio(file_url, output_path)

        print(f"Sample {i}:")
        print(f"  BPM: {result['metas']['bpm']}")
        print(f"  Key: {result['metas']['keyscale']}")
        print(f"  Seed: {result['seed_value']}")
```

### 고급 예제 (메타데이터 제어)

```python
def create_music_advanced(
    prompt,
    lyrics="",
    bpm=None,
    keyscale="",
    duration=None,
    thinking=True,
    temperature=0.85,
):
    """고급 파라미터로 음악 생성"""
    url = f"{API_BASE}/release_task"

    payload = {
        "prompt": prompt,
        "lyrics": lyrics,
        "thinking": thinking,
        "inference_steps": 8,
        "batch_size": 2,
        "lm_temperature": temperature,
        "lm_cfg_scale": 2.5,
    }

    # 메타데이터 추가 (지정된 경우)
    if bpm:
        payload["bpm"] = bpm
    if keyscale:
        payload["key_scale"] = keyscale
    if duration:
        payload["audio_duration"] = duration

    response = requests.post(url, json=payload)
    response.raise_for_status()

    return response.json()["data"]["task_id"]


# 사용 예시
task_id = create_music_advanced(
    prompt="melancholic indie folk with acoustic guitar",
    lyrics="[Verse 1]\nWalking alone...",
    bpm=72,
    keyscale="Am",
    duration=120,
    thinking=True,
    temperature=0.9,
)
```

### Cover Generation 예제

```python
def create_cover(src_audio_path, prompt, strength=0.8):
    """커버 생성"""
    url = f"{API_BASE}/release_task"

    payload = {
        "task_type": "cover",
        "src_audio_path": src_audio_path,
        "prompt": prompt,
        "audio_cover_strength": strength,
        "thinking": True,
        "inference_steps": 8,
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    return response.json()["data"]["task_id"]


# 사용 예시
task_id = create_cover(
    src_audio_path="/path/to/original.mp3",
    prompt="orchestral symphonic arrangement",
    strength=0.7
)
```

### File Upload 예제

```python
def create_music_with_file(prompt, src_audio_file):
    """로컬 파일 업로드하여 생성"""
    url = f"{API_BASE}/release_task"

    files = {
        "src_audio": open(src_audio_file, "rb")
    }

    data = {
        "prompt": prompt,
        "task_type": "cover",
        "thinking": "true",
    }

    response = requests.post(url, files=files, data=data)
    response.raise_for_status()

    return response.json()["data"]["task_id"]


# 사용 예시
task_id = create_music_with_file(
    prompt="jazz piano version",
    src_audio_file="./my_song.mp3"
)
```

---

## cURL 예제

### 기본 생성

```bash
#!/bin/bash

# 1. 작업 생성
TASK_ID=$(curl -s -X POST http://localhost:8001/release_task \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "upbeat pop song",
    "thinking": true,
    "inference_steps": 8
  }' | jq -r '.data.task_id')

echo "Task ID: $TASK_ID"

# 2. 완료 대기 (polling)
while true; do
  STATUS=$(curl -s -X POST http://localhost:8001/query_result \
    -H 'Content-Type: application/json' \
    -d "{\"task_id_list\": [\"$TASK_ID\"]}" \
    | jq -r '.data[0].status')

  if [ "$STATUS" == "1" ]; then
    echo "Task completed!"
    break
  elif [ "$STATUS" == "2" ]; then
    echo "Task failed!"
    exit 1
  fi

  echo "Status: $STATUS (waiting...)"
  sleep 2
done

# 3. 결과 다운로드
curl -s -X POST http://localhost:8001/query_result \
  -H 'Content-Type: application/json' \
  -d "{\"task_id_list\": [\"$TASK_ID\"]}" \
  | jq -r '.data[0].result' \
  | jq -r '.[0].file' \
  | xargs -I {} curl -s "http://localhost:8001{}" -o output.mp3

echo "Downloaded: output.mp3"
```

### 인증 포함

```bash
# API Key 설정
API_KEY="sk-your-secret-key"

# Authorization header 사용
curl -X POST http://localhost:8001/release_task \
  -H 'Authorization: Bearer sk-your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "jazz piano trio",
    "thinking": true
  }'

# 또는 request body에 포함
curl -X POST http://localhost:8001/release_task \
  -H 'Content-Type: application/json' \
  -d '{
    "ai_token": "sk-your-secret-key",
    "prompt": "jazz piano trio",
    "thinking": true
  }'
```

---

## 인증 및 보안

### API Key 설정

```bash
# 환경변수로 설정
export ACESTEP_API_KEY=sk-your-secret-key

# 또는 .env 파일에 추가
echo "ACESTEP_API_KEY=sk-your-secret-key" >> .env

# 서버 재시작
uv run acestep-api
```

### 인증 방법

**방법 A: Authorization Header (권장)**

```bash
curl -X POST http://localhost:8001/release_task \
  -H 'Authorization: Bearer sk-your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "jazz"}'
```

**방법 B: Request Body**

```bash
curl -X POST http://localhost:8001/release_task \
  -H 'Content-Type: application/json' \
  -d '{
    "ai_token": "sk-your-secret-key",
    "prompt": "jazz"
  }'
```

### 보안 Best Practices

```yaml
1. API Key 사용:
   - 프로덕션 환경에서는 필수
   - 강력한 랜덤 키 생성
   - 주기적으로 로테이션

2. HTTPS 사용:
   - Reverse proxy (nginx) 사용
   - SSL/TLS 인증서 설정

3. Rate Limiting:
   - Queue 크기 제한 (ACESTEP_QUEUE_MAXSIZE)
   - IP 기반 제한 (reverse proxy)

4. Network 제한:
   - 내부 네트워크만 허용
   - Firewall 설정

5. 로그 관리:
   - API 요청 로깅
   - 에러 모니터링
```

---

## Studio UI 소개

### Studio UI란?

**Studio UI**는 REST API를 위한 선택적 HTML 프론트엔드로, DAW(Digital Audio Workstation) 스타일의 인터페이스를 제공합니다.

```
┌────────────────────────────────────────────────────┐
│  ACE-Step Studio UI (ui/studio.html)               │
├────────────────────────────────────────────────────┤
│                                                     │
│  Browser ←→ studio.html ←→ REST API (localhost:8001)│
│                                                     │
│  특징:                                               │
│  • Gradio UI와 동일한 백엔드 사용                      │
│  • 프론트엔드만 존재 (정적 HTML)                       │
│  • 다중 트랙 관리                                     │
│  • 프로젝트 세션 관리                                 │
│                                                     │
└────────────────────────────────────────────────────┘
```

### Studio UI 시작

```bash
# 1. API 서버 시작
uv run acestep-api

# 2. 브라우저에서 studio.html 열기
# 방법 A: 직접 열기
open ui/studio.html

# 방법 B: 로컬 서버 사용
cd ui
python -m http.server 8080
# http://localhost:8080/studio.html

# 3. API URL 설정
# Studio UI에서 "Settings" 클릭
# API URL: http://localhost:8001
# API Key: (있는 경우 입력)
```

### Studio UI vs Gradio UI

```yaml
Gradio UI:
  장점:
    - 올인원 패키지 (서버 + UI 통합)
    - 빠른 시작
    - LoRA 훈련 지원

  용도:
    - 로컬 실험
    - 빠른 프로토타입
    - 단일 사용자

Studio UI:
  장점:
    - DAW 스타일 인터페이스
    - 프로젝트 세션 관리
    - 다중 트랙 관리
    - REST API 직접 사용

  용도:
    - 프로덕션 워크플로우
    - 복잡한 프로젝트
    - 팀 협업 (API 공유)
```

---

## 기타 엔드포인트

### POST /format_input (Caption/Lyrics 향상)

```bash
curl -X POST http://localhost:8001/format_input \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "pop rock",
    "lyrics": "Walking down the street",
    "param_obj": "{\"duration\": 180, \"language\": \"en\"}"
  }'
```

**응답:**

```json
{
  "data": {
    "caption": "Upbeat pop rock with electric guitars and driving drums",
    "lyrics": "[Verse 1]\nWalking down the street today...",
    "bpm": 120,
    "key_scale": "C Major",
    "time_signature": "4",
    "duration": 180,
    "vocal_language": "en"
  },
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

### POST /create_random_sample (랜덤 샘플)

```bash
curl -X POST http://localhost:8001/create_random_sample \
  -H 'Content-Type: application/json' \
  -d '{"sample_type": "simple_mode"}'
```

**응답:**

```json
{
  "data": {
    "caption": "Upbeat pop song with guitar accompaniment",
    "lyrics": "[Verse 1]\nSunshine on my face...",
    "bpm": 120,
    "key_scale": "G Major",
    "time_signature": "4",
    "duration": 180,
    "vocal_language": "en"
  },
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

### GET /v1/models (모델 목록)

```bash
curl http://localhost:8001/v1/models
```

**응답:**

```json
{
  "data": {
    "models": [
      {
        "name": "acestep-v15-turbo",
        "is_default": true
      },
      {
        "name": "acestep-v15-base",
        "is_default": false
      }
    ],
    "default_model": "acestep-v15-turbo"
  },
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

### GET /v1/stats (서버 통계)

```bash
curl http://localhost:8001/v1/stats
```

**응답:**

```json
{
  "data": {
    "jobs": {
      "total": 100,
      "queued": 5,
      "running": 1,
      "succeeded": 90,
      "failed": 4
    },
    "queue_size": 5,
    "queue_maxsize": 200,
    "avg_job_seconds": 8.5
  },
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

### GET /health (헬스 체크)

```bash
curl http://localhost:8001/health
```

**응답:**

```json
{
  "data": {
    "status": "ok",
    "service": "ACE-Step API",
    "version": "1.0"
  },
  "code": 200,
  "error": null,
  "timestamp": 1700000000000,
  "extra": null
}
```

---

## Error Handling

### HTTP Status Codes

```yaml
200: Success
400: Invalid request (잘못된 JSON, 필수 필드 누락)
401: Unauthorized (API key 누락/잘못됨)
404: Resource not found
415: Unsupported Content-Type
429: Server busy (queue full)
500: Internal server error
```

### Error Response Format

```json
{
  "detail": "Error message describing the issue"
}
```

### Python Error Handling

```python
def create_music_safe(prompt, **kwargs):
    """에러 핸들링 포함 음악 생성"""
    url = f"{API_BASE}/release_task"
    payload = {"prompt": prompt, **kwargs}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        if data["code"] != 200:
            raise RuntimeError(f"API Error: {data['error']}")

        return data["data"]["task_id"]

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("Error: Invalid API key")
        elif e.response.status_code == 429:
            print("Error: Server busy, try again later")
        else:
            print(f"HTTP Error: {e}")
        raise

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        raise
```

---

## Best Practices

### 1. thinking=True 사용

```python
# 권장: LM으로 품질 향상
payload = {
    "prompt": "upbeat pop song",
    "thinking": True,
    "lm_temperature": 0.85,
}

# 빠른 생성이 필요한 경우에만 False
payload = {
    "prompt": "upbeat pop song",
    "thinking": False,
}
```

### 2. 배치 작업 쿼리

```python
# 여러 작업을 한 번에 쿼리
task_ids = [task_id_1, task_id_2, task_id_3]

response = requests.post(
    f"{API_BASE}/query_result",
    json={"task_id_list": task_ids}
)

results = response.json()["data"]
for result in results:
    print(f"Task {result['task_id']}: status={result['status']}")
```

### 3. 서버 부하 확인

```python
# 작업 제출 전 서버 상태 확인
def check_server_load():
    response = requests.get(f"{API_BASE}/v1/stats")
    stats = response.json()["data"]

    queue_usage = stats["queue_size"] / stats["queue_maxsize"]

    if queue_usage > 0.8:
        print("Warning: Server is busy")
        return False

    return True


if check_server_load():
    task_id = create_music(...)
```

### 4. Multi-Model 활용

```python
# 환경변수로 여러 모델 설정
ACESTEP_CONFIG_PATH=acestep-v15-turbo
ACESTEP_CONFIG_PATH2=acestep-v15-base

# 요청 시 모델 선택
task_id_turbo = create_music(
    prompt="fast generation",
    model="acestep-v15-turbo"
)

task_id_base = create_music(
    prompt="high quality",
    model="acestep-v15-base"
)
```

### 5. Timeout 설정

```python
def wait_for_completion(task_id, timeout=600):
    """타임아웃 설정"""
    start = time.time()

    while time.time() - start < timeout:
        task = query_task(task_id)
        if task["status"] in [1, 2]:
            return task
        time.sleep(2)

    raise TimeoutError(f"Task {task_id} timed out")
```

---

## 전체 워크플로우 예제

```python
import requests
import json
import time
from pathlib import Path

API_BASE = "http://localhost:8001"
OUTPUT_DIR = Path("./generated_music")
OUTPUT_DIR.mkdir(exist_ok=True)


class ACEStepClient:
    def __init__(self, api_base=API_BASE, api_key=None):
        self.api_base = api_base
        self.api_key = api_key
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {api_key}"
            })

    def create_music(self, **kwargs):
        """음악 생성 작업 생성"""
        url = f"{self.api_base}/release_task"
        response = self.session.post(url, json=kwargs)
        response.raise_for_status()
        return response.json()["data"]["task_id"]

    def query_task(self, task_id):
        """작업 상태 쿼리"""
        url = f"{self.api_base}/query_result"
        response = self.session.post(url, json={"task_id_list": [task_id]})
        response.raise_for_status()
        return response.json()["data"][0]

    def wait_for_completion(self, task_id, timeout=600, poll_interval=2):
        """작업 완료 대기"""
        start = time.time()

        while time.time() - start < timeout:
            task = self.query_task(task_id)

            if task["status"] == 1:
                return json.loads(task["result"])
            elif task["status"] == 2:
                raise RuntimeError(f"Task {task_id} failed")

            time.sleep(poll_interval)

        raise TimeoutError(f"Task {task_id} timed out")

    def download_audio(self, file_url, output_path):
        """오디오 다운로드"""
        url = f"{self.api_base}{file_url}"
        response = self.session.get(url)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

    def generate_and_download(self, output_dir, **kwargs):
        """생성 + 다운로드 통합"""
        # 작업 생성
        task_id = self.create_music(**kwargs)
        print(f"Task created: {task_id}")

        # 완료 대기
        results = self.wait_for_completion(task_id)
        print(f"Generated {len(results)} samples")

        # 다운로드
        downloaded = []
        for i, result in enumerate(results, 1):
            filename = f"{task_id}_{i}.mp3"
            filepath = Path(output_dir) / filename

            self.download_audio(result["file"], filepath)

            downloaded.append({
                "path": str(filepath),
                "metadata": result["metas"],
                "seed": result["seed_value"],
            })

        return downloaded


# 사용 예시
if __name__ == "__main__":
    client = ACEStepClient(api_key="sk-your-key")

    # 예시 1: 기본 생성
    results = client.generate_and_download(
        output_dir=OUTPUT_DIR,
        prompt="upbeat electronic dance music",
        thinking=True,
        batch_size=2,
    )

    for result in results:
        print(f"Downloaded: {result['path']}")
        print(f"  BPM: {result['metadata']['bpm']}")
        print(f"  Key: {result['metadata']['keyscale']}")

    # 예시 2: 고급 파라미터
    results = client.generate_and_download(
        output_dir=OUTPUT_DIR,
        prompt="melancholic indie folk",
        lyrics="[Verse 1]\nWalking alone...",
        bpm=72,
        key_scale="Am",
        duration=120,
        thinking=True,
        lm_temperature=0.9,
    )
```

---

## 정리

### 핵심 개념

```yaml
비동기 워크플로우:
  1. POST /release_task → task_id 받기
  2. POST /query_result로 polling
  3. status==1이면 result 파싱
  4. GET /v1/audio로 다운로드

파라미터 우선순위:
  thinking=True: 권장 (품질 향상)
  batch_size: 2-4 (다양성)
  inference_steps: 8 (Turbo 기본값)

인증:
  프로덕션: API Key 필수
  로컬 테스트: 선택
```

### REST API vs Gradio UI

```yaml
REST API:
  용도:
    - 프로그래밍 방식 통합
    - 자동화 워크플로우
    - 서비스 배포
    - 팀 협업

  장점:
    - 언어 독립적 (Python, JS, cURL 등)
    - 비동기 처리
    - 확장 가능

Gradio UI:
  용도:
    - 대화형 실험
    - 빠른 프로토타입
    - 수동 제어
    - LoRA 훈련

  장점:
    - 즉시 시작
    - 비주얼 인터페이스
    - 실시간 피드백
```

---

## 다음 챕터

이제 ACE-Step 1.5의 기본 사용법을 모두 마스터했습니다. 다음 챕터에서는 더 고급 주제를 다룹니다:

- Python Inference API 가이드
- LoRA 훈련 심화
- 멀티 트랙 & 편집 워크플로우
- 성능 최적화 및 튜닝

**[다음: 06장 - Python Inference API (예정)](/ace-step-guide-06-python-api/)**
