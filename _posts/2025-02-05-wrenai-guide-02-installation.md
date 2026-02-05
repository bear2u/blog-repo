---
layout: post
title: "WrenAI 완벽 가이드 (2) - 설치 및 환경 설정"
date: 2025-02-05
permalink: /wrenai-guide-02-installation/
author: Canner
categories: [AI 에이전트, WrenAI]
tags: [WrenAI, Docker, Installation, Setup, Configuration]
original_url: "https://github.com/Canner/WrenAI"
excerpt: "WrenAI를 Docker Compose로 설치하고 환경을 설정하는 방법을 안내합니다."
---

## 설치 방법 개요

WrenAI는 여러 가지 방법으로 설치할 수 있습니다:

| 방법 | 용도 | 난이도 |
|------|------|--------|
| **Docker Compose** | 로컬 개발, 소규모 운영 | 쉬움 |
| **Wren Launcher** | 원클릭 설치 | 매우 쉬움 |
| **Kubernetes** | 프로덕션 대규모 배포 | 보통 |
| **수동 설치** | 개발/커스터마이징 | 어려움 |

---

## 사전 요구사항

```bash
# 필수 소프트웨어
- Docker 20.10+
- Docker Compose v2
- 8GB+ RAM
- 10GB+ 디스크 공간

# LLM API 키 (하나 이상 필요)
- OpenAI API Key (gpt-4o-mini 권장)
- 또는 Azure OpenAI
- 또는 Google AI Studio
- 또는 기타 LiteLLM 지원 제공자
```

---

## Docker Compose 설치

### 1단계: 레포지토리 클론

```bash
git clone https://github.com/Canner/WrenAI.git
cd WrenAI/docker
```

### 2단계: 환경 파일 복사

```bash
cp .env.example .env.local
cp config.example.yaml config.yaml
```

### 3단계: config.yaml 설정

```yaml
# config.yaml

# LLM 설정 (필수)
type: llm
provider: litellm_llm
timeout: 120
models:
  - alias: default
    model: gpt-4o-mini          # 모델 선택
    context_window_size: 128000
    kwargs:
      temperature: 0
      max_tokens: 4096
      seed: 0

# 임베딩 모델 설정
type: embedder
provider: litellm_embedder
models:
  - model: text-embedding-3-large
    alias: default
    dimension: 3072
    timeout: 120

# 엔진 설정
type: engine
provider: wren_ui
endpoint: http://wren-ui:3000

type: engine
provider: wren_ibis
endpoint: http://ibis-server:8000

# 벡터 DB 설정
type: document_store
provider: qdrant
location: http://qdrant:6333
embedding_model_dim: 3072
timeout: 120
recreate_index: true

# 파이프라인 설정
type: pipeline
pipes:
  - name: sql_generation
    llm: litellm_llm.default
    engine: wren_ui
    document_store: qdrant
  - name: chart_generation
    llm: litellm_llm.default
    document_store: qdrant
```

### 4단계: 환경변수 설정

```bash
# .env.local 편집
OPENAI_API_KEY=sk-your-api-key-here

# 선택적 설정
TELEMETRY_ENABLED=true
USER_UUID=$(uuidgen)
```

### 5단계: 서비스 실행

```bash
docker compose --env-file .env.local up -d
```

### 6단계: 상태 확인

```bash
# 컨테이너 상태 확인
docker compose ps

# 예상 출력:
# NAME                STATUS
# wren-ui            running (healthy)
# wren-ai-service    running (healthy)
# wren-engine        running
# qdrant             running
# ibis-server        running
```

### 7단계: 접속

- **UI**: http://localhost:3000
- **AI Service API**: http://localhost:5555/docs

---

## Wren Launcher로 설치 (원클릭)

```bash
# macOS
curl -L https://github.com/Canner/WrenAI/releases/latest/download/wren-launcher-darwin-amd64 -o wren-launcher
chmod +x wren-launcher

# Linux
curl -L https://github.com/Canner/WrenAI/releases/latest/download/wren-launcher-linux-amd64 -o wren-launcher
chmod +x wren-launcher

# 설치 및 실행
./wren-launcher install  # 대화형 설정
./wren-launcher start    # 서비스 시작
./wren-launcher logs     # 로그 확인
```

---

## LLM 제공자별 설정

### OpenAI

```yaml
# config.yaml
type: llm
provider: litellm_llm
models:
  - alias: default
    model: gpt-4o-mini
    kwargs:
      temperature: 0
```

```bash
# .env.local
OPENAI_API_KEY=sk-...
```

### Azure OpenAI

```yaml
type: llm
provider: litellm_llm
models:
  - alias: default
    model: azure/gpt-4o-mini
    kwargs:
      api_base: https://your-resource.openai.azure.com
      api_version: "2024-02-15-preview"
```

```bash
AZURE_API_KEY=your-azure-key
```

### Google AI Studio (Gemini)

```yaml
type: llm
provider: litellm_llm
models:
  - alias: default
    model: gemini/gemini-1.5-flash
```

```bash
GOOGLE_API_KEY=your-google-key
```

### Ollama (로컬)

```yaml
type: llm
provider: litellm_llm
models:
  - alias: default
    model: ollama/llama3.1
    kwargs:
      api_base: http://host.docker.internal:11434
```

---

## 데이터베이스 설정

### SQLite (기본)

```bash
# .env.local - 별도 설정 불필요
DB_TYPE=sqlite
```

### PostgreSQL

```bash
# .env.local
DB_TYPE=pg
PG_HOST=localhost
PG_PORT=5432
PG_USER=postgres
PG_PASSWORD=your-password
PG_DATABASE=wrenai
```

---

## 포트 설정

| 서비스 | 기본 포트 | 환경변수 |
|--------|----------|----------|
| Wren UI | 3000 | `WREN_UI_PORT` |
| AI Service | 5555 | `WREN_AI_SERVICE_PORT` |
| Wren Engine | 8080 | `WREN_ENGINE_PORT` |
| Ibis Server | 8000 | `IBIS_SERVER_PORT` |
| Qdrant | 6333 | `QDRANT_PORT` |

---

## 문제 해결

### 컨테이너가 시작되지 않음

```bash
# 로그 확인
docker compose logs wren-ai-service
docker compose logs wren-ui

# 재시작
docker compose down
docker compose --env-file .env.local up -d
```

### LLM 연결 오류

```bash
# API 키 확인
echo $OPENAI_API_KEY

# config.yaml 문법 검사
python3 -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

### 메모리 부족

```bash
# Docker 리소스 확인
docker stats

# 권장: Docker Desktop에서 8GB+ RAM 할당
```

---

## 개발 모드 실행

```bash
# 개발용 docker-compose 사용
docker compose -f docker-compose-dev.yaml up -d

# 또는 개별 서비스 실행
cd wren-ui && yarn dev
cd wren-ai-service && just dev
```

---

*다음 글에서는 WrenAI의 아키텍처를 심층 분석합니다.*
