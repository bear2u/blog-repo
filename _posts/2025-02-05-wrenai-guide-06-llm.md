---
layout: post
title: "WrenAI 완벽 가이드 (6) - LLM 연동"
date: 2025-02-05
permalink: /wrenai-guide-06-llm/
author: Canner
categories: [AI 에이전트, WrenAI]
tags: [WrenAI, LLM, OpenAI, Azure, LiteLLM, Ollama]
original_url: "https://github.com/Canner/WrenAI"
excerpt: "WrenAI에서 다양한 LLM 제공자(OpenAI, Azure, Ollama 등)를 연동하는 방법을 안내합니다."
---

## LLM 연동 개요

WrenAI는 **LiteLLM**을 통해 다양한 LLM 제공자를 지원합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    지원 LLM 제공자                          │
├─────────────────────────────────────────────────────────────┤
│  ☁️  클라우드: OpenAI, Azure, Google, Anthropic, AWS       │
│  🖥️  로컬: Ollama, LM Studio, vLLM                         │
│  🏢  엔터프라이즈: Databricks, Snowflake Cortex            │
└─────────────────────────────────────────────────────────────┘
```

---

## 기본 설정 구조

### config.yaml LLM 섹션

```yaml
# LLM 제공자 설정
type: llm
provider: litellm_llm
timeout: 120
models:
  - alias: default          # 별칭 (필수)
    model: gpt-4o-mini      # 모델명 (필수)
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
```

---

## 제공자별 설정

### OpenAI

```yaml
type: llm
provider: litellm_llm
models:
  - alias: default
    model: gpt-4o-mini      # 또는 gpt-4o, gpt-4-turbo
    context_window_size: 128000
    kwargs:
      temperature: 0
      max_tokens: 4096

type: embedder
provider: litellm_embedder
models:
  - model: text-embedding-3-large
    alias: default
    dimension: 3072
```

```bash
# .env.local
OPENAI_API_KEY=sk-your-key-here
```

---

### Azure OpenAI

```yaml
type: llm
provider: litellm_llm
models:
  - alias: default
    model: azure/your-deployment-name
    context_window_size: 128000
    kwargs:
      api_base: https://your-resource.openai.azure.com
      api_version: "2024-02-15-preview"
      temperature: 0

type: embedder
provider: litellm_embedder
models:
  - model: azure/your-embedding-deployment
    alias: default
    dimension: 1536
    kwargs:
      api_base: https://your-resource.openai.azure.com
      api_version: "2024-02-15-preview"
```

```bash
# .env.local
AZURE_API_KEY=your-azure-key
```

---

### Google AI Studio (Gemini)

```yaml
type: llm
provider: litellm_llm
models:
  - alias: default
    model: gemini/gemini-1.5-flash    # 또는 gemini-1.5-pro
    context_window_size: 1000000
    kwargs:
      temperature: 0

type: embedder
provider: litellm_embedder
models:
  - model: gemini/text-embedding-004
    alias: default
    dimension: 768
```

```bash
# .env.local
GOOGLE_API_KEY=your-google-key
```

---

### Google Vertex AI

```yaml
type: llm
provider: litellm_llm
models:
  - alias: default
    model: vertex_ai/gemini-1.5-flash
    context_window_size: 1000000
    kwargs:
      vertex_project: your-project-id
      vertex_location: us-central1
```

```bash
# .env.local
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

---

### Anthropic Claude

```yaml
type: llm
provider: litellm_llm
models:
  - alias: default
    model: claude-3-5-sonnet-20241022
    context_window_size: 200000
    kwargs:
      temperature: 0
      max_tokens: 4096
```

```bash
# .env.local
ANTHROPIC_API_KEY=sk-ant-...
```

---

### AWS Bedrock

```yaml
type: llm
provider: litellm_llm
models:
  - alias: default
    model: bedrock/anthropic.claude-3-sonnet-20240229-v1:0
    context_window_size: 200000
    kwargs:
      aws_region_name: us-east-1
```

```bash
# .env.local
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION_NAME=us-east-1
```

---

### DeepSeek

```yaml
type: llm
provider: litellm_llm
models:
  - alias: default
    model: deepseek/deepseek-chat
    context_window_size: 64000
    kwargs:
      temperature: 0
```

```bash
# .env.local
DEEPSEEK_API_KEY=your-deepseek-key
```

---

### Ollama (로컬)

```yaml
type: llm
provider: litellm_llm
models:
  - alias: default
    model: ollama/llama3.1:8b
    context_window_size: 128000
    kwargs:
      api_base: http://host.docker.internal:11434
      temperature: 0

type: embedder
provider: litellm_embedder
models:
  - model: ollama/nomic-embed-text
    alias: default
    dimension: 768
    kwargs:
      api_base: http://host.docker.internal:11434
```

```bash
# Ollama 실행 (Docker 외부)
ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

---

## 다중 모델 설정

```yaml
type: llm
provider: litellm_llm
models:
  # 기본 모델 (빠른 응답)
  - alias: default
    model: gpt-4o-mini
    context_window_size: 128000
    kwargs:
      temperature: 0

  # 고성능 모델 (복잡한 쿼리)
  - alias: advanced
    model: gpt-4o
    context_window_size: 128000
    kwargs:
      temperature: 0

  # 비용 절약 모델
  - alias: budget
    model: gpt-3.5-turbo
    context_window_size: 16000
    kwargs:
      temperature: 0
```

### 파이프라인에서 모델 선택

```yaml
type: pipeline
pipes:
  - name: sql_generation
    llm: litellm_llm.default     # 기본 모델 사용

  - name: chart_generation
    llm: litellm_llm.advanced    # 고성능 모델 사용
```

---

## 프록시 및 커스텀 엔드포인트

### OpenAI 호환 API

```yaml
type: llm
provider: litellm_llm
models:
  - alias: default
    model: openai/custom-model
    kwargs:
      api_base: https://your-custom-endpoint.com/v1
      api_key: your-custom-key
```

### LiteLLM 프록시

```yaml
type: llm
provider: litellm_llm
models:
  - alias: default
    model: your-model-name
    kwargs:
      api_base: http://litellm-proxy:4000
```

---

## 비용 추적 (LangFuse)

```yaml
settings:
  langfuse_enable: true
  langfuse_host: https://cloud.langfuse.com
```

```bash
# .env.local
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
```

### 추적되는 정보

- 모델명
- 입력/출력 토큰 수
- 예상 비용
- 응답 시간
- 성공/실패 상태

---

## 권장 모델

| 용도 | 권장 모델 | 이유 |
|------|----------|------|
| **기본** | gpt-4o-mini | 빠름, 저렴, 충분한 성능 |
| **복잡한 SQL** | gpt-4o | 더 정확한 추론 |
| **비용 절약** | gpt-3.5-turbo | 가장 저렴 |
| **프라이버시** | Ollama/llama3.1 | 로컬 실행 |
| **한국어** | gpt-4o | 다국어 지원 우수 |

---

## 문제 해결

### API 키 오류

```bash
# 키 확인
echo $OPENAI_API_KEY

# 테스트
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### 타임아웃 오류

```yaml
type: llm
provider: litellm_llm
timeout: 180  # 기본 120초에서 증가
```

### Rate Limit 오류

```yaml
kwargs:
  max_retries: 3
  retry_on_timeout: true
```

---

*다음 글에서는 프론트엔드 구조를 살펴봅니다.*
