---
layout: post
title: "UltraRAG 완벽 가이드 (06) - 생성(Generation) 모듈"
date: 2026-02-15
permalink: /ultrarag-guide-06-generation/
author: UltraRAG Team
categories: [AI 에이전트, RAG]
tags: [RAG, MCP, LLM, UltraRAG, Generation, LLM, vLLM]
original_url: "https://github.com/OpenBMB/UltraRAG"
excerpt: "UltraRAG의 생성 모듈: OpenAI API, vLLM, HuggingFace 모델 연동과 프롬프트 관리까지 상세 가이드입니다."
---

## 생성 모듈 개요

UltraRAG의 생성(Generation) 모듈은 다양한 LLM 백엔드를 지원합니다:

- **OpenAI API**: GPT-4, GPT-3.5-Turbo 등
- **vLLM**:高性能 LLM 서빙
- **HuggingFace**: 로컬 모델 지원
- **VLM**: 비전-언어 모델 (이미지 이해)

---

## 초기화 방법

### OpenAI API

```yaml
pipeline:
  - generation.generation_init:
      provider: "openai"
      model: "gpt-4o"
      api_key: "${OPENAI_API_KEY}"
      base_url: "https://api.openai.com/v1"  # 기본값
      temperature: 0.7
      max_tokens: 2048
```

### vLLM 서버

```yaml
pipeline:
  - generation.generation_init:
      provider: "vllm"
      model: "Qwen/Qwen2.5-7B-Instruct"
      api_url: "http://localhost:8000/v1"
      temperature: 0.7
      max_tokens: 2048
```

vLLM 서버를 먼저 실행해야 합니다:

```shell
# vLLM 서버 시작
vllm serve Qwen/Qwen2.5-7B-Instruct --api-key token-abc123
```

### HuggingFace 모델

```yaml
pipeline:
  - generation.generation_init:
      provider: "huggingface"
      model: "meta-llama/Llama-3.1-8B-Instruct"
      device: "cuda"
      temperature: 0.7
      max_tokens: 2048
```

---

## 프롬프트 관리

UltraRAG는 `servers/prompt`를 통해 프롬프트 템플릿을 관리합니다.

### 기본 QA 프롬프트

```yaml
pipeline:
  - prompt.qa_rag:
      template: "rag_basic"

  - generation.generate:
      query: "RAG란 무엇인가요?"
```

### 사용 가능한 프롬프트 템플릿

| 템플릿 | 설명 |
|--------|------|
| `qa_rag` | 기본 RAG QA |
| `qa_rag_boxed` | 인용 포함 QA |
| `qa_rag_direct` | 직접 답변 |
| `qa_rag_analysis` | 분석적 답변 |
| `qa_rag_creative` | 창작적 답변 |

### 커스텀 프롬프트

```yaml
pipeline:
  - prompt.custom:
      template: |
        다음 문서를 기반으로 질문에 답변해주세요.

        문서:
        {{context}}

        질문: {{query}}

        답변:
```

---

## 생성 실행

### 기본 생성

```yaml
pipeline:
  - retriever.retriever_search:
      query: "UltraRAG의 장점은?"
      top_k: 5

  - prompt.qa_rag_boxed

  - generation.generate:
      query: "UltraRAG의 장점은?"
      temperature: 0.7
      max_tokens: 1000
```

### Chat 형식

```yaml
pipeline:
  - prompt.chat:
      system: "당신은helpful한 어시스턴트입니다."
      messages:
        - role: "user"
          content: "{{query}}"

  - generation.chat:
      model: "gpt-4o"
      temperature: 0.7
```

---

## Reasoning 모델

 Chain-of-Thought (CoT) reasoning을 지원하는 모델용:

### search_o1.yaml

```yaml
# examples/search_o1.yaml
pipeline:
  - retriever.retriever_init

  - generation.generation_init:
      provider: "openai"
      model: "o1-preview"

  - generation.reasoning_generate:
      query: "RAG 시스템의 검색 정확도를 높이는 방법은?"
      max_tokens: 4096
```

### search_r1.yaml

```yaml
# examples/search_r1.yaml
pipeline:
  - retriever.retriever_search
  - generation.reasoning_generate:
      query: "검색 결과를 기반으로 분석해주세요"
      model: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
      do_sample: false
      max_tokens: 4096
```

---

## Multimodal (멀티모달)

VLM(Vision-Language Model)을 사용한 이미지 이해:

```yaml
# examples/vanilla_vlm.yaml
pipeline:
  - generation.generation_init:
      provider: "openai"
      model: "gpt-4o"  # 또는 다른 VLM

  - generation.generate_with_image:
      query: "이 이미지에서 무엇을 볼 수 있나요?"
      image_url: "https://example.com/image.jpg"
```

---

## 파라미터 설명

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `temperature` | 생성 다양성 (높을수록 창의적) | 0.7 |
| `max_tokens` | 최대 생성 토큰 수 | 2048 |
| `top_p` | Nucleus sampling | 1.0 |
| `frequency_penalty` | 반복 페널티 | 0.0 |
| `presence_penalty` | 존재 페널티 | 0.0 |
| `stop` | 중지 시퀀스 | null |

---

## 응답 형식

생성 결과는 다음과 같은 형식으로 반환됩니다:

```json
{
  "text": "UltraRAG는 Model Context Protocol(MCP)을 기반으로...",
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 200,
    "total_tokens": 300
  },
  "finish_reason": "stop"
}
```

---

## AgentCPM-Report 모델

UltraRAG는 **AgentCPM-Report** 모델과 연동하여 Deep Research 기능을 제공합니다:

```yaml
# examples/AgentCPM-Report.yaml
pipeline:
  - generation.generation_init:
      provider: "huggingface"
      model: "openbmb/AgentCPM-Report-8B"

  - generation.deep_research:
      query: "MCP 프로토콜의 최신 연구 동향"
      max_words: 50000  # 생성할 리포트 길이
      search_iterations: 10
```

---

*다음 글에서는 평가(Evaluation) 시스템에 대해 자세히 살펴보겠습니다.*
