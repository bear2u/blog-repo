---
layout: post
title: "UltraRAG 완벽 가이드 (09) - Deep Research 파이프라인"
date: 2026-02-15
permalink: /ultrarag-guide-09-deepresearch/
author: UltraRAG Team
categories: [AI 에이전트, RAG]
tags: [RAG, MCP, LLM, UltraRAG, Deep Research, AgentCPM]
original_url: "https://github.com/OpenBMB/UltraRAG"
excerpt: "UltraRAG의flagship 기능: AgentCPM-Report 모델과 Deep Research 파이프라인으로 자동 조사 리포트를 생성하는 방법을 알아봅니다."
---

## Deep Research 개요

UltraRAG의 **Deep Research** 파이프라인은 대규모 자동 조사 리포트 생성 기능입니다:

- **다단계 검색**: 반복적 검색으로 심층 정보 수집
- **정보 통합**: 여러 소스의 정보를 종합
- **장문 생성**:数万 단어의 리포트 자동 생성
- **AgentCPM-Report**: On-device 에이전트 (8B 파라미터)

---

## AgentCPM-Report 모델

清华大学에서 개발한 **AgentCPM-Report**는 경량化了한 Deep Research 모델입니다:

| 모델 | 파라미터 | 특징 |
|------|----------|------|
| AgentCPM-Report-8B | 8B | 로컬 실행 가능, On-device |

### 특징

- **로컬 실행**: 고성능 GPU 없이도 실행 가능
- **장문 생성**:数万 단어의 상세 리포트 생성
- **다단계 추론**: 복잡한 조사 작업 처리
- **开源**: HuggingFace에서 다운로드 가능

### 모델 다운로드

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "openbmb/AgentCPM-Report",
    device_map="auto"
)
```

---

## Deep Research 파이프라인

### 기본 구성

```yaml
# examples/LightResearch.yaml
servers:
  retriever: servers/retriever
  generation: servers/generation
  custom: servers/custom

pipeline:
  # 1. 검색기/생성기 초기화
  - retriever.retriever_init
  - generation.generation_init

  # 2. 반복적 검색 및 생성
  - loop:
      max_iterations: 10
      body:
        # 웹 검색
        - retriever.web_search

        # 관련 문서 검색
        - retriever.retriever_search

        # 검색 결과 기반으로 사고 체인 생성
        - generation.generate_cot

        # 새로운 검색어 추출
        - custom.extract_new_queries

        # 종료 조건 확인
        - custom.check_completeness

  # 3. 최종 리포트 생성
  - generation.generate_report:
      max_words: 30000
      structure: "academic"
```

---

## R1 Searcher

DeepSeek-R1 기반 검색 에이전트:

```yaml
# examples/search_r1.yaml
pipeline:
  - retriever.retriever_init

  - generation.generation_init:
      model: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
      provider: "huggingface"

  - loop:
      max_iterations: 5
      body:
        - retriever.retriever_search
        - generation.reasoning_search:
            query: "{{query}}"
            context: "{{previous_context}}"
```

---

## IRCoT (Iterative Retrieval CoT)

검색과 사고 체인의 반복적 결합:

```yaml
# examples/ircot.yaml
pipeline:
  - retriever.retriever_init
  - generation.generation_init

  # 초기화
  - custom.init_research:
      topic: "{{query}}"

  # 반복적 검색 + CoT
  - loop:
      max_iterations: 5
      body:
        #检索
        - retriever.retriever_search

        # CoT 기반 분석
        - generation.cot_generate:
            prompt: "검색 결과를 분석하고 새로운研究方向를 제시하세요"

        # 다음 검색 쿼리 생성
        - custom.generate_next_query

        # 종료 조건
        - custom.should_stop:
            threshold: 0.8

  # 최종 리포트
  - generation.finalize_report:
      format: "markdown"
      min_words: 10000
```

---

## RankCoT

순위付けを考慮した 검색:

```yaml
# examples/rankcot.yaml
pipeline:
  - retriever.retriever_init
  - generation.generation_init

  - loop:
      max_iterations: 3
      body:
        # 초기 검색
        - retriever.retriever_search

        # 중요도 순위 매기기
        - generation.rank_cot:
            criteria: "정보의 중요도와 신뢰도"

        # 상위 결과 상세 분석
        - custom.analyze_top_results

  # 종합 리포트
  - generation.synthesize_report
```

---

## IterRetGen (Iterative Retrieval Generation)

반복적 검색과 생성의 결합:

```yaml
# examples/iterretgen.yaml
pipeline:
  - retriever.retriever_init
  - generation.generation_init

  # 초기 답변 생성
  - generation.generate:
      prompt: "초기 답변을 생성하세요"

  - loop:
      max_iterations: 3
      body:
        # 부족한 정보 식별
        - custom.identify_gaps

        # 보완 검색
        - retriever.retriever_search

        # 답변 개선
        - generation.refine:
            strategy: "incremental"

  # 최종 답변
  - generation.final_answer
```

---

## 웹 검색 통합

Deep Research에 웹 검색을 통합:

```yaml
# examples/webnote.yaml
pipeline:
  - retriever.web_search:
      engine: "tavily"
      query: "{{topic}}"
      max_results: 20

  - retriever.retriever_search:
      collection: "web_index"

  - generation.research_generate:
      max_words: 50000
      sections:
        - introduction
        - background
        - methods
        - results
        - discussion
        - conclusion
```

---

## 실행 예시

```shell
# CLI로 Deep Research 실행
ultrarag run examples/LightResearch.yaml \
  --query "MCP 프로토콜의 최신 연구 동향과 미래 전망" \
  --output ./research_report.md
```

### 출력 예시

```
# MCP 프로토콜의 최신 연구 동향과 미래 전망

## 1. 서론
Model Context Protocol (MCP)은 AI 시스템이 외부 도구와 통신하기 위한...

## 2. 관련 연구
...

## 3. 기술적 분석
...

## 4. 미래 전망
...
```

---

## 성능 최적화

| 기법 | 설명 | 효과 |
|------|------|------|
| **캐싱** | 검색 결과 캐싱 | 중복 검색 방지 |
| **병렬 처리** | 병렬 검색/생성 | 속도 향상 |
| **Streaming** | 실시간 출력 | 사용자 경험 개선 |
| **토큰 관리** | 컨텍스트 압축 | 비용 절감 |

---

*마지막 글에서는 확장 및 커스터마이징 방법에 대해 살펴보겠습니다.*
