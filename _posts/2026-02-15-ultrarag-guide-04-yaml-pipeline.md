---
layout: post
title: "UltraRAG 완벽 가이드 (04) - YAML 파이프라인 설정"
date: 2026-02-15
permalink: /ultrarag-guide-04-yaml-pipeline/
author: UltraRAG Team
categories: [AI 에이전트, RAG]
tags: [RAG, MCP, LLM, UltraRAG, YAML, Pipeline]
original_url: "https://github.com/OpenBMB/UltraRAG"
excerpt: "UltraRAG의 YAML 설정 파일을 활용한 파이프라인 구성 방법을 단계별로 설명합니다."
---

## YAML 파이프라인 개요

UltraRAG의 핵심 강점은 **YAML 설정만으로 복잡한 RAG 파이프라인을 구축**할 수 있다는 것입니다. 개발자는 코드를 작성하지 않고도 순차, 루프, 조건 분기 등 다양한 제어 구조를 구현할 수 있습니다.

---

## 기본 구조

UltraRAG YAML 설정 파일은 두 가지 주요 섹션으로 구성됩니다:

```yaml
# 1. MCP Server 정의
servers:
  server_name: path/to/server

# 2. Pipeline 정의
pipeline:
  - server.component.method  # 실행할 작업
```

---

## 기본 예제: Hello World

가장 간단한 예제来看看:

```yaml
# examples/sayhello.yaml
pipeline:
  - sayhello.hello
```

실행:

```shell
ultrarag run examples/sayhello.yaml
```

출력:

```
Hello, UltraRAG v3!
```

---

## 기본 RAG 파이프라인

가장 흔한 RAG 워크플로우입니다:

```yaml
# MCP Server 정의
servers:
  benchmark: servers/benchmark
  retriever: servers/retriever
  prompt: servers/prompt
  generation: servers/generation
  custom: servers/custom

# Pipeline 정의
pipeline:
  # 1. 데이터 로드
  - benchmark.get_data

  # 2. 검색기 초기화
  - retriever.retriever_init

  # 3. 생성기 초기화
  - generation.generation_init

  # 4. 검색 수행
  - retriever.retriever_search

  # 5. 인용 ID 할당
  - custom.assign_citation_ids

  # 6. 프롬프트 선택
  - prompt.qa_rag_boxed

  # 7. 생성 수행
  - generation.generate
```

---

## 파라미터 설정

각 단계에서 파라미터를 전달할 수 있습니다:

```yaml
pipeline:
  # 검색기 초기화 with 파라미터
  - retriever.retriever_init:
      model: "BAAI/bge-m3"
      device: "cuda"
      top_k: 5

  # 검색 수행 with 파라미터
  - retriever.retriever_search:
      query: "什么是RAG?"
      collection: "my_knowledge_base"
      top_k: 10
```

---

## 루프 (Loops)

반복적인 검색이 필요한 경우 루프를 사용할 수 있습니다:

```yaml
# examples/rag_loop.yaml
pipeline:
  - retriever.retriever_init
  - generation.generation_init

  # 루프 시작 (최대 3회)
  - loop:
      max_iterations: 3
      body:
        - retriever.retriever_search
        - generation.generate
        # 이전 결과 기반으로 검색어 수정
        - custom.modify_query
```

---

## 조건 분기 (Conditional Branches)

쿼리 유형이나 결과에 따라 다른 처리를 할 수 있습니다:

```yaml
# examples/rag_branch.yaml
pipeline:
  - router.classify_query

  # 조건 분기
  - branch:
      conditions:
        - if: "query_type == 'factual'"
          then:
            - retriever.retriever_search
            - prompt.qa_rag_direct

        - if: "query_type == 'analytical'"
          then:
            - retriever.retriever_search
            - prompt.qa_rag_analysis
            - generation.generate_with_reasoning

        - if: "query_type == 'creative'"
          then:
            - prompt.qa_rag_creative
            - generation.generate_creative
```

---

## 고급 예제: IRCoT (Iterative Retrieval CoT)

반복적 검색과 사고 체인 결합:

```yaml
# examples/ircot.yaml
pipeline:
  - retriever.retriever_init
  - generation.generation_init

  - loop:
      max_iterations: 5
      body:
        # 검색
        - retriever.retriever_search

        #思考 체인 생성
        - generation.generate_cot

        # 종료 조건 확인
        - custom.check_completeness
```

---

## 예제: 하이브리드 검색

밀도 검색과 희소 검색을 결합:

```yaml
# examples/hybrid_search.yaml
servers:
  retriever: servers/retriever

pipeline:
  - retriever.retriever_init:
      method: "hybrid"

  - retriever.retriever_search:
      dense_weight: 0.7
      sparse_weight: 0.3

  - retriever.rerank:
      model: "BAAI/bge-reranker-base"
```

---

## 예제: 멀티턴 대화

대화 맥락을 유지하는 RAG:

```yaml
# examples/multiturn_chat.yaml
pipeline:
  - retriever.retriever_init
  - generation.generation_init

  # 대화 이력 로드
  - benchmark.load_conversation_history

  # 현재 쿼리 검색
  - retriever.retriever_search

  # 맥락 포함 생성
  - prompt.qa_rag_with_history
  - generation.generate
```

---

## 실행 방법

```shell
# 파이프라인 실행
ultrarag run <yaml_file>

# 파라미터와 함께 실행
ultrarag run <yaml_file> --query "질문" --top-k 5

# 서버 모드로 실행 (API 제공)
ultrarag serve <yaml_file>
```

---

## 팁 및 베스트 프랙티스

1. **단계별 테스트**: 복잡한 파이프라인은 단계별로 테스트하세요
2. **파라미터 관리**: 파라미터는 별도 YAML 파일로 분리하여 관리하세요
3. **디버깅**: `debug: true` 설정으로 상세 로그를 확인하세요
4. **모듈화**: 재사용 가능한 서브 파이프라인을 별도 파일로 분리하세요

---

*다음 글에서는 검색(Retriever) 모듈에 대해 자세히 살펴보겠습니다.*
