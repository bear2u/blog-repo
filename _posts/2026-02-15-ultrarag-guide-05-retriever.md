---
layout: post
title: "UltraRAG 완벽 가이드 (05) - 검색(Retriever) 모듈"
date: 2026-02-15
permalink: /ultrarag-guide-05-retriever/
author: UltraRAG Team
categories: [AI 에이전트, RAG]
tags: [RAG, MCP, LLM, UltraRAG, Retriever, 검색, Embedding]
original_url: "https://github.com/OpenBMB/UltraRAG"
excerpt: "UltraRAG의 검색 모듈: Dense Retrieval, Sparse Retrieval, Hybrid Search, 웹 검색까지 상세 가이드입니다."
---

## 검색 모듈 개요

UltraRAG의 검색(Retriever) 모듈은 다양한 검색 전략을 지원합니다:

- **Dense Retrieval**: 임베딩 기반 벡터 검색
- **Sparse Retrieval**: BM25 기반 텍스트 검색
- **Hybrid Search**: 밀도 + 희소 검색 결합
- **Web Search**: Tavily, Exa 기반 웹 검색
- **Reranking**: 검색 결과 재순위화

---

## Dense Retrieval (밀도 검색)

임베딩 모델을 사용하여 쿼리와 문서의语义적 유사도를 계산합니다.

### 초기화

```yaml
pipeline:
  - retriever.retriever_init:
      method: "dense"
      model: "BAAI/bge-m3"       # 임베딩 모델
      device: "cuda"             # CPU 또는 CUDA
      batch_size: 32
      normalize: true
      use_fp16: true
```

### 사용 가능한 모델

| 모델 | 설명 | 속도 |
|------|------|------|
| BAAI/bge-m3 | 멀티언어, 멀티모달 지원 | 중간 |
| BAAI/bge-base-en | 영어 전용 | 빠름 |
| intfloat/e5-mistral | 고성능 임베딩 | 중간 |
| sentence-transformers/all-MiniLM-L6-v2 | 경량 모델 | 매우 빠름 |

### 검색 수행

```yaml
pipeline:
  - retriever.retriever_search:
      query: "UltraRAG是什么?"
      collection: "my_knowledge_base"
      top_k: 10
      score_threshold: 0.5
```

---

## Sparse Retrieval (희소 검색)

BM25 알고리즘을 사용한 전통적인 텍스트 검색입니다.

### 초기화

```yaml
pipeline:
  - retriever.retriever_init:
      method: "sparse"
      model: "bm25s"
      language: "en"  # 또는 "ko", "zh"
```

### 인덱스 생성

```yaml
# 텍스트 코퍼스 인덱싱
pipeline:
  - retriever.build_bm25_index:
      corpus_path: "./data/text_corpus/"
      index_path: "./indexes/bm25/"
```

### 검색 수행

```yaml
pipeline:
  - retriever.bm25_search:
      query: "RAG framework"
      top_k: 20
```

---

## Hybrid Search (하이브리드 검색)

밀도와 희소 검색을 결합하여 더 나은 검색 결과를 제공합니다.

### 설정

```yaml
# examples/hybrid_search.yaml
servers:
  retriever: servers/retriever

pipeline:
  - retriever.retriever_init:
      method: "hybrid"
      dense_model: "BAAI/bge-m3"
      sparse_model: "bm25s"

  - retriever.retriever_search:
      query: "什么是RAG?"
      top_k: 10
      dense_weight: 0.7    # 밀도 검색 가중치
      sparse_weight: 0.3   # 희소 검색 가중치
```

### 가중치 조정 가이드

| 시나리오 | dense_weight | sparse_weight |
|----------|--------------|---------------|
|语义적 검색 위주 | 0.8 | 0.2 |
|키워드 매칭 위주 | 0.3 | 0.7 |
|균형잡힌 결과 | 0.5 | 0.5 |

---

## Vector Database 연동

### Milvus

```yaml
pipeline:
  - retriever.retriever_init:
      method: "dense"
      model: "BAAI/bge-m3"
      vector_db: "milvus"
      connection:
        host: "localhost"
        port: 19530
      collection: "my_docs"
```

### Faiss

```yaml
pipeline:
  - retriever.retriever_init:
      method: "dense"
      model: "BAAI/bge-m3"
      vector_db: "faiss"
      index_type: "IVF_FLAT"
      nlist: 100
```

---

## Reranking (재순위화)

초기 검색 결과를 다시 정렬하여 더 정확한 순서를 제공합니다.

```yaml
pipeline:
  - retriever.retriever_search:
      query: "RAG란?"
      top_k: 100  # 더 많은 결과 검색

  - retriever.rerank:
      model: "BAAI/bge-reranker-base"
      top_k: 10   # 상위 10개만 반환
```

### 사용 가능한 Reranker 모델

| 모델 | 설명 |
|------|------|
| BAAI/bge-reranker-base | 경량 reranker |
| BAAI/bge-reranker-large |高性能 reranker |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 경량 교차 인코더 |

---

## Web Search (웹 검색)

온라인 검색 결과를 RAG 파이프라인에 통합합니다.

### Tavily Search

```yaml
pipeline:
  - retriever.web_search:
      engine: "tavily"
      query: "latest RAG research 2025"
      max_results: 5
      api_key: "${TAVILY_API_KEY}"
```

### Exa Search

```yaml
pipeline:
  - retriever.web_search:
      engine: "exa"
      query: "MCP protocol documentation"
      max_results: 10
      api_key: "${EXA_API_KEY}"
```

---

## 코퍼스 인덱싱

문서를 검색 가능한 형태로 변환합니다.

### 텍스트 코퍼스

```yaml
# examples/build_text_corpus.yaml
pipeline:
  - corpus.load_text:
      path: "./data/documents/"
      extensions: [".txt", ".md"]

  - corpus.chunk:
      chunk_size: 512
      overlap: 50

  - corpus.index:
      method: "hybrid"
      index_path: "./indexes/my_corpus/"
```

### PDF 코퍼스

```yaml
# examples/build_mineru_corpus.yaml
pipeline:
  - corpus.load_pdf:
      path: "./data/pdfs/"

  - corpus.extract_text:
      use_ocr: true
      languages: ["ko", "en", "zh"]

  - corpus.index:
      method: "dense"
      model: "BAAI/bge-m3"
```

---

## 검색 결과 형식

검색 결과는 다음과 같은 형식으로 반환됩니다:

```json
{
  "results": [
    {
      "id": "doc_001",
      "text": "UltraRAG는 MCP 기반 RAG 프레임워크입니다...",
      "score": 0.95,
      "metadata": {
        "source": "docs/intro.md",
        "page": 1,
        "title": "UltraRAG 소개"
      }
    }
  ]
}
```

---

*다음 글에서는 생성(Generation) 모듈에 대해 자세히 살펴보겠습니다.*
