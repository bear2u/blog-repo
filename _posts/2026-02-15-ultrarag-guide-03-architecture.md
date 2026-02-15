---
layout: post
title: "UltraRAG 완벽 가이드 (03) - MCP 아키텍처 및 핵심 구성요소"
date: 2026-02-15
permalink: /ultrarag-guide-03-architecture/
author: UltraRAG Team
categories: [AI 에이전트, RAG]
tags: [RAG, MCP, LLM, UltraRAG, 아키텍처, Server]
original_url: "https://github.com/OpenBMB/UltraRAG"
excerpt: "UltraRAG의 MCP 기반 모듈식 아키텍처와 핵심 구성요소(Server)들을详细介绍합니다."
---

## MCP 아키텍처 개요

UltraRAG는 **Model Context Protocol (MCP)**을 기반으로 설계된 모듈식 RAG 프레임워크입니다. MCP는 AI 모델이 외부 도구, 데이터 소스, 컴포넌트와 통신하기 위한 표준화된 프로토콜입니다.

### 왜 MCP인가?

기존 RAG 시스템들은 컴포넌트들이 긴밀하게 결합되어 있어 확장性和 재사용성이 제한적이었습니다. UltraRAG는 각 핵심 기능을 독립적인 **MCP Server**로 분리하여 다음과 같은 장점을 제공합니다:

- **모듈화**: 각 서버가 독립적으로 개발, 테스트, 배포 가능
- **재사용성**: 함수 레벨의 Tool로 등록하여 워크플로우에 원활하게 통합
- **확장성**: 새로운 기능 추가가 용이
- **유지보수성**: 개별 컴포넌트 수정 시 다른 부분에 영향 최소화

---

## 핵심 구성요소 (MCP Servers)

UltraRAG는 다양한 기능의 MCP Server를 제공합니다:

### 1. Retriever Server (`servers/retriever`)

검색 기능 담당 서버입니다.

| 기능 | 설명 |
|------|------|
| **Dense Retrieval** | Sentence-Transformers, Infinity 임베딩 기반 벡터 검색 |
| **Sparse Retrieval** | BM25s 기반 텍스트 검색 |
| **Hybrid Search** | Dense + Sparse 결합 검색 |
| **Vector DB** | Milvus, Faiss 연동 |
| **Web Search** | Tavily, Exa 기반 웹 검색 |

### 2. Generation Server (`servers/generation`)

LLM 생성 기능 담당 서버입니다.

| 기능 | 설명 |
|------|------|
| **OpenAI API** | GPT-4, GPT-3.5-Turbo 등 지원 |
| **vLLM** | 자체 vLLM 서버 연동 |
| **HuggingFace** | 로컬 HF 모델 지원 |
| **VLM** | 비전-언어 모델 지원 |

### 3. Reranker Server (`servers/reranker`)

검색 결과 재순위화 서버입니다.

### 4. Router Server (`servers/router`)

쿼리 라우팅 및 분류 서버입니다.

### 5. Corpus Server (`servers/corpus`)

코퍼스 전처리 및 인덱싱 서버입니다.

| 기능 | 설명 |
|------|------|
| **Text Chunking** | 텍스트 분할 |
| **PDF Parsing** | PDF 문서 처리 |
| **Image OCR** | 이미지에서 텍스트 추출 (Mineru) |
| **Indexing** | 벡터 인덱스 생성 |

### 6. Benchmark Server (`servers/benchmark`)

평가 데이터셋 관리 서버입니다.

### 7. Evaluation Server (`servers/evaluation`)

결과 평가 서버입니다.

| 지표 | 설명 |
|------|------|
| **Rouge** | 텍스트 유사도 평가 |
| **Precision/Recall** | 검색 정밀도/재현율 |
| **Custom** | 사용자 정의 지표 |

### 8. Prompt Server (`servers/prompt`)

프롬프트 템플릿 관리 서버입니다.

### 9. Custom Server (`servers/custom`)

사용자 정의 기능 서버입니다.

---

## MCP Client 및 Pipeline

UltraRAG의 핵심은 **MCP Client**입니다. Client는 YAML 설정 파일을 기반으로 Pipeline을 오케스트레이션합니다.

### Pipeline 구조

```yaml
# MCP Server 정의
servers:
  benchmark: servers/benchmark
  retriever: servers/retriever
  prompt: servers/prompt
  generation: servers/generation
  evaluation: servers/evaluation
  custom: servers/custom

# MCP Client Pipeline
pipeline:
  - benchmark.get_data          # 1. 데이터 로드
  - retriever.retriever_init    # 2. 검색기 초기화
  - generation.generation_init   # 3. 생성기 초기화
  - retriever.retriever_search  # 4. 검색 수행
  - custom.assign_citation_ids  # 5. 인용 ID 할당
  - prompt.qa_rag_boxed         # 6. 프롬프트 선택
  - generation.generate         # 7. 생성 수행
```

### 제어 구조

UltraRAG는 복잡한 제어 구조를 지원합니다:

| 구조 | 설명 | 예시 |
|------|------|------|
| **Sequential** | 순차 실행 | 데이터 로드 → 검색 → 생성 |
| **Loop** | 반복 실행 | 다단계 검색 반복 |
| **Conditional** | 조건 분기 | 쿼리 유형에 따른 라우팅 |

---

## 디렉토리 구조

```
UltraRAG/
├── src/ultrarag/          # 핵심 라이브러리
│   ├── client.py          # MCP Client
│   ├── server.py          # MCP Server 基底
│   ├── cli.py             # CLI 인터페이스
│   └── api.py             # API 엔드포인트
├── servers/               # MCP Server 구현
│   ├── retriever/         # 검색 서버
│   ├── generation/        # 생성 서버
│   ├── reranker/          # 리랭커 서버
│   ├── router/            # 라우터 서버
│   ├── corpus/            # 코퍼스 처리 서버
│   ├── benchmark/         # 벤치마크 서버
│   ├── evaluation/        # 평가 서버
│   ├── prompt/            # 프롬프트 서버
│   └── custom/            # 커스텀 서버
├── examples/              # 예제 YAML 파이프라인
├── ui/                    # 웹 UI
├── docs/                  # 문서
└── script/                # 유틸리티 스크립트
```

---

## 코드 통합

UltraRAG 컴포넌트를 Python 코드에서 직접 호출할 수도 있습니다:

```python
from ultrarag import UltraRAGClient

# 클라이언트 초기화
client = UltraRAGClient(config_path="config.yaml")

# 검색 수행
results = client.retriever.search(query="질문", top_k=10)

# 생성 수행
answer = client.generate(
    query="질문",
    context=results
)
```

---

*다음 글에서는 YAML 파이프라인 설정 방법에 대해 자세히 살펴보겠습니다.*
