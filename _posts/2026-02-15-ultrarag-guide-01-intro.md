---
layout: post
title: "UltraRAG 완벽 가이드 (01) - 소개 및 개요"
date: 2026-02-15
permalink: /ultrarag-guide-01-intro/
author: UltraRAG Team
categories: [AI 에이전트, RAG]
tags: [RAG, MCP, LLM, UltraRAG, 검색증강생성]
original_url: "https://github.com/OpenBMB/UltraRAG"
excerpt: "清华大学, NEUIR, OpenBMB, AI9stars가 함께 만든 MCP 기반 경량 RAG 개발 프레임워크를 소개합니다."
---

## UltraRAG란?

**UltraRAG**는 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/docs/getting-started/intro) 아키텍처를 기반으로 설계된 최초의 경량 RAG(Retrieval-Augmented Generation) 개발 프레임워크입니다.清华大学(Tsinghua University) THUNLP, Northeastern University NEUIR, [OpenBMB](https://www.openbmb.cn/home), AI9stars가 공동으로 개발했습니다.

### 주요 비전

> **Less Code, Lower Barrier, Faster Deployment**

UltraRAG는 연구 탐색과 산업 프로토타이핑을 위해 설계되었으며, 핵심 RAG 컴포넌트(Retriever, Generation 등)를 독립적인 **MCP Server**로 표준화했습니다. **MCP Client**의 강력한 워크플로우 오케스트레이션 기능과 결합하여 개발자는 조건 분기, 루프 등 복잡한 제어 구조를 YAML 설정만으로 정밀하게 오케스트레이션할 수 있습니다.

---

## 주요 특징

### 🚀 Low-Code 복잡한 워크플로우 오케스트레이션

추론 오케스트레이션은 순차, 루프, 조건 분기 등 제어 구조를 네이티브로 지원합니다. 개발자는 복잡한 반복 RAG 로직을 수십 줄의 YAML 설정 파일로 구현할 수 있습니다.

### ⚡ 모듈식 확장 및 재현

**원자 서버(Atomic Servers)**: MCP 아키텍처 기반으로 함수를 독립적인 Server로 분리합니다. 새로운 기능은 함수 레벨의 Tool로 등록하기만 하면 워크플로우에 원활하게 통합되어 극高的 재사용성을 달성합니다.

### 📊 통합 평가 및 벤치마크 비교

연구 효율성을 위해 구축된 표준화된 평가 워크플로우와 즉시 사용 가능한 주요 연구 벤치마크를 제공합니다. 통합 지표 관리와 베이스라인 통합을 통해 실험 재현성 및 비교 효율성을 크게 향상시킵니다.

### 🎯 빠른 대화형 프로토타입 생성

繁琐한 UI 개발에 작별을 고합니다. 단 한 줄의 명령으로 Pipeline 로직을 즉각적으로 대화형 웹 UI로 변환하여 알고리즘에서 데모까지의 거리를 단축합니다.

---

## 버전 히스토리

| 버전 | 출시일 | 주요 내용 |
|------|--------|----------|
| **3.0** | 2026.01.23 | "블랙 박스" 개발Say No - 모든 추론 로직이 선명하게 보이도록 |
| **2.1** | 2025.11.11 | 지식 수집 및 멀티모달 지원 강화, 통합 평가 시스템 |
| **2.0** | 2025.08.28 |高性能 RAG를 수십 줄의 코드로 구축 |
| **1.0** | 2025.01.23 | 초대형 모델이 지식 베이스를 더 잘 이해하고 활용할 수 있도록 |

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    UltraRAG Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Retriever  │    │  Generation │    │   Reranker │     │
│  │   Server    │    │   Server    │    │   Server   │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            │                                  │
│                    ┌───────▼───────┐                         │
│                    │ MCP Client    │                         │
│                    │ (YAML Config) │                         │
│                    └───────┬───────┘                         │
│                            │                                  │
│                    ┌───────▼───────┐                         │
│                    │   Pipeline    │                         │
│                    │   Orchestration│                        │
│                    └───────────────┘                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 지원 기능

### 검색 (Retrieval)
- **Dense Retrieval**: 임베딩 기반 벡터 검색
- **Sparse Retrieval**: BM25 기반 텍스트 검색
- **Hybrid Search**: 밀도/희소 검색 결합
- **Reranking**: 검색 결과 재순위화

### 생성 (Generation)
- **OpenAI API**: GPT 시리즈 연동
- **vLLM**: 자체 LLM 서버 연동
- **本地 모델**: HuggingFace 모델 지원

### 평가 (Evaluation)
- **표준 벤치마크**: TREC 등 주요 데이터셋
- **사용자 정의 평가**: 커스텀 지표 지원

---

## 관련 링크

- [GitHub 저장소](https://github.com/OpenBMB/UltraRAG)
- [공식 문서](https://ultrarag.openbmb.cn/pages/en/getting_started/introduction)
- [홈페이지](https://ultrarag.github.io/)
- [데이터셋](https://modelscope.cn/datasets/UltraRAG/UltraRAG_Benchmark)

---

*다음 글에서는 UltraRAG 설치 및 환경 설정 방법을 살펴보겠습니다.*
