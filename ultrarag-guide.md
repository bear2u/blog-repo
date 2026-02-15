---
layout: page
title: UltraRAG 가이드
permalink: /ultrarag-guide/
icon: fas fa-database
---

# UltraRAG 완벽 가이드

> *清华大学, NEUIR, OpenBMB, AI9stars가 함께 만든 MCP 기반 경량 RAG 개발 프레임워크*

**UltraRAG**는 Model Context Protocol (MCP) 아키텍처를 기반으로 설계된 최초의 경량 RAG(Retrieval-Augmented Generation) 개발 프레임워크입니다. "Less Code, Lower Barrier, Faster Deployment"라는 비전 아래, 연구자와 개발자가 복잡한 RAG 시스템을 쉽게 구축할 수 있도록 합니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/ultrarag-guide-01-intro/) | UltraRAG란? 주요 특징, 버전 히스토리 |
| 02 | [설치](/blog-repo/ultrarag-guide-02-installation/) | uv 설치, Docker 배포, 요구사항 |
| 03 | [아키텍처](/blog-repo/ultrarag-guide-03-architecture/) | MCP Servers, Client, Pipeline 구조 |
| 04 | [YAML 파이프라인](/blog-repo/ultrarag-guide-04-yaml-pipeline/) | 설정 파일로 복잡한 RAG 구축 |
| 05 | [검색 모듈](/blog-repo/ultrarag-guide-05-retriever/) | Dense, Sparse, Hybrid Search |
| 06 | [생성 모듈](/blog-repo/ultrarag-guide-06-generation/) | OpenAI, vLLM, HF 모델 연동 |
| 07 | [평가 시스템](/blog-repo/ultrarag-guide-07-evaluation/) | 벤치마크, 지표, 분석 도구 |
| 08 | [UI 활용](/blog-repo/ultrarag-guide-08-ui/) | Pipeline Builder, 지식 베이스 관리 |
| 09 | [Deep Research](/blog-repo/ultrarag-guide-09-deepresearch/) | AgentCPM-Report, 자동 조사 리포트 |
| 10 | [확장](/blog-repo/ultrarag-guide-10-extension/) | 커스텀 Server, Plugin, 코드 통합 |

---

## 주요 특징

- **Low-Code 오케스트레이션**: YAML 설정만으로 복잡한 RAG 파이프라인 구축
- **MCP 기반 모듈식 아키텍처**: 검색, 생성, 재순위화等功能이 독립적인 Server로 분리
- **통합 평가 시스템**: 표준화된 벤치마크와 지표로 실험 재현성 확보
- **대화형 UI**: 코딩 없이 시각적으로 Pipeline 구성
- **Deep Research**: 자동 조사 리포트 생성 기능

---

## 빠른 시작

```bash
# 설치 (uv 권장)
git clone https://github.com/OpenBMB/UltraRAG.git --depth 1
cd UltraRAG
uv sync --all-extras

# 실행
ultrarag run examples/sayhello.yaml
```

### Docker로 실행

```bash
docker run -it --gpus all -p 5050:5050 hdxin2002/ultrarag:v0.3.0
```

브라우저에서 `http://localhost:5050`에 접속하여 UI를 사용할 수 있습니다.

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    UltraRAG Architecture                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Retriever │  │  Generation │  │   Reranker │        │
│  │   Server    │  │   Server    │  │   Server   │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           │                                 │
│                   ┌───────▼───────┐                         │
│                   │ MCP Client    │                         │
│                   │ (YAML Config) │                         │
│                   └───────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 기술 | 용도 |
|------|------|
| Python 3.11-3.12 | 주요 언어 |
| FastMCP | MCP Server 프레임워크 |
| Milvus | 벡터 데이터베이스 |
| BM25s | 희소 검색 |
| vLLM | LLM 서빙 |
| Flask | 웹 UI |

---

## 버전 히스토리

| 버전 | 출시일 | 주요 내용 |
|------|--------|----------|
| **3.0** | 2026.01.23 | 추론 로직 시각화, 개선된 디버깅 |
| **2.1** | 2025.11.11 | 지식 수집 및 멀티모달 지원 강화 |
| **2.0** | 2025.08.28 | Low-code RAG 구축 |
| **1.0** | 2025.01.23 | 첫 출시 |

---

## 관련 링크

- [GitHub 저장소](https://github.com/OpenBMB/UltraRAG)
- [공식 문서](https://ultrarag.openbmb.cn/pages/en/getting_started/introduction)
- [홈페이지](https://ultrarag.github.io/)
- [데이터셋](https://modelscope.cn/datasets/UltraRAG/UltraRAG_Benchmark)
- [Discord](https://discord.gg/yRFFjjJnnS)
