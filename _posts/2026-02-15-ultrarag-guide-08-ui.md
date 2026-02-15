---
layout: post
title: "UltraRAG 완벽 가이드 (08) - UltraRAG UI 활용"
date: 2026-02-15
permalink: /ultrarag-guide-08-ui/
author: UltraRAG Team
categories: [AI 에이전트, RAG]
tags: [RAG, MCP, LLM, UltraRAG, UI, Web, Pipeline Builder]
original_url: "https://github.com/OpenBMB/UltraRAG"
excerpt: "UltraRAG의 시각적 RAG IDE: Pipeline Builder, Knowledge Base Management, 대화형 데모 시스템을 소개합니다."
---

## UltraRAG UI 개요

UltraRAG UI는 기존 채팅 인터페이스의 한계를 뛰어넘는 **시각적 RAG 통합 개발 환경(IDE)**입니다:

- **Pipeline Builder**: 코드 편집과 캔버스 구성의 양방향 실시간 동기화
- **지식 베이스 관리**: 커스텀 지식 베이스 구축
- **대화형 데모**: 파이프라인을 대화 시스템으로 원클릭 변환
- **AI 어시스턴트**: 파라미터 튜닝, 프롬프트 생성 지원

---

## 시작하기

### 로컬 실행

```shell
# 소스 코드 설치 후
cd UltraRAG
uv sync --extra ui

# UI 서버 시작
ultrarag ui
```

브라우저에서 `http://localhost:5050`에 접속합니다.

### Docker 실행

```shell
docker run -it --gpus all -p 5050:5050 hdxin2002/ultrarag:v0.3.0
```

---

## Pipeline Builder

Pipeline Builder는 UltraRAG의 핵심 기능으로, 코딩 없이 시각적으로 파이프라인을 구성할 수 있습니다:

### 주요 기능

| 기능 | 설명 |
|------|------|
| **Drag & Drop** | 컴포넌트를 캔버스로 드래그 |
| **양방향 동기화** | 캔버스 ↔ 코드 실시간 동기화 |
| **실시간 파라미터 조정** | 온라인에서 파라미터 수정 |
| **프롬프트 편집** | 프롬프트 템플릿 시각적 편집 |

### 사용 방법

1. **새 파이프라인 생성**: "New Pipeline" 클릭
2. **컴포넌트 추가**: 좌측 패널에서 Retriever, Generation 등 드래그
3. **연결**: 컴포넌트間の 연결선으로 데이터 흐름 설정
4. **파라미터 설정**: 각 컴포넌트 클릭하여 파라미터 편집
5. **테스트**: "Run" 버튼으로 파이프라인 테스트

---

## 지식 베이스 관리

### 지식 베이스 생성

1. **새 KB 생성**: "Knowledge Base" → "Create New"
2. **문서 업로드**: PDF, TXT, Markdown 등 지원
3. **인덱싱**: 자동으로 벡터 인덱스 생성
4. **검색 테스트**: 샘플 쿼리로 검색 결과 확인

### 지원 형식

| 형식 | 지원 여부 |
|------|----------|
| PDF | ✅ |
| TXT | ✅ |
| Markdown | ✅ |
| DOCX | ✅ |
| HTML | ✅ |
| 이미지 (OCR) | ✅ |

---

## 대화형 데모

### 데모 생성

구성한 파이프라인을 대화형 웹 UI로 변환:

```yaml
# examples/rag_deploy.yaml
pipeline:
  - retriever.retriever_init
  - generation.generation_init
  - retriever.retriever_search
  - prompt.qa_rag_boxed
  - generation.generate

# UI로 배포
ultrarag deploy examples/rag_deploy.yaml --port 8080
```

### 사용자 정의 UI

```yaml
# examples/rag_wow.yaml
pipeline:
  - retriever.retriever_init
  - generation.generation_init
  - retriever.retriever_search
  - generation.generate

ui:
  theme: "dark"
  title: "RAG 데모"
  show_sources: true  # 참조 문서 표시
  streaming: true    # 실시간 스트리밍
```

---

## Multi-Agent 구성

### 다중 에이전트 설정

```yaml
#examples/agent_crew.yaml
agents:
  - name: "Researcher"
    role: "정보 검색 전문가"
    tools:
      - retriever.search
      - web_search

  - name: "Analyzer"
    role: "정보 분석 전문가"
    tools:
      - generation.analyze

  - name: "Writer"
    role: "글쓰기 전문가"
    tools:
      - generation.write

workflow:
  - Researcher: "최신 AI 연구 동향"
  - Analyzer: "연구 결과 분석"
  - Writer: "리포트 작성"
```

---

## 관리자 모드

고급 구성 옵션:

```shell
# 관리자 모드로 시작
ultrarag ui --admin
```

| 옵션 | 설명 |
|------|------|
| API 키 관리 | LLM API 키 설정 |
| 모델 선택 | 기본 모델 변경 |
| 파이프라인 관리 | 파이프라인 저장/불러오기 |
| 로그 분석 | 디버그 로그 확인 |

---

## 배포 가이드

### 프로덕션 배포

```yaml
# Docker Compose
services:
  ultrarag:
    image: hdxin2002/ultrarag:v0.3.0
    ports:
      - "5050:5050"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MILVUS_HOST=milvus
    volumes:
      - ./data:/app/data

  milvus:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
```

### Retriever/LLM 설정

UI 관리자 모드에서 다음을 설정:

1. **Retriever**: Milvus 서버 주소, 임베딩 모델
2. **Generation**: LLM (OpenAI/vLLM) API 엔드포인트
3. **Vector DB**: Milvus 연결 정보

---

## 팁 및 베스트 프랙티스

1. **파라미터 조정**: UI에서 실시간 조정하며 최적값 찾기
2. **프롬프트 테스트**: 다양한 프롬프트로 결과 비교
3. **캐싱 활용**: 반복 쿼리에는 캐싱하여 응답 시간 단축
4. **로그 분석**: 디버그 모드로 문제 원인 파악

---

*다음 글에서는 Deep Research 파이프라인에 대해 살펴보겠습니다.*
