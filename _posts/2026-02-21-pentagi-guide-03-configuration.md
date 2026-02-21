---
layout: post
title: "PentAGI 가이드 (03) - 환경설정: .env 핵심 변수 맵"
date: 2026-02-21
permalink: /pentagi-guide-03-configuration/
author: PentAGI Team
categories: ['AI 에이전트', '보안']
tags: [PentAGI, Env, LLM, OpenAI, Anthropic, Gemini, Ollama, Graphiti]
original_url: "https://github.com/vxcontrol/pentagi"
excerpt: "PentAGI의 설정은 대부분 .env로 결정됩니다. 반드시 알아야 할 변수들을 영역별로 묶어 설명합니다."
---

## 왜 `.env`가 가장 중요한가?

PentAGI는 “하나의 서버”가 아니라, **여러 서비스와 외부 API를 연결하는 운영 스택**입니다.  
그래서 코드보다 먼저 확인해야 하는 것이 `.env.example`입니다.

```text
.env
├─ LLM Provider 키/엔드포인트
├─ 검색 엔진 API 키/설정
├─ Docker 실행기 설정(소켓/네트워크/이미지)
├─ 저장소/보안 관련 시크릿
└─ (옵션) Langfuse/OTEL/Graphiti 설정
```

---

## 1) “최소 동작”을 위한 변수

README 기준, 최소 동작을 위해서는 아래 중 **하나 이상**의 LLM provider가 필요합니다.

- OpenAI (`OPEN_AI_KEY`)
- Anthropic (`ANTHROPIC_API_KEY`)
- Gemini (`GEMINI_API_KEY`)
- Bedrock(키/시크릿)
- Ollama(로컬)
- 혹은 커스텀 OpenAI 호환 엔드포인트(`LLM_SERVER_*`)

또한 기본 스택은 PostgreSQL/pgvector를 포함하므로 DB는 내부로 연결됩니다.

---

## 2) LLM Provider 구성 패턴

PentAGI는 “여러 provider를 동시에” 구성할 수 있고, 작업 성격에 따라 모델을 선택합니다(프로젝트 설정/기본값에 따름).

### A. OpenAI/Anthropic/Gemini처럼 “고정 Provider”

키와 엔드포인트가 핵심입니다.

```bash
OPEN_AI_KEY=...
OPEN_AI_SERVER_URL=https://api.openai.com/v1
```

### B. OpenAI 호환 “Custom Provider”

OpenAI 호환 API를 쓰는 경우 아래 형태를 사용합니다.

```bash
LLM_SERVER_URL=https://your-openai-compatible-endpoint/v1
LLM_SERVER_KEY=...
LLM_SERVER_MODEL=...
```

### C. 로컬 LLM: Ollama

Ollama는 **비용/프라이버시** 측면에서 장점이 있지만, 컨텍스트 길이/성능/자원 요구를 꼭 고려해야 합니다.

```bash
OLLAMA_SERVER_URL=http://localhost:11434
OLLAMA_SERVER_MODEL=llama3.1:8b-instruct-q8_0
```

---

## 3) 검색 엔진/웹 인텔리전스 설정

PentAGI는 여러 검색 엔진 API를 옵션으로 지원합니다.  
“정확도/커버리지”는 키를 얼마나 붙이느냐에 따라 달라집니다.

예:

- DuckDuckGo(단순 on/off)
- Google Custom Search
- Tavily / Traversaal / Perplexity
- Searxng(자체 메타 검색)

```bash
DUCKDUCKGO_ENABLED=true
TAVILY_API_KEY=...
TRAVERSAAL_API_KEY=...
PERPLEXITY_API_KEY=...
SEARXNG_URL=http://your-searxng:8080
```

또한 “격리 브라우저”용 스크래퍼 URL도 설정 포인트입니다.

```bash
SCRAPER_PUBLIC_URL=...
SCRAPER_PRIVATE_URL=...
```

---

## 4) Docker 실행기(격리) 관련 설정

PentAGI의 중요한 설계는 “실행을 Docker로 넘긴다”는 점입니다.  
따라서 아래는 운영 품질을 크게 좌우합니다.

- `DOCKER_HOST` / `DOCKER_TLS_VERIFY` / `DOCKER_CERT_PATH`: 원격 도커(워커 노드 포함)
- `DOCKER_DEFAULT_IMAGE`: 일반 작업 기본 이미지
- `DOCKER_DEFAULT_IMAGE_FOR_PENTEST`: 보안 작업 기본 이미지(프로젝트 기본값 존재)
- `DOCKER_NET_ADMIN`: 네트워크 스캔 도구에 필요한 권한(보안 영향 큼)

---

## 5) Graphiti(지식 그래프) 옵션

Graphiti는 **선택 기능**이며 기본적으로 꺼져 있습니다.

```bash
GRAPHITI_ENABLED=true
GRAPHITI_URL=http://graphiti:8000
NEO4J_URI=bolt://neo4j:7687
```

Graphiti를 켜는 순간:

- 추가 서비스(neo4j, graphiti)가 필요하고
- 저장되는 데이터 범위/정책을 고민해야 하며
- 리소스 사용량도 늘어납니다.

따라서 “먼저 기본 스택으로 안정화” 후 켜는 것을 권장합니다.

---

## 6) 운영에서 꼭 바꿔야 하는 보안 변수

`.env.example`에는 개발 편의 기본값이 포함될 수 있습니다.  
프로덕션에선 최소한 아래는 재설정하는 편이 안전합니다.

- 쿠키 서명/세션 관련 시크릿(`COOKIE_SIGNING_SALT` 등)
- DB/Neo4j 계정 정보
- 공개 URL/SSL 인증서 경로(`PUBLIC_URL`, `SERVER_SSL_*`)

---

## 참고 링크

- 인덱스(전체 목차): `/blog-repo/pentagi-guide/`
- `.env.example`: `https://github.com/vxcontrol/pentagi/blob/master/.env.example`

---

다음 글에서는 Compose 스택 관점에서 **컨테이너/네트워크/볼륨**이 어떻게 격리를 구성하는지 정리합니다.

