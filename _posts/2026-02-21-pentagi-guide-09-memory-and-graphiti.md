---
layout: post
title: "PentAGI 가이드 (09) - 메모리 & Graphiti: 벡터 스토어와 지식 그래프"
date: 2026-02-21
permalink: /pentagi-guide-09-memory-and-graphiti/
author: PentAGI Team
categories: ['AI 에이전트', '보안']
tags: [PentAGI, pgvector, Embeddings, Memory, Graphiti, Neo4j]
original_url: "https://github.com/vxcontrol/pentagi"
excerpt: "PentAGI의 장기 기억은 pgvector에서 시작하고, 선택적으로 Graphiti(Neo4j)로 관계를 확장합니다. 무엇이 저장되는지 관점으로 정리합니다."
---

## “기억”이 중요한 이유

에이전트가 실행을 반복할수록 시스템이 좋아지려면:

- 이전 실행에서 “무엇이 효과가 있었는지”
- 어떤 결과가 어떤 판단으로 이어졌는지
- 같은 유형의 작업에서 재사용할 수 있는 패턴이 무엇인지

가 남아야 합니다.

PentAGI는 이를 위해 최소한:

- 실행 로그를 DB에 남기고
- 임베딩 기반 검색(pgvector)을 제공하며
- (옵션) 관계를 그래프로 확장(Graphiti + Neo4j)

하는 구조를 갖습니다.

---

## 1) pgvector: 텍스트를 검색 가능한 기억으로

기본 스택의 `pgvector`는 단순 저장소가 아니라:

- 실행 로그/메시지/검색 결과를 축적하고
- 임베딩을 저장해 “유사도 검색”을 가능하게 합니다.

이 방식의 장점은 “자연어 기반 회상”입니다.

- “비슷한 케이스에서 어떤 툴이 유효했는가?”
- “이전 실행에서 관측한 핵심 사실이 무엇인가?”

같은 질문에 대해, 키워드가 완벽히 일치하지 않아도 근접 검색이 가능합니다.

---

## 2) 임베딩 공급자(Embedding Provider)

임베딩도 결국 외부 모델 호출이기 때문에, `.env`에서 별도로 설정합니다.

```text
EMBEDDING_URL
EMBEDDING_KEY
EMBEDDING_MODEL
EMBEDDING_PROVIDER
```

운영에서는 다음을 함께 고려해야 합니다.

- 배치 크기(비용/지연)
- 줄바꿈 처리(문서 형태에 따라 검색 품질 영향)
- 데이터 보관 정책(민감 정보)

---

## 3) 요약기(Summarizer): 컨텍스트 폭발을 관리하는 장치

메시지/툴 출력이 누적되면, 결국 모델 호출 비용이 커집니다.

PentAGI는 요약 설정을 환경 변수로 노출해:

- 얼마나 많이 보존할지
- 어떤 단위로 줄일지

를 튜닝할 수 있게 합니다.

---

## 4) Graphiti + Neo4j(옵션): “관계”를 저장하는 방식

벡터 스토어는 “유사한 텍스트”에 강합니다.  
하지만 “관계”를 명시적으로 추적하려면 그래프가 유리할 수 있습니다.

Graphiti는 에이전트 상호작용에서 엔티티/관계를 추출해 저장하는 지식 그래프 계층으로 소개됩니다.

### 켜는 방법(개념)

1) `.env`에서 Graphiti 활성화
2) `docker-compose-graphiti.yml`을 기본 compose와 함께 실행

Graphiti는 선택 기능이므로, 운영 정책(저장 범위/익명화/보관 기간)을 먼저 정한 뒤 켜는 것을 권장합니다.

---

## 무엇이 저장되는가(운영 관점)

저장 범위는 구성에 따라 달라지지만, 큰 범주는 다음입니다.

- 에이전트 메시지(질문/응답/결론)
- 툴 실행 로그(요청/결과)
- 검색 결과/스크래핑 결과
- 작업 구조(Flow/Task/Subtask)

운영에서는 “저장되는 것”보다 “저장하지 말아야 할 것”을 먼저 정의하는 게 더 중요합니다.

---

## 참고 링크

- 인덱스(전체 목차): `/blog-repo/pentagi-guide/`
- Graphiti compose: `https://github.com/vxcontrol/pentagi/blob/master/docker-compose-graphiti.yml`

---

다음 글에서는 Langfuse/OTEL/Grafana 같은 **관측성 스택**, 그리고 `ctester/ftester` 같은 **테스트·빌드 파이프라인**을 정리합니다.

