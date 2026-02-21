---
layout: post
title: "PentAGI 가이드 (01) - 소개: 무엇을 자동화하는가"
date: 2026-02-21
permalink: /pentagi-guide-01-intro/
author: PentAGI Team
categories: ['AI 에이전트', '보안']
tags: [PentAGI, Security, AI Agent, Docker, pgvector, GraphQL]
original_url: "https://github.com/vxcontrol/pentagi"
excerpt: "PentAGI의 목적/핵심 기능/안전 모델을 정리하고, 이 시리즈가 다룰 범위를 확정합니다."
---

## PentAGI는 무엇인가?

**PentAGI**(Penetration testing Artificial General Intelligence)는 Docker로 격리된 환경에서 보안 테스트 워크플로우를 자동화하는 멀티 에이전트 시스템입니다.

이 가이드 시리즈는 “어떻게 공격할까?”가 아니라:

- **어떻게 안전하게 배포하고**
- **어떻게 올바르게 설정하고**
- **어떻게 운영/관측하고**
- **어떻게 코드베이스를 이해하고 확장할지**

를 중심으로 정리합니다.

---

## 사용 전제(중요)

PentAGI는 보안 테스트 도구를 포함할 수 있습니다. 따라서 이 시리즈에서 다루는 사용 시나리오는 아래를 전제로 합니다.

- **작성된 승인/계약**이 있는 합법적인 보안 점검
- **자체 소유/통제 가능한 실습 환경**(예: 로컬 랩, CTF, 내부 테스트용 시스템)
- 서비스 운영 환경에 적용 시, **격리·권한·네트워크 경계**를 우선 검토

---

## 핵심 기능 한 장 요약

README 기준 PentAGI가 강조하는 포인트는 다음과 같습니다.

1. **격리 실행**: 모든 작업을 Docker 샌드박스에서 수행
2. **자율 에이전트**: 여러 역할(리서치/코딩/실행 등)의 에이전트로 업무를 분업
3. **검색/브라우저 내장**: 외부 검색 엔진 + 격리 브라우저(스크래퍼)로 최신 정보 수집
4. **영속 메모리**: PostgreSQL + pgvector에 실행 로그/결과를 저장해 재사용
5. **관측성/분석**: Langfuse·Grafana·OTEL 같은 스택과 연동(선택)
6. **지식 그래프**: Graphiti + Neo4j로 관계를 구조화(선택)

---

## 저장소 지도(Repo Map)

루트 디렉토리 기준 주요 폴더는 크게 4개로 나뉩니다.

```text
pentagi/
├─ backend/              # Go 서버(REST/GraphQL), 에이전트/툴 실행 로직
├─ frontend/             # React + Vite SPA
├─ observability/        # (옵션) Grafana/OTEL 등 관측성 구성 리소스
├─ docker-compose*.yml   # 기본/확장 스택(관측성, Langfuse, Graphiti)
├─ .env.example          # 환경 변수 샘플(중요!)
└─ Dockerfile            # 배포 이미지 빌드(프론트+백엔드 멀티스테이지)
```

`examples/`는 “설치 후 무엇을 확인할지”에 도움이 되는 샘플들이 모여 있습니다.

```text
examples/
├─ configs/   # Provider 설정 예시(YAML)
├─ guides/    # 워커 노드 등 운영 가이드
├─ tests/     # 에이전트 테스트 리포트 예시
└─ reports/   # 리포트 출력 예시(샘플)
```

---

## PentAGI가 “제품”으로 제공하는 것

이 프로젝트는 단순 라이브러리라기보다 **운영 가능한 self-hosted 스택**에 가깝습니다.

- `docker-compose.yml`로 기본 스택(pentagi + pgvector + scraper 등)을 기동
- 추가 compose로 Langfuse/Graphiti/Observability를 확장
- 프론트/백엔드를 한 이미지로 묶은 `vxcontrol/pentagi` 배포 이미지

즉, “코드만 읽는 것”보다 “스택으로 배포하고 설정하는 것”이 중요합니다.

---

## 시리즈에서 다룰 범위

이 시리즈는 다음 질문에 답하도록 구성합니다.

- 설치/배포는 어떤 경로가 있고, 어떤 환경 변수가 핵심인가?
- LLM Provider/검색 엔진/브라우저는 어떤 방식으로 연결되는가?
- 백엔드(Go)는 어떤 레이어로 구성되고, 어디에서 무엇이 결정되는가?
- 프론트엔드(React)는 어떤 API/구독 모델로 실시간 UI를 구성하는가?
- 메모리/임베딩/요약/지식 그래프는 어떤 조건에서 켜고, 무엇을 저장하는가?
- 운영 관측성은 어떤 포인트를 봐야 하는가?

---

## 참고 링크

- 저장소: `https://github.com/vxcontrol/pentagi`
- 인덱스(전체 목차): `/blog-repo/pentagi-guide/`

---

다음 글에서는 **가장 빠른 설치 경로(인스톨러/수동)**와 첫 실행 체크리스트를 정리합니다.

