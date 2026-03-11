---
layout: post
title: "MiroFish 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-10
permalink: /mirofish-guide-01-intro/
author: 666ghj
categories: [AI 에이전트, MiroFish]
tags: [Trending, GitHub, MiroFish, Multi-Agent, Simulation, Flask, Vue, Zep]
original_url: "https://github.com/666ghj/MiroFish"
excerpt: "MiroFish의 문제 정의(입력→그래프→시뮬레이션→리포트)와 저장소 지도를 근거 기반으로 정리합니다."
---

## 이 문서의 목적

- MiroFish의 “무엇을/왜/어떻게”를 **코드/설정/README 근거**로 정리합니다.
- 다음 챕터(설치/아키텍처/사용/운영)를 읽기 위한 **지도(Repo Map + 흐름)**를 제공합니다.

---

## 빠른 요약

- MiroFish는 “문서(시드) + 요구사항”에서 **본체(ontology)**를 만들고(`backend/app/services/ontology_generator.py`), 텍스트를 **Zep Cloud**에 적재해 그래프를 생성한 뒤(`backend/app/services/graph_builder.py`), 그래프 엔티티를 기반으로 **OASIS 시뮬레이션(Twitter/Reddit)**을 준비/실행하고(`backend/app/services/simulation_manager.py`, `backend/app/services/simulation_runner.py`), 마지막에 **Report Agent**가 보고서를 생성합니다(`backend/app/services/report_agent.py`, `backend/app/api/report.py`).
- 프론트는 **Vue 3 + Vite**(`frontend/package.json`), 백엔드는 **Flask**(`backend/run.py`, `backend/app/__init__.py`)입니다.
- 필수 키는 최소 **LLM_API_KEY**, **ZEP_API_KEY**입니다(`backend/app/config.py`, `.env.example`).

---

## MiroFish는 무엇인가? (프로덕트 관점)

README의 표현을 개발자 관점으로 번역하면, MiroFish는 대략 아래 파이프라인을 “제품화”한 것입니다.

1) **문서 업로드**(PDF/MD/TXT) + “무엇을 예측/시뮬레이션할지” 요구사항 입력  
2) LLM으로 **사회 시뮬레이션용 본체(ontology) 설계**  
3) Zep에 텍스트를 넣어 **지식 그래프 생성**  
4) 그래프 엔티티를 읽고 **플랫폼별 에이전트 프로필 생성**  
5) Twitter/Reddit **이중 플랫폼 시뮬레이션** 실행(라운드/액션 로그)  
6) 시뮬레이션 결과와 그래프를 바탕으로 **리포트 생성 + 대화(Report Chat)**  

---

## 저장소 지도(Repo Map)

```text
MiroFish/
├─ frontend/                 # Vue 3 + Vite UI
├─ backend/                  # Flask API + 시뮬레이션/리포트 서비스
├─ static/                   # 로고/스크린샷 등 정적 리소스(README에 사용)
├─ Dockerfile                # 단일 컨테이너(프론트+백엔드) 빌드/실행
├─ docker-compose.yml        # 이미지 실행 + 포트(3000/5001) + uploads 볼륨
├─ .env.example              # 필수/선택 환경 변수 샘플
└─ package.json              # dev/setup 스크립트(루트에서 한번에 실행)
```

---

## 전체 흐름(개발자 시점)

```mermaid
flowchart TD
  U[사용자/운영자] -->|문서 업로드 + 요구사항| FE[Frontend (Vue)]
  FE -->|HTTP| BE[Backend (Flask)]
  BE -->|LLM 호출| LLM[LLM API (OpenAI SDK 호환)]
  BE -->|GraphRAG/Graph| ZEP[Zep Cloud]

  subgraph BackendPipeline[Backend 파이프라인]
    O[OntologyGenerator] --> G[GraphBuilderService]
    G --> S[SimulationManager]
    S --> R[SimulationRunner]
    R --> RA[ReportAgent]
  end

  BE --> O
  BE --> G
  BE --> S
  BE --> R
  BE --> RA
```

---

## 이 시리즈에서 다룰 것

- (02) **설치/실행**: `.env` 구성, `npm run dev`, Docker 배포(`Dockerfile`, `docker-compose.yml`)
- (03) **아키텍처**: 프론트↔백엔드 경계, API 흐름, 상태 저장(프로젝트/태스크/시뮬레이션)
- (04) **실전 사용 패턴**: 문서→본체→그래프→시뮬레이션→리포트까지의 “한 번 돌려보기” 체크리스트
- (05) **운영/확장**: 로그/리소스/키 관리, 실패 지점 디버깅, 커스터마이징 포인트

---

## 근거(파일/경로)

- 제품 개요/워크플로우: `README.md`의 “工作流程/快速开始” 섹션
- 실행 스크립트: `package.json` (`setup:all`, `dev`, `backend`, `frontend`)
- 백엔드 엔트리: `backend/run.py`, `backend/app/__init__.py`
- 설정/필수 키: `backend/app/config.py`, `.env.example`
- 그래프/본체: `backend/app/services/ontology_generator.py`, `backend/app/services/graph_builder.py`
- 시뮬레이션: `backend/app/services/simulation_manager.py`, `backend/app/services/simulation_runner.py`
- 리포트: `backend/app/api/report.py`, `backend/app/services/report_agent.py`

---

## 주의사항/함정

- **키 미설정 시 백엔드가 바로 종료**됩니다: `Config.validate()`가 `LLM_API_KEY`, `ZEP_API_KEY`를 검사합니다(`backend/run.py`, `backend/app/config.py`).
- Docker는 기본적으로 `backend/uploads`를 볼륨으로 매핑합니다(`docker-compose.yml`). 로컬 실행 시에도 업로드/시뮬레이션 파일이 여기에 쌓입니다.
- 프론트/백엔드를 한 컨테이너에서 같이 띄우는 구조라(`Dockerfile`의 `CMD ["npm","run","dev"]`), “prod 빌드/서빙” 방식은 별도 확인이 필요합니다.

---

## TODO/확인 필요

- README는 “在线 Demo” 링크를 제공합니다. 데모/프로덕션 환경의 배포 방식과 보안 경계(인증/권한)는 코드에서 추가 확인이 필요합니다.
- 시뮬레이션에서 사용하는 OASIS 스크립트/데이터 포맷의 버전 호환성(외부 의존)은 `backend/scripts/*` 및 `backend/pyproject.toml` 기준으로 점검 필요.

---

## 위키 링크

- [[MiroFish Guide - Installation]] / [설치 및 빠른 시작](/blog-repo/mirofish-guide-02-installation/)
- [[MiroFish Guide - Architecture]] / [핵심 개념과 아키텍처](/blog-repo/mirofish-guide-03-architecture/)
- [[MiroFish Guide - Usage]] / [실전 사용 패턴](/blog-repo/mirofish-guide-04-usage/)
- [[MiroFish Guide - Best Practices]] / [운영/확장/베스트 프랙티스](/blog-repo/mirofish-guide-05-best-practices/)

다음 글에서는 **로컬 실행(uv + Node)과 Docker 실행**을 단계별로 정리합니다.
