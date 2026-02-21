---
layout: post
title: "PentAGI 가이드 (02) - Quick Start: 설치와 첫 접속"
date: 2026-02-21
permalink: /pentagi-guide-02-quick-start/
author: PentAGI Team
categories: ['AI 에이전트', '보안']
tags: [PentAGI, Docker, Docker Compose, Installation, SSL]
original_url: "https://github.com/vxcontrol/pentagi"
excerpt: "인스톨러(권장)와 수동 설치(Compose)를 비교하고, 첫 실행/기본 접속까지의 흐름을 정리합니다."
---

## 이 글에서 다룰 것

- 설치 경로 2가지: **인스톨러** vs **수동(Compose)**
- 최소 요구사항(리소스/네트워크)
- 첫 실행 후 “무엇이 정상인지” 확인하는 체크리스트

---

## 최소 요구사항(README 기준)

- Docker + Docker Compose
- 최소 2 vCPU / 4GB RAM / 20GB 디스크
- 이미지 다운로드를 위한 인터넷 연결

운영/보안 관점에서 중요한 점은 **Docker 권한**입니다.

- 로컬 개발 편의로 `docker` 그룹을 사용하는 경우가 많지만,
- `docker` 그룹은 사실상 **root 급 권한**을 의미하므로
- 프로덕션에서는 더 보수적으로 권한/격리를 설계해야 합니다.

---

## 설치 경로 A: 인스톨러(권장)

README는 “대화형 인스톨러”를 권장합니다. 인스톨러는 대략 아래를 처리합니다.

- 시스템 체크(Docker 접근, 네트워크)
- `.env` 생성/기본값 셋업
- LLM provider/검색 엔진 설정
- 보안 관련 시크릿 생성
- Compose 기동

리눅스 예시(개념 흐름만):

```bash
mkdir -p pentagi && cd pentagi
wget -O installer.zip https://pentagi.com/downloads/linux/amd64/installer-latest.zip
unzip installer.zip
./installer
```

프로덕션 성격의 환경이라면 인스톨러를 **root로 실행**하는 방식이 보안상 더 단순할 수 있습니다(“docker 그룹 부여”를 피하기 위해).

---

## 설치 경로 B: 수동 설치(Compose)

“코드를 복제하거나, `.env`와 compose 파일을 받아서” 구성합니다.

```bash
mkdir -p pentagi && cd pentagi

# 1) 환경 변수
curl -o .env https://raw.githubusercontent.com/vxcontrol/pentagi/master/.env.example

# 2) 기본 스택
curl -O https://raw.githubusercontent.com/vxcontrol/pentagi/master/docker-compose.yml

# 3) 실행
docker compose up -d
```

수동 설치는 장점이 분명합니다.

- IaC/운영 자동화에 맞춰 **변경 이력 관리**가 쉽고
- 환경별로 `.env`를 명시적으로 관리할 수 있으며
- 확장 스택(langfuse/graphiti/observability)을 단계적으로 켤 수 있습니다.

---

## 첫 접속 체크리스트

기본값 기준으로 PentAGI는 HTTPS 포트로 노출됩니다(예: 8443).

1) 컨테이너 상태 확인

```bash
docker compose ps
docker compose logs -f --tail=200 pentagi
```

2) DB/벡터 스토어(예: pgvector) 기동 확인

```bash
docker compose logs -f --tail=200 pgvector
```

3) 스크래퍼(격리 브라우저) 기동 확인

```bash
docker compose logs -f --tail=200 scraper
```

---

## “확장 스택”을 한 번에 켜지 않는 이유

PentAGI는 기본 스택만으로도 동작하지만, 운영/분석 기능을 위해 확장 스택을 붙일 수 있습니다.

- Langfuse: LLM 관측/분석
- Graphiti + Neo4j: 지식 그래프
- Observability: Grafana/OTEL 등

처음부터 전부 켜면 장애 지점이 늘어납니다.  
따라서 **기본 스택 → 확장 스택** 순으로 점진 적용하는 편이 디버깅에 유리합니다.

---

## 참고 링크

- 인덱스(전체 목차): `/blog-repo/pentagi-guide/`
- 저장소: `https://github.com/vxcontrol/pentagi`

---

다음 글에서는 `.env`를 중심으로 **LLM Provider/검색 엔진/SSL**을 어떻게 구성하는지 정리합니다.

