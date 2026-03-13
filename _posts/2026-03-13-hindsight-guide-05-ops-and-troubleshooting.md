---
layout: post
title: "Hindsight 완벽 가이드 (05) - 운영/배포/트러블슈팅"
date: 2026-03-13
permalink: /hindsight-guide-05-ops-and-troubleshooting/
author: vectorize-io
categories: [AI 에이전트, hindsight]
tags: [Trending, GitHub, hindsight, Helm, Docker, Troubleshooting, GitHub Trending]
original_url: "https://github.com/vectorize-io/hindsight"
excerpt: "README.md(Docker run/provider), docker/docker-compose, helm/ 디렉토리를 근거로 운영 체크리스트를 정리합니다."
---

## 이 문서의 목적

- Hindsight를 팀/서비스 환경에서 운영할 때 필요한 “설정/배포/장애 대응” 포인트를 정리합니다.
- Docker 단일 실행과 외부 PostgreSQL(Compose), Kubernetes(Helm) 경로를 구분합니다.

---

## 빠른 요약

- 단일 Docker run은 로컬 볼륨(`$HOME/.hindsight-docker`)을 `/home/hindsight/.pg0`에 마운트합니다. (`README.md`)
- 외부 PostgreSQL은 `docker/docker-compose`에서 `docker compose up` 흐름입니다. (`README.md`, `docker/docker-compose` 디렉토리 존재)
- Kubernetes 배포는 `helm/` 디렉토리가 존재합니다.
- LLM Provider는 `HINDSIGHT_API_LLM_PROVIDER`로 변경할 수 있습니다. (`README.md`)

---

## 1) 설정 포인트(환경 변수)

README 기준 핵심:

- `HINDSIGHT_API_LLM_API_KEY` (예: OpenAI 키)
- `HINDSIGHT_API_LLM_PROVIDER` (openai/anthropic/gemini/groq/ollama/lmstudio)

근거:
- `README.md` Quick Start

---

## 2) 저장소(데이터) 전략

### 로컬 단일 실행(pg0 볼륨)

README의 docker run 예시는:

- `$HOME/.hindsight-docker`를 `/home/hindsight/.pg0`에 마운트

즉, 컨테이너 재기동 시에도 데이터가 유지되는 구성을 가정합니다.

### 외부 PostgreSQL

README는 `docker/docker-compose`에서 외부 PostgreSQL을 사용하는 예시를 제공합니다.

운영 팁:

- 개발/테스트 환경에서는 docker-compose로 분리된 DB가 디버깅이 쉽습니다.
- 프로덕션은 백업/복구/모니터링 가능한 PostgreSQL(관리형 등)로 옮기는 것이 일반적입니다(레포는 helm도 제공).

---

## 3) 배포 경로 선택 가이드

```mermaid
flowchart TD
  A[Start] --> B{Single host?}
  B -- Yes --> C[Docker run]
  B -- No --> D{Kubernetes?}
  D -- Yes --> E[helm/]
  D -- No --> F[docker/docker-compose (external PG)]
```

---

## 4) 트러블슈팅 체크리스트(README 기반 + 운영 관점)

- API/UI 포트 확인: `8888/9999` 점유/방화벽
- LLM 키/Provider 확인: `HINDSIGHT_API_LLM_API_KEY`, `HINDSIGHT_API_LLM_PROVIDER`
- 외부 DB 구성 시: `HINDSIGHT_DB_PASSWORD` 등 compose 환경 변수 누락 여부(README 예시)

근거:
- `README.md`

---

## TODO / 확인 필요

- 모니터링/로깅/레이트 리밋 등 운영 기능은 `monitoring/` 디렉토리 및 helm values를 읽고 “실제 파라미터(리소스/오토스케일)”로 확정하는 것이 좋습니다.

---

## 위키 링크

- `[[Hindsight Guide - Index]]` → [가이드 목차](/blog-repo/hindsight-guide/)
- `[[Hindsight Guide - Components]]` → [02. 구성요소 맵](/blog-repo/hindsight-guide-02-components/)
- `[[Hindsight Guide - Quickstart]]` → [03. 빠른 시작(로컬)](/blog-repo/hindsight-guide-03-quickstart/)
- `[[Hindsight Guide - Memory Design]]` → [04. 메모리 설계/데이터 흐름](/blog-repo/hindsight-guide-04-memory-design/)

