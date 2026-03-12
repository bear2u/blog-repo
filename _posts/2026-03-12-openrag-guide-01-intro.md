---
layout: post
title: "openrag 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-12
permalink: /openrag-guide-01-intro/
author: langflow-ai
categories: [AI 에이전트, openrag]
tags: [Trending, GitHub, openrag, RAG, Langflow, OpenSearch, GitHub Trending]
original_url: "https://github.com/langflow-ai/openrag"
excerpt: "문서 업로드→검색/대화까지 한 번에 제공하는 OpenRAG 플랫폼을 개요부터 정리합니다."
---

## OpenRAG란?

GitHub Trending(daily, 2026-03-12 기준) 상위에 오른 **langflow-ai/openrag**를 한국어로 정리합니다.

- **한 줄 요약(Trending 표시)**: OpenRAG is a comprehensive, single package Retrieval-Augmented Generation platform built on Langflow, Docling, and Opensearch.
- **언어(Trending 표시)**: Python
- **오늘 스타(Trending 표시)**: +191
- **원본**: https://github.com/langflow-ai/openrag

---

## 이 문서의 목적

- OpenRAG가 제공하는 “RAG 플랫폼” 범위를 README 기준으로 정리합니다.
- 빠른 시작(Quickstart) 경로와 레포 구조(백엔드/프론트/도커)를 다음 챕터로 이어질 수 있게 연결합니다.

---

## 빠른 요약 (README 기반)

- OpenRAG는 문서를 업로드/처리한 뒤, 채팅 인터페이스로 질의/대화를 하는 RAG 플랫폼을 지향합니다. (`README.md`)
- 구성요소로 Langflow(OpenRAG 워크플로우), OpenSearch(검색), Docling(문서 처리)을 언급합니다. (`README.md`)
- “Built with”로 FastAPI(백엔드)와 Next.js(프론트)를 명시합니다. (`README.md`)

---

## 바로 시작하기(공식 문서)

- Documentation: https://docs.openr.ag/
- Quickstart: https://docs.openr.ag/quickstart
- Install options: https://docs.openr.ag/install-options
- Docker/Podman: https://docs.openr.ag/docker

---

## 근거(파일/경로)

- 개요/특징/가이드 링크: `README.md`
- 백엔드/파이썬 패키징 단서: `src/`, `pyproject.toml`, `uv.lock`
- 프론트엔드: `frontend/`
- 워크플로우/예시: `flows/`
- 배포/로컬 실행 단서: `docker-compose.yml`, `Dockerfile*`
- 테스트: `tests/`

---

## 레포 구조(상위)

```text
openrag/
  src/
  frontend/
  flows/
  docs/
  tests/
  docker-compose.yml
  pyproject.toml
```

---

## 위키 링크

- `[[OpenRAG Guide - Index]]` → [가이드 목차](/blog-repo/openrag-guide/)

---

*다음 글에서는 문서(quickstart/도커) 기준으로 “가장 짧은 실행 루트”를 정리합니다.*

