---
layout: post
title: "hindsight 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-12
permalink: /hindsight-guide-01-intro/
author: vectorize-io
categories: [AI 에이전트, hindsight]
tags: [Trending, GitHub, hindsight, Memory, Agents, GitHub Trending]
original_url: "https://github.com/vectorize-io/hindsight"
excerpt: "대화 기록이 아니라 '학습'에 초점을 둔 에이전트 메모리 시스템 Hindsight를 개요부터 정리합니다."
---

## Hindsight란?

GitHub Trending(daily, 2026-03-12 기준) 상위에 오른 **vectorize-io/hindsight**를 한국어로 정리합니다.

- **한 줄 요약(Trending 표시)**: Hindsight: Agent Memory That Learns
- **언어(Trending 표시)**: Python
- **오늘 스타(Trending 표시)**: +95
- **원본**: https://github.com/vectorize-io/hindsight

---

## 이 문서의 목적

- README가 정의하는 “Hindsight가 해결하려는 메모리 문제”를 개요 수준에서 정리합니다.
- 레포가 멀티 컴포넌트(서버/CLI/클라이언트 등) 형태인지 구조를 잡고, 다음 챕터의 분해 기준을 만듭니다.

---

## 빠른 요약 (README 기반)

- Hindsight는 “대화 히스토리를 회상”하는 메모리보다, **시간이 지날수록 학습하는 에이전트**에 초점을 둔다고 설명합니다. (`README.md`)
- 문서/논문/쿡북/클라우드 링크가 README 상단에 제공됩니다. (`README.md`)
- 레포는 API/CLI/클라이언트 등 다수 디렉토리로 구성된 모노레포 형태입니다. (예: `hindsight-api/`, `hindsight-cli/`, `hindsight-clients/`)

---

## 바로 시작하기(README 링크)

- Documentation: https://hindsight.vectorize.io
- Paper: https://arxiv.org/abs/2512.12818
- Cookbook: https://hindsight.vectorize.io/cookbook

---

## 근거(파일/경로)

- 개요/링크/방향성: `README.md`
- 파이썬 패키징 단서: `pyproject.toml`, `uv.lock`
- Node 패키징 단서(클라이언트 등): `package.json`, `package-lock.json`
- 주요 컴포넌트 디렉토리: `hindsight/`, `hindsight-api/`, `hindsight-cli/`, `hindsight-clients/`
- 예제/운영 단서: `cookbook/`, `docker/`, `helm/`, `monitoring/`

---

## 레포 구조(상위)

```text
hindsight/
  hindsight/
  hindsight-api/
  hindsight-cli/
  hindsight-clients/
  cookbook/
  docker/
  helm/
  pyproject.toml
  package.json
```

---

## 위키 링크

- `[[Hindsight Guide - Index]]` → [가이드 목차](/blog-repo/hindsight-guide/)

---

*다음 글에서는 “컴포넌트별(서버/CLI/클라이언트)로 어떻게 시작할지”를 정리합니다.*

