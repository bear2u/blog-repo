---
layout: page
title: MiroFish 가이드
permalink: /mirofish-guide/
icon: fas fa-robot
---

# MiroFish 완벽 가이드

> **A Simple and Universal Swarm Intelligence Engine, Predicting Anything. 简洁通用的群体智能引擎，预测万物**

**MiroFish**를 빠르게 훑고, 설치부터 활용/확장까지 핵심을 정리한 시리즈입니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/blog-repo/mirofish-guide-01-intro/) | 소개 및 개요 |
| 02 | [설치 및 빠른 시작](/blog-repo/mirofish-guide-02-installation/) | 설치 및 빠른 시작 |
| 03 | [핵심 개념과 아키텍처](/blog-repo/mirofish-guide-03-architecture/) | 핵심 개념과 아키텍처 |
| 04 | [실전 사용 패턴](/blog-repo/mirofish-guide-04-usage/) | 실전 사용 패턴 |
| 05 | [운영/확장/베스트 프랙티스](/blog-repo/mirofish-guide-05-best-practices/) | 운영/확장/베스트 프랙티스 |
| 06 | [문서 점검 자동화](/blog-repo/mirofish-guide-06-doc-automation/) | routes/env/경로 계약 점검 |

---

## 이 시리즈에서 얻는 것

- **입력→그래프→시뮬레이션→리포트** 파이프라인을 “코드/설정 근거”로 이해
- 로컬/도커 실행과, 실패 시 **어디(태스크/로그/상태 파일)를 먼저 볼지**의 기준
- 가이드 자체가 깨지지 않도록 하는 **문서 점검 자동화 체크리스트**

---

## 빠른 시작

```bash
git clone https://github.com/666ghj/MiroFish.git
cd MiroFish

cp .env.example .env
# .env에 LLM_API_KEY / ZEP_API_KEY 설정

npm run setup:all
npm run dev
```

Docker로 실행:

```bash
cp .env.example .env
docker compose up -d
```

---

## 기술 스택(요약)

| 구성 | 기술 |
|---|---|
| Frontend | Vue 3 + Vite |
| Backend | Flask + Python(uv) |
| 외부 연동 | LLM API(OpenAI SDK 호환), Zep Cloud |

## 관련 링크

- GitHub 저장소: https://github.com/666ghj/MiroFish
