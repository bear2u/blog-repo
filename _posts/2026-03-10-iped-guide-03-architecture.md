---
layout: post
title: "IPED 완벽 가이드 (03) - 핵심 개념과 아키텍처"
date: 2026-03-10
permalink: /iped-guide-03-architecture/
author: sepinf-inc
categories: [개발 도구, iped]
tags: [Trending, GitHub, IPED]
original_url: "https://github.com/sepinf-inc/IPED"
excerpt: "처리(Processing)와 분석(Analysis) 축으로 구조를 잡습니다."
---

## 처리 vs 분석(README 기준)

- **Processing**: 디스크/이미지/파일 시스템을 디코딩하고, 확장/파싱/인덱싱을 수행해 “케이스”를 만든다
- **Analysis**: 생성된 케이스를 UI에서 탐색/검색/타임라인/갤러리 등으로 분석한다

---

## 개념도

```mermaid
flowchart LR
  A[증거 데이터\n(이미지/디스크/파일)] --> B[Processing]
  B --> C[인덱스/케이스]
  C --> D[Analysis UI]
  D --> E[검색/필터/리포트]
```

---

## 다음에 볼 것

- IPED Wiki: Beginner's Start Guide / Linux 섹션
- README의 Features 목록에서 “자주 쓰는 기능”을 먼저 골라 실습 흐름을 만들기

---

*다음 글에서는 실전 사용 패턴을 정리합니다.*

