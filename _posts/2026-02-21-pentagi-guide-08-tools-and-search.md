---
layout: post
title: "PentAGI 가이드 (08) - 툴 실행 & 검색: Docker 실행기와 웹 인텔리전스"
date: 2026-02-21
permalink: /pentagi-guide-08-tools-and-search/
author: PentAGI Team
categories: ['AI 에이전트', '보안']
tags: [PentAGI, Tools, Docker, Scraper, Search, Tavily, Searxng]
original_url: "https://github.com/vxcontrol/pentagi"
excerpt: "PentAGI는 LLM만으로 일하지 않습니다. Docker 기반 실행기와 검색/브라우저 툴을 ‘함수 호출’로 묶는 구조를 정리합니다."
---

## 큰 그림: “툴 = 함수 호출”

PentAGI에서 툴은 보통 “LLM function calling”으로 연결됩니다.

- 모델이 툴 호출을 결정
- 서버가 해당 툴 핸들러를 실행
- 결과를 로그/DB에 저장
- UI에 스트리밍/이벤트로 전달

이 설계는 다음을 가능하게 합니다.

- 툴 호출을 **감사 가능(auditable)**하게 남김
- 동일 작업을 **재현 가능(reproducible)**하게 만듦
- 검색/브라우저/터미널 같은 이질적 작업을 동일한 실행 모델로 통합

---

## 코드에서 보는 실행기: `pkg/docker` / `pkg/tools`

핵심은 두 축입니다.

1) **Docker client**: 컨테이너 생성/중지/삭제/exec/파일 복사 등  
2) **Tools executor**: “어떤 툴을 어떤 컨텍스트에서 허용하고” 실행 결과를 어떻게 기록할지

대략의 구조:

```text
backend/pkg/docker/   # 컨테이너 오케스트레이션
backend/pkg/tools/    # 검색/브라우저/코드/메모리 등의 툴 핸들러
```

---

## 검색 엔진 툴: 여러 공급자를 ‘동일 인터페이스’로

`backend/pkg/tools`에는 여러 검색 엔진 핸들러가 존재합니다.

- DuckDuckGo
- Google Custom Search
- Tavily
- Traversaal
- Perplexity
- Searxng

핵심은 “어떤 엔진을 쓰든”:

- 입력: query/옵션
- 출력: 결과(텍스트/요약)

형태로 정규화해, 상위(에이전트) 레이어가 엔진 차이를 덜 보게 만드는 것입니다.

---

## 스크래퍼(격리 브라우저)

Compose에 포함된 `scraper` 서비스는 웹 상호작용을 “격리된 브라우저”로 제공합니다.

- 위험한 콘텐츠/스크립트 실행의 영향을 분리하고
- 내부망/외부망 접근 정책을 분리하며
- 수집 결과를 구조화해 에이전트에 제공하는 식의 확장 여지가 생깁니다.

---

## 로깅: “툴 호출 로그”를 남기는 이유

PentAGI는 메시지/검색/터미널/벡터스토어 같은 로그 컨트롤러를 두고,
각 실행 단위를 DB에 남깁니다.

이런 구조가 있으면:

- 결과 리포트 생성이 쉬워지고
- 실패한 실행을 추적하기 쉬우며
- 관측성 스택과 결합했을 때 분석이 가능해집니다.

---

## 운영 팁: 툴/권한을 ‘최소’로 시작하기

초기 배포에서 가장 안전한 접근은:

- 필요한 검색 엔진만 붙이고
- 필요할 때만 스크래퍼/확장 스택을 켜며
- Docker 권한/네트워크 권한을 최소화하는 것입니다.

PentAGI는 다양한 기능을 제공하지만, 운영 환경의 “신뢰 경계”는 사용자 책임입니다.

---

## 참고 링크

- 인덱스(전체 목차): `/blog-repo/pentagi-guide/`
- 도커 클라이언트: `backend/pkg/docker/`
- 툴 핸들러: `backend/pkg/tools/`

---

다음 글에서는 PentAGI의 “기억”에 해당하는 **pgvector/임베딩/요약/Graphiti**를 연결해 설명합니다.

