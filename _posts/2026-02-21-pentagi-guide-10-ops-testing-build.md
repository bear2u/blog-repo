---
layout: post
title: "PentAGI 가이드 (10) - 관측성/테스트/빌드: 운영 품질 체크리스트"
date: 2026-02-21
permalink: /pentagi-guide-10-ops-testing-build/
author: PentAGI Team
categories: ['AI 에이전트', '보안']
tags: [PentAGI, Langfuse, OpenTelemetry, Grafana, ctester, ftester, Dockerfile]
original_url: "https://github.com/vxcontrol/pentagi"
excerpt: "운영에서 중요한 것은 ‘돌아간다’가 아니라 ‘관측/검증/재현’입니다. PentAGI의 관측성 스택과 테스트 유틸, Dockerfile 빌드 구조를 정리합니다."
---

## 관측성(Observability)은 선택이 아니라 보험이다

에이전트 시스템은 “결과가 맞았다/틀렸다”만으로 디버깅하기 어렵습니다.

- 어떤 프롬프트/모델 설정이었는지
- 어떤 툴 호출이 있었는지
- 어디서 지연/에러가 났는지

가 남아야 운영이 가능합니다.

PentAGI는 이를 위해 **선택 가능한 스택**을 제공합니다.

- Langfuse: LLM 관측/분석
- OpenTelemetry + Grafana 스택: 시스템/트레이스/로그

---

## 1) Langfuse 연동(옵션)

저장소는 `docker-compose-langfuse.yml`로 확장 스택을 제공합니다.

운영 포인트는 간단합니다.

- `.env`에 Langfuse 관련 시크릿/키를 넣고
- PentAGI가 Langfuse로 이벤트를 내보내도록 설정합니다.

Langfuse를 켜면 “에이전트 호출 단위”로 추적이 쉬워져,
프롬프트/모델 튜닝의 기준 데이터를 만들 수 있습니다.

---

## 2) OpenTelemetry + Grafana(옵션)

`docker-compose-observability.yml`을 붙이면:

- 메트릭/트레이스/로그 수집 파이프라인을 만들 수 있습니다.

실무에서는 아래 질문에 답할 수 있어야 합니다.

- 평균/최대 지연은 어디서 발생하는가?
- LLM 호출 비율이 늘 때 병목은 어디인가?
- 컨테이너/DB 리소스는 어떤 패턴으로 치솟는가?

---

## 3) 테스트 유틸리티: `ctester` / `ftester` / `etester`

README는 여러 테스트 유틸리티를 소개합니다.

- `ctester`: 에이전트 타입별 LLM 응답/함수 호출 등 “호환성”을 확인
- `ftester`: 기능/함수 호출 시나리오 테스트
- `etester`: (구성에 따라) 임베딩/검색 등 특정 기능 검증

로컬 Go 환경이 없더라도, 배포 이미지 안에 바이너리가 포함되어 있어 Docker로 실행할 수 있습니다.

```bash
# 예시: 배포 이미지로 ctester 실행(개념)
docker run --rm vxcontrol/pentagi /opt/pentagi/bin/ctester -verbose
```

운영에서 중요한 것은 “테스트를 통과했다”보다:

- 어떤 provider/모델 조합이 안정적인지
- 어떤 옵션이 실패를 줄이는지

를 찾는 것입니다.

---

## 4) Dockerfile 빌드 구조: 멀티스테이지(프론트+백엔드)

루트 `Dockerfile`은 크게 3단계입니다.

1) Node 스테이지: 프론트 빌드(Vite)  
2) Go 스테이지: 백엔드 + 유틸 바이너리 빌드  
3) Alpine 런타임: 최소 런타임 이미지 구성

이 방식의 장점:

- 런타임 이미지를 작게 유지
- 빌드 환경/런타임 환경 분리
- 배포 이미지에 테스트 유틸(`ctester` 등)을 함께 포함

---

## 마무리: 운영 체크리스트

이 시리즈를 다 읽고 나면, 최소한 아래를 점검할 수 있어야 합니다.

- `.env`에서 어떤 변수가 기능/안전/비용에 영향을 주는가?
- Compose 스택을 “기본 → 확장” 순으로 안정화할 수 있는가?
- LLM provider/검색 엔진을 바꿔도 구조가 유지되는가?
- 로그/관측 데이터가 남아 “왜 실패했는지” 추적 가능한가?

---

## 참고 링크

- 인덱스(전체 목차): `/blog-repo/pentagi-guide/`
- Dockerfile: `https://github.com/vxcontrol/pentagi/blob/master/Dockerfile`

