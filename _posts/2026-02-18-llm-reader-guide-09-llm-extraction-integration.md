---
layout: post
title: "LLM Reader 가이드 (09) - LLM 추출 연동: 프롬프트, 길이 제어, 비용 최적화"
date: 2026-02-18
permalink: /llm-reader-guide-09-llm-extraction-integration/
author: m92vyas
categories: ['개발 도구', '웹 스크래핑']
tags: [Prompt Engineering, Token Budget, Extraction, llm-reader, OpenAI]
original_url: "https://github.com/m92vyas/llm-reader"
excerpt: "전처리된 텍스트를 LLM 추출 파이프라인에 붙일 때 필요한 프롬프트 설계와 비용 최적화 기준을 정리합니다."
---

## README 예제의 핵심 아이디어

저장소 예제는 아래 패턴을 권장합니다.

1. llm-reader로 전처리된 텍스트 확보
2. 명시적 JSON 스키마를 프롬프트에 포함
3. 모델에 보내기 전 길이 제한 적용

이 3단계만 지켜도 추출 결과 안정성이 크게 올라갑니다.

---

## 프롬프트 템플릿 팁

- 출력 형식을 JSON으로 고정
- 필수 필드 누락 시 빈 문자열 규칙 정의
- 중복 항목 제거 기준 명시

```text
반드시 JSON으로만 답변하고, 필드는 product_name/product_link/image_link/price를 사용해라.
```

---

## 길이 제어 전략

README 예시처럼 문자열 길이로 1차 절단을 할 수 있지만, 운영에서는 토큰 기준 절단이 더 안전합니다.

- 긴 페이지: 섹션 단위 chunk 분할
- 추출 대상이 명확할 때: 키워드 주변 우선 추출
- 모델별 context 한도에 맞춘 동적 자르기

---

## 수집 비용과 추론 비용 분리

`Cheaper_Alternative.md`는 HTML 수집 API를 pay-as-you-go로 쓰고,
후처리는 llm-reader 오픈소스로 처리하는 비용 모델을 제안합니다.

즉 "페이지 수집 비용"과 "LLM 토큰 비용"을 분리 최적화하는 접근입니다.

다음 장에서는 한계와 운영 트러블슈팅을 정리합니다.
