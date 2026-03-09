---
layout: post
title: "Prompt Engineering Guide 가이드 (05) - 응용 패턴과 PAL"
date: 2026-02-20
permalink: /prompt-engineering-guide-05-application-patterns-and-pal/
author: Elvis Saravia (DAIR.AI)
categories: ['LLM 학습', '프롬프트 엔지니어링']
tags: [Applications, Data Generation, PAL, Code Generation]
original_url: "https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/guides/prompts-applications.md"
excerpt: "데이터 생성/라벨링/코드 생성 같은 응용 패턴과 Program-Aided Language Models 접근을 연결해 설명합니다."
---

## 응용 패턴의 핵심

`prompts-applications.md`는 "프롬프트를 기능으로 바꾸는" 관점에 집중합니다.

대표 패턴:

- 예시 데이터 생성
- 라벨링/증강 데이터 생성
- 코드/쿼리 생성
- 도구와 결합한 계산/검증

---

## 데이터 생성 프롬프트

가이드 예시는 감성분석용 샘플, NER 좌표가 포함된 JSON 데이터 생성까지 다룹니다.

실무에서는 아래를 반드시 추가해야 합니다.

1. 스키마 명시
2. 클래스 분포 명시
3. 품질 검증 규칙 명시

---

## PAL(Program-Aided LM)

PAL은 모델이 최종 답을 바로 말하게 하는 대신, 중간 단계를 프로그램으로 외부 실행해 정답을 검증하는 접근입니다.

```text
자연어 문제 해석 -> 코드 생성 -> 인터프리터 실행 -> 결과 반환
```

이 방식은 계산/날짜 추론처럼 오류가 잦은 문제에서 효과가 큽니다.

---

## 운영 팁

- 생성 데이터는 곧바로 학습에 넣지 말고 샘플링 검수
- 모델 출력과 실행 결과를 분리 저장
- 실패 케이스를 프롬프트 개선 데이터로 재사용

다음 장에서 채팅형 모델 설계와 역할 메시지 구조를 다룹니다.
