---
layout: post
title: "Prompt Engineering Guide 가이드 (07) - 신뢰성 엔지니어링과 평가"
date: 2026-02-20
permalink: /prompt-engineering-guide-07-reliability-and-evaluation/
author: Elvis Saravia (DAIR.AI)
categories: ['LLM 학습', '프롬프트 엔지니어링']
tags: [Reliability, Factuality, Bias, Evaluation]
original_url: "https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/guides/prompts-reliability.md"
excerpt: "사실성/바이어스 문제를 프롬프트와 예시 설계로 어떻게 완화하고, 무엇을 평가 지표로 삼을지 정리합니다."
---

## 왜 신뢰성이 먼저인가

프롬프트 성능은 단일 데모보다 실제 분포에서 안정적으로 유지되는지가 중요합니다.

`prompts-reliability.md`는 특히 다음을 강조합니다.

- 사실이 아닌 답을 그럴듯하게 말하는 문제
- few-shot 예시 분포/순서에 의한 편향

---

## 사실성 개선 기본

가이드에서 제안하는 핵심 방법:

1. 근거 문맥(ground truth) 제공
2. 불확실 시 모른다고 답하게 지시
3. 확률 설정을 보수적으로 조정

운영 환경에서는 여기에 출처 검증 단계를 추가해야 합니다.

---

## 예시 분포와 순서

few-shot에서 모델 편향은 종종 "예시 구성"에서 발생합니다.

- 라벨 비율이 치우치면 출력도 치우침
- 같은 예시라도 순서를 바꾸면 결과가 변함

따라서 예시 셋은 균형/랜덤화/회귀검증이 필요합니다.

---

## 평가 루틴 제안

- 고정 테스트셋(사실성/안전성/형식 준수)
- 모델/프롬프트 버전별 비교 로그
- 실패 케이스를 프롬프트 개선 루프로 환원

다음 장에서는 공격 관점(Injection/Leak/Jailbreak)으로 전환합니다.
