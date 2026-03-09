---
layout: post
title: "Prompt Engineering Guide 가이드 (04) - 고급 프롬프팅 패턴"
date: 2026-02-20
permalink: /prompt-engineering-guide-04-advanced-prompting-patterns/
author: Elvis Saravia (DAIR.AI)
categories: ['LLM 학습', '프롬프트 엔지니어링']
tags: [Zero-shot, Few-shot, Chain-of-Thought, Self-Consistency]
original_url: "https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/guides/prompts-advanced-usage.md"
excerpt: "Zero-shot/Few-shot의 경계와 CoT, Zero-shot CoT, Self-Consistency를 어떤 조건에서 써야 하는지 정리합니다."
---

## Zero-shot에서 시작

단순 분류/요약/추출은 Zero-shot으로 먼저 시도하는 것이 비용 대비 효율이 좋습니다.

하지만 복잡한 추론(수학/다단계 추론)에서는 쉽게 실패합니다.

---

## Few-shot의 역할

Few-shot은 정답 자체보다 **출력 형식과 라벨 공간을 고정**하는 데 특히 효과적입니다.

- 원하는 라벨 케이스(예: `neutral`) 고정
- 출력 템플릿 고정
- 예시 분포와 형식 일관성 확보

---

## CoT와 Zero-shot CoT

복잡한 문제에서는 중간 추론 단계를 노출하는 CoT가 효과적입니다.

- Few-shot CoT: 예시와 함께 단계 추론 유도
- Zero-shot CoT: "Let's think step by step" 같은 간단 트리거

가이드 예시에서도 산술 문제 정확도가 크게 개선됩니다.

---

## Self-Consistency

단일 경로 추론 대신 여러 경로를 샘플링해 다수결/일관성으로 답을 선택하는 방식입니다.

- 장점: 산술/상식 추론 안정화
- 비용: 토큰 사용량 증가

실무에서는 "중요 질문에만 적용"하는 선택적 전략이 적절합니다.

다음 장에서 고급 기법이 실제 응용 패턴으로 어떻게 연결되는지 봅니다.
