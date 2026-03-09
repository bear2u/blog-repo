---
layout: post
title: "Prompt Engineering Guide 가이드 (08) - 공격 시나리오와 방어 전략"
date: 2026-02-20
permalink: /prompt-engineering-guide-08-adversarial-prompting-and-defense/
author: Elvis Saravia (DAIR.AI)
categories: ['LLM 학습', '프롬프트 엔지니어링']
tags: [Prompt Injection, Prompt Leaking, Jailbreak, Safety]
original_url: "https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/guides/prompts-adversarial.md"
excerpt: "Prompt Injection, Prompt Leaking, Jailbreaking 패턴과 현실적인 방어 전술을 제품 관점에서 정리합니다."
---

## 핵심 공격 유형

`prompts-adversarial.md`는 세 가지를 중심으로 설명합니다.

1. Prompt Injection: 기존 지시를 덮어쓰기
2. Prompt Leaking: 시스템/예시 프롬프트 유출
3. Jailbreaking: 정책 우회 유도

이 셋은 독립적이라기보다 연쇄적으로 결합됩니다.

---

## 방어는 단일 기법으로 끝나지 않는다

가이드에서 제시하는 방어 방향:

- 지시문에 방어 문맥 추가
- 입력/지시 분리(파라미터화)
- 인코딩/인용/포맷 강화
- 탐지 전용 평가 모델(필터) 사용

하지만 어떤 방식도 완전하지 않으므로 **계층형 방어**가 필요합니다.

---

## 제품 레벨 최소 방어선

1. 입력 전처리(길이, 패턴, 금칙어, escape)
2. 시스템 정책 고정 및 템플릿화
3. 고위험 요청에 이중 판정(본 모델 + 검사 모델)
4. 감사 로그 저장

---

## 중요한 현실

모델이 업데이트되면 과거 방어가 무력화되거나 반대로 과잉 차단이 생길 수 있습니다.

따라서 보안 프롬프트는 코드처럼 버전 관리하고 회귀 테스트해야 합니다.

다음 장에서 최신 흐름인 컨텍스트 엔지니어링과 에이전트 설계를 다룹니다.
