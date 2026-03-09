---
layout: post
title: "Prompt Engineering Guide 가이드 (09) - 컨텍스트 엔지니어링과 에이전트"
date: 2026-02-20
permalink: /prompt-engineering-guide-09-context-engineering-and-agents/
author: Elvis Saravia (DAIR.AI)
categories: ['LLM 학습', 'AI 에이전트']
tags: [Context Engineering, Agents, RAG, Reasoning LLMs]
original_url: "https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/pages/guides/context-engineering-guide.en.mdx"
excerpt: "최신 문서 기준으로 프롬프트를 넘어 컨텍스트 엔지니어링, 에이전트 구성요소, Reasoning LLM 활용 패턴을 연결합니다."
---

## 프롬프트에서 컨텍스트 엔지니어링으로

최근 가이드는 프롬프트 작성만으로는 에이전트 품질을 설명하기 어렵다고 봅니다.

핵심은 "LLM에 무엇을, 언제, 어떤 구조로 넣을지"를 시스템적으로 설계하는 것입니다.

---

## 에이전트 3요소

`pages/agents` 문서군은 아래를 기본 축으로 둡니다.

- Planning
- Tool Use
- Memory

단일 호출보다, 다중 단계에서 상태를 유지하고 도구를 호출하는 능력이 중요해집니다.

---

## 컨텍스트 엔지니어링 체크포인트

`context-engineering-guide.en.mdx`와 `context-engineering.en.mdx` 기반 핵심:

1. 시스템 지시와 사용자 입력을 구조적으로 분리
2. 출력 스키마(JSON/XML) 명시
3. 시간/상태/과거 실행 기록을 컨텍스트에 반영
4. 에러 처리/재시도 규칙까지 프롬프트에 내재화

---

## Reasoning LLM 활용

`reasoning-llms.en.mdx`는 reasoning 모델을 모든 단계에 쓰지 말고, 추론 병목 구간에 선택적으로 적용하라고 권장합니다.

- planning
- LLM-as-a-judge
- agentic RAG
- 복잡한 코드/리서치 태스크

정확도, 비용, 지연의 균형이 핵심입니다.

---

## Deep Research 맥락

`deep-research.en.mdx`는 멀티스텝 검색/분석/합성 워크플로우를 보여주는 사례입니다.

이 장의 결론은 단순합니다. 좋은 결과는 좋은 모델만이 아니라, **좋은 컨텍스트 설계**에서 나옵니다.

다음 장에서 실제 학습·운영 로드맵으로 시리즈를 마무리합니다.
