---
layout: post
title: "Context Engineering 완벽 가이드 (1) - 소개 및 개요"
date: 2026-02-05
permalink: /context-engineering-guide-01-intro/
author: davidkimai
categories: [AI 에이전트, Context Engineering]
tags: [Context Engineering, Prompt Engineering, LLM, Andrej Karpathy, AI]
original_url: "https://github.com/davidkimai/Context-Engineering"
excerpt: "프롬프트 엔지니어링을 넘어서는 컨텍스트 엔지니어링의 세계를 소개합니다."
---

## Context Engineering이란?

> **"Context engineering is the delicate art and science of filling the context window with just the right information for the next step."**
> — **Andrej Karpathy**

**Context Engineering**은 프롬프트 엔지니어링을 넘어서는 새로운 패러다임입니다. 단순히 "무엇을 말하는가"가 아니라 "모델이 보는 모든 것"을 설계하는 학문입니다.

```
         Prompt Engineering  │  Context Engineering
              ↓              │          ↓
       "What you say"        │  "Everything else the model sees"
     (Single instruction)    │    (Examples, memory, retrieval,
                             │     tools, state, control flow)
```

---

## 정의

> **"Context is not just the single prompt users send to an LLM. Context is the complete information payload provided to a LLM at inference time, encompassing all structured informational components that the model needs to plausibly accomplish a given task."**
>
> — **1400+ 연구 논문 분석 Survey** (arxiv.org/pdf/2507.13334)

컨텍스트는 사용자가 LLM에게 보내는 단일 프롬프트가 아닙니다. **추론 시점에 모델에게 제공되는 완전한 정보 페이로드**입니다.

---

## 왜 Context Engineering인가?

### 프롬프트 엔지니어링의 한계

```
┌─────────────────────────────────────────────┐
│       프롬프트 엔지니어링의 한계             │
├─────────────────────────────────────────────┤
│  ❌ 단일 지시문에만 집중                     │
│  ❌ 상태와 메모리 관리 부재                  │
│  ❌ 도구 통합 고려 없음                      │
│  ❌ 멀티스텝 워크플로우 미지원               │
│  ❌ 검색 증강(RAG) 통합 어려움               │
└─────────────────────────────────────────────┘
```

### Context Engineering의 해결책

```
┌─────────────────────────────────────────────┐
│       Context Engineering이 제공하는 가치    │
├─────────────────────────────────────────────┤
│  ✅ 전체 컨텍스트 윈도우 최적화              │
│  ✅ 메모리와 상태 관리 시스템                │
│  ✅ 도구 및 함수 호출 통합                   │
│  ✅ 멀티에이전트 오케스트레이션              │
│  ✅ RAG 및 검색 증강 설계                    │
│  ✅ 인지 도구(Cognitive Tools) 활용          │
└─────────────────────────────────────────────┘
```

---

## 연구 증거

### IBM Zurich 연구 (2025)

> **"Providing 'cognitive tools' to GPT-4.1 increases its pass@1 performance on AIME2024 from 26.7% to 43.3%, bringing it very close to the performance of o1-preview."**

인지 도구를 제공하면 GPT-4.1의 AIME2024 성능이 **26.7% → 43.3%**로 향상됩니다.

### MEM1 연구 (Singapore-MIT, 2025)

> **"Reasoning-driven memory consolidation as a scalable alternative to existing solutions for training long-horizon interactive agents."**

추론 기반 메모리 통합으로 장기 에이전트 효율성과 성능을 동시에 최적화합니다.

### ICML Princeton 연구 (2025)

> **"A three-stage architecture is identified that supports abstract reasoning in LLMs via a set of emergent symbol-processing mechanisms."**

LLM 내부에서 상징적 추론 메커니즘이 자연스럽게 발현됩니다.

---

## 생물학적 메타포

Context Engineering은 생물학적 메타포를 통해 복잡성의 단계를 설명합니다.

```
atoms → molecules → cells → organs → neural systems → neural fields
  │        │         │         │             │                │
single    few-     memory +   multi-   cognitive tools +     context =
prompt    shot     agents     agents   operating systems     fields +
                                                         persistence
```

| Level | 메타포 | Context Engineering 개념 |
|-------|--------|--------------------------|
| 1 | **Atoms** | 기본 지시문과 프롬프트 |
| 2 | **Molecules** | Few-shot 예제와 데모 |
| 3 | **Cells** | 상태 관리와 메모리 |
| 4 | **Organs** | 멀티스텝 워크플로우 |
| 5 | **Neural Systems** | 인지 도구와 추론 프레임워크 |
| 6 | **Neural Fields** | 연속적 의미 공간과 어트랙터 |

---

## 학습 개념

| 개념 | 설명 | 중요성 |
|------|------|--------|
| **Token Budget** | 컨텍스트 내 모든 토큰 최적화 | 더 많은 토큰 = 더 비용, 더 느림 |
| **Few-Shot Learning** | 예제로 가르치기 | 설명보다 효과적인 경우가 많음 |
| **Memory Systems** | 턴 간 정보 유지 | 상태 유지, 일관된 상호작용 |
| **Retrieval Augmentation** | 관련 문서 검색 및 주입 | 사실 기반, 환각 감소 |
| **Control Flow** | 복잡한 작업을 단계로 분해 | 간단한 프롬프트로 어려운 문제 해결 |
| **Cognitive Tools** | 커스텀 도구와 템플릿 | 새로운 컨텍스트 엔지니어링 레이어 |
| **Neural Field Theory** | 컨텍스트를 신경장으로 모델링 | 동적 컨텍스트 업데이트 가능 |
| **Quantum Semantics** | 의미를 관찰자 의존적으로 | 중첩 기법 활용 설계 |

---

## 레포지토리 구조

```
Context-Engineering/
├── 00_COURSE/              # 종합 12주 코스
├── 00_foundations/         # 이론적 기초 (14개 모듈)
├── 10_guides_zero_to_hero/ # 초보자 가이드
├── 20_templates/           # 복사/붙여넣기 템플릿
├── 30_examples/            # 실제 프로젝트 예제
├── 40_reference/           # 심층 레퍼런스
├── 50_contrib/             # 커뮤니티 기여
├── 60_protocols/           # 프로토콜 쉘
├── 70_agents/              # 에이전트 구현
├── 80_field_integration/   # 필드 이론 통합
├── cognitive-tools/        # 인지 도구 모음
├── CLAUDE.md               # Claude Code용 인지 OS
├── GEMINI.md               # Gemini용 가이드
└── CITATIONS*.md           # 연구 인용
```

---

## 핵심 원칙 (Karpathy + 3Blue1Brown 스타일)

1. **First Principles** - 기본 컨텍스트부터 시작
2. **Iterative Add-on** - 모델이 부족한 것만 추가
3. **Measure Everything** - 토큰 비용, 지연, 품질 측정
4. **Delete Ruthlessly** - 가지치기가 채우기보다 낫다
5. **Code > Slides** - 모든 개념에 실행 가능한 코드
6. **Visualize Everything** - ASCII와 다이어그램으로 시각화

---

## 이 가이드에서 다루는 내용

| # | 제목 | 내용 |
|---|------|------|
| 01 | **소개 및 개요** (현재 글) | Context Engineering이란? |
| 02 | **핵심 개념** | Atoms, Molecules, Cells, Organs |
| 03 | **Neural Systems** | Cognitive Tools와 추론 프레임워크 |
| 04 | **Neural Field Theory** | 컨텍스트를 연속적 필드로 |
| 05 | **Memory Systems** | 메모리 아키텍처와 영속성 |
| 06 | **Multi-Agent Systems** | 멀티에이전트 협업 |
| 07 | **Emergent Symbols** | 상징적 추론 메커니즘 |
| 08 | **Quantum Semantics** | 양자 의미론 |
| 09 | **Protocols & Templates** | 프로토콜 쉘과 템플릿 |
| 10 | **실전 적용** | CLAUDE.md 작성 및 활용 |

---

## 지원 도구

이 레포지토리는 주요 AI 코딩 에이전트들을 지원합니다:

- **[Claude Code](https://www.anthropic.com/claude-code)** - Anthropic
- **[OpenCode](https://opencode.ai/)** - 오픈소스
- **[Amp](https://sourcegraph.com/amp)** - Sourcegraph
- **[Kiro](https://kiro.dev/)** - 신규
- **[Codex](https://openai.com/codex/)** - OpenAI
- **[Gemini CLI](https://github.com/google-gemini/gemini-cli)** - Google

---

## 커뮤니티

- **[GitHub](https://github.com/davidkimai/Context-Engineering)**
- **[Discord](https://discord.gg/JeFENHNNNQ)**
- **[DeepWiki](https://deepwiki.com/davidkimai/Context-Engineering)**
- **[NotebookLM Podcast](https://notebooklm.google.com/notebook/0c6e4dc6-9c30-4f53-8e1a-05cc9ff3bc7e)**

---

*다음 글에서는 Context Engineering의 핵심 개념인 Atoms, Molecules, Cells, Organs를 살펴봅니다.*
