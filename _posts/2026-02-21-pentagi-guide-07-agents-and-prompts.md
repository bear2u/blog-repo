---
layout: post
title: "PentAGI 가이드 (07) - 에이전트/프롬프트: 역할 분리와 컨텍스트 제어"
date: 2026-02-21
permalink: /pentagi-guide-07-agents-and-prompts/
author: PentAGI Team
categories: ['AI 에이전트', '보안']
tags: [PentAGI, Agents, Prompts, Templates, Summarizer, Langfuse]
original_url: "https://github.com/vxcontrol/pentagi"
excerpt: "PentAGI는 ‘한 모델’이 아니라 ‘역할을 분리한 팀’으로 동작합니다. 템플릿 구조와 컨텍스트 제어 지점을 정리합니다."
---

## 왜 멀티 에이전트인가?

보안 테스트 워크플로우는 보통 한 번에 끝나지 않습니다.

- 정보를 모으고(리서치/검색)
- 가능한 가설을 세우고
- 필요한 도구를 고르고(실행)
- 결과를 정리하고(리포팅)
- 다음 루프를 설계합니다.

PentAGI는 이를 **역할(Agent Type)**로 분리해,
각 역할에 맞는 프롬프트/툴 세트를 제공하는 방식으로 설계되어 있습니다.

---

## 프롬프트 템플릿이 있는 위치

백엔드에는 템플릿 디렉토리가 존재합니다.

```text
backend/pkg/templates/
└─ prompts/
   ├─ primary_agent.tmpl
   ├─ assistant.tmpl
   ├─ adviser.tmpl
   ├─ coder.tmpl
   ├─ searcher.tmpl
   ├─ memorist.tmpl
   ├─ reporter.tmpl
   ├─ summarizer.tmpl
   └─ ... (질문 템플릿, fixer 등)
```

이 템플릿들은 “어떤 도구를 쓸 수 있는지”, “어떤 형식으로 답해야 하는지” 같은
실행 제약을 포함합니다.

---

## 컨텍스트가 커지는 문제와 ‘요약기’

에이전트 시스템에서 가장 흔한 운영 문제는 **컨텍스트 폭발**입니다.

- 로그/툴 출력/검색 결과가 누적되며
- 모델 호출 비용과 지연이 증가하고
- 중요한 정보가 뒤로 밀립니다.

PentAGI는 이를 위해:

- “대화 체인”을 구조화하고
- 오래된 구간을 부분 요약하며
- 마지막 구간은 보존하는

형태의 요약 로직(설정 가능)을 포함합니다.

`.env`에서 관련 옵션을 찾을 수 있습니다.

```text
SUMMARIZER_*
ASSISTANT_SUMMARIZER_*
```

---

## 로그/관측성과 프롬프트의 결합

PentAGI는 Langfuse 같은 LLM 관측 도구와 연동할 수 있습니다.

이때 유용한 점은:

- 에이전트별 호출/지연/에러를 분리해 보고
- 어떤 프롬프트가 어떤 결과를 만들었는지 추적하며
- 운영 중 튜닝할 포인트를 찾기 쉬워진다는 것입니다.

---

## 안전한 운영을 위한 실무 포인트

프롬프트/에이전트 설계는 “기능”만이 아니라 “운영 안전성”에도 연결됩니다.

- 어떤 툴을 기본 활성화할지(혹은 막을지)
- 어떤 작업은 사용자 확인(ASK_USER)을 강제할지
- 어떤 네트워크/권한을 부여할지(Docker, NET_ADMIN 등)

이 선택이 결국 시스템의 위험도를 결정합니다.

---

## 참고 링크

- 인덱스(전체 목차): `/blog-repo/pentagi-guide/`
- 템플릿 폴더: `backend/pkg/templates/`

---

다음 글에서는 에이전트가 실제로 사용하는 **툴 실행기(Docker) + 검색/브라우저 도구**가 어떻게 묶이는지 살펴봅니다.

