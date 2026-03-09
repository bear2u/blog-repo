---
layout: page
title: Prompt Engineering Guide 가이드
permalink: /prompt-engineering-guide/
icon: fas fa-brain
---

# Prompt Engineering Guide 완벽 가이드

> **LLM 프롬프트 설계부터 컨텍스트 엔지니어링, 에이전트 응용까지 연결하는 실전 지식베이스**

`dair-ai/Prompt-Engineering-Guide`는 단일 튜토리얼이 아니라, 프롬프트 기초/기법/리스크/응용/에이전트/리서치 자료를 한곳에 모아 계속 업데이트하는 레퍼런스입니다.

- 원문 저장소: https://github.com/dair-ai/Prompt-Engineering-Guide
- 웹 버전: https://www.promptingguide.ai/
- 로컬 실행 기반: Next.js + Nextra 문서 사이트

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개와 저장소 지도](/blog-repo/prompt-engineering-guide-01-intro-and-repo-map/) | 프로젝트 목적, 콘텐츠 범위, 디렉토리 구조 |
| 02 | [로컬 실행과 사이트 구조](/blog-repo/prompt-engineering-guide-02-local-setup-and-site-structure/) | Node/pnpm 실행, 다국어/페이지 구성 |
| 03 | [프롬프트 기본기](/blog-repo/prompt-engineering-guide-03-prompt-basics-and-settings/) | 프롬프트 요소, LLM 설정, 기본 설계 원칙 |
| 04 | [고급 프롬프팅 패턴](/blog-repo/prompt-engineering-guide-04-advanced-prompting-patterns/) | Zero/Few-shot, CoT, Self-Consistency |
| 05 | [응용 패턴과 PAL](/blog-repo/prompt-engineering-guide-05-application-patterns-and-pal/) | 데이터 생성, 분류/추출, Program-Aided LM |
| 06 | [채팅 모델 설계](/blog-repo/prompt-engineering-guide-06-chat-models-and-conversation-design/) | 멀티턴/싱글턴, 역할 메시지, 대화 시스템화 |
| 07 | [신뢰성 엔지니어링](/blog-repo/prompt-engineering-guide-07-reliability-and-evaluation/) | 사실성, 바이어스, 예시 분포/순서, 평가 관점 |
| 08 | [공격과 방어](/blog-repo/prompt-engineering-guide-08-adversarial-prompting-and-defense/) | Injection/Leak/Jailbreak와 방어 전략 |
| 09 | [컨텍스트 엔지니어링과 에이전트](/blog-repo/prompt-engineering-guide-09-context-engineering-and-agents/) | 도구/메모리/구조화 출력 기반 에이전트 설계 |
| 10 | [학습 로드맵과 실무 적용](/blog-repo/prompt-engineering-guide-10-learning-roadmap-and-practical-workflow/) | 논문·노트북·실험 루틴으로 연결하는 운영 방법 |

---

## 이 시리즈가 다루는 핵심

- 프롬프트 문장 작성에서 끝나지 않고, **시스템 설계 문제**로 확장하는 관점
- 성능뿐 아니라 신뢰성/보안/비용/운영을 함께 보는 실무 기준
- Prompt Engineering에서 Context Engineering으로 넘어가는 최신 흐름

---

## 빠른 시작

```bash
pnpm i
pnpm dev
```

브라우저에서 `http://localhost:3000`으로 문서를 확인할 수 있습니다.
