---
layout: post
title: "Tambo 완벽 가이드 (03) - 작동 원리"
date: 2026-02-12
permalink: /tambo-guide-03-how-it-works/
author: Tambo AI
categories: [AI 에이전트, 웹 개발]
tags: [Tambo, Zod, 도구, 스트리밍, 프롭스]
original_url: "https://github.com/tambo-ai/tambo"
excerpt: "컴포넌트 선택과 프롭스 스트리밍 흐름"
---

## 핵심 아이디어: 컴포넌트를 "LLM 도구"로 만든다

Tambo의 핵심은 다음 흐름으로 요약됩니다.

1. 개발자가 React 컴포넌트를 등록한다
2. props를 Zod 스키마로 정의한다
3. 이 스키마가 LLM의 tool definition이 된다
4. LLM이 어떤 컴포넌트를 쓸지 결정하고 props를 생성한다
5. Tambo가 프롭스를 스트리밍으로 전달하고 UI를 렌더링한다

---

## 생성형 vs 상호작용

README는 컴포넌트를 2가지로 나눠 설명합니다.

- **생성형 컴포넌트**: 메시지에 반응해 "한 번" 렌더링 (차트/요약/시각화)
- **상호작용 컴포넌트**: 렌더 후에도 유지되며 사용자 요청에 따라 업데이트 (장바구니/스프레드시트/작업보드)

이 구분이 설계에서 중요합니다.
- "한 번 보여주면 끝"인지
- "사용자와 상호작용하며 상태가 변해야" 하는지

---

## 전체 흐름(개념도)

```
사용자 메시지
   ↓
LLM (컴포넌트/툴 선택)
   ↓
프롭스 생성(스트리밍)
   ↓
Tambo 런타임(검증/상태/에러복구)
   ↓
React UI 렌더링
```

*다음 글에서는 생성형 컴포넌트 등록 예시를 살펴봅니다.*
