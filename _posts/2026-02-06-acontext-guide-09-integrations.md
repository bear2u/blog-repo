---
layout: post
title: "Acontext 완벽 가이드 (9) - 프레임워크 연동"
date: 2026-02-06
permalink: /acontext-guide-09-integrations/
author: memodb-io
categories: [AI 에이전트, Acontext]
tags: [Acontext, Integrations, Agno, OpenAI, Vercel AI SDK]
original_url: "https://docs.acontext.io/integrations/intro"
excerpt: "Acontext를 기존 에이전트 프레임워크에 붙이는 방법을 정리합니다."
---

## 연동 기본 원칙

공식 문서의 공통 패턴:

1. 템플릿 프로젝트 생성(`acontext create ... --template-path ...`)
2. 환경변수 설정(`OPENAI_API_KEY`, `ACONTEXT_API_KEY`)
3. 에이전트 입출력을 Acontext Session에 저장

## 주요 연동 대상

- Agno
- OpenAI Python SDK
- OpenAI TypeScript SDK
- OpenAI Agents SDK
- Vercel AI SDK

## OpenAI Agents SDK 주의점

공식 가이드는 Agents SDK 출력을 `Converter.items_to_messages()`로 변환해 Acontext 호환 메시지로 저장하는 흐름을 제시합니다.

## 템플릿 우선 전략

직접 구현보다 제공 템플릿으로 시작하면 초기 구조를 빠르게 검증할 수 있습니다.

- Python 예제 템플릿
- TypeScript 예제 템플릿
