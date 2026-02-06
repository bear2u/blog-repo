---
layout: post
title: "Acontext 완벽 가이드 (6) - 컨텍스트 엔지니어링"
date: 2026-02-06
permalink: /acontext-guide-06-context-engineering/
author: memodb-io
categories: [AI 에이전트, Acontext]
tags: [Acontext, Context Engineering, Session Summary, Context Editing, Prompt Cache]
original_url: "https://docs.acontext.io/engineering/whatis"
excerpt: "Acontext의 Context Engineering 기능을 실전 관점에서 정리합니다."
---

## 핵심 기능 3가지

1. **Session Summary**: 세션을 토큰 효율적 요약으로 압축
2. **Context Editing**: 조회 시점 전략으로 컨텍스트 길이 제어
3. **Prompt Cache Stability**: 편집 전략 고정점으로 캐시 히트율 보호

## Session Summary

- 세션 작업 히스토리를 짧은 요약으로 주입
- 시스템 프롬프트에 붙여 장기 맥락 유지

## Context Editing

원본 메시지는 유지하고, `get_messages` 응답 시 전략을 적용합니다.

- `token_limit` 등 전략 조합
- 응답의 `this_time_tokens`로 현재 길이 측정
- 길이 임계치 기반 동적 편집

## Cache 안정화

편집 전략이 매번 다른 위치에 적용되면 프롬프트 prefix가 흔들려 캐시 miss가 증가합니다.

Acontext는 `pin_editing_strategies_at_message`로 적용 기준점을 고정해 prefix 안정성을 높이는 방법을 제시합니다.

## Agent Skills와 연결

컨텍스트 엔지니어링은 단순 메시지 압축만이 아니라, 스킬을 사용해 문제 해결 컨텍스트를 구조화하는 전략과 함께 사용됩니다.
