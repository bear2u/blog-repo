---
layout: post
title: "Acontext 완벽 가이드 (1) - 소개 및 구조"
date: 2026-02-06
permalink: /acontext-guide-01-intro/
author: memodb-io
categories: [AI 에이전트, Acontext]
tags: [Acontext, Context Storage, Agent Infrastructure, Observability]
original_url: "https://github.com/memodb-io/Acontext"
excerpt: "Acontext가 해결하는 문제와 시스템 구조를 빠르게 이해합니다."
---

## Acontext란?

Acontext는 **프로덕션 AI 에이전트용 컨텍스트 데이터 플랫폼**입니다. 공식 설명대로, Supabase가 앱 데이터를 다루듯 Acontext는 에이전트 컨텍스트를 다룹니다.

핵심 목적은 세 가지입니다.

- 컨텍스트 데이터 저장(Session, Disk, Skill, Sandbox)
- 컨텍스트 엔지니어링(Context Editing, Session Summary)
- 에이전트 관측성(Task/Trace/대시보드)

## 왜 필요한가?

에이전트를 운영하면 보통 다음 문제가 발생합니다.

- 메시지/파일/스킬 데이터가 분산 저장됨
- 장기 실행 에이전트의 컨텍스트 관리 비용이 큼
- 멀티모달/멀티LLM 상태 추적이 어려움

Acontext는 이를 통합 계층으로 해결합니다.

## 상위 아키텍처

공식 문서 구조를 기준으로 보면 Acontext는 아래 4계층으로 이해하면 쉽습니다.

1. **Storage**: Session Messages, Disk, Skill, Sandbox
2. **Engineering**: Context Editing, Session Summary, Cache 안정화
3. **Observability**: Agent Tasks, Buffer, Dashboard, Tracing
4. **Integration/API**: SDK/프레임워크 연동 + REST API

## 문서 구조 읽는 법

가이드는 공식 `docs/docs.json`의 내비게이션을 그대로 챕터화했습니다.

- Getting Started
- Context Storage
- Agent Tools
- Context Engineering
- Agent Observability
- Customization
- Integrations
- API Reference

다음 장부터는 실제 적용 순서(설치 → 저장 → 편집 → 관측 → 운영)로 정리합니다.
