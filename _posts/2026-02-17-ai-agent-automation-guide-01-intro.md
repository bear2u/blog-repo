---
layout: post
title: "AI Agent Automation 가이드 (01) - 소개/포지셔닝: 로컬 우선 AI 워크플로우 엔진"
date: 2026-02-17
permalink: /ai-agent-automation-guide-01-intro/
author: vmDeshpande
categories: ['AI 에이전트', '워크플로우 자동화']
tags: [AI Agent Automation, Workflow Engine, Local First, Task Runner]
original_url: "https://github.com/vmDeshpande/ai-agent-automation"
excerpt: "ai-agent-automation이 어떤 문제를 해결하는지, 왜 deterministic 실행 모델을 강조하는지 핵심 포지셔닝을 정리합니다."
---

## 이 프로젝트의 핵심 메시지

README가 강조하는 키워드는 세 가지입니다.

- Local-first
- Deterministic execution
- Full visibility

즉 "프롬프트 데모"가 아니라, 실행 가능한 워크플로우 엔진을 로컬/셀프호스트 중심으로 운영하는 방향입니다.

---

## 워크플로우 모델

기본 동작은 다음 순서입니다.

1. 사용자가 워크플로우(ordered steps)를 정의
2. 실행 시 Task가 생성
3. Agent Runner가 step-by-step으로 실행
4. 각 step 입력/출력/성공 여부가 기록

이 구조 덕분에 실패 지점을 스텝 단위로 추적하기 쉽습니다.

---

## 어떤 팀에 맞나

README 기준 타깃은 아래와 같습니다.

- AI 자동화 백엔드를 직접 만들고 싶은 개발팀
- 실행 이력/실패 원인 분석이 중요한 운영팀
- SaaS 의존보다 셀프호스트를 선호하는 팀

n8n/Zapier/Temporal류 워크플로우 도구와 유사한 문제를 AI-native 방식으로 다룹니다.

---

## 기술 스택 한눈에 보기

- Backend: Node.js + Express + MongoDB + node-cron
- Frontend: Next.js + React + Tailwind
- AI: Groq/OpenAI/Gemini/HF 키 기반 확장 구조

다음 장에서는 실제 로컬 실행 경로와 필수 환경변수를 정리합니다.

