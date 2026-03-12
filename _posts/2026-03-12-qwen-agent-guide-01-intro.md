---
layout: post
title: "Qwen-Agent 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-12
permalink: /qwen-agent-guide-01-intro/
author: QwenLM
categories: [AI 에이전트, Qwen-Agent]
tags: [Trending, GitHub, Qwen-Agent, Python, MCP, RAG, GitHub Trending]
original_url: "https://github.com/QwenLM/Qwen-Agent"
excerpt: "Qwen 기반 LLM 애플리케이션/에이전트를 만드는 프레임워크(Qwen-Agent)의 구성과 시작점을 정리합니다."
---

## Qwen-Agent란?

GitHub Trending(weekly) 기준으로 주목받는 **QwenLM/Qwen-Agent**를 한국어로 정리합니다.

- **한 줄 요약(README 기반)**: Qwen의 계획/도구사용/메모리 역량을 바탕으로 LLM 애플리케이션을 만드는 프레임워크이며, Browser Assistant/Code Interpreter 같은 예제 앱도 포함합니다.
- **언어(Trending 표시)**: Python
- **이번 주 스타(Trending 표시)**: +1,887
- **원본**: https://github.com/QwenLM/Qwen-Agent

---

## 빠른 시작(README 기반)

- PyPI 설치(옵션 extras 포함): `pip install -U "qwen-agent[gui,rag,code_interpreter,mcp]"`
- 소스 설치: `pip install -e ./"[gui,rag,code_interpreter,mcp]"`
- 모델 서비스 준비:
  - DashScope 사용 시 `DASHSCOPE_API_KEY` 환경변수
  - 또는 OpenAI 호환 API(vLLM/Ollama 등) 엔드포인트 구성

---

## 저장소 구조(디렉토리 기반)

- `qwen_agent/`: 코어 라이브러리(LLM/Tools/Agents/Memory/GUI 등)
- `examples/`: 기능별 예제(툴콜/에이전트/코드 인터프리터 등)
- `run_server.py`, `qwen_server/`: 서버/웹 UI 관련 구성
- `benchmark/`: 에이전트 평가/벤치마크(예: DeepPlanning)
- `tests/`: 테스트

---

## 위키 링크

- `[[Qwen-Agent Guide - Index]]` → [가이드 목차](/blog-repo/qwen-agent-guide/)

