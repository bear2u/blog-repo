---
layout: post
title: "ClawRouter 완벽 가이드 (01) - 소개: 왜 ClawRouter인가?"
date: 2026-02-15
permalink: /clawrouter-guide-01-intro/
author: BlockRun
categories: [AI 에이전트, ClawRouter]
tags: [ClawRouter, BlockRun, OpenClaw, LLM Router, Cost Optimization, x402, USDC]
original_url: "https://github.com/BlockRunAI/ClawRouter"
excerpt: "ClawRouter의 목표(최저가 모델 라우팅), 핵심 아이디어(로컬 라우팅 + 지갑 서명 결제), OpenRouter와의 차이를 정리합니다."
---

## ClawRouter는 무엇인가?

ClawRouter는 OpenAI 호환 요청(`POST /v1/chat/completions` 등)을 받아서,

- 요청 내용을 기반으로 **로컬에서 라우팅 결정을 내리고**
- 선택된 모델로 요청을 프록시하며
- 결제가 필요한 경우 **x402(402 Payment Required)** 프로토콜로 **USDC 마이크로페이먼트**를 서명해 재시도

까지를 묶어서 제공하는 **스마트 LLM 라우터**입니다.

---

## 왜 "API 키 없는" 라우팅을 강조하나?

ClawRouter는 비교 문서에서 기존 API 키 방식의 문제를 다음처럼 요약합니다.

- 공유 시크릿(API key)은 노출/유출 표면이 큼
- 자동화(에이전트) 관점에서는 사람이 키를 발급/주입해야 하는 흐름이 병목

ClawRouter는 이를 "지갑 서명"으로 바꿉니다.

- 인증: API 키 대신 **지갑 서명(암호학적 증명)**
- 결제: 선불 잔고가 아니라 **요청 단위 USDC 결제(비수탁)**

---

## OpenRouter와의 차이(요지)

문서가 말하는 핵심 차이는 2가지입니다.

1. **라우팅이 로컬에서 결정된다**
- 서버 호출 없이, 규칙/가중치 기반 스코어러로 티어/모델을 선택

2. **결제와 인증이 하나로 합쳐진다(x402)**
- 402 응답(가격 포함) -> 서명 -> 재요청

---

## 이 시리즈에서 다룰 범위

- OpenClaw 플러그인으로 설치하고 `blockrun/auto`로 라우팅 시작하기
- auto/eco/premium/free 프로필과 티어(SIMPLE/MEDIUM/COMPLEX/REASONING)
- 설정(지갑/포트/오버라이드/스코어링 가중치)
- 아키텍처(프록시, dedup, 스트리밍 하트비트, 폴백)
- /stats, /wallet 등 운영 명령
- 흔한 오류 해결

---

## 다음 글

다음 글에서는 OpenClaw에 ClawRouter를 붙이는 가장 빠른 경로(설치/헬스체크/기본 모델 설정)를 정리합니다.

- 다음: [ClawRouter (02) - 설치와 빠른 시작](/blog-repo/clawrouter-guide-02-installation/)
