---
layout: page
title: ClawRouter 가이드
permalink: /clawrouter-guide/
icon: fas fa-route
---

# ClawRouter 완벽 가이드

> **OpenClaw용 로컬 LLM 라우터: 30+ 모델을 한 지갑으로, 요청마다 최저가/최적 모델로 라우팅**

**ClawRouter**는 OpenClaw(또는 OpenAI 호환 클라이언트)의 요청을 받아, **로컬에서(외부 호출 없이) 라우팅 결정을 내리고** 필요 시 **x402(402 Payment Required) 마이크로페이먼트**로 BlockRun API를 통해 다양한 모델(OpenAI/Anthropic/Google/DeepSeek/xAI/Moonshot 등)로 연결해주는 라우터입니다.

- 원문 저장소: https://github.com/BlockRunAI/ClawRouter
- 공식 사이트/문서: https://blockrun.ai/docs

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/clawrouter-guide-01-intro/) | 왜 ClawRouter인가, OpenRouter와의 차이 |
| 02 | [설치와 빠른 시작](/blog-repo/clawrouter-guide-02-installation/) | OpenClaw 플러그인 설치, 지갑 생성/펀딩, 헬스체크 |
| 03 | [라우팅 프로필과 티어](/blog-repo/clawrouter-guide-03-routing-profiles/) | auto/eco/premium/free, SIMPLE/MEDIUM/COMPLEX/REASONING |
| 04 | [설정 레퍼런스](/blog-repo/clawrouter-guide-04-configuration/) | 환경변수, proxy 포트, overrides, scoring weights |
| 05 | [아키텍처](/blog-repo/clawrouter-guide-05-architecture/) | 프록시, dedup, 라우팅 엔진, 스트리밍 하트비트 |
| 06 | [지갑과 결제(x402)](/blog-repo/clawrouter-guide-06-wallet-x402/) | 402 플로우, 키 보관/백업, /wallet |
| 07 | [관측과 비용 절감](/blog-repo/clawrouter-guide-07-observability-stats/) | /stats, 로그에서 라우팅 확인, 로컬 저장 경로 |
| 08 | [신뢰성/폴백](/blog-repo/clawrouter-guide-08-reliability-failover/) | provider 오류 폴백, subscription+failover 패턴 |
| 09 | [보안 모델](/blog-repo/clawrouter-guide-09-security/) | 키가 네트워크로 나가지 않는 이유, 스캐너 경고 해석 |
| 10 | [트러블슈팅/운영](/blog-repo/clawrouter-guide-10-troubleshooting/) | 흔한 에러, 포트 충돌, 업데이트/재설치 |

---

## 한 문장 요약

> ClawRouter는 **라우팅은 로컬(<1ms)**, **결제는 지갑 서명(x402)**, **모델 호출은 BlockRun API**라는 분리로 "API 키 없이" 라우팅/과금/폴백을 단순화합니다.
