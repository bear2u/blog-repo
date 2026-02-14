---
layout: post
title: "Explain OpenClaw 완벽 가이드 (06) - 배포 2: Moltworker/로컬 모델"
date: 2026-02-14
permalink: /explain-openclaw-guide-06-deploy-moltworker-local-models/
author: centminmod
categories: [AI 에이전트, OpenClaw]
tags: [OpenClaw, Cloudflare, Moltworker, Serverless, Local LLM, Docker Model Runner]
original_url: "https://github.com/centminmod/explain-openclaw"
excerpt: "Cloudflare Moltworker(서버리스 Gateway)와 Docker Model Runner(로컬 LLM) 배포의 장단점과 신뢰 경계를 정리합니다."
---
## Moltworker(Cloudflare): 편의성 대신 제어 감소

Explain OpenClaw는 Moltworker를 "내 서버가 아니라 Cloudflare에서 Gateway를 구동"하는 PoC 배포로 설명합니다.

구성 요소(요약):
- Workers + Sandbox SDK 컨테이너: Gateway 런타임
- R2: config/sessions/credentials 영속 저장
- AI Gateway: 모델 프록시/캐싱/폴백
- Browser Rendering: 헤드리스 브라우저 도구

문서가 강조하는 운영 리스크(요지):
- 서버리스는 편하지만 실행 환경/네트워크 제어가 제한된다.
- 프롬프트 인젝션이 성공했을 때 egress가 구조적으로 위험해질 수 있다.

---

## Docker Model Runner(DMR): 로컬 LLM로 비용 0/데이터 0

Explain OpenClaw의 DMR 런북 요지는 이렇습니다.

- Docker Desktop이 로컬 LLM을 실행
- OpenAI 호환 API로 노출
- OpenClaw는 그 엔드포인트를 provider로 붙인다

예시(요약):
```bash
docker model pull glm-4.7-flash
docker model run glm-4.7-flash "What is a recursive function?"

openclaw config set provider.name openai
openclaw config set provider.baseUrl http://model-runner.docker.internal/v1
openclaw config set provider.model glm-4.7-flash
openclaw config set provider.apiKey "not-needed"
```

---

## 다음 글

다음 글에서는 위협 모델과 하드닝 체크리스트, 고프라이버시 설정 예시를 묶어 정리합니다.

- 다음: [Explain OpenClaw (07) - 프라이버시/하드닝 체크리스트](/blog-repo/explain-openclaw-guide-07-privacy-hardening/)
