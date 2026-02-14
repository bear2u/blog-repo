---
layout: page
title: Explain OpenClaw 가이드
permalink: /explain-openclaw-guide/
icon: fas fa-book
---

# Explain OpenClaw 완벽 가이드

> **OpenClaw(구 Moltbot/Clawdbot) 운영, 보안, 배포를 위한 통합 지식베이스**

**Explain OpenClaw**는 OpenClaw를 처음 접하는 사람을 위한 쉬운 설명부터, 운영자가 실제로 부딪히는 배포/하드닝/보안 감사/최적화까지 한 번에 묶어 둔 문서 레포지토리입니다.

- 원문 저장소: https://github.com/centminmod/explain-openclaw
- OpenClaw 공식 문서(정본): https://docs.openclaw.ai

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/explain-openclaw-guide-01-intro/) | 이 레포가 다루는 범위, 읽는 순서, 핵심 개념 |
| 02 | [핵심 개념과 용어](/blog-repo/explain-openclaw-guide-02-concepts-glossary/) | Gateway/Channel/Session/Tool, trust boundary, 기본 보안 원칙 |
| 03 | [CLI 빠른 참조와 보안 감사](/blog-repo/explain-openclaw-guide-03-cli-and-audit/) | 자주 쓰는 명령, 트러블슈팅, `openclaw security audit` 읽는 법 |
| 04 | [아키텍처와 레포 맵](/blog-repo/explain-openclaw-guide-04-architecture-repo-map/) | 메시지가 응답으로 변하는 흐름, 코드에서 어디를 볼지 |
| 05 | [배포 1: Mac mini/VPS](/blog-repo/explain-openclaw-guide-05-deploy-mac-vps/) | 로컬 우선 vs 원격 운영, SSH/Tailscale, 기본 하드닝 |
| 06 | [배포 2: Moltworker/로컬 모델](/blog-repo/explain-openclaw-guide-06-deploy-moltworker-local-models/) | Cloudflare 서버리스 Gateway, Docker Model Runner로 로컬 LLM |
| 07 | [프라이버시/하드닝 체크리스트](/blog-repo/explain-openclaw-guide-07-privacy-hardening/) | 위협 모델, 안전한 기본값, 고프라이버시 설정 예시 |
| 08 | [최악의 시나리오와 운영 최적화](/blog-repo/explain-openclaw-guide-08-worst-case-ops/) | 프롬프트 인젝션, 공급망 리스크, 인시던트 대응, 비용/리소스 최적화 |

---

## 이 시리즈의 핵심 한 문장

> OpenClaw는 "모델"이 아니라 **Gateway(항상 켜져 있는 제어면)**가 핵심이고, 보안은 대부분 **Gateway 호스트와 설정**에서 결정됩니다.
