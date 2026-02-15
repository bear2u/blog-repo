---
layout: post
title: "PicoClaw 가이드 (01) - 소개: $10 하드웨어에서 도는 초경량 개인 AI 어시스턴트"
date: 2026-02-15
permalink: /picoclaw-guide-01-intro/
author: Sipeed
categories: [AI 에이전트, 개발 도구]
tags: ["PicoClaw", "Go", "Low Footprint", "SBC", "Single Binary"]
original_url: "https://github.com/sipeed/picoclaw"
excerpt: "PicoClaw는 Go로 만든 초경량 개인 AI 어시스턴트로, 10MB 미만 RAM과 빠른 부팅을 목표로 합니다. 이 장에서는 제품의 포지션과 전체 구조를 잡습니다."
---

## PicoClaw는 무엇인가

**PicoClaw**는 “개인 AI 어시스턴트”를 **초저사양/저비용 하드웨어에서도** 굴릴 수 있게 만드는 프로젝트입니다.

README에서 강조하는 포인트:

- Go 기반 단일 바이너리
- x86_64 / ARM64 / RISC-V 등 다양한 아키텍처 지향
- “에이전트(CLI)”와 “게이트웨이(채팅 봇)” 두 실행 경로
- 도구 실행(파일/명령)과 웹 검색 같은 기능
- 워크스페이스 기반의 메모리/세션/스케줄링(cron, heartbeat)

---

## 어떤 문제를 풀고 있나

“에이전트” 제품이 많아질수록, 실행 비용과 운영 복잡도가 커집니다.

PicoClaw는 다음 방향을 택합니다.

- **작게**: 작은 바이너리, 작은 RAM 풋프린트, 빠른 기동
- **넓게**: 여러 LLM 프로바이더(OpenRouter/Zhipu/OpenAI 등) 및 채널(Telegram/Discord/LINE 등)
- **안전하게**: 기본은 워크스페이스 내부로만 접근하도록 제한(샌드박스)

---

## “에이전트”와 “게이트웨이” 모드

README 기준으로 PicoClaw는 크게 2가지 실행 패턴이 있습니다.

1. `picoclaw agent`
   - 단발 질문(`-m`) 또는 대화형 CLI
2. `picoclaw gateway`
   - Telegram/Discord/LINE 같은 채팅 채널을 붙여 “상시 봇”으로 운영

이 시리즈는 이 두 모드를 모두 다루되, 처음에는 **agent 모드로 최소 동작**을 확인하고 그 다음에 gateway로 확장하는 흐름으로 진행합니다.

---

## 보안 공지(중요)

README에는 다음 종류의 공지가 들어 있습니다.

- 공식 도메인 사칭/유사 도메인 주의
- 토큰/코인 관련 사기 경고
- v1.0 이전에는 네트워크 보안 이슈가 있을 수 있으니 프로덕션 배포 주의

다음 장에서는 설치부터 잡고, 로컬에서 `onboard` 후 첫 `agent` 실행까지 바로 가보겠습니다.

