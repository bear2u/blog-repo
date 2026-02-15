---
layout: page
title: PicoClaw 가이드
permalink: /picoclaw-guide/
icon: fas fa-robot
---

# PicoClaw 완벽 가이드

> **Go로 만든 초경량 개인 AI 어시스턴트: $10 하드웨어, 10MB 미만 RAM, 1초 부팅을 목표로**

**PicoClaw**는 Sipeed가 공개한 초경량 개인 AI 어시스턴트로, Go로 구현된 단일 바이너리를 다양한 아키텍처(x86_64/ARM64/RISC-V)에서 실행하는 것을 목표로 합니다. CLI로 “에이전트(단발/대화형)”를 실행할 수 있고, Telegram/Discord/LINE 같은 채팅 채널을 붙여 **게이트웨이(봇)** 형태로도 운영할 수 있습니다.

- 원문 저장소: https://github.com/sipeed/picoclaw
- 공식 사이트(README 기준): https://picoclaw.io

> 참고: README에 보안 공지가 있습니다. 공식 사이트/도메인 사칭 및 토큰/코인 사기 주의, 그리고 v1.0 이전에는 프로덕션 배포를 조심하라는 경고가 포함돼 있습니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/picoclaw-guide-01-intro/) | PicoClaw의 목표, “초경량”의 의미 |
| 02 | [설치(바이너리/소스)](/blog-repo/picoclaw-guide-02-install/) | 릴리즈 바이너리, `make build/install` |
| 03 | [Quick Start](/blog-repo/picoclaw-guide-03-quick-start/) | `picoclaw onboard`, 첫 `agent` 실행 |
| 04 | [Docker Compose](/blog-repo/picoclaw-guide-04-docker-compose/) | `agent`/`gateway` 컨테이너 구성 |
| 05 | [설정 파일](/blog-repo/picoclaw-guide-05-config/) | `~/.picoclaw/config.json` 구조 해부 |
| 06 | [프로바이더/모델](/blog-repo/picoclaw-guide-06-providers-and-models/) | OpenRouter/Zhipu/Groq 등, 모델 선택 |
| 07 | [채팅 채널](/blog-repo/picoclaw-guide-07-channels/) | Telegram/Discord/LINE/Slack/OneBot |
| 08 | [워크스페이스/메모리](/blog-repo/picoclaw-guide-08-workspace-and-memory/) | workspace 레이아웃, IDENTITY/SOUL/MEMORY |
| 09 | [샌드박스/보안](/blog-repo/picoclaw-guide-09-sandbox-and-safety/) | `restrict_to_workspace`, exec 가드레일 |
| 10 | [운영/트러블슈팅](/blog-repo/picoclaw-guide-10-ops-and-troubleshooting/) | cron/heartbeat, 자주 겪는 이슈 |

