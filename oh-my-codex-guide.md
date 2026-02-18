---
layout: page
title: oh-my-codex 가이드
permalink: /oh-my-codex-guide/
icon: fas fa-robot
---

# oh-my-codex 완벽 가이드

> **Codex CLI 위에 멀티-에이전트 오케스트레이션, 팀 실행, MCP 상태 관리 레이어를 올리는 OMX 실전 해설**

**oh-my-codex(OMX)**는 OpenAI Codex CLI를 확장해 역할 프롬프트, 스킬, 팀 모드, 훅 플러그인, HUD, MCP 상태/메모리 서버를 통합하는 오케스트레이션 도구입니다.

- 원문 저장소: https://github.com/Yeachan-Heo/oh-my-codex
- npm 패키지: https://www.npmjs.com/package/oh-my-codex

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개/포지셔닝](/blog-repo/oh-my-codex-guide-01-intro/) | OMX가 Codex CLI에 추가하는 핵심 레이어 |
| 02 | [설치/부트스트랩](/blog-repo/oh-my-codex-guide-02-installation-and-bootstrap/) | `omx setup`, `omx doctor`로 초기 환경 구성 |
| 03 | [CLI 엔트리/런치 플래그](/blog-repo/oh-my-codex-guide-03-cli-entry-and-launch-flags/) | `omx` 명령 분기, reasoning/launch 정책 |
| 04 | [Setup 파이프라인](/blog-repo/oh-my-codex-guide-04-setup-pipeline-and-generated-assets/) | prompts/skills/config/AGENTS.md 생성 흐름 |
| 05 | [Team 모드 런타임](/blog-repo/oh-my-codex-guide-05-team-mode-runtime-and-tmux/) | tmux 기반 다중 워커 오케스트레이션 |
| 06 | [Team 상태/MCP 연동](/blog-repo/oh-my-codex-guide-06-team-state-and-mcp-ops/) | mailbox/task/state 툴과 운영 모델 |
| 07 | [Hooks 확장 런타임](/blog-repo/oh-my-codex-guide-07-hooks-extension-runtime/) | `.omx/hooks/*.mjs` 플러그인 이벤트 파이프라인 |
| 08 | [Config/모델 라우팅](/blog-repo/oh-my-codex-guide-08-config-model-routing-and-reasoning/) | `config.toml` 병합, 모델/추론 레벨 제어 |
| 09 | [HUD/알림/세션 라이프사이클](/blog-repo/oh-my-codex-guide-09-hud-notifications-and-session-lifecycle/) | 관측성, 알림 채널, 세션 제어 |
| 10 | [운영 체크리스트/트러블슈팅](/blog-repo/oh-my-codex-guide-10-operations-checklist-and-troubleshooting/) | 실서비스 적용 시 위험 포인트와 대응 |

---

## 핵심 특징

- **Codex 네이티브 확장**: 포크가 아니라 프롬프트/스킬/AGENTS.md 주입 방식
- **팀 오케스트레이션**: tmux 워커 + 태스크/메일박스 + 팀 상태 모델
- **MCP 기반 상태 계층**: `omx_state`, `omx_memory`, `omx_code_intel`, `omx_trace`
- **훅 플러그인 확장**: `session-start`, `turn-complete` 등 이벤트 기반 자동화
- **운영 툴링**: `doctor`, `status`, `cancel`, HUD, 알림으로 런타임 가시성 강화

---

## 빠른 시작

```bash
npm install -g oh-my-codex
omx setup
omx doctor
```

```bash
# 권장 신뢰 환경 런치 프로필
omx --xhigh --madmax
```

다음 글부터 내부 구조를 코드 기준으로 순서대로 분석합니다.
