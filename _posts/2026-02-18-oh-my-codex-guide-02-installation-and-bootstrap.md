---
layout: post
title: "oh-my-codex 가이드 (02) - 설치/부트스트랩: omx setup과 doctor로 시작하기"
date: 2026-02-18
permalink: /oh-my-codex-guide-02-installation-and-bootstrap/
author: Yeachan Heo
categories: ['AI 코딩', '설치 가이드']
tags: [OMX Setup, omx doctor, Node.js, Codex CLI, Bootstrap]
original_url: "https://github.com/Yeachan-Heo/oh-my-codex"
excerpt: "OMX 요구사항과 설치 순서, setup/doctor 명령이 검증하는 항목을 기준으로 빠른 초기 세팅 경로를 정리합니다."
---

## 요구사항

README 기준 필수 조건은 아래와 같습니다.

- macOS/Linux(Windows는 WSL2 권장)
- Node.js >= 20
- `@openai/codex` 전역 설치
- Codex 인증 완료

---

## 기본 설치 순서

```bash
npm install -g oh-my-codex
omx setup
omx doctor
```

`DEMO.md` 기준 `omx setup`은 7단계로 프롬프트/스킬/설정/AGENTS.md/HUD를 구성합니다.

---

## 권장 런치 프로필

```bash
omx --xhigh --madmax
```

- `--xhigh`: 추론 강도 상향
- `--madmax`: Codex bypass 플래그 매핑(신뢰 환경 전용)

보안/컴플라이언스가 중요한 환경에서는 `--madmax`를 기본값으로 두지 않는 편이 안전합니다.

---

## `omx doctor`가 보는 것

`src/cli/doctor.ts` 기준 주요 점검은 다음입니다.

1. Codex CLI 설치 여부
2. Node 버전
3. `~/.codex` 및 `config.toml`
4. prompts/skills 설치 수량
5. 프로젝트 `AGENTS.md`
6. `.omx/state` 및 MCP 구성

즉 "실행 전 점검 자동화"가 설치 흐름에 포함됩니다.

---

## 팀 진단 모드

`omx doctor --team`은 별도 진단 경로를 사용해 팀 런타임 문제를 찾습니다.

- stale heartbeat
- shutdown ack 지연
- orphan tmux session
- resume blocker

다음 장에서는 실제 CLI 엔트리와 명령 분기 구조를 봅니다.
