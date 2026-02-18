---
layout: post
title: "oh-my-codex 가이드 (04) - Setup 파이프라인: prompts/skills/config/AGENTS.md 생성"
date: 2026-02-18
permalink: /oh-my-codex-guide-04-setup-pipeline-and-generated-assets/
author: Yeachan Heo
categories: ['AI 코딩', '환경 자동화']
tags: [omx setup, AGENTS.md, Skills, Prompts, Config]
original_url: "https://github.com/Yeachan-Heo/oh-my-codex"
excerpt: "src/cli/setup.ts 7단계를 기준으로 OMX setup이 사용자 환경에 쓰는 파일과 보호 로직을 정리합니다."
---

## `setup.ts` 7단계

코드 기준 설치 순서는 아래와 같습니다.

1. 디렉토리 생성(`~/.codex`, `~/.agents/skills`, `.omx/*`)
2. 프롬프트 설치(`prompts/*.md`)
3. 스킬 설치(`skills/*/SKILL.md`)
4. `config.toml` 병합
5. 프로젝트 `AGENTS.md` 생성/갱신
6. 알림 훅 구성
7. HUD 기본 설정 생성

---

## AGENTS.md 보호 로직

활성 세션이 감지되면 `--force`라도 무조건 덮어쓰지 않고 경고 후 스킵합니다.

이 설계는 실행 중 오버레이 파일이 깨지는 위험을 줄이기 위한 안전장치입니다.

---

## config 병합 포인트

`mergeConfig()`는 사용자 설정을 최대한 보존하면서 OMX 블록만 갱신합니다.

- top-level: `notify`, `model_reasoning_effort`, `developer_instructions`
- `[features]`: `multi_agent`, `child_agents_md`
- `[mcp_servers.*]`, `[tui]`

---

## 설치 산출물 관점

README/DEMO 기준 핵심 산출물:

- `~/.codex/prompts/` (약 30개)
- `~/.agents/skills/` (약 40개)
- `~/.codex/config.toml` OMX 블록
- 프로젝트 루트 `AGENTS.md`
- `.omx/` 상태/로그/HUD 설정

---

## 실무 팁

`omx setup --force`는 편하지만, 팀 저장소의 기존 `AGENTS.md` 정책과 충돌할 수 있습니다.

프로젝트 정책이 이미 확립된 곳에서는 템플릿 차이(diff)를 먼저 검토한 뒤 적용하는 것이 안전합니다.

다음 장에서 팀 모드 런타임을 살펴봅니다.
