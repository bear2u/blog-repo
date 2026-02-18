---
layout: post
title: "oh-my-codex 가이드 (01) - 소개/포지셔닝: Codex CLI를 멀티-에이전트 시스템으로 확장하기"
date: 2026-02-18
permalink: /oh-my-codex-guide-01-intro/
author: Yeachan Heo
categories: ['AI 코딩', 'Codex CLI']
tags: [oh-my-codex, OMX, Codex CLI, Multi-Agent, Orchestration]
original_url: "https://github.com/Yeachan-Heo/oh-my-codex"
excerpt: "OMX가 단일 Codex 세션을 어떻게 구조화된 멀티-에이전트 실행 계층으로 확장하는지 전체 관점을 정리합니다."
---

## OMX의 한 줄 정의

README 기준 OMX는 "Codex CLI용 멀티-에이전트 오케스트레이션 레이어"입니다.

핵심은 Codex를 대체하는 것이 아니라, 아래 확장면을 추가하는 데 있습니다.

- 역할 프롬프트 (`/prompts:name`)
- 워크플로우 스킬 (`$name`)
- tmux 팀 실행 (`omx team`, `$team`)
- MCP 기반 상태/메모리 지속화

---

## 왜 필요한가

Codex CLI 단독 사용은 빠른 단일 작업에 강하지만, 대규모 작업에서는 구조가 필요합니다.

OMX는 이 간극을 다음으로 메웁니다.

1. 단계형 실행 파이프라인(`team-plan -> team-prd -> team-exec -> team-verify -> team-fix`)
2. `.omx/state/` 중심 모드 상태 추적
3. 장기 세션용 메모리/노트/로그 표면 제공

---

## 저장소 구조 관점

레포 핵심 폴더는 다음입니다.

- `src/cli`: 사용자 명령 엔트리
- `src/team`: 팀 런타임/태스크/워커 통신
- `src/mcp`: 상태/메모리/트레이스 서버
- `src/hooks`: 이벤트/플러그인 확장
- `skills/`, `prompts/`, `templates/AGENTS.md`: Codex 확장 자산

도구라기보다 "운영 프레임워크"에 가깝습니다.

---

## 이 시리즈에서 다룰 범위

이 가이드는 단순 설치 문서가 아니라, 코드 기준으로 다음을 해설합니다.

- CLI 명령 분기와 런치 정책
- setup이 실제로 쓰는 파일/설정
- 팀 모드 상태 모델과 MCP 계약
- hooks 플러그인 런타임
- 운영 체크리스트

다음 장에서 설치/초기 검증 경로를 정리합니다.
