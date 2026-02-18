---
layout: post
title: "oh-my-codex 가이드 (03) - CLI 엔트리/런치 플래그: omx index.ts 읽기"
date: 2026-02-18
permalink: /oh-my-codex-guide-03-cli-entry-and-launch-flags/
author: Yeachan Heo
categories: ['AI 코딩', 'CLI 아키텍처']
tags: [CLI, Launch Flags, Reasoning, OMX, TypeScript]
original_url: "https://github.com/Yeachan-Heo/oh-my-codex"
excerpt: "bin/omx.js와 src/cli/index.ts를 기준으로 omx 명령이 어떻게 해석되고 Codex 실행으로 이어지는지 분석합니다."
---

## 실행 진입점

- `bin/omx.js`: `dist/cli/index.js` 존재 시 로드
- 빌드 산출물이 없으면 실행 중단(안내 메시지 출력)

즉 배포 패키지 관점에서 TypeScript 소스 직접 실행보다 컴파일 결과를 우선합니다.

---

## 명령 라우팅 구조

`src/cli/index.ts`의 `main(args)`는 `switch`로 커맨드를 분기합니다.

- `setup`, `doctor`, `team`, `status`, `cancel`, `reasoning`
- `tmux-hook`, `hooks`, `hud`
- 명시적 커맨드가 없으면 `launch` 취급

이 패턴 덕분에 `omx --xhigh --madmax` 같은 "플래그 단독 호출"이 바로 런치로 연결됩니다.

---

## 런치 정책

`resolveCodexLaunchPolicy()`는 `TMUX` 환경변수로 정책을 나눕니다.

- tmux 내부: HUD pane 관리 포함
- 일반 쉘: direct 실행

또한 `launchWithHud()`에서 preLaunch/run/postLaunch를 분리해 훅/세션 정리를 단계적으로 수행합니다.

---

## reasoning 서브커맨드

`omx reasoning <low|medium|high|xhigh>`는 `config.toml`의 `model_reasoning_effort`를 갱신합니다.

설정이 없을 때 현재 상태를 읽어주는 조회 모드도 제공해 운영 편의성이 좋습니다.

---

## 플래그 정리

README 기준 핵심 플래그는 다음입니다.

- `--yolo`
- `--high`, `--xhigh`
- `--madmax`
- `--dry-run`, `--force`, `--verbose`

다음 장에서 `omx setup`이 실제로 무엇을 설치/변경하는지 파일 단위로 해설합니다.
