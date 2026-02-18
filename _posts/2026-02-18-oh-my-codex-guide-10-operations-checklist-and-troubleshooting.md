---
layout: post
title: "oh-my-codex 가이드 (10) - 운영 체크리스트/트러블슈팅: 팀 실행 안정화 가이드"
date: 2026-02-18
permalink: /oh-my-codex-guide-10-operations-checklist-and-troubleshooting/
author: Yeachan Heo
categories: ['AI 코딩', '운영 가이드']
tags: [Troubleshooting, Team Mode, tmux, MCP, OMX]
original_url: "https://github.com/Yeachan-Heo/oh-my-codex"
excerpt: "OMX를 실제 프로젝트에 적용할 때 발생하는 대표 장애 유형과 점검 루틴을 정리합니다."
---

## 대표 실패 유형

1. tmux 미설치 또는 tmux 외부에서 team 실행
2. setup 미완료로 prompts/skills/MCP 누락
3. stale team state로 resume/monitor 불일치
4. hooks 플러그인 타임아웃/유효하지 않은 export
5. 과도한 `--madmax` 상시 사용으로 안전장치 우회

---

## 시작 전 체크리스트

```bash
omx version
omx doctor
omx doctor --team
omx status
```

- `doctor` fail이 1개라도 있으면 먼저 복구
- active mode가 남아 있으면 새 team 시작 전 정리

---

## 팀 모드 운영 루틴

- 시작: `omx team 3:executor "..."`
- 모니터: `omx team status <team-name>`
- 비정상 종료 시: `omx team shutdown <team-name>` 후 상태 검증

"in_progress 태스크가 남은 채 shutdown"은 의도적 중단 상황에서만 허용하는 것이 안전합니다.

---

## 설정/확장 변경 시 루틴

- `omx setup --force` 전후 `config.toml` diff 확인
- `.omx/hooks/*.mjs`는 `omx hooks validate` 선행
- 모델 라우팅 변경 후 작은 팀 태스크로 smoke test

---

## 안정적 도입 전략

```text
Phase 1: setup + doctor 안정화
Phase 2: 단일 세션에서 prompts/skills 적응
Phase 3: team 모드 소규모 도입(2~3 workers)
Phase 4: hooks/알림/MCP 운영 자동화 확장
```

OMX는 기능이 많은 만큼, 한 번에 전부 켜기보다 단계 도입이 실패 비용을 낮춥니다.

---

여기까지 10챕터를 완료했습니다. 이후에는 실제 팀 개발 워크플로우에 맞춰 AGENTS.md 정책과 모드별 모델 라우팅을 커스텀하면 효과가 커집니다.
