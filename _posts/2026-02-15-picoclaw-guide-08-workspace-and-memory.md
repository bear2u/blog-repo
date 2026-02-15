---
layout: post
title: "PicoClaw 가이드 (08) - 워크스페이스/메모리: 세션, 장기 기억, 스킬과 프롬프트 파일"
date: 2026-02-15
permalink: /picoclaw-guide-08-workspace-and-memory/
author: Sipeed
categories: [AI 에이전트, 개발 도구]
tags: ["PicoClaw", "Workspace", "Memory", "Skills", "Sessions"]
original_url: "https://github.com/sipeed/picoclaw#workspace-layout"
excerpt: "PicoClaw는 workspace 디렉토리에 세션/메모리/스킬/상태/스케줄 DB를 저장합니다. AGENT/IDENTITY/SOUL/USER/MEMORY 파일을 어떻게 활용할지 정리합니다."
---

## workspace는 “에이전트의 홈 디렉토리”다

README는 기본 워크스페이스를 `~/.picoclaw/workspace`로 안내하며, 여기에 다음이 들어간다고 설명합니다.

- `sessions/`: 대화 세션/히스토리
- `memory/`: 장기 메모리
- `state/`: 지속 상태(마지막 채널 등)
- `cron/`: 스케줄 작업 DB
- `skills/`: 커스텀 스킬
- `AGENTS.md`, `IDENTITY.md`, `SOUL.md`, `USER.md` 등 에이전트 “행동/정체성” 파일

즉, PicoClaw는 “설정(config.json)”과 “작업공간(workspace)”를 분리해 운영합니다.

---

## 기본 프롬프트 파일들(레포에 포함된 템플릿)

레포의 `workspace/` 디렉토리는 다음 템플릿을 제공합니다.

- `AGENT.md`: 에이전트 지침(간결/정확/도구 활용/명확한 설명 등)
- `IDENTITY.md`: 이름/목적/철학(초경량, 사용자 통제, 투명성 등)
- `SOUL.md`: 성격/가치
- `USER.md`: 사용자 선호/정보 템플릿
- `memory/MEMORY.md`: 장기 메모리 템플릿

실전에서는:

1. `USER.md`를 본인 스타일에 맞게 먼저 채우고
2. `AGENT.md`에 “항상 지킬 규칙”을 추가하고
3. `MEMORY.md`에 장기적으로 유지될 사실/선호만 적는

순서가 운영 난이도를 낮춥니다.

---

## skills 디렉토리

README의 workspace 레이아웃에는 `skills/`가 포함돼 있습니다.

- 반복 작업을 “스킬”로 묶어두면
- gateway/agent 어디서든 재사용이 쉬워집니다.

다음 장에서는 workspace를 중심으로 동작하는 “샌드박스/가드레일”을 정리합니다.

