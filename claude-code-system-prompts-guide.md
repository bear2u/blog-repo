---
layout: page
title: Claude Code System Prompts 가이드
permalink: /claude-code-system-prompts-guide/
icon: fas fa-robot
---

# Claude Code System Prompts 완벽 가이드

> **Claude Code의 “진짜 시스템 프롬프트”는 하나가 아니라, 110+ 조각들입니다.**

이 시리즈는 `Piebald-AI/claude-code-system-prompts` 레포를 기반으로, Claude Code가 내부적으로 사용하는 **System Prompt / Tool Description / System Reminder / Agent Prompt**를 “어떻게 쪼개져 있고, 어떤 이벤트에서 주입되며, 버전 업에서 무엇이 바뀌는지”를 한국어로 구조화해 정리합니다.

- 원문 저장소: https://github.com/Piebald-AI/claude-code-system-prompts
- 기준 버전(원문 README): Claude Code `v2.1.42` (2026-02-13)
- 참고 도구: `tweakcc`(프롬프트 패치), `CHANGELOG.md`(98개 버전 변화 추적)

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/claude-code-system-prompts-guide-01-intro/) | 왜 “시스템 프롬프트가 여러 개”인가 |
| 02 | [프롬프트 카탈로그 구조](/blog-repo/claude-code-system-prompts-guide-02-taxonomy-and-metadata/) | 파일 네이밍, 메타데이터, 변수(보간) |
| 03 | [추출/업데이트 파이프라인](/blog-repo/claude-code-system-prompts-guide-03-extraction-pipeline/) | `tools/updatePrompts.js`로 README/토큰 카운트 갱신 |
| 04 | [메인 System Prompt 해부](/blog-repo/claude-code-system-prompts-guide-04-system-prompt-anatomy/) | “정체성/톤/안전/학습모드”가 어떻게 분리되는가 |
| 05 | [Builtin Tool Description](/blog-repo/claude-code-system-prompts-guide-05-builtin-tool-descriptions/) | Bash/Read/Edit/Write/Task 등 도구 설명이 UX를 만든다 |
| 06 | [System Reminders](/blog-repo/claude-code-system-prompts-guide-06-system-reminders/) | Plan Mode, 파일 이벤트, 훅 이벤트가 만드는 숨은 규칙 |
| 07 | [Agent Prompts](/blog-repo/claude-code-system-prompts-guide-07-agent-prompts/) | Explore/Plan/Task 서브에이전트의 역할 분리 |
| 08 | [CHANGELOG와 커스터마이징](/blog-repo/claude-code-system-prompts-guide-08-changelog-and-customization/) | 버전 변화 읽기, `tweakcc`로 로컬 패치하기 |

---

## 이 시리즈를 어떻게 읽으면 좋나

- Claude Code를 쓰고 있다면: 5~7장을 먼저 읽고 “도구/모드/서브에이전트가 내 대화 품질에 어떤 영향을 주는지”를 빠르게 감 잡는 편이 효율적입니다.
- Claude Code를 커스터마이징하고 싶다면: 2~3장 → 8장 순서가 좋습니다.
- “왜 이런 동작을 하지?” 디버깅 관점이라면: 6장의 System Reminders가 가장 큰 힌트를 주는 경우가 많습니다.

