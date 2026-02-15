---
layout: post
title: "Claude Code System Prompts 가이드 (07) - Agent Prompts: Explore/Plan/Task 서브에이전트의 역할 분리"
date: 2026-02-15
permalink: /claude-code-system-prompts-guide-07-agent-prompts/
author: Piebald AI
categories: [AI 코딩 에이전트, Claude Code]
tags: [Claude Code, Agent Prompts, Explore, Plan Mode, Task Tool, Subagents]
original_url: "https://github.com/Piebald-AI/claude-code-system-prompts"
excerpt: "Explore/Plan/Task 등 서브에이전트 프롬프트가 어떻게 역할·권한·출력 기대치를 분리하는지 정리합니다."
---

## Agent Prompt란 무엇인가

`system-prompts/agent-prompt-*.md`는 Claude Code가 내부적으로 띄우는:

- 서브에이전트(Explore/Plan 등)
- 특수 목적 유틸리티(요약/분석/문서 생성 등)
- 슬래시 커맨드(/review-pr, /security-review 등)

에 부여되는 시스템 프롬프트를 모아둔 것입니다.

이 파일들을 읽으면, Claude Code가 “한 모델이 다 한다”가 아니라:

```text
메인 조율자 + 역할 특화 에이전트(권한/규칙 다름)
```

로 설계돼 있다는 걸 확인할 수 있습니다.

---

## Explore: read-only 탐색 전문 에이전트

`agent-prompt-explore.md`는 매우 강하게 “읽기 전용”을 선언합니다.

금지 목록이 길게 나오는 이유는 단순합니다.

- 탐색 에이전트가 실수로 상태를 바꾸면(파일 생성/수정/삭제)
- 사용자는 “탐색 결과”를 기대했는데 “변경”이 섞여 사고가 납니다.

그래서 Explore는 다음을 강제합니다.

- 파일 생성/수정/삭제 금지
- 리다이렉트(`>`, `>>`)나 heredoc으로 파일 쓰기 금지
- 상태 변경 명령 금지(`git add/commit`, `npm install` 등)
- 가능한 한 Glob/Grep/Read 같은 전용 툴 사용

이 설계는 “탐색”과 “실행/편집”을 분리해서, 역할별 안전을 확보하는 방식입니다.

---

## Task tool 에이전트: ‘서브에이전트 기본 인격’

`agent-prompt-task-tool.md`는 Task 도구로 생성되는 서브에이전트에게 주는 기본 프롬프트입니다.

여기서 흥미로운 대목:

- “딱 요청받은 것만 하라. 더도 말고 덜도 말고.”
- 파일 경로는 절대 경로로 반환하라
- 이모지 사용을 피하라
- 문서 파일을 “자기 판단으로 만들지 말라” 같은 제약이 들어 있음

즉, Task 에이전트는:

- 빠르게 일을 끝내고
- 결과를 자세히 보고(writeup)하며
- 불필요한 산출물을 만들지 않는

방향으로 튜닝돼 있습니다.

---

## Plan 모드/Plan 서브에이전트: “계획 프로토콜”을 프롬프트로 강제

이 레포에는 “Plan” 관련 prompt가 여러 형태로 존재합니다.

- System Reminder로서의 Plan Mode 활성화 규칙
- Agent Prompt로서의 Plan 서브에이전트(강화 버전 등)

이 둘을 같이 보면 Plan은 단순히 “계획을 잘 써라”가 아니라:

- 탐색을 하고
- 발견을 plan 파일에 누적하고
- 사용자 질문을 배치로 묻고
- 충분히 수렴하면 승인 툴로 넘어가는

작업 흐름이 프로토콜처럼 고정돼 있다는 걸 알 수 있습니다.

---

## 왜 이렇게 역할을 나눴을까

이 레포의 Agent Prompt 분리는, Claude Code의 제품 목표를 드러냅니다.

- 메인 모델은 사용자와 “대화/의사결정/통합”에 집중
- Explore 같은 에이전트는 “탐색”을 안전하게, 빠르게 수행
- 보안 리뷰/PR 리뷰 같은 에이전트는 “특수 체크리스트”를 적용

즉, “프롬프트를 잘 쓰면 된다”가 아니라:

```text
역할 분리 + 상태 제약 + 도구 계약
```

조합으로 품질을 만드는 방향입니다.

---

## 다음 장 예고: CHANGELOG와 커스터마이징

여기까지 보면, 이 레포가 왜 “카탈로그”인지 이해가 됩니다.

마지막 장에서는:

- `CHANGELOG.md`에서 버전별 변화를 어떻게 읽는지
- 무엇이 자주 바뀌고(특히 Tool/Reminder/Plan 관련)
- 로컬에서 특정 조각을 바꾸고 싶으면 어떤 접근이 현실적인지(`tweakcc`)

를 정리합니다.

---

*다음 글에서는 CHANGELOG의 실제 항목을 예로 들어 변화 유형을 분류하고, tweakcc로 로컬 설치를 패치하는 워크플로우를 설명합니다.*

