---
layout: post
title: "Cline 완벽 가이드 (07) - Rules & Workflows: 정책과 자동화를 파일로 고정하기"
date: 2026-02-15
permalink: /cline-guide-07-rules-and-workflows/
author: Cline Bot Inc.
categories: [AI 코딩 에이전트, Cline]
tags: [Cline, clinerules, AGENTS, Workflows, Automation, BestPractices]
original_url: "https://docs.cline.bot/features/cline-rules"
excerpt: ".clinerules/AGENTS.md로 행동 규칙을 고정하고, Workflows(.md)로 반복 작업을 자동화하는 방법을 정리합니다."
---

## Rules: “항상 적용되는” 프로젝트 가드레일

**Cline Rules**는 프로젝트에 대한 지속적인 지침을 파일로 고정해, 매번 채팅으로 반복하지 않게 합니다.

문서의 로딩 순서는 다음을 제시합니다.

1. `.clinerules/` 폴더 내부의 모든 `.md`
2. 단일 `.clinerules` 파일
3. `AGENTS.md` (agents.md 표준, 재귀 탐색)

실전적으로는 `.clinerules/` 폴더를 권장합니다.

```text
your-project/
├── .clinerules/
│   ├── 01-coding-standards.md
│   ├── 02-architecture.md
│   └── 03-testing.md
└── ...
```

---

## 조건부 룰: “관련 파일에만” 룰을 붙이기

룰은 YAML front matter로 특정 경로에만 활성화되도록 제한할 수 있습니다.

예(개념):

```yaml
---
paths:
  - "src/components/**"
---
```

이 방식은 컨텍스트를 낭비하지 않고 “필요할 때만” 규칙을 주입하는 데 유용합니다.

---

## Workflows: “필요할 때 호출하는” 자동화 스크립트

Workflows는 `.md` 파일로 정의하는 “단계별 작업 절차”입니다.

- 채팅에서 `/파일이름.md`로 호출
- 반복 작업(릴리즈, PR 리뷰, 배포 등)을 일관되게 수행
- CLI 도구, Cline 도구, MCP 도구를 함께 엮을 수 있음

문서가 제시하는 저장 위치:

- 프로젝트 전용: `.clinerules/workflows/`
- 전역(모든 프로젝트): `~/Documents/Cline/Workflows/` (플랫폼별 경로 상이)

---

## Rules vs Workflows: 무엇을 어디에 넣을까

문서의 구분을 실전 기준으로 바꾸면:

- **Rules**: “항상 지켜야 하는 정책”
  - 예: TypeScript 강제, 테스트 파일 위치, 에러 처리 규칙, 금지 API
- **Workflows**: “반복되는 절차”
  - 예: PR 리뷰 루틴, 릴리즈 체크리스트, 배포 절차, 일일 변경 로그 작성

이 분리를 잘 해두면, Cline은:

- 평소에는 Rules로 일관된 행동을 유지하고
- 특정 작업 때는 Workflows로 빠르게 자동화

하는 구조를 갖게 됩니다.

---

*다음 글에서는 Hooks와 Auto Approve로, “자동화는 빠르게, 위험한 행동은 안전하게” 만드는 가드레일을 다룹니다.*

