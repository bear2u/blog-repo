---
layout: post
title: "Cline 완벽 가이드 (09) - 서브에이전트/스킬/CLI: 병렬 탐색과 자동화를 붙이기"
date: 2026-02-15
permalink: /cline-guide-09-subagents-skills-cli/
author: Cline Bot Inc.
categories: [AI 코딩 에이전트, Cline]
tags: [Cline, Subagents, Skills, CLI, Automation, Headless]
original_url: "https://docs.cline.bot/features/subagents"
excerpt: "서브에이전트로 병렬 탐색을 하고, Skills로 지식을 온디맨드로 로드하며, Cline CLI로 자동화/CI 파이프라인에 연결하는 방법을 정리합니다."
---

## Subagents: 메인 컨텍스트를 더럽히지 않는 병렬 탐색

문서에서 **Subagents**는 “읽기 전용 연구 에이전트”로 설명됩니다.

핵심 설계:

- 각 서브에이전트는 별도의 컨텍스트 윈도우/토큰 예산을 갖는다
- 파일 탐색/검색/읽기/읽기 전용 명령 실행까지만 허용된다
- 파일 편집, 브라우저, MCP 사용, 서브에이전트 중첩 생성은 금지된다

실전에서 유용한 순간:

- 처음 보는 레포에서 빠르게 구조를 파악해야 할 때
- 인증/DB/배포 등 서로 다른 축을 동시에 조사해야 할 때
- “어디를 먼저 읽을지”를 결정하기 위한 넓은 탐색이 필요할 때

---

## Skills: ‘항상’이 아니라 ‘필요할 때만’ 지식을 로드

문서에서 **Skills**는 온디맨드 지식 묶음입니다.

- Rules는 항상 활성(또는 조건부 활성)
- Skills는 요청이 트리거될 때만 로드

스킬은 보통 `SKILL.md`(YAML 메타 + 본문 지침)를 가진 디렉토리로 구성되고,
추가 문서/템플릿/스크립트도 같이 번들링할 수 있습니다.

이 구조는 “컨텍스트를 아끼면서도, 도메인 지식은 깊게” 넣는 방향에 맞습니다.

---

## Cline CLI: 터미널에서 쓰는 Cline(대화형 + 자동화)

문서는 CLI를 크게 두 모드로 나눕니다.

1. **Interactive Mode**
   - `cline` 또는 TTY 환경에서 대화형 UI로 협업
   - Plan/Act 전환, 슬래시 커맨드, 파일 멘션(@) 등 지원
2. **Headless Mode**
   - `-y/--yolo`, `--json`, 파이프 입력 등에서 비대화형 실행
   - 자동화/CI/CD에 적합

예(개념):

```bash
# 자동 실행(주의: YOLO)
cline -y "Run tests and fix any failures"

# diff를 파이프로 넘겨 리뷰 자동화
git diff | cline -y "Review these changes for bugs and security issues"
```

CLI는 “에이전트 기반 자동화”를 셸 파이프라인에 자연스럽게 연결할 수 있는 형태로 설계되어 있습니다.

---

*다음 글에서는 터미널 통합, 네트워크/프록시, 히스토리 복구 같은 실전 이슈를 트러블슈팅 관점에서 정리합니다.*

