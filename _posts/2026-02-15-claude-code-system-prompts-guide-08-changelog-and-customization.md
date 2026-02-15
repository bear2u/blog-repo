---
layout: post
title: "Claude Code System Prompts 가이드 (08) - CHANGELOG와 커스터마이징: 버전 변화 읽기와 tweakcc"
date: 2026-02-15
permalink: /claude-code-system-prompts-guide-08-changelog-and-customization/
author: Piebald AI
categories: [AI 코딩 에이전트, Claude Code]
tags: [Claude Code, Changelog, Versions, tweakcc, Customization, Diff]
original_url: "https://github.com/Piebald-AI/claude-code-system-prompts"
excerpt: "CHANGELOG로 프롬프트 변화를 추적하는 법과, 로컬 Claude Code 설치에서 특정 프롬프트 조각을 안전하게 수정하는 접근(tweakcc)을 정리합니다."
---

## CHANGELOG.md는 “제품의 숨은 릴리즈 노트”다

이 레포의 `CHANGELOG.md`는 단순한 변경 기록이 아니라:

- 어떤 프롬프트 조각이 추가/삭제/변경됐는지
- 토큰이 얼마나 늘거나 줄었는지
- 변화가 어떤 기능(Plan Mode, Tool policy, Skills 등)과 연결되는지

를 버전별로 요약합니다.

Claude Code는 소스가 공개되지 않기 때문에, 실제로는 CHANGELOG가 “시스템 프롬프트 관점의 릴리즈 노트” 역할을 합니다.

---

## CHANGELOG를 읽는 요령

변화는 대체로 아래 타입으로 나타납니다.

1. **NEW**: 새로운 프롬프트 조각이 등장
2. **REMOVED**: 특정 조각이 사라짐
3. **MODIFIED**: 기존 조각의 문구/구조가 바뀜(토큰 증감)

그리고 가장 유용한 질문은 이겁니다.

- “이 변화가 실제 사용자 체감 동작을 바꾸는가?”

예:

- Tool usage policy가 바뀌면: 도구 선택/병렬 호출/금지 패턴이 바뀔 수 있음
- Plan mode reminder가 바뀌면: 계획 루프/질문 방식/서브에이전트 사용 조건이 바뀔 수 있음
- Bash tool description이 바뀌면: 안전/권장 커맨드 패턴이 바뀔 수 있음

---

## 실제 예: 2.1.42(2026-02-13) 항목에서 볼 수 있는 것

README 기준 이 레포는 Claude Code `v2.1.42` (릴리즈 날짜: 2026-02-13) 시점을 포함합니다.

CHANGELOG의 2.1.42에서는 예를 들어:

- 어떤 에이전트 프롬프트가 삭제되거나(REMOVED)
- WebSearch 관련 변수/예시가 단순화되는 식의 변경(MODIFIED)

이 기록은 “새 기능이 추가됐다”라기보다:

- 내부 프롬프트 구조가 계속 다듬어지고
- 변수/조건부 섹션이 정리되며
- 토큰 비용이 최적화되는

제품 성숙 과정을 보여줍니다.

---

## “이 레포를 수정하면 Claude Code가 바뀌나?”

아닙니다.

`CLAUDE.md`에서 강조하듯, 이 레포의 파일은 “추출된 참조 자료(reference material)”입니다.

- 여기 파일을 편집해도 Claude Code 동작은 바뀌지 않습니다.
- 실제 동작을 바꾸려면 Claude Code 설치본(바이너리/npx 패키지)을 패치해야 합니다.

---

## tweakcc: ‘프롬프트 조각 단위 패치’를 위한 현실적인 접근

README에서 추천하는 방식은 `tweakcc`입니다.

개념적으로는:

1. Claude Code 설치본에 들어 있는 “정확히 같은 문자열 조각”을 찾고
2. 특정 조각을 마크다운 파일로 오버라이드/패치하고
3. Claude Code가 업데이트되면, 내 변경과 upstream 변경의 충돌을 관리한다

는 워크플로우를 제공합니다.

왜 이런 도구가 필요한가?

- Claude Code의 프롬프트는 110+ 조각으로 쪼개져 있고
- 번들 JS에서 문자열 위치가 계속 바뀌기 때문에
- “수동으로 패치”는 거의 유지보수가 불가능합니다.

이 레포는 `tweakcc`가 패치해야 하는 “원본 문자열”을 관측 가능한 형태로 제공해 준다는 점에서도 의미가 있습니다.

---

## 커스터마이징 가이드(현실적인 기준)

로컬에서 프롬프트를 바꾸고 싶다면, 아래를 권합니다.

1. “큰 조각”부터 손대지 말고, 최소 변경으로 시작
  - 예: Tone/Style의 일부 문구, Tool usage policy의 일부 제한 등
2. CHANGELOG에서 해당 조각이 “자주 바뀌는지” 먼저 확인
  - 자주 바뀌는 조각은 충돌 관리 비용이 커집니다
3. 업데이트 시엔 “diff 기반”으로 유지
  - upstream 변경을 수용하고, 내 변경은 최소로 유지

그리고 가장 중요한 전제:

- 프롬프트 변경은 곧 “제품 동작 변경”입니다.
- 팀/조직에서 쓴다면, 변경 이력과 근거를 남기는 게 좋습니다.

---

## 마무리: 이 레포를 ‘디버깅 도구’로 쓰기

이 레포는 단순히 “재미로 보는 시스템 프롬프트” 모음이 아니라:

- Claude Code의 이상 행동(?)을 재현/설명하고
- 버전 업으로 생긴 변화의 근거를 찾고
- 로컬 커스터마이징의 충돌을 줄이는

디버깅 도구에 가깝습니다.

Claude Code의 프롬프트 변경을 Anthropic에 요청하고 싶다면:

- `anthropics/claude-code` 이슈 트래커를 통해 제안하는 흐름이 안내돼 있습니다.

---

*시리즈를 읽고 “내 환경에서 어떤 조각을 바꾸면 효과가 있을지” 더 구체적으로 정리하고 싶다면, CHANGELOG에서 자주 변하는 파일을 먼저 추려보는 것부터 추천합니다.*

