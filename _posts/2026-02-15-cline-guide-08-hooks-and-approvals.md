---
layout: post
title: "Cline 완벽 가이드 (08) - Hooks & 승인 정책: 가드레일을 코드로 만들기"
date: 2026-02-15
permalink: /cline-guide-08-hooks-and-approvals/
author: Cline Bot Inc.
categories: [AI 코딩 에이전트, Cline]
tags: [Cline, Hooks, AutoApprove, Safety, Policies, YOLO]
original_url: "https://docs.cline.bot/features/hooks/index"
excerpt: "Hooks로 워크플로우에 체크포인트를 넣고, Auto Approve/YOLO 모드를 안전하게 다루는 기준을 정리합니다."
---

## Hooks: 자동화의 체크포인트

문서에서 **Hooks**는 “특정 이벤트에 자동으로 실행되는 로직”으로 설명됩니다.

핵심 효과는 3가지입니다.

1. 문제가 되는 작업을 **실행 전에 차단**할 수 있다
2. 도구 사용/성과를 **관측**하고 로그로 남길 수 있다
3. 필요한 추가 컨텍스트를 주입해 “AI의 판단”을 **형태**로 만들 수 있다

예를 들면:

- TypeScript 프로젝트에서 `.js` 파일 생성 시도를 막기
- 특정 폴더(예: `infra/`, `secrets/`) 변경을 차단하기
- 특정 커맨드(`rm -rf`, 위험한 `curl | sh`) 실행을 경고/차단하기

---

## Hook 위치: 전역 vs 프로젝트

문서 기준 Hooks는 보통 두 범주로 관리됩니다.

- 전역(개인): 모든 프로젝트에 적용
- 프로젝트: 해당 레포에만 적용(팀 공유 가능)

프로젝트 훅을 `.clinerules/hooks/`에 두면 버전 관리가 가능해져서,
팀 단위 가드레일을 만들기에 유리합니다.

---

## Auto Approve: “반복 승인 클릭”을 줄이되, 범위를 잘라라

**Auto Approve**는 도구 호출(파일 읽기/편집/커맨드 실행/브라우저/MCP)을 범주별로 자동 승인하는 기능입니다.

문서가 강조하는 실전 포인트:

- “Read all files / Edit all files” 같은 확장 옵션은 기본 토글이 켜져 있어야 의미가 있다
- 터미널 커맨드는 “안전(safe)” vs “승인 필요(requires approval)”로 나뉘고, Auto Approve는 그 플래그를 기준으로 동작한다

추천 기본값(문서 톤에 맞춘 최소주의):

- `Read project files`만 먼저 켜고
- 편집/커맨드/MCP/브라우저 자동 승인은 필요해졌을 때 범위를 좁게 켠다

---

## YOLO 모드: 빠르지만 위험하다

**YOLO mode**는 “모든 것을 자동 승인”합니다.

문서도 명확히 경고합니다.

- 파일/커맨드/브라우저/MCP/모드 전환까지 자동 승인
- 안전장치가 꺼진 상태라, 되돌림 전략이 없으면 사고가 나기 쉽다

실전 권장:

- 격리된 샌드박스/테스트 브랜치에서만
- 체크포인트(스냅샷/깃) 같은 롤백 수단을 먼저 준비
- 요청을 아주 구체적으로(모호한 지시 + 무제한 권한은 위험)

---

*다음 글에서는 서브에이전트/스킬/CLI로 “병렬 탐색”과 “자동화 파이프라인”을 구성하는 방법을 정리합니다.*

