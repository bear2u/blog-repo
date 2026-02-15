---
layout: post
title: "Cline 완벽 가이드 (02) - 설치와 시작: 설치, 모델 선택, 첫 작업"
date: 2026-02-15
permalink: /cline-guide-02-installation-and-first-run/
author: Cline Bot Inc.
categories: [AI 코딩 에이전트, Cline]
tags: [Cline, Installation, VSCode, JetBrains, ModelSelection, OpenAI, Anthropic]
original_url: "https://docs.cline.bot/getting-started/installing-cline"
excerpt: "Cline 설치부터 모델 선택, 첫 번째 프로젝트 생성까지 한 번에 진행할 수 있게 정리합니다."
---

## 1) 설치(IDE 별)

문서 기준으로 Cline은 여러 IDE에서 동작합니다.

- VS Code/Cursor/VSCodium/Windsurf 계열
- JetBrains IDEs

### VS Code 계열 설치(요약)

1. 확장(Extensions)에서 **Cline**을 검색해 설치
2. 사이드바의 Cline 아이콘으로 열거나, 커맨드 팔레트에서 `Cline: Open In New Tab`
3. “확장 실행 허용” 프롬프트가 뜨면 허용

### JetBrains 설치(요약)

1. Settings → Plugins → Marketplace에서 **Cline** 설치
2. IDE 재시작 후 Tool Window에서 Cline을 열기

---

## 2) 모델 선택: 가장 쉬운 시작은 “Cline Provider”

Cline은 여러 API 프로바이더를 지원합니다. 가장 쉬운 시작으로 문서가 추천하는 흐름은:

1. Cline 설정(gear 아이콘) 열기
2. API Provider를 **Cline**으로 선택
3. 모델 선택(예: Claude Sonnet 계열, DeepSeek 계열 등)

대안으로는 OpenAI Codex 로그인(키 없이 OAuth) 같은 선택지도 안내합니다.

핵심은 “모델은 언제든 바꿀 수 있다”는 점입니다.

---

## 3) 첫 작업: 한 파일 웹페이지 만들기

문서의 “첫 프로젝트” 튜토리얼은 작은 작업으로 루프를 체험하게 합니다.

예시 프롬프트(의도만 유지하고 간단히):

```text
단일 HTML 파일로 간단한 웹페이지를 만들어줘.
그라데이션 배경, 버튼 클릭 시 테마 변경, CSS/JS는 같은 파일에 포함.
```

Cline은 보통 다음 순서로 진행합니다.

1. 파일 생성 계획 제시
2. diff 형태로 변경 제안
3. 사용자가 승인하면 파일 생성
4. 완료 후 결과 확인 방법(파일 열기 등) 제시

이 때 중요한 감각은 “승인 기반”입니다.
자동 실행이 아니라, 사용자가 통제권을 갖고 루프를 굴립니다.

---

## 4) 시작 직후 세팅 팁(짧게)

- Cline 패널을 사이드바/오른쪽에 두면, 파일 트리와 함께 보면서 작업 흐름을 추적하기 편합니다.
- 터미널 통합이 불안하면, 먼저 “Background Execution Mode”를 고려해보는 게 빠른 해결책이 될 수 있습니다(트러블슈팅 장에서 자세히 다룸).

---

*다음 글에서는 Cline의 핵심 사용법인 Plan & Act 모드를, “언제/어떻게” 전환해야 생산성이 올라가는지 기준으로 정리합니다.*

