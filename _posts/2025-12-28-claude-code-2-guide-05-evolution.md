---
layout: post
title: "Claude Code 2.0 가이드 (5) - Claude Code의 진화와 QoL 개선"
date: 2025-12-28
permalink: /claude-code-2-guide-05-evolution/
author: Sankalp
categories: [AI 코딩 에이전트, Claude Code]
tags: [Claude Code, AI, QoL, 기능, 업데이트]
original_url: "https://sankalp.bearblog.dev/my-experience-with-claude-code-20-and-how-to-get-better-at-using-coding-agents/"
excerpt: "Claude Code 2.0의 다양한 QoL(삶의 질) 개선 사항들을 살펴봅니다. 구문 강조, 체크포인팅, 프롬프트 제안 등 유용한 기능들을 소개합니다."
---

## Claude Code의 진화

Claude Code는 7월 이후로 많은 AI 기능과 삶의 질(QoL) 개선이 있었습니다. 유용하다고 느낀 것들을 살펴보겠습니다. 모든 변경 사항은 [Changelog](https://docs.anthropic.com/en/docs/claude-code/changelog)에서 볼 수 있습니다.

---

## Claude Code 2.0 QoL 개선 사항

### 1. 구문 강조 (v2.0.71)

구문 강조가 최근 2.0.71에 추가되었습니다. 저는 80%의 시간을 Claude Code CLI에서 보내기 때문에 이 변화가 저에게 기쁨이었습니다. 저는 대부분의 것을 Claude Code에서 한 번 검토하는 것을 좋아합니다. Opus 4.5가 정말 좋다는 것 외에도, 이 기능이 코드를 검토하기 위해 Cursor를 전혀 열지 않게 된 큰 기여자였습니다.

### 2. 팁 표시

Claude가 생각하는 동안 보여지는 팁들에서 많이 배웠습니다.

### 3. 피드백 UI

피드백을 요청하는 이 방식은 꽤 우아합니다. 한동안 있었습니다. 가끔 팝업되고 숫자 키(1: Bad, 2: Fine, 3: Good)로 빠르게 응답하거나 0으로 닫을 수 있습니다. 비침입적인 특성이 좋습니다.

### 4. 질문 모드 옵션

또 다른 좋아하는 점은 세 번째 옵션 - "Type here to tell Claude what to do differently"입니다. 재미있는 사실: 이것들은 실제로 모델을 위한 프롬프트이고 그 출력은 다른 도구 호출에 의해 파싱되어 이 방식으로 보여집니다.

### 5. Ultrathink

어려운 작업이나 Opus 4.5가 더 철저하기를 원할 때 ultrathink를 자주 사용합니다 - 예를 들어, 무언가를 설명받거나, 자체 변경 검토 등.

```bash
/ultrathink  # 복잡한 작업에 더 철저한 분석 요청
```

### 6. 생각 토글

생각을 켜고 끄는 Tab 토글이 좋은 기능이었습니다. 최근 Alt/Option + Tab으로 변경되었지만 Mac에서 작동하지 않는 버그가 있습니다. 어쨌든 CC는 settings.json에서 확인하면 기본적으로 thinking을 항상 true로 설정합니다.

### 7. /context

`/context`로 현재 컨텍스트 사용량을 볼 수 있습니다. 저는 이것을 꽤 자주 사용합니다. 복잡한 것을 빌드할 때 총 60%에 도달하면 handoff나 compact를 합니다.

```
컨텍스트 사용량 모니터링:
├── /context - 현재 사용량 확인
├── 60% 도달 시 - handoff 또는 compact 권장
└── 복잡한 작업 전 - 새 세션 시작 고려
```

### 8. /usage와 /stats

`/usage`로 사용량을, `/stats`로 통계를 볼 수 있습니다. 자주 사용하지는 않습니다.

---

## 체크포인팅 (Checkpointing)

`Esc + Esc` 또는 `/rewind` 옵션으로 이제 Cursor에서 했던 것처럼 특정 체크포인트로 돌아갈 수 있습니다. **코드와 대화 모두** 되돌릴 수 있습니다. 이것은 저에게 주요 기능 요청이었습니다.

```bash
Esc + Esc  # 빠른 되돌리기
/rewind    # 특정 체크포인트로 복원
```

---

## 프롬프트 관련 기능

### 프롬프트 제안 (v2.0.73)

프롬프트 제안이 최근 추가되었고 예측이 꽤 괜찮습니다. Claude Code는 이 시점에서 토큰 소비 기계입니다. 제가 본 가장 간단한 프롬프트일 것입니다.

### 프롬프트 히스토리 검색

`Ctrl + R`로 프롬프트를 검색할 수 있습니다(터미널 백서치와 유사). 2.0.74에 있습니다. 프로젝트 전체 대화를 검색할 수 있습니다. `Ctrl + R`을 반복해서 결과를 순환할 수 있습니다.

### 커서 순환

프롬프트의 시작/끝에 도달했을 때, 위/아래를 눌러 순환할 수 있습니다.

### 메시지 큐 네비게이션

이제 대기 중인 메시지와 이미지 첨부파일(2.0.73)을 탐색할 수 있습니다.

---

## 기타 개선 사항

### 퍼지 파일 검색

파일 제안이 3배 빨라지고 퍼지 검색을 지원합니다(2.0.72).

### LSP 지원

최근 추가되었습니다. 플러그인을 통해 접근합니다.

---

## 새로운 통합

새로운 통합도 있습니다:

- **Slack 통합**
- **Claude Web (베타)**
- **Claude Chrome 확장 프로그램**

이것들은 꽤 명백하므로 다루지 않겠습니다. 특히 Claude Web이 많은 분들께 흥미로울 것 같습니다(iOS/Android에서도 작업을 시작할 수 있으므로).

---

*다음 글에서는 명령어(Commands)와 서브 에이전트(Sub-agents)에 대해 심층적으로 다룹니다.*
