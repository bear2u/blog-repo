---
layout: post
title: "GitHub Agentic Workflows(gh-aw) 가이드 (05) - Safe Outputs/보안: 읽기 전용 에이전트 + 쓰기 분리"
date: 2026-02-15
permalink: /gh-aw-guide-05-safe-outputs-and-security/
author: GitHub
categories: [개발 도구, GitHub]
tags: [gh-aw, SafeOutputs, Security, LeastPrivilege, ThreatDetection, StrictMode]
original_url: "https://github.github.com/gh-aw/reference/safe-outputs/"
excerpt: "gh-aw의 핵심은 safe-outputs로 ‘쓰기 권한’을 에이전트에서 분리하는 것입니다. 이 장에서는 safe-outputs의 목적, 권한 설계, strict 모드, threat detection을 요약합니다."
---

## safe-outputs가 필요한 이유

gh-aw에서 에이전트 워크플로우를 안전하게 운영하려면,
“LLM이 직접 `issues: write`로 이슈를 만들게 하는 방식”을 피하는 게 기본입니다.

**safe-outputs**는 다음을 가능하게 합니다.

- 에이전트 단계는 최소 권한(보통 read-only)
- 에이전트는 “무엇을 하고 싶다”를 구조화된 출력으로 요청
- 별도의 후처리 job이 필요한 write 권한으로 실제 GitHub API 작업을 수행

이 방식은 최소 권한, 감사 가능성, 프롬프트 인젝션 방어에 유리합니다.

---

## safe-outputs 선언 예시(개념)

{% raw %}
```yaml
permissions:
  contents: read

safe-outputs:
  create-issue:
    labels: [automation]
```
{% endraw %}

여기서 중요한 것은:

- safe-outputs가 있다고 해서 에이전트 job이 write 권한을 갖는 게 아니라
- safe-output 전용 job이 별도 권한으로 실행되는 구조라는 점입니다.

---

## strict 모드: “좋은 보안 관행”을 컴파일 단계에서 강제

문서는 strict 모드를 “프로덕션 워크플로우의 기본값”처럼 다룹니다.

대표적으로 strict에서 요구/권장되는 것들:

- write 권한을 직접 주지 말고 safe-outputs 사용
- 네트워크 접근을 명시(`network`)
- 허술한 와일드카드/비핀ning 액션 등을 제한
- deprecated 필드 사용을 막고 자동 fix/upgrade 흐름을 제공

즉, 보안이 “운영 정책”이 아니라 “컴파일 검증”으로 내려오는 설계입니다.

---

## threat detection(선택): 출력과 쓰기 사이의 보안 게이트

문서의 compilation process에서도 강조되듯,
safe-outputs 앞에 detection job을 두는 패턴이 있습니다.

목적:

- 에이전트 출력이 안전한지(인젝션/악성 지시/데이터 유출 시도 등)
- 쓰기 job을 돌리기 전에 한 번 더 검사

이 레이어는 “job-level gating”으로 분리되어야 의미가 커집니다.

---

## 운영 팁: 안전 설계는 ‘권한 + 네트워크 + 도구’의 곱

safe-outputs만 써도 충분하지 않은 경우가 있습니다.

- 도구(toolsets)가 과하게 열려 있으면 읽기만으로도 민감 정보에 접근 가능
- 네트워크 허용이 넓으면 외부로 데이터가 나갈 수 있음
- 트리거(예: 이슈 코멘트)에 대한 입력 sanitization이 약하면 인젝션이 쉬움

그래서 gh-aw에서는 보통 아래를 한 세트로 봅니다.

1. `permissions`(최소)
2. `tools`(최소)
3. `network`(명시)
4. `safe-outputs`(쓰기 분리)
5. `--strict`/검증 스캐너(컴파일 단계)

---

*다음 글에서는 `.md`를 `.lock.yml`로 만드는 컴파일/락파일 운영(`gh aw compile`)을 정리합니다.*

