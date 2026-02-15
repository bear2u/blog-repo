---
layout: post
title: "GitHub Agentic Workflows(gh-aw) 가이드 (07) - 실행/운영: 트리거, gh aw run, 인터랙티브 런"
date: 2026-02-15
permalink: /gh-aw-guide-07-running-workflows/
author: GitHub
categories: [개발 도구, GitHub]
tags: [gh-aw, Run, workflow_dispatch, InteractiveMode, Inputs, Operations]
original_url: "https://github.github.com/gh-aw/setup/cli/#run"
excerpt: "워크플로우 실행은 GitHub Actions 트리거와 gh aw run으로 이뤄집니다. 입력이 있는 workflow_dispatch와 gh aw run의 인터랙티브 모드를 중심으로 운영 흐름을 정리합니다."
---

## 실행은 결국 GitHub Actions다

gh-aw 워크플로우는 컴파일되면 일반적인 GitHub Actions로 돌아갑니다.

- 이벤트 트리거(`on:`)
- 수동 실행(`workflow_dispatch`)
- 스케줄

등의 방식으로 실행되고, 실행 결과는 Actions 탭에서 추적합니다.

---

## gh aw run: 즉시 실행

기본 실행:

```bash
gh aw run my-workflow
```

반복 실행이나 브랜치 지정 등도 상황에 따라 사용합니다(문서의 run 옵션 참고).

---

## 인터랙티브 런 모드: gh aw run (인자 없이)

문서에는 `gh aw run`을 인자 없이 호출하면 “가이드된 실행 UI”가 뜨는 흐름이 있습니다.

- 실행 가능한 워크플로우 목록을 보여주고
- 입력이 있는 경우(required/optional) 값을 수집하고
- 마지막에 확인 후 dispatch를 수행
- 그리고 “동일 실행을 재현할 수 있는 커맨드”를 출력합니다

이 모드는 “입력 값이 많거나, 사람이 매번 정확히 넣어야 하는 운영 워크플로우”에서 특히 편합니다.

---

## workflow_dispatch 입력값 설계 팁

입력값이 있는 워크플로우는 “사용자 입력이 곧 공격 표면”이 됩니다.

실전 팁:

- 입력값은 최소화하고, 가능한 선택지(choice)로 제한
- 입력 텍스트는 sanitization/검증(safe-inputs 등) 흐름과 함께 설계
- 중요한 실행은 roles/manual approval 같은 가드레일을 함께 사용

---

*다음 글에서는 gh-aw 문서가 제공하는 패턴/예시를 통해, 어떤 자동화를 어떤 형태로 만들면 좋은지 감을 잡아봅니다.*

