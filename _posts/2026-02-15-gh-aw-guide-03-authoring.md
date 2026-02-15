---
layout: post
title: "GitHub Agentic Workflows(gh-aw) 가이드 (03) - 워크플로우 작성: .md 포맷, frontmatter, 편집/리컴파일 기준"
date: 2026-02-15
permalink: /gh-aw-guide-03-authoring/
author: GitHub
categories: [개발 도구, GitHub]
tags: [gh-aw, Workflow, Frontmatter, Markdown, Compilation, GitHub Actions]
original_url: "https://github.github.com/gh-aw/guides/editing-workflows/"
excerpt: "gh-aw 워크플로우는 마크다운 본문과 YAML frontmatter로 구성됩니다. 어떤 변경이 즉시 반영되고(본문), 어떤 변경이 컴파일을 요구하는지(frontmatter) 기준을 잡습니다."
---

## 파일 구조: .md가 소스, .lock.yml이 실행물

gh-aw 워크플로우의 기본은 1:1 매핑입니다.

```text
.github/workflows/my-workflow.md        # 사람이 쓰는 소스(자연어 + frontmatter)
.github/workflows/my-workflow.lock.yml  # 컴파일 산출물(실행되는 Actions)
```

원칙:

- `.md`를 수정하면 “논리”가 바뀐다
- `.lock.yml`은 **직접 편집하지 않는다**
- frontmatter 변경은 컴파일로 반영한다

---

## .md 포맷: YAML frontmatter + 마크다운 본문

워크플로우는 대략 이런 형태입니다.

{% raw %}
```markdown
---
on:
  workflow_dispatch:
permissions:
  contents: read
tools:
  github:
    toolsets: [default]
safe-outputs:
  create-issue:
---

# Workflow Title

여기에 자연어로 절차/정책/출력 포맷을 적습니다.
```
{% endraw %}

frontmatter는 “실행 환경(트리거/권한/도구/네트워크/엔진/안전장치)”이고,
본문은 “에이전트가 해야 할 일(판단/절차/형식)”입니다.

---

## 편집 규칙: 본문 vs frontmatter

문서의 핵심 요약:

- **마크다운 본문**은 런타임에 로드되므로, 수정 후 즉시 효과가 날 수 있다(다음 실행부터).
- **frontmatter**는 컴파일 결과에 포함되므로, 수정 후 `gh aw compile`이 필요하다.

컴파일이 필요한 것들(대표):

- `on`, `permissions`
- `tools`, `engine`, `network`
- `safe-outputs`, `safe-inputs`, `imports`
- `strict`, `roles`, `jobs` 등

반대로 “지침/템플릿/출력 포맷” 같은 본문 변경은 빠르게 반복하기 좋습니다.

---

## 표현식과 안전

GitHub Actions 표현식(예: {% raw %}`${{ ... }}`{% endraw %})은 본문에 넣을 수 있지만,
보안/검증 정책에 의해 제한될 수 있습니다.

실전 팁:

- 사용자 입력을 그대로 표현식으로 쓰기보다, 문서가 안내하는 “sanitize된 출력”을 활용
- 모호한 데이터 바인딩을 줄이고, 안전한 입력/출력(safe-inputs/safe-outputs)을 중심으로 설계

---

*다음 글에서는 frontmatter의 `tools:`를 중심으로, GitHub toolsets와 MCP 서버를 어떻게 붙이는지 정리합니다.*
