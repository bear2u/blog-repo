---
layout: post
title: "GitHub Agentic Workflows(gh-aw) 가이드 (10) - 트러블슈팅: 설치/정책/컴파일/도구 이슈"
date: 2026-02-15
permalink: /gh-aw-guide-10-troubleshooting/
author: GitHub
categories: [개발 도구, GitHub]
tags: [gh-aw, Troubleshooting, Enterprise, Policies, Compilation, MCP]
original_url: "https://github.github.com/gh-aw/troubleshooting/common-issues/"
excerpt: "gh-aw 운영에서 자주 나오는 문제(확장 설치 실패, 엔터프라이즈 Actions 정책, 컴파일/락파일, MCP/툴 설정)를 증상별로 정리합니다."
---

## 1) 설치가 안 된다: standalone installer로 우회

`gh extension install`이 인증/권한 문제로 실패하면 문서는 standalone installer를 권합니다.

```bash
curl -sL https://raw.githubusercontent.com/github/gh-aw/main/install-gh-aw.sh | bash
```

---

## 2) 엔터프라이즈 Actions 정책에 막힌다

조직 정책으로 특정 액션(`github/gh-aw/actions/...`)이 차단되는 경우가 있습니다.

이건 워크플로우 문제가 아니라 “조직 Actions allowlist 정책” 이슈라,
조직 설정에서 `github/gh-aw@*` 같은 항목을 허용해야 해결됩니다(문서의 조직 정책 섹션 참고).

---

## 3) 컴파일이 실패한다

가장 흔한 원인:

- frontmatter YAML 문법 오류(들여쓰기/콜론/타입)
- deprecated 필드 사용
- strict 모드에서 금지된 네트워크/권한 설정

실전 디버깅 루틴:

```bash
gh aw fix --write
gh aw compile --validate --strict -v
```

---

## 4) lock file이 없다/오래됐다

- `.lock.yml`이 없으면 실행이 안 됩니다.
- `.md`를 바꿨는데 `.lock.yml`을 안 갱신하면 실행물이 구버전일 수 있습니다.

해결:

```bash
gh aw compile
```

삭제된 워크플로우의 lock가 남아 있으면:

```bash
gh aw compile --purge
```

---

## 5) GitHub 도구가 “없다”

GitHub API 연동은 `tools.github.toolsets`로 여는 방식이 권장됩니다.

{% raw %}
```yaml
tools:
  github:
    toolsets: [default, actions]
```
{% endraw %}

필요한 API 그룹이 빠지면 toolset을 추가하거나 조합합니다.

---

## 6) Playwright를 require하려고 하다가 터진다

문서의 대표 사례:

- 워크플로우 안에서 `require('playwright')` 같은 방식으로 npm 패키지를 기대하면 실패할 수 있음
- gh-aw는 Playwright를 MCP 도구 형태로 제공하는 흐름을 전제로 함

따라서 브라우저 자동화는 “패키지 import”가 아니라 “MCP 도구 호출” 관점으로 설계해야 합니다.

---

*시리즈를 읽고 실제 레포에 적용할 때는, 먼저 작은 manual workflow부터 시작해서 safe-outputs/네트워크/툴 경계를 조금씩 열어가면 운영 리스크가 크게 줄어듭니다.*

