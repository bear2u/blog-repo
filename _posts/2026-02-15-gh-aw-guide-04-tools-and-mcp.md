---
layout: post
title: "GitHub Agentic Workflows(gh-aw) 가이드 (04) - 도구/MCP: toolsets로 GitHub API를 열고, 커스텀 MCP를 붙이기"
date: 2026-02-15
permalink: /gh-aw-guide-04-tools-and-mcp/
author: GitHub
categories: [개발 도구, GitHub]
tags: [gh-aw, Tools, MCP, GitHub API, Toolsets, Playwright]
original_url: "https://github.github.com/gh-aw/reference/tools/"
excerpt: "gh-aw는 frontmatter의 tools로 에이전트가 사용할 수 있는 능력을 제한/확장합니다. GitHub toolsets, 웹 도구, Playwright, 커스텀 MCP 서버까지 핵심 사용법을 정리합니다."
---

## tools: 에이전트의 능력 표면을 설계하는 곳

gh-aw에서 `tools:`는 “에이전트가 할 수 있는 것”의 경계를 정합니다.

{% raw %}
```yaml
tools:
  edit:
  bash: true
```
{% endraw %}

핵심은 “필요 최소”입니다. 불필요하게 도구를 열면:

- 프롬프트 인젝션에 대한 공격 표면이 커지고
- 권한/네트워크 요구가 커지며
- 워크플로우가 불안정해질 수 있습니다.

---

## GitHub 도구: allowed보다 toolsets를 우선

GitHub API 연동은 `tools.github`로 설정합니다.
문서는 `allowed:`(툴 이름 allowlist)보다 **toolsets**를 권장합니다.

{% raw %}
```yaml
tools:
  github:
    mode: remote
    toolsets: [default, actions]
    read-only: true
```
{% endraw %}

toolsets는 “능력 묶음”이라, MCP 서버 버전/툴 이름 변경에 덜 흔들립니다.

---

## 웹 도구/브라우저 자동화

워크플로우가 문서/리서치/검증을 해야 한다면 `web-fetch`, `web-search` 같은 도구를 켤 수 있습니다.

브라우저 자동화가 필요하면 Playwright 도구를 사용합니다.

핵심은 네트워크 허용과 함께 설계해야 한다는 점입니다(다음 글에서 보안/네트워크와 같이 설명).

---

## 내장 MCP 도구들: agentic-workflows, repo-memory, cache-memory

문서에는 gh-aw 생태계에서 유용한 내장 도구들이 있습니다.

- `agentic-workflows`: 워크플로우 인트로스펙션/로그/디버깅
- `repo-memory`: 레포 단위 메모리 저장
- `cache-memory`: 실행 간 추세/상태를 저장하는 캐시

이런 도구는 “장기 운영”에서 특히 도움이 됩니다.

---

## 커스텀 MCP 서버 붙이기

외부 시스템(Slack, Jira 등)과 연결하고 싶다면 `mcp-servers:`로 정의합니다.

{% raw %}
```yaml
mcp-servers:
  slack:
    command: "npx"
    args: ["-y", "@slack/mcp-server"]
    env:
      SLACK_BOT_TOKEN: "${{ secrets.SLACK_BOT_TOKEN }}"
    allowed: ["send_message"]
```
{% endraw %}

포인트:

- `env`로 시크릿을 주입하고
- `allowed`로 도구 호출 범위를 제한합니다.

---

*다음 글에서는 safe-outputs와 strict/security 모델을 중심으로, “에이전트는 읽기 전용으로, 쓰기는 통제된 후처리로” 설계하는 방법을 정리합니다.*

