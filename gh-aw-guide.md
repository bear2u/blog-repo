---
layout: page
title: GitHub Agentic Workflows(gh-aw) 가이드
permalink: /gh-aw-guide/
icon: fas fa-tools
---

# GitHub Agentic Workflows(gh-aw) 완벽 가이드

> **자연어 마크다운으로 워크플로우를 작성하고, `gh aw`로 컴파일해 GitHub Actions에서 안전하게 실행하기**

**gh-aw**는 GitHub CLI 확장(`gh aw`)으로, 마크다운 + YAML frontmatter로 작성한 “에이전트 워크플로우”를 GitHub Actions 워크플로우(`.lock.yml`)로 컴파일해서 실행할 수 있게 합니다. 핵심은 **가드레일(권한/네트워크/샌드박스/검증)**과 **safe-outputs(쓰기 권한 분리)**로, AI가 실제로 레포 작업을 하더라도 안전 경계를 유지하는 방식입니다.

- 원문 저장소: https://github.com/github/gh-aw
- 공식 문서: https://github.github.com/gh-aw/

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/gh-aw-guide-01-intro/) | gh-aw가 해결하는 문제, 전체 그림 |
| 02 | [설치/초기화](/blog-repo/gh-aw-guide-02-install-and-init/) | 설치, `gh aw init`, 엔진/시크릿 |
| 03 | [워크플로우 작성](/blog-repo/gh-aw-guide-03-authoring/) | `.md` 포맷, frontmatter, 편집/리컴파일 기준 |
| 04 | [도구/MCP](/blog-repo/gh-aw-guide-04-tools-and-mcp/) | `tools:`/toolsets, Playwright, 커스텀 MCP |
| 05 | [Safe Outputs/보안](/blog-repo/gh-aw-guide-05-safe-outputs-and-security/) | 읽기 전용 에이전트 + 쓰기 분리, threat detection |
| 06 | [컴파일/락파일](/blog-repo/gh-aw-guide-06-compilation-and-lockfiles/) | `compile`, `--watch`, `--strict`, 스캐너, 핀ning |
| 07 | [실행/운영](/blog-repo/gh-aw-guide-07-running-workflows/) | 트리거, `gh aw run`, 인터랙티브 런 |
| 08 | [패턴/예시](/blog-repo/gh-aw-guide-08-patterns-and-examples/) | issueops/dailyops 등 패턴, 샘플 워크플로우 읽기 |
| 09 | [업그레이드/유지](/blog-repo/gh-aw-guide-09-upgrading-and-maintenance/) | `upgrade`, codemod, lockfile 운영 |
| 10 | [트러블슈팅](/blog-repo/gh-aw-guide-10-troubleshooting/) | 설치/정책/컴파일/툴 이슈 정리 |

