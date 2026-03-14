---
layout: page
title: Claude Code Plugins Directory 가이드
permalink: /claude-plugins-official-guide/
icon: fas fa-plug
---

# 🔌 Claude Code Plugins Directory 완벽 가이드

> **Claude Code Plugins Directory — 고품질 플러그인(내장/외부) 카탈로그 + 설치/개발/검증 흐름**

**anthropics/claude-plugins-official**은 Claude Code에서 설치할 수 있는 플러그인 디렉터리를 “레포 형태”로 제공하는 저장소입니다.

- 내부 플러그인: `plugins/` (Anthropic 유지보수)
- 외부 플러그인: `external_plugins/` (파트너/커뮤니티)
- 마켓플레이스 카탈로그: `.claude-plugin/marketplace.json` (플러그인 목록/메타/소스)

이 시리즈는 “어떻게 설치하고”, “어떻게 구조를 이해하고”, “어떻게 안전하게 운영/검증하는지”를 **레포 근거(파일/경로)** 기준으로 위키형으로 정리합니다.

---

## 📚 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 & 위키 맵](/blog-repo/claude-plugins-official-guide-01-intro-and-wiki-map/) | 디렉터리 목적, 위험 경고, 전체 탐색 지도 |
| 02 | [설치 & Discover](/blog-repo/claude-plugins-official-guide-02-install-and-discover/) | `/plugin install ...`와 Discover 흐름, 설치 전 점검 |
| 03 | [marketplace.json & 메타데이터](/blog-repo/claude-plugins-official-guide-03-marketplace-json-and-metadata/) | 카탈로그 구조, `source`(로컬/URL), 카테고리/태그 |
| 04 | [플러그인 패키지 구조](/blog-repo/claude-plugins-official-guide-04-plugin-package-structure/) | `.claude-plugin/plugin.json`, `.mcp.json`, `commands/`, `skills/` |
| 05 | [example-plugin 딥다이브](/blog-repo/claude-plugins-official-guide-05-example-plugin-deep-dive/) | 커맨드/스킬/MCP 예제 기반으로 “한 번에 감 잡기” |
| 06 | [LSP 플러그인 딥다이브](/blog-repo/claude-plugins-official-guide-06-lsp-plugins-deep-dive/) | `lspServers` 설정과 언어 서버 설치(typescript/pyright) |
| 07 | [외부 MCP 플러그인 딥다이브(GitHub)](/blog-repo/claude-plugins-official-guide-07-external-mcp-plugin-github/) | `.mcp.json` 헤더/환경변수/토큰 패턴 |
| 08 | [보안·문제해결·자동화](/blog-repo/claude-plugins-official-guide-08-security-troubleshooting-and-automation/) | 신뢰/리스크, CI 검증(Frontmatter), PR 정책, 운영 체크리스트 |

---

## 관련 링크

- GitHub 저장소: https://github.com/anthropics/claude-plugins-official
- 공식 문서: https://code.claude.com/docs/en/plugins

