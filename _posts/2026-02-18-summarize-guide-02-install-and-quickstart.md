---
layout: post
title: "Summarize 가이드 (02) - 설치/빠른 시작: CLI 단독 사용과 확장+daemon 연결"
date: 2026-02-18
permalink: /summarize-guide-02-install-and-quickstart/
author: steipete
categories: ['개발 도구', '설치 가이드']
tags: [Node.js, pnpm, summarize daemon, Chrome Extension, Firefox Sidebar]
original_url: "https://github.com/steipete/summarize/blob/main/README.md"
excerpt: "README와 extension 문서를 기반으로 CLI 설치, 확장 로드, daemon pairing까지 실제 사용 순서로 정리합니다."
---

## 기본 요구사항

`package.json` 기준 Node 요구사항은 `>=22`입니다.

CLI 설치는 크게 3가지 경로를 제공합니다.

```bash
npx -y @steipete/summarize "https://example.com"
npm i -g @steipete/summarize
brew install steipete/tap/summarize
```

---

## CLI만 쓰는 최소 경로

브라우저 확장을 쓰지 않는다면 daemon 설치는 불필요합니다.

```bash
summarize "https://example.com"
```

입력은 URL, 로컬 파일, stdin(`-`) 모두 지원합니다.

---

## 확장(Chrome/Firefox) 빠른 시작

확장 측 문서(`apps/chrome-extension/README.md`, `docs/chrome-extension.md`) 기준 흐름:

1. 확장 설치(스토어 또는 unpacked)
2. 사이드패널/사이드바에서 토큰 확인
3. 터미널에서 daemon 설치

```bash
summarize daemon install --token <TOKEN>
summarize daemon status
```

daemon은 localhost(`127.0.0.1`)에서만 동작하며, 토큰으로 인증합니다.

---

## 개발자 빌드 경로

레포에서 직접 확장을 테스트할 때:

```bash
pnpm -C apps/chrome-extension build
pnpm summarize daemon install --token <TOKEN> --dev
```

`--dev`는 dist 빌드 대신 `src/cli.ts` 경로를 서비스 엔트리로 사용합니다.

---

## 운영 팁

문서 기준 기본 점검 커맨드는 아래 두 개입니다.

```bash
summarize daemon status
summarize daemon restart
```

다음 장에서 입력 타입별 실행 파이프라인을 코드 기준으로 봅니다.

