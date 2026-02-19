---
layout: post
title: "Open Mercato 위키 가이드 (03) - 모노레포/패키지 그래프: apps + packages + turbo"
date: 2026-02-19
permalink: /open-mercato-guide-03-monorepo-and-package-graph/
author: open-mercato
categories: ['개발 도구', '아키텍처']
tags: [Monorepo, Turbo, Workspaces, Open Mercato, Packages]
original_url: "https://github.com/open-mercato/open-mercato"
excerpt: "Open Mercato 모노레포의 apps/packages 역할 분리와 turbo task 체계를 위키 관점으로 시각화합니다."
---

## 디렉토리 핵심

- `apps/mercato`: 메인 Next.js 앱 (`@open-mercato/app`)
- `apps/docs`: Docusaurus 문서 사이트
- `packages/*`: 코어 기능/SDK/툴링 패키지

워크스페이스 구성은 루트 `package.json`의 `apps/*`, `packages/*`로 선언됩니다.

---

## 주요 패키지 역할

- `@open-mercato/core`: 비즈니스 모듈(auth/customers/sales 등)
- `@open-mercato/shared`: 공용 유틸/타입/DSL
- `@open-mercato/ui`: 공용 UI/CRUD 컴포넌트
- `@open-mercato/cli`: `mercato` 명령/생성기
- `@open-mercato/search`, `events`, `queue`, `cache`
- `@open-mercato/ai-assistant`: MCP/AI chat

---

## 패키지 그래프 (Mermaid)

```mermaid
graph TD
  A[@open-mercato/app] --> B[@open-mercato/core]
  A --> C[@open-mercato/ui]
  A --> D[@open-mercato/shared]
  A --> E[@open-mercato/cli]
  A --> F[@open-mercato/search]
  A --> G[@open-mercato/events]
  A --> H[@open-mercato/queue]
  A --> I[@open-mercato/cache]
  A --> J[@open-mercato/ai-assistant]
  J --> F
  B --> D
  C --> D
```

---

## turbo task 체계

`turbo.json` 기준 핵심:

- `build`, `generate`, `initialize`: 캐시 비활성(정합성 우선)
- `dev`, `watch`, `mcp:*`: persistent task
- `test`, `typecheck`: 산출물 없이 검증

대규모 모듈 변경 시 `yarn build:packages -> yarn generate -> yarn dev` 순으로 보는 편이 안정적입니다.

---

## 실무적 해석

Open Mercato는 "앱 + 코어 패키지" 분리로 업그레이드/커스터마이징 경계를 명확히 둡니다.

- 코어 변경 최소화
- 앱 레이어에서 모듈 추가/오버라이드
- 필요 시 eject로 깊은 커스텀

다음 장에서 모듈 오토디스커버리 메커니즘을 봅니다.

---

## 위키 링크

- `[[Open Mercato Wiki - Setup Bootstrap]]` → [02 로컬 설치/부트스트랩](/blog-repo/open-mercato-guide-02-local-setup-and-bootstrap/)
- `[[Open Mercato Wiki - Module Auto Discovery]]` → [04 모듈 시스템/오토디스커버리](/blog-repo/open-mercato-guide-04-module-system-and-auto-discovery/)
- `[[Open Mercato Wiki - Customization Eject]]` → [12 커스터마이징/Eject 로드맵](/blog-repo/open-mercato-guide-12-customization-eject-and-extension-roadmap/)
