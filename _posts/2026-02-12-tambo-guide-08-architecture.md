---
layout: post
title: "Tambo 완벽 가이드 (08) - 모노레포 아키텍처"
date: 2026-02-12
permalink: /tambo-guide-08-architecture/
author: Tambo AI
categories: [AI 에이전트, 웹 개발]
tags: [Tambo, Turborepo, 모노레포, Next.js, NestJS]
original_url: "https://github.com/tambo-ai/tambo"
excerpt: "Turborepo 기반 구성과 패키지 구조"
---

## 큰 그림

Tambo는 Turborepo 기반 모노레포로, "프레임워크 패키지"와 "클라우드 플랫폼"이 함께 들어 있습니다(레포 내 AGENTS.md).

---

## 프레임워크 패키지(워크스페이스)

AGENTS.md에 정리된 주요 디렉토리:

- `react-sdk/`: React SDK (`@tambo-ai/react`)
- `cli/`: CLI(`tambo`)
- `create-tambo-app/`: 프로젝트 부트스트랩퍼
- `showcase/`: 데모 앱(문서 기준 포트 8262)
- `docs/`: 문서 사이트(문서 기준 포트 8263)
- `packages/`: 공유 설정/유틸(ESLint, TS config 등)

---

## Tambo Cloud(플랫폼)

- `apps/web`: Next.js 웹
- `apps/api`: NestJS API
- `packages/db`: Drizzle ORM 스키마/마이그레이션
- `packages/backend`, `packages/core`: 공용 로직

---

## 주요 개발 명령

루트 `package.json`의 대표 스크립트:

```bash
npm run dev        # showcase + docs
npm run dev:cloud  # cloud web + api
npm run lint
npm run check-types
npm test
```

*다음 글에서는 Docker 기반 self-host와 운영 포인트를 정리합니다.*
