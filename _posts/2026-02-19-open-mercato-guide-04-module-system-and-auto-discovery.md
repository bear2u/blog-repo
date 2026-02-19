---
layout: post
title: "Open Mercato 위키 가이드 (04) - 모듈 시스템/오토디스커버리: 생성기와 라우팅 규칙"
date: 2026-02-19
permalink: /open-mercato-guide-04-module-system-and-auto-discovery/
author: open-mercato
categories: ['개발 도구', '프레임워크']
tags: [Module System, Auto Discovery, Generator, Mercato CLI, Registry]
original_url: "https://github.com/open-mercato/open-mercato"
excerpt: "module-registry 생성기가 페이지/API/CLI/이벤트를 어떻게 자동 등록하는지 코드 기준으로 정리합니다."
---

## 핵심 아이디어

Open Mercato는 모듈 파일 규약을 기반으로 등록 코드를 자동 생성합니다.

관련 코드:

- `packages/cli/src/lib/generators/module-registry.ts`
- `packages/cli/src/lib/generators/openapi.ts`
- `packages/cli/src/lib/resolver.ts`

---

## 자동 발견 대상

모듈 디렉토리(`src/modules/<module>`)에서 대표적으로 스캔되는 요소:

- `frontend/`, `backend/` 페이지
- `api/*` 라우트
- `cli.ts` 명령
- `setup.ts`, `acl.ts`, `di.ts`
- `search.ts`, `events.ts`, `notifications.ts`
- `widgets/injection/*`

즉 모듈 계약을 맞추면 등록 코드를 직접 손대지 않고도 시스템에 편입됩니다.

---

## 생성 파이프라인 (Mermaid)

```mermaid
flowchart TD
  A[src/modules.ts enabledModules] --> B[resolver가 @app/@open-mercato/* 경로 해석]
  B --> C[module-registry generator 스캔]
  C --> D[.mercato/generated/modules.generated.ts]
  C --> E[dashboard/injection/search/events generated files]
  C --> F[openapi.generated.json]
  D --> G[앱 런타임 라우팅/DI 등록]
```

---

## 실무 체크포인트

1. 새 모듈 추가 후 `yarn generate` 누락하지 않기
2. app override와 package 기본 구현의 우선순위 이해
3. 라우트/메타 파일 명명 규약 유지
4. generated 산출물 수동 편집 금지

---

## 확장성 관점

이 구조 덕분에 신규 모듈은 "파일 배치 + modules.ts 등록"으로 시작할 수 있고,
장기적으로는 팀별 vertical 모듈을 병렬로 개발하기 좋습니다.

다음 장에서 멀티테넌시·RBAC·암호화 등 데이터 경계를 다룹니다.

---

## 위키 링크

- `[[Open Mercato Wiki - Package Graph]]` → [03 모노레포/패키지 그래프](/blog-repo/open-mercato-guide-03-monorepo-and-package-graph/)
- `[[Open Mercato Wiki - Data Security]]` → [05 데이터/테넌시/RBAC/암호화](/blog-repo/open-mercato-guide-05-data-model-tenancy-rbac-and-encryption/)
- `[[Open Mercato Wiki - API OpenAPI]]` → [06 API/OpenAPI/Query Engine](/blog-repo/open-mercato-guide-06-api-openapi-and-query-engine/)
