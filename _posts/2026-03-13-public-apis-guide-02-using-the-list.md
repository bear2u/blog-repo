---
layout: post
title: "public-apis 완벽 가이드 (02) - 리스트 활용법 (검색/해석/프로그램적 소비)"
date: 2026-03-13
permalink: /public-apis-guide-02-using-the-list/
author: public-apis
categories: [개발 도구, public-apis]
tags: [Trending, GitHub, public-apis, API, Markdown]
original_url: "https://github.com/public-apis/public-apis"
excerpt: "README의 카테고리/테이블을 빠르게 찾고, 필드를 해석하고, 로컬에서 프로그램적으로 소비하는 실전 팁을 정리합니다."
---

## README 테이블의 의미 (레포 기준)

`README.md`에는 카테고리별로 다음 컬럼을 가진 테이블이 반복됩니다. (`CONTRIBUTING.md`의 “Formatting” 섹션)

| 컬럼 | 의미 |
|---|---|
| API | 이름(문서 링크) |
| Description | 100자 이내 설명 |
| Auth | `OAuth` / `apiKey` / `X-Mashape-Key` / `No` / `User-Agent` 등 |
| HTTPS | HTTPS 지원 여부 |
| CORS | `Yes` / `No` / `Unknown` |
| Call this API | (선택) Postman Run 버튼 등 |

---

## 빠른 검색 (로컬에서)

대부분의 “탐색”은 `README.md`에서 시작합니다.

```bash
git clone https://github.com/public-apis/public-apis.git
cd public-apis

# 카테고리 탐색
rg -n "^### " README.md | head

# 특정 도메인/키워드 탐색
rg -n "\\bWeather\\b|\\bMaps\\b|\\bCurrency\\b" README.md

# Auth가 없는 API만 보고 싶을 때(정밀 필터링은 별도 파싱이 편함)
rg -n "\\| No \\|" README.md | head
```

---

## 프로그램적으로 소비하기 (최소 접근)

public-apis는 별도 JSON 내보내기를 공식으로 제공한다기보다, **README를 검증 가능한 원본(소스 오브 트루스)**로 유지하는 구조입니다. 따라서 “프로그램적 소비”는 보통 아래 흐름이 됩니다.

```mermaid
flowchart LR
  A[README.md] --> B[Markdown 파서/스크레이퍼]
  B --> C[내부 인덱스(JSON/DB)]
  C --> D[검색/추천/분류 UI]
  A --> E[검증 스크립트]
  E --> A
```

- **원본 유지**: `README.md`
- **품질 보장**: `scripts/validate/*` + GitHub Actions
- **내부 활용**: README를 파싱해 사내 도구/노트/검색 UI로 변환

---

## (실무 팁) 데이터 품질 체크 포인트

- `Auth` / `CORS`는 “사용 가능성”을 크게 좌우하므로, 내 프로젝트 요구사항에 맞는 필터부터 정합니다.
- CORS가 `No`이거나 미상인 경우, **브라우저 직접 호출**이 아니라 **서버 사이드 프록시**가 필요할 수 있습니다. (`CONTRIBUTING.md`에 CORS 관련 안내가 포함)
- 링크는 깨질 수 있으니, 기여 전에는 로컬에서 **중복 링크 검사**라도 돌리는 것이 안전합니다. (`scripts/README.md`의 `-odlc`)

---

## 위키 링크

- `[[public-apis Guide - Index]]` → [가이드 목차](/blog-repo/public-apis-guide/)
- `[[public-apis Guide - Contributing]]` → [03. 기여(Contributing)](/blog-repo/public-apis-guide-03-contributing/)

---

*다음 글에서는 `CONTRIBUTING.md` 기준으로 PR 규칙/포맷을 정리합니다.*

