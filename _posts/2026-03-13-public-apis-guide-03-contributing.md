---
layout: post
title: "public-apis 완벽 가이드 (03) - 기여(Contributing) 규칙과 포맷"
date: 2026-03-13
permalink: /public-apis-guide-03-contributing/
author: public-apis
categories: [개발 도구, public-apis]
tags: [Trending, GitHub, public-apis, Contribution, GitHub Actions]
original_url: "https://github.com/public-apis/public-apis"
excerpt: "마케팅 PR을 거르는 규칙, 테이블 포맷, 중복 방지, PR 운영 팁을 CONTRIBUTING.md 기준으로 정리합니다."
---

## 기여의 전제 (레포가 명시한 원칙)

`CONTRIBUTING.md`는 public-apis를 “마케팅 채널”로 악용하는 PR을 거절한다는 점을 분명히 합니다. 또한 ‘완전 무료’ 또는 ‘무료 티어’가 명확해야 하며, 디바이스 구매 같은 선행 조건이 있는 API는 거절될 수 있습니다.

---

## 테이블 포맷(스키마) 재정리

`CONTRIBUTING.md` 기준으로, 신규 항목은 다음 컬럼을 따릅니다.

```text
| API | Description | Auth | HTTPS | CORS | Call this API |
```

허용되는 `Auth` / `CORS` 값도 문서에 제한되어 있습니다. (예: `Auth`는 `OAuth`, `apiKey`, `X-Mashape-Key`, `No`, `User-Agent`)

---

## PR 가이드라인 (핵심만)

`CONTRIBUTING.md`에 명시된 운영 규칙 중, 실제로 자주 막히는 포인트:

- 이미 존재하는 API의 “다른 버전/업데이트”를 또 올리지 말 것
- 섹션 내 **알파벳 순 정렬** 유지
- 설명(Description)은 **100자 이내**
- PR 하나에 링크 하나(“Add one link per Pull Request”)
- PR 타이틀 형식(예: `Add Api-name API`) 준수
- 중복 PR/이슈가 없는지 사전 검색

---

## 로컬에서 최소 검증 루틴

레포는 검증 스크립트를 제공하므로, PR 전에는 로컬에서 최소한 아래를 돌리는 게 안전합니다. (`scripts/README.md`)

```bash
python -m pip install -r scripts/requirements.txt

# 포맷 검증
python scripts/validate/format.py README.md

# 링크 “중복”만 우선 체크(전체 링크 체크는 오래 걸릴 수 있음)
python scripts/validate/links.py README.md -odlc
```

---

## 위키 링크

- `[[public-apis Guide - Index]]` → [가이드 목차](/blog-repo/public-apis-guide/)
- `[[public-apis Guide - Validation & CI]]` → [04. 검증 스크립트 & CI](/blog-repo/public-apis-guide-04-validation-and-ci/)

---

*다음 글에서는 `scripts/validate/*`와 GitHub Actions 워크플로우를 연결해서 “어떻게 품질을 보장하는지”를 정리합니다.*

