---
layout: post
title: "public-apis 완벽 가이드 (05) - 베스트 프랙티스 & 문제해결 체크리스트"
date: 2026-03-13
permalink: /public-apis-guide-05-best-practices-and-troubleshooting/
author: public-apis
categories: [개발 도구, public-apis]
tags: [Trending, GitHub, public-apis, Best Practices, Troubleshooting]
original_url: "https://github.com/public-apis/public-apis"
excerpt: "public-apis에 기여하거나 내부 인덱스로 사용할 때 자주 깨지는 지점과, 레포가 제공하는 검증 도구로 해결하는 루틴을 정리합니다."
---

## 자주 겪는 문제 유형(레포 문서 기반)

`CONTRIBUTING.md`와 검증 스크립트 구조를 보면, 유지보수 난점은 대체로 아래로 수렴합니다.

1) **마케팅/유료 전환 유도성 PR**
2) **중복 링크/중복 항목**
3) **포맷(테이블 정렬/공백/컬럼) 깨짐**
4) **깨진 링크(죽은 도메인/경로 변경)**

---

## 로컬 체크리스트 (최소)

```bash
# 1) 포맷 검증
python scripts/validate/format.py README.md

# 2) 링크 중복만 우선 체크(빠름)
python scripts/validate/links.py README.md -odlc

# 3) 필요 시 전체 링크 체크(느릴 수 있음)
python scripts/validate/links.py README.md

# 4) validate 패키지 테스트(스크립트 쪽 변경 시)
cd scripts
python -m unittest discover tests/ --verbose
```

---

## (내부 도구화) 안전한 운영 팁

README를 파싱해 내부 인덱스로 만들 때, “신뢰도/변경” 문제를 줄이려면:

- **원본은 README로 고정**하고, 파생 데이터(JSON/DB)는 재생성 가능하게 유지
- “신규 API 추가” 자동화는 PR의 인간 검토를 전제로 설계(레포 자체가 수동 큐레이션)
- CORS/인증 방식(`Auth`, `CORS`)을 우선 필터로 삼아, 사용 불가 항목을 먼저 배제

---

## 위키 링크

- `[[public-apis Guide - Index]]` → [가이드 목차](/blog-repo/public-apis-guide/)
- `[[public-apis Guide - Intro]]` → [01. 소개 & 위키 맵](/blog-repo/public-apis-guide-01-intro-and-wiki-map/)

---

*이 시리즈는 추후 “README 파싱 → 로컬 검색 UI” 같은 확장 챕터로 늘릴 수 있도록 구성했습니다.*

