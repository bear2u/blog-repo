---
layout: page
title: public-apis 가이드
permalink: /public-apis-guide/
icon: fas fa-book
---

# 📚 public-apis 완벽 가이드

> **A collective list of free APIs**

**public-apis/public-apis**는 커뮤니티가 유지하는 “무료(또는 무료 티어) API 목록” 레포입니다. 이 시리즈는 단순 링크 나열이 아니라, **검색/활용 방법**, **기여 규칙**, **검증 자동화(링크/포맷)**까지 “레포 자체를 도구처럼” 쓰는 방법을 위키형으로 정리합니다.

---

## 📚 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 & 위키 맵](/blog-repo/public-apis-guide-01-intro-and-wiki-map/) | 레포 목적, 용어, 탐색 경로 |
| 02 | [리스트 활용법](/blog-repo/public-apis-guide-02-using-the-list/) | 카테고리/필드 해석, 빠른 검색, 프로그램적 소비 |
| 03 | [기여(Contributing)](/blog-repo/public-apis-guide-03-contributing/) | PR 규칙, 포맷, 중복 방지 |
| 04 | [검증 스크립트 & CI](/blog-repo/public-apis-guide-04-validation-and-ci/) | `scripts/validate/*`와 GitHub Actions |
| 05 | [베스트 프랙티스 & 문제해결](/blog-repo/public-apis-guide-05-best-practices-and-troubleshooting/) | 깨진 링크, 중복, 로컬 체크리스트 |

---

## 빠른 시작 (레포를 “로컬 인덱스”로 쓰기)

```bash
git clone https://github.com/public-apis/public-apis.git
cd public-apis

# README에서 원하는 키워드/카테고리를 찾아보기
rg -n "Weather|Currency|Maps" README.md

# (기여 목적) 포맷/링크 검증 스크립트 실행
python -m pip install -r scripts/requirements.txt
python scripts/validate/format.py README.md
python scripts/validate/links.py README.md -odlc
```

---

## 관련 링크

- GitHub 저장소: https://github.com/public-apis/public-apis
- 기여 가이드: `CONTRIBUTING.md`
- 검증 스크립트 문서: `scripts/README.md`

