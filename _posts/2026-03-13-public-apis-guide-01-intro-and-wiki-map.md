---
layout: post
title: "public-apis 완벽 가이드 (01) - 소개 & 위키 맵"
date: 2026-03-13
permalink: /public-apis-guide-01-intro-and-wiki-map/
author: public-apis
categories: [개발 도구, public-apis]
tags: [Trending, GitHub, public-apis, API, Directory]
original_url: "https://github.com/public-apis/public-apis"
excerpt: "무료(또는 무료 티어) Public API 디렉토리 레포 public-apis를 탐색/기여/검증까지 위키형으로 정리합니다."
---

## public-apis란?

GitHub Trending(daily, 2026-03-13 기준) 상위에 오른 **public-apis/public-apis**를 “레포 사용법” 관점에서 정리합니다.

- **한 줄 요약(Trending 표시)**: A collective list of free APIs
- **언어(Trending 표시)**: Python (레포 내 검증 스크립트 기준)
- **오늘 스타(Trending 표시)**: +895
- **원본**: https://github.com/public-apis/public-apis

---

## 이 시리즈의 범위

이 레포는 “라이브러리/프레임워크”라기보다 **지식 베이스(디렉토리)**에 가깝습니다. 그래서 본 시리즈는 아래를 중심으로 구성합니다.

- README의 **표 구조/필드(Auth/HTTPS/CORS)**를 어떻게 해석하고 빠르게 찾을지
- “내 프로젝트에서 쓰기 위해” **로컬 인덱스**로 만드는 방법
- PR을 위한 **포맷 규칙/중복 방지 규칙**
- 레포가 제공하는 **검증 스크립트 + GitHub Actions 자동화**

---

## 레포에서 ‘사실’로 확인되는 핵심 파일

- 전체 디렉토리 본체: `README.md`
- 기여 규칙/포맷: `CONTRIBUTING.md`
- 검증 스크립트 안내: `scripts/README.md`
- 링크/포맷 검증 구현: `scripts/validate/links.py`, `scripts/validate/format.py`
- PR 변경 검증 스크립트: `scripts/github_pull_request.sh`
- CI(예: 링크 검증 스케줄): `.github/workflows/validate_links.yml`

---

## (위키 맵) 문서 구조

```text
01. 소개 & 위키 맵
02. 리스트 활용법 (검색/해석/프로그램적 소비)
03. 기여(Contributing) (규칙/포맷/PR)
04. 검증 스크립트 & CI (format/links/tests/workflows)
05. 베스트 프랙티스 & 문제해결 (중복/깨진 링크/로컬 체크)
```

---

## public-apis 레포 구조(상위)

```text
public-apis/
  README.md
  CONTRIBUTING.md
  scripts/
    README.md
    requirements.txt
    validate/
      format.py
      links.py
    tests/
      test_validate_format.py
      test_validate_links.py
  .github/
    workflows/
      validate_links.yml
      test_of_push_and_pull.yml
      test_of_validate_package.yml
```

---

## 위키 링크

- `[[public-apis Guide - Index]]` → [가이드 목차](/blog-repo/public-apis-guide/)
- `[[public-apis Guide - Using the List]]` → [02. 리스트 활용법](/blog-repo/public-apis-guide-02-using-the-list/)

---

*다음 글에서는 README 테이블을 “검색 가능한 데이터”처럼 활용하는 방법을 정리합니다.*

