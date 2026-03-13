---
layout: post
title: "public-apis 완벽 가이드 (04) - 검증 스크립트 & CI (format/links/tests)"
date: 2026-03-13
permalink: /public-apis-guide-04-validation-and-ci/
author: public-apis
categories: [개발 도구, public-apis]
tags: [Trending, GitHub, public-apis, CI, Validation]
original_url: "https://github.com/public-apis/public-apis"
excerpt: "README 품질을 유지하는 validate 패키지(format/links)와 GitHub Actions 워크플로우를 레포 기준으로 정리합니다."
---

## 검증 스크립트의 “정체”

public-apis는 README를 원본 데이터로 삼습니다. 그래서 품질은 **README 변경을 막는 자동화**로 유지됩니다.

레포 기준으로 확인되는 검증 구성:

- validate 패키지: `scripts/validate/format.py`, `scripts/validate/links.py`
- 의존성: `scripts/requirements.txt`
- 단위 테스트: `scripts/tests/test_validate_format.py`, `scripts/tests/test_validate_links.py`
- PR/푸시 검증 워크플로우: `.github/workflows/test_of_push_and_pull.yml`
- validate 패키지 테스트 워크플로우: `.github/workflows/test_of_validate_package.yml`
- 정기 링크 점검: `.github/workflows/validate_links.yml` (schedule)

---

## 로컬 실행 (레포 문서 그대로)

`scripts/README.md`에 안내된 루틴:

```bash
python -m pip install -r scripts/requirements.txt

# 포맷 검증
python scripts/validate/format.py README.md

# 링크 검증(전체)
python scripts/validate/links.py README.md

# 링크 중복만(빠름)
python scripts/validate/links.py README.md -odlc

# 단위 테스트
cd scripts
python -m unittest discover tests/ --verbose
```

---

## CI 파이프라인 개요(레포 기준)

```mermaid
flowchart TB
  P[Pull Request / Push] --> A[GitHub Actions]
  A --> F[format.py]
  A --> L[links.py (dup only or full)]
  A --> T[unittest (validate package)]
  A --> R[PR change validator (github_pull_request.sh)]
```

- PR에서는 `scripts/github_pull_request.sh`를 통해 PR 범위의 변경을 검사하도록 구성됩니다. (`.github/workflows/test_of_push_and_pull.yml`)
- 링크 “전체” 검사는 스케줄 워크플로우로 분리되어 있습니다. (`.github/workflows/validate_links.yml`)

---

## 위키 링크

- `[[public-apis Guide - Index]]` → [가이드 목차](/blog-repo/public-apis-guide/)
- `[[public-apis Guide - Best Practices]]` → [05. 베스트 프랙티스 & 문제해결](/blog-repo/public-apis-guide-05-best-practices-and-troubleshooting/)

---

*다음 글에서는 실제로 깨지기 쉬운 지점(중복/깨진 링크/포맷)과 로컬 체크리스트를 정리합니다.*

