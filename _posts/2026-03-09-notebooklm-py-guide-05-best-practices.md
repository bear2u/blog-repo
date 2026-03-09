---
layout: post
title: "notebooklm-py 완벽 가이드 (05) - 운영/확장/베스트 프랙티스"
date: 2026-03-09
permalink: /notebooklm-py-guide-05-best-practices/
author: teng-lin
categories: [개발 도구, notebooklm-py]
tags: [notebooklm-py, Python, GitHub Trending]
original_url: "https://github.com/teng-lin/notebooklm-py"
excerpt: "notebooklm-py 운영/확장 시 체크리스트"
---

## 안정성/운영 체크리스트

- **비공식 API** 전제: 프로덕션 핵심 경로에 쓰기 전에 “깨져도 감당 가능한” 설계로 감싸세요(재시도/폴백/알림).
- **레이트 리밋 대비**: 호출 폭주를 막기 위해 큐/백오프를 두세요.
- **스토리지/상태 파일**: `--storage PATH`로 실험/운영 프로필을 분리하면 디버깅이 편합니다.

---

## 보안 체크리스트

- 인증 상태 파일/토큰은 최소 권한으로 관리하고, 로그에 남기지 마세요.
- CI/CD에서 파일 쓰기 없이 인증하려면 문서의 환경변수(`NOTEBOOKLM_AUTH_JSON` 등) 가이드를 우선 확인하세요.

---

## 트러블슈팅 가이드

- `docs/troubleshooting.md`의 증상별 체크리스트를 먼저 따라가세요.
- 디버깅이 필요하면 문서의 RPC 디버그 옵션을 확인하세요.

---

*시리즈를 마칩니다. 필요하면 `docs/`를 따라 더 깊게 파고드는 확장 챕터를 추가해도 좋습니다.*

