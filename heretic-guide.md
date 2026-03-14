---
layout: page
title: Heretic 가이드
permalink: /heretic-guide/
icon: fas fa-shield-halved
---

# Heretic 가이드 (연구/보안 관점)

> **주의: 이 프로젝트는 LLM 안전 정렬(안전 필터)을 약화/제거하는 용도로 악용될 수 있습니다.**
>
> 이 시리즈는 “작동 방법의 이해/코드 아키텍처 분석/방어 관점”에 초점을 두며, 오남용을 조장하는 실행 절차(모델 비정상 활용, 안전장치 우회, 비인가 배포 등)는 다루지 않습니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 안전 고지](/blog-repo/heretic-guide-01-intro-and-safety/) | 프로젝트 개요, 위험/윤리/컴플라이언스 |
| 02 | [레포 구조 & 설정 스키마](/blog-repo/heretic-guide-02-repo-structure-and-config/) | `src/heretic/*`, config 파일 위치와 역할 |
| 03 | [아키텍처](/blog-repo/heretic-guide-03-architecture/) | Analyzer/Model/Evaluator/Optuna 파이프라인 개요 |
| 04 | [최적화 파이프라인(개념)](/blog-repo/heretic-guide-04-optimization-pipeline/) | TPE/Trial/Pruning, 재현성/리스크 포인트 |
| 05 | [방어적 활용 & 재현성](/blog-repo/heretic-guide-05-defensive-use-and-reproducibility/) | 안전 평가/가드레일, 로그/벤치마크 설계 |

---

## 관련 링크

- [GitHub 저장소](https://github.com/p-e-w/heretic)
- [설정 파일 템플릿](https://github.com/p-e-w/heretic/blob/main/config.default.toml)

