---
layout: post
title: "Heretic 가이드 (05) - 방어적 활용 & 재현성: 안전 평가/가드레일/로그"
date: 2026-03-14
permalink: /heretic-guide-05-defensive-use-and-reproducibility/
author: p-e-w
categories: [AI, LLM 안전]
tags: [Trending, GitHub, heretic, Safety, Evaluation, Reproducibility]
original_url: "https://github.com/p-e-w/heretic"
excerpt: "Heretic 같은 ‘안전 정렬 약화’ 도구를 다룰 때 필요한 연구 윤리/컴플라이언스/가드레일과, 재현성 있는 로그/벤치마크 설계를 정리합니다."
---

## 안전 고지(요약)

이 문서는 “방어/평가/연구 거버넌스”에 초점을 둡니다. 오남용을 조장하는 실행/배포/튜닝 방법은 제공하지 않습니다. (`README.md`)

---

## 이 문서의 목적

- 조직/프로젝트 차원에서 “무엇을 하면 안 되는지” 경계선을 명확히 합니다.
- 재현성/감사 가능성(auditability)을 확보하기 위한 로그/체크포인트/데이터 관리 체크리스트를 제공합니다.

---

## 1) 사용 범위 가드레일(권장)

- **목적 제한**: 안전 연구(레드팀), 방어 메커니즘 검증, 모델 취약성 평가 등으로 범위를 제한
- **데이터 통제**: 민감/유해 데이터셋은 접근 통제·감사 로그·보관 정책을 갖춘 환경에서만 사용
- **배포 금지/격리**: 네트워크로 제공되는 서비스에 적용하지 말 것(특히 AGPL 의무/법무 리스크 포함). (`LICENSE`)
- **휴먼 리뷰**: 자동 지표만으로 결론을 내리지 말고, 리뷰 절차를 강제

---

## 2) 재현성 체크리스트(실험 기록)

코드상으로 최소한 아래가 “실험 결과”를 좌우합니다.

- 설정: `src/heretic/config.py` (Settings 스키마)
- 의존성/락: `pyproject.toml`, `uv.lock`
- 체크포인트/저장: `src/heretic/main.py` (Optuna journaling storage 단서)
- 장비/메모리: `src/heretic/main.py` (디바이스 탐지/메모리 관련 출력 단서)

권장 기록 항목:

- 실행 시각/커밋 해시(레포 버전)
- GPU/드라이버/프레임워크 버전
- 설정 파일 스냅샷(민감 정보는 별도 보관)
- 실험 로그 + 체크포인트 디렉토리의 해시/메타데이터

---

## 3) “방어적 평가” 설계 힌트(코드 근거)

Heretic는 설정 스키마에 “데이터셋 스펙”을 모델링하는 타입이 존재합니다. (`DatasetSpecification`, `src/heretic/config.py`)

방어 관점에서는:

- 유해 행위 조장 프롬프트 자체를 확산시키지 않도록, 사내/폐쇄 환경에서 자체 평가셋을 관리하고,
- 결과를 “행동 가이드”가 아니라 “취약성 리포트(재현 가능한 증거 + 대응)”로 문서화하는 편이 안전합니다.

---

## 근거(파일/경로)

- 안전/의도: `README.md`
- 라이선스: `LICENSE`
- 설정/데이터셋 스키마: `src/heretic/config.py`
- 오케스트레이션/체크포인트 단서: `src/heretic/main.py`
- 의존성/락: `pyproject.toml`, `uv.lock`

---

## 주의사항/함정

- 자동화된 “성능 지표”가 사용자 피해를 정당화하지 않습니다. 연구 결과는 항상 안전/윤리 검토와 함께 취급해야 합니다.
- 레포가 공개되어 있어도, 2차 가이드가 오남용을 확대할 수 있습니다. 문서를 공개 블로그에 올릴 때는 범위/예시/용어 선택을 특히 보수적으로 하세요.

---

## TODO/확인 필요

- `src/heretic/evaluator.py`의 평가 지표 정의/출력 포맷을 감사(Audit) 목적에 맞게 요약하기
- `src/heretic/utils.py`에서 “프롬프트 로딩”이 어떤 경로를 허용하는지(로컬 파일/원격) 보안 관점으로 검토

---

## 위키 링크

- `[[Heretic Guide - Index]]` → [가이드 목차](/blog-repo/heretic-guide/)
- `[[Heretic Guide - Repo Structure & Config]]` → [02. 레포 구조 & 설정 스키마](/blog-repo/heretic-guide-02-repo-structure-and-config/)

