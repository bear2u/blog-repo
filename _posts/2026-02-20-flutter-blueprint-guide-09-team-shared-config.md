---
layout: post
title: "flutter_blueprint 가이드 (09) - 팀 공유 설정(share) 운영"
date: 2026-02-20
permalink: /flutter-blueprint-guide-09-team-shared-config/
author: chirag640
categories: ['개발 도구', 'Flutter']
tags: [flutter_blueprint, ShareCommand, Team Config, YAML]
original_url: "https://github.com/chirag640/flutter_blueprint-Package"
excerpt: "share 명령과 SharedBlueprintConfig를 사용해 팀 표준 생성 설정을 배포/검증하는 방법을 정리합니다."
---

## 왜 공유 설정이 필요한가

개발자별로 `init` 플래그 조합이 달라지면 코드베이스 편차가 커집니다.

`share` 명령은 이 편차를 줄이기 위해 구성 파일을 관리합니다.

---

## 핵심 서브커맨드

```bash
flutter_blueprint share list
flutter_blueprint share import ./company_standard.yaml
flutter_blueprint share export company_standard ./out/company_standard.yaml
flutter_blueprint share validate ./company_standard.yaml
```

팀은 `import`로 표준을 배포하고, 신규 프로젝트는 `--from-config`로 동일 기준을 재사용할 수 있습니다.

---

## 구성 모델 포인트

`SharedBlueprintConfig`에는 다음 정보가 들어갑니다.

- 기본 상태관리/플랫폼/CI
- 코드 스타일 규칙
- 아키텍처 선호
- 필수 패키지/메타데이터

즉, 이 파일은 "생성 옵션 모음"이 아니라 팀 정책 문서 역할까지 수행합니다.

---

## 운영 팁

1. 설정 파일 버전을 SemVer로 관리
2. 변경 시 `validate`를 CI에서 자동 실행
3. 프로젝트 템플릿 변경 이력과 함께 배포 노트 작성

다음 글에서 시리즈를 마무리하며 릴리스/업데이트 체크리스트를 정리합니다.

