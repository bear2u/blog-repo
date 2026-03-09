---
layout: post
title: "flutter_blueprint 가이드 (01) - 소개: 왜 이 CLI가 필요한가"
date: 2026-02-20
permalink: /flutter-blueprint-guide-01-intro/
author: chirag640
categories: ['개발 도구', 'Flutter']
tags: [flutter_blueprint, Flutter, CLI, Scaffolding, Architecture]
original_url: "https://github.com/chirag640/flutter_blueprint-Package"
excerpt: "flutter_blueprint가 어떤 개발 병목을 줄이고, 어떤 기준으로 실무형 Flutter 프로젝트를 생성하는지 정리합니다."
---

## 문제 정의: Flutter 시작 비용은 생각보다 크다

Flutter 앱을 새로 시작할 때 실제 시간이 많이 드는 구간은 UI가 아니라 초기 구조입니다.

- 상태관리 선택(Provider/Riverpod/Bloc)
- API 계층/인증/스토리지/에러 처리 결정
- 테스트/CI/환경설정 같은 운영 기반 구성

`flutter_blueprint`는 이 초기 설계 결정을 명령 옵션으로 표준화합니다.

---

## flutter_blueprint의 포지셔닝

이 프로젝트는 "코드 생성기"이지만, 지향점은 단순 보일러플레이트가 아닙니다.

- **CLI 실행 환경**: Windows/Linux/macOS
- **생성 결과물 대상**: Android/iOS/Web/Desktop
- **중점**: 생성 직후 바로 기능 개발을 이어갈 수 있는 구조

README 기준으로 `--api`를 사용하면 인증/프로필/홈/설정 기능 골격이 포함된 형태를 만들 수 있습니다.

---

## 핵심 가치 3가지

1. **초기 아키텍처 결정 단축**  
   CLI 옵션으로 팀 합의 항목을 빠르게 고정합니다.
2. **일관된 생성물**  
   같은 옵션이면 비슷한 구조가 나와 코드 리뷰 비용을 낮춥니다.
3. **운영 고려 내장**  
   테스트, CI, 보안 헤더 등 "나중에 붙일 것"을 처음부터 반영합니다.

---

## 빠른 예시

```bash
flutter_blueprint init my_app --state riverpod --api --theme --tests
```

한 줄이지만 이 명령은 앱 구조, API 계층, 테마, 테스트 기반을 동시에 생성합니다.

다음 글에서는 설치와 첫 실행 흐름을 실제 명령 순서로 정리합니다.

