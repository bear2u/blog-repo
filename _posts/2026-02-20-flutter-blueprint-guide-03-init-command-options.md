---
layout: post
title: "flutter_blueprint 가이드 (03) - init 옵션 설계 읽기"
date: 2026-02-20
permalink: /flutter-blueprint-guide-03-init-command-options/
author: chirag640
categories: ['개발 도구', 'Flutter']
tags: [flutter_blueprint, InitCommand, Riverpod, Bloc, Provider]
original_url: "https://github.com/chirag640/flutter_blueprint-Package"
excerpt: "InitCommand의 플래그 구성을 기반으로 상태관리/플랫폼/API/확장 옵션을 어떻게 고를지 정리합니다."
---

## `init`이 받는 핵심 입력

`lib/src/cli/commands/init_command.dart`를 보면 `init`은 크게 세 묶음의 옵션을 받습니다.

- **아키텍처 축**: `--state`, `--platforms`, `--ci`
- **기능 축**: `--api`, `--theme`, `--tests`, `--hive`, `--pagination`
- **확장 축**: `--analytics`, `--websocket`, `--push-notifications`, `--maps`

---

## 상태관리 선택 기준

```bash
flutter_blueprint init my_app --state provider
flutter_blueprint init my_app --state riverpod
flutter_blueprint init my_app --state bloc
```

- **Provider**: 작은 팀/빠른 진입
- **Riverpod**: 명확한 의존성 추적과 테스트성
- **Bloc**: 이벤트/상태 경계가 분명한 대규모 앱

팀 표준이 있으면 개인 취향보다 표준 우선이 유지보수에 유리합니다.

---

## 템플릿/미리보기/최신 의존성

`init`에는 운영 편의를 위한 옵션도 있습니다.

- `--template <preset>`
- `--preview`
- `--latest-deps`
- `--from-config <name>`

특히 `--preview`는 생성 전 구조를 확인해 재생성 비용을 줄이는 데 유용합니다.

---

## 실무 권장 조합 예시

```bash
flutter_blueprint init commerce_app \
  --state riverpod \
  --platforms mobile,web \
  --api --theme --env --tests --hive \
  --ci github
```

이 조합은 MVP 이후 운영 전환까지 고려한 기본형으로 쓰기 좋습니다.

다음 글에서는 실제 생성된 프로젝트 디렉토리를 읽는 방법을 다룹니다.

