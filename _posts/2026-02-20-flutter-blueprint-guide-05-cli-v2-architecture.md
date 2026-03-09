---
layout: post
title: "flutter_blueprint 가이드 (05) - CLI v2 아키텍처"
date: 2026-02-20
permalink: /flutter-blueprint-guide-05-cli-v2-architecture/
author: chirag640
categories: ['개발 도구', 'Flutter']
tags: [flutter_blueprint, CliRunnerV2, CommandRegistry, Result]
original_url: "https://github.com/chirag640/flutter_blueprint-Package"
excerpt: "v2에서 도입된 CommandRegistry, Result 타입, DI 구조가 유지보수성을 어떻게 개선했는지 정리합니다."
---

## 엔트리포인트는 얇아졌다

`bin/flutter_blueprint.dart`는 `CliRunnerV2`만 호출합니다.

```dart
final runner = CliRunnerV2();
await runner.run(arguments);
```

핵심은 거대한 단일 러너에서 명령 디스패처 구조로 옮겨갔다는 점입니다.

---

## 디스패치 흐름

`CliRunnerV2`는 다음 역할만 수행합니다.

1. 전역 플래그 파싱(`--help`, `--version`)
2. 명령 이름 추출
3. `CommandRegistry`로 위임
4. 업데이트 알림 처리

비즈니스 로직은 개별 커맨드로 분리돼 테스트와 변경이 쉬워집니다.

---

## `Result<T, E>`로 예외 흐름을 명시화

커맨드 실행 결과는 `Result<CommandResult, CommandError>`를 반환합니다.

- 성공/실패 분기 강제
- try/catch 남용 감소
- 호출자에서 실패 처리 정책 통일

이 패턴은 CLI뿐 아니라 생성 파이프라인에도 동일하게 적용됩니다.

---

## DI 포인트

`CommandContext`와 `ServiceLocator`를 통해 로거/프롬프터 같은 의존성을 주입합니다.

직접 생성보다 주입 중심으로 바뀌면 테스트 더블 교체가 쉬워지고, 커맨드 단위 테스트 비용이 낮아집니다.

다음 글에서는 템플릿 레지스트리와 프로젝트 생성 파이프라인을 이어서 봅니다.

