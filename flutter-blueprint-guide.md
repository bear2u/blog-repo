---
layout: page
title: flutter_blueprint 가이드
permalink: /flutter-blueprint-guide/
icon: fas fa-terminal
---

# flutter_blueprint 완벽 가이드

> **Flutter 프로젝트를 빠르게 시작하되, 운영 단계까지 고려한 구조로 생성하는 CLI**

`flutter_blueprint`는 단순 템플릿 복사가 아니라, 상태관리/플랫폼/보안/API/테스트/CI 옵션을 조합해 실무형 Flutter 프로젝트를 생성하는 스캐폴딩 도구입니다.

- 원문 저장소: https://github.com/chirag640/flutter_blueprint-Package
- 패키지: https://pub.dev/packages/flutter_blueprint
- 현재 분석 기준 버전: `2.0.2`

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/flutter-blueprint-guide-01-intro/) | 이 도구가 해결하는 문제와 핵심 가치 |
| 02 | [설치와 첫 실행](/blog-repo/flutter-blueprint-guide-02-installation-and-first-run/) | 설치, `init` 첫 실행, 생성 확인 |
| 03 | [init 옵션 해설](/blog-repo/flutter-blueprint-guide-03-init-command-options/) | 상태관리/플랫폼/API/테스트 플래그 설계 |
| 04 | [생성 결과 구조](/blog-repo/flutter-blueprint-guide-04-generated-project-structure/) | 생성된 앱 디렉토리와 파일 계층 읽는 법 |
| 05 | [CLI v2 아키텍처](/blog-repo/flutter-blueprint-guide-05-cli-v2-architecture/) | `CliRunnerV2`, `CommandRegistry`, `Result` |
| 06 | [템플릿 렌더링 파이프라인](/blog-repo/flutter-blueprint-guide-06-template-registry-and-generator/) | `TemplateRegistry`와 `ProjectGenerator` 동작 |
| 07 | [API/보안 설정](/blog-repo/flutter-blueprint-guide-07-api-config-and-security/) | 백엔드 프리셋, 보안 인터셉터, 인증 흐름 |
| 08 | [분석·최적화·리팩토링](/blog-repo/flutter-blueprint-guide-08-analyze-optimize-refactor/) | `analyze`, `optimize`, `refactor` 실무 사용 |
| 09 | [팀 공유 설정](/blog-repo/flutter-blueprint-guide-09-team-shared-config/) | `share` 명령과 조직 표준 템플릿 운영 |
| 10 | [운영 체크리스트](/blog-repo/flutter-blueprint-guide-10-release-operations-checklist/) | 릴리스/업데이트/품질 루틴 정리 |

---

## 이 시리즈에서 얻는 것

- 생성 옵션을 감으로 고르지 않고, 프로젝트 성격에 맞춰 선택하는 기준
- CLI 내부 구조를 이해해 확장 포인트를 빠르게 찾는 방법
- 보안/API/품질 명령을 개발 흐름에 붙이는 실전 운영 패턴

---

## 빠른 시작

```bash
dart pub global activate flutter_blueprint
flutter_blueprint init my_app --state riverpod --api --theme --tests
```

---

## 기술 포인트

| 영역 | 핵심 |
|------|------|
| CLI 계층 | `CliRunnerV2` + `CommandRegistry` 기반 분리 |
| 에러 처리 | `Result<T, E>` 중심의 명시적 실패 처리 |
| 템플릿 계층 | `ITemplateRenderer` / `TemplateRegistry`로 선택 |
| 프로젝트 생성 | 병렬 파일 쓰기 + CI 템플릿 + `blueprint.yaml` |
| 보안 | OWASP 헤더, 인증 인터셉터, 레이트 리밋, SSRF 방어 |

