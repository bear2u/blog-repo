---
layout: post
title: "diVine(OpenVine) 완벽 가이드 (12) - 빌드, 릴리스, 배포"
date: 2026-02-13
permalink: /divine-mobile-guide-12-build-release-deploy/
author: divinevideo
categories: [모바일, Flutter]
tags: [Build, Release, Deploy, iOS, Android, Web, macOS, Codemagic]
original_url: "https://github.com/divinevideo/divine-mobile"
excerpt: "`mobile/`의 빌드 스크립트와 CocoaPods 동기화 전략, CI 구성(codemagic.yaml)을 중심으로 배포 흐름을 정리합니다."
---

## 빌드 스크립트는 `mobile/`에 모여 있다

diVine(OpenVine)은 플랫폼별 빌드/배포를 스크립트로 정리해 둡니다.
대표적으로:

- `mobile/build_android.sh`
- `mobile/build_ios.sh`
- `mobile/build_macos.sh`
- `mobile/build_native.sh` (iOS/macOS 통합)
- `mobile/build_web_optimized.sh`

이 레포를 “실제로 빌드하는 사람”이라면, 문서보다 먼저 `mobile/`의 스크립트 목록을 확인하는 것이 빠릅니다.

---

## iOS/macOS: CocoaPods 동기화(문서 요약)

`docs/BUILD_SCRIPTS_README.md`는 iOS/macOS 빌드에서 흔히 겪는:

- “Podfile.lock과 sandbox 불일치”

같은 문제를 줄이기 위해, 빌드 과정에서 필요하면 `pod install`을 자동으로 수행하는 전략을 설명합니다.

즉, “수동으로 pods를 맞추는 것”이 아니라, 빌드 루프 안에 “동기화 체크”를 포함시킨 구성입니다.

---

## 개발 실행: `run_dev.sh`

개발 실행은:

- `mobile/run_dev.sh`

가 진입점입니다.
디바이스 선택(Chrome/iOS/Android/macOS)과 `.env` 기반 `--dart-define` 구성을 흡수해서,
반복 실행 비용을 줄이는 역할을 합니다.

---

## CI/CD 힌트: codemagic.yaml

레포 루트에 `codemagic.yaml`이 있어, CI/CD 파이프라인의 힌트를 줍니다.
“어떤 플랫폼을 빌드하고 어떤 아티팩트를 만들려는지”를 파악할 때 유용합니다.

블로그 독자 관점에서는:

- 로컬 스크립트(`mobile/*.sh`)와
- CI 정의(`codemagic.yaml`)

가 서로 어떤 관계인지(중복/재사용/차이)를 비교해서 보는 것이 도움이 됩니다.

---

## 배포 체크리스트 문서

`docs/RELEASE_CHECKLIST.md` 같은 문서는 “릴리스 전 확인할 것”을 정리해 둡니다.
다만 이 레포는 모바일 앱/SDK/백엔드 문서가 함께 존재하므로,
내가 배포하려는 대상(iOS/Android/Web/SDK)이 무엇인지 먼저 정해두고 체크리스트를 읽는 게 안전합니다.

---

## 마무리

이 시리즈의 핵심은 “코드베이스를 처음 보는 사람이, 어디서부터 읽고 어떻게 연결해야 하는지”를 빠르게 잡는 것입니다.

- provider에서 의존성 그래프를 잡고
- 서비스에서 도메인 로직을 확인하고
- 문서로 설계 의도를 보강하고
- 스크립트/CI로 실제 운영 경로를 확인하는 방식으로 접근하면,
큰 레포도 비교적 빠르게 온보딩할 수 있습니다.

