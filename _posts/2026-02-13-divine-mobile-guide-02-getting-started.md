---
layout: post
title: "diVine(OpenVine) 완벽 가이드 (02) - 개발 환경과 빠른 시작"
date: 2026-02-13
permalink: /divine-mobile-guide-02-getting-started/
author: divinevideo
categories: [모바일, Flutter]
tags: [diVine, OpenVine, Flutter, iOS, Android, Web, macOS]
original_url: "https://github.com/divinevideo/divine-mobile"
excerpt: "`mobile/` 디렉토리에서 앱을 실행하고, `.env`와 빌드 스크립트를 이해합니다."
---

## 전제: Flutter가 설치되어 있어야 한다

이 레포의 제품 앱은 `mobile/` 아래에 있는 Flutter 프로젝트입니다.
가장 먼저 확인할 것은 언제나:

```bash
flutter doctor
```

---

## 기본 실행(가장 빠른 루트)

레포 루트가 아니라 `mobile/`에서 시작합니다.

```bash
git clone https://github.com/divinevideo/divine-mobile
cd divine-mobile/mobile

flutter pub get
./run_dev.sh chrome debug
```

`mobile/run_dev.sh`는 개발 실행을 단순화합니다.

- 기본 디바이스: 인자를 주지 않으면 `chrome`
- iOS/Android 지정 시: `flutter devices --machine` 결과에서 적당한 디바이스를 골라 실행
- `.env`가 있으면: 특정 값들을 `--dart-define=...`로 주입

---

## `.env` 구성(선택)

레포에는 `mobile/.env.example`가 있습니다.
여기에는 (예: Zendesk/Proofmode 같은) 기능을 위해 필요한 값들이 들어갈 수 있습니다.

```bash
cd mobile
cp .env.example .env
```

실제 값은 환경/팀 설정에 따라 달라서, 공개 블로그에서는 “무엇이 들어가는지”만 알고 있으면 됩니다.
커밋에는 절대 `.env`를 포함하지 않는 것이 안전합니다.

---

## 문서가 말하는 업로드 토큰(CF_STREAM_TOKEN)에 대해

`docs/README.md`와 `docs/CF_STREAM_SETUP.md`는 업로드 관련 설정으로 `CF_STREAM_TOKEN`을 강조합니다.
하지만 현재 코드 전체에서 `CF_STREAM_TOKEN` 문자열은 문서에만 등장합니다.

확인 방법(레포에서 직접 체크):

```bash
rg -n "CF_STREAM_TOKEN" -S .
```

따라서 “업로드가 안 된다” 같은 문제를 디버깅할 때는 문서만 믿기보다는:

- `mobile/run_dev.sh` (실제 `--dart-define` 구성)
- `mobile/lib/services/upload_manager.dart` (업로드 상태/메타데이터 처리)
- `mobile/lib/services/video_event_publisher.dart` (퍼블리시 단계)
- `mobile/lib/services/blossom_upload_service.dart` (업로드 구현)

을 같이 봐야 합니다.

---

## 플랫폼별 빌드 스크립트(요약)

`mobile/`에는 플랫폼/배포 타깃별 스크립트가 여러 개 있습니다.

- `build_android.sh`
- `build_ios.sh`
- `build_macos.sh`
- `build_native.sh` (iOS/macOS 통합 빌드 스크립트)
- `build_web_optimized.sh`

특히 iOS/macOS는 CocoaPods 동기화 문제가 자주 터지기 때문에, 레포 문서(`docs/BUILD_SCRIPTS_README.md`)는 “Podfile.lock과 sandbox 불일치”를 자동으로 처리하도록 스크립트를 제공한다고 설명합니다.

---

## 체크리스트(최소)

```bash
cd divine-mobile/mobile
flutter pub get
flutter analyze
flutter test
```

이 3개가 통과하면, 최소한의 개발/테스트 루프는 돈다고 보면 됩니다.

---

*다음 글에서는 `docs/`와 `mobile/lib/`가 어떻게 나뉘어 있고, 무엇을 어디서 찾아야 하는지 레포 구조를 정리합니다.*

