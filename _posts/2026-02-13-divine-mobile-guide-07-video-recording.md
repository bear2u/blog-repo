---
layout: post
title: "diVine(OpenVine) 완벽 가이드 (07) - 비디오 녹화 UX와 파이프라인"
date: 2026-02-13
permalink: /divine-mobile-guide-07-video-recording/
author: divinevideo
categories: [모바일, Flutter]
tags: [Video, Camera, Recorder, Flutter, Riverpod, diVine]
original_url: "https://github.com/divinevideo/divine-mobile"
excerpt: "비디오 녹화 화면이 어떻게 초기화되고, 클립/오토세이브가 어떤 방식으로 엮이는지 코드 경로 중심으로 정리합니다."
---

## 시작 지점: VideoRecorderScreen

비디오 녹화 기능의 가장 좋은 진입점은:

- `mobile/lib/screens/video_recorder_screen.dart`

입니다. 이 화면은 “카메라 프리뷰 + 녹화 컨트롤 + 세그먼트(클립) 편집”을 한 화면에 모아서 제공합니다.

---

## 초기화 흐름(권한/리소스/오토세이브)

`VideoRecorderScreen`의 `initState()`를 보면 핵심 단계가 정리되어 있습니다.

1. `WidgetsBindingObserver` 등록(라이프사이클 대응)
2. post-frame 콜백에서:
   - `_initializeCamera()`
   - `_checkAutosavedChanges()`

여기서 중요한 디테일은 “카메라 초기화 전에 리소스 정리”가 들어간다는 점입니다.

- `_disposeVideoControllers()`를 먼저 호출해 기존 비디오 컨트롤러를 해제

이건 모바일에서 흔히 생기는 “카메라/플레이어 리소스 잠금” 문제를 회피하는 전형적인 패턴입니다.

---

## Riverpod 연결: videoRecorderProvider / clipManagerProvider

화면은 Riverpod provider를 통해 상태와 로직을 가져옵니다.

- `videoRecorderProvider`: 카메라 초기화/파괴/라이프사이클 핸들링
- `clipManagerProvider`: 클립(세그먼트) 상태 관리

즉, 화면은 “UI”만 갖고, 녹화의 핵심 상태는 notifier/provider로 내려갑니다.

---

## 카메라 서비스 구현 위치

카메라 구현은 한 파일에 몰려있지 않고, 플랫폼별로 분리되어 있습니다.

- `mobile/lib/services/video_recorder/camera/camera_base_service.dart`
- `mobile/lib/services/video_recorder/camera/camera_mobile_service.dart`
- `mobile/lib/services/video_recorder/camera/camera_macos_service.dart`

그리고 별도로 macOS 관련 네이티브 연동이:

- `mobile/lib/services/camera/native_macos_camera.dart`

에 있습니다.

이 구조의 장점은 “플랫폼별 API/권한/장치 관리”를 서비스 레이어에서 흡수하고,
UI는 같은 방식으로 provider만 호출할 수 있다는 점입니다.

---

## UI 구성 요소(위젯 단위로 분해)

`VideoRecorderScreen`의 `build()`는 큰 레이아웃만 잡고, 실제 기능은 위젯으로 쪼개져 있습니다.

- `VideoRecorderCameraPreview`: 프리뷰 렌더링
- `VideoRecorderSegmentBar`: 세그먼트 진행/클립 표시
- `VideoRecorderTopBar`: 상단 컨트롤(닫기/확정 등)
- `RecordButton`: 녹화 버튼
- `VideoRecorderBottomBar`: 하단 옵션/컨트롤
- `VideoRecorderCountdownOverlay`: 카운트다운 UI

“녹화 UX를 바꾸고 싶다”면, 이 위젯들부터 보면 됩니다.

---

## 오토세이브 복구(사용자 경험)

`_checkAutosavedChanges()`는 오토세이브된 드래프트를 찾아 복구 시트를 띄웁니다.
여기서 확인할 파일들:

- `mobile/lib/services/draft_storage_service.dart`
- `mobile/lib/widgets/video_clip_editor/sheets/video_editor_restore_autosave_sheet.dart`

즉, 녹화 UX는 단순히 “카메라 실행”이 아니라:

1. 클립 상태가 이미 로드되어 있는지 체크
2. 아니면 오토세이브 드래프트를 확인
3. 있으면 복구 UI를 노출

이라는 “세션 복원 UX”까지 포함합니다.

---

*다음 글에서는 녹화된 비디오가 업로드되고, 결국 kind 32222 이벤트로 퍼블리시되는 과정(`UploadManager`, `VideoEventPublisher`)을 따라갑니다.*

