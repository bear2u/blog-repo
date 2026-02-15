---
layout: page
title: scrcpy-mobile(Scrcpy Remote) 가이드
permalink: /scrcpy-mobile-guide/
icon: fas fa-mobile
---

# scrcpy-mobile(Scrcpy Remote) 완벽 가이드

> **아이폰(또는 모바일)에서 Wi-Fi ADB로 Android 기기를 원격 제어하는 scrcpy 모바일 포팅**

**scrcpy-mobile**은 Genymobile의 `scrcpy`를 모바일 플랫폼으로 포팅한 프로젝트로, iPhone(또는 모바일 기기)에서 Android 디바이스를 원격으로 조작할 수 있게 합니다. README 기준으로는 **iOS에서 Android를 제어하는 흐름이 중심**이며(향후 Android→Android도 지원 예정), 기본 연결은 **ADB over Wi-Fi(무선 디버깅/5555)** 또는 **VNC(noVNC 기반)** 모드를 제공합니다.

- 원문 저장소: https://github.com/wsvn53/scrcpy-mobile
- 기반 프로젝트(scrcpy): https://github.com/Genymobile/scrcpy
- App Store: https://apps.apple.com/us/app/scrcpy-remote/id1629352527

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/scrcpy-mobile-guide-01-intro/) | scrcpy-mobile이 하는 일, 지원 범위/제약 |
| 02 | [설치와 준비](/blog-repo/scrcpy-mobile-guide-02-install-and-prereqs/) | App 설치, Android 개발자 옵션, 네트워크 준비 |
| 03 | [연결 모드](/blog-repo/scrcpy-mobile-guide-03-connection-modes/) | ADB 모드 vs VNC 모드, 언제 무엇을 쓰나 |
| 04 | [ADB Wi-Fi 시작](/blog-repo/scrcpy-mobile-guide-04-adb-tcpip-5555/) | `adb tcpip 5555` 설정, 인증/연결 흐름 |
| 05 | [페어링 코드](/blog-repo/scrcpy-mobile-guide-05-pairing-code/) | Android 10+ 무선 디버깅 페어링(코드)로 안전하게 연결 |
| 06 | [URL Scheme/단축어](/blog-repo/scrcpy-mobile-guide-06-url-scheme-and-shortcuts/) | `scrcpy2://`로 빠른 접속, iOS Shortcuts 구성 |
| 07 | [조작/입력 기능](/blog-repo/scrcpy-mobile-guide-07-controls-and-input/) | 제스처, 내비게이션 버튼, 키보드/클립보드, 오디오 |
| 08 | [adbkey 관리](/blog-repo/scrcpy-mobile-guide-08-adbkey-management/) | 키 생성/가져오기/내보내기, 키 회전/보안 팁 |
| 09 | [소스 빌드](/blog-repo/scrcpy-mobile-guide-09-building-from-source/) | `make libs`, `scrcpy-server` 빌드, Xcode 실행 |
| 10 | [트러블슈팅](/blog-repo/scrcpy-mobile-guide-10-troubleshooting/) | 연결 실패, 인증/포트/네트워크, 성능/오디오 이슈 |

