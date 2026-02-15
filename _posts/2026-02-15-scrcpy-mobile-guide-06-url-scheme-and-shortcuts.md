---
layout: post
title: "scrcpy-mobile(Scrcpy Remote) 가이드 (06) - URL Scheme/단축어: scrcpy2://로 1탭 접속 만들기"
date: 2026-02-15
permalink: /scrcpy-mobile-guide-06-url-scheme-and-shortcuts/
author: wsvn53
categories: [개발 도구, 모바일]
tags: [scrcpy-mobile, iOS, URL Scheme, Shortcuts, Automation]
original_url: "https://github.com/wsvn53/scrcpy-mobile"
excerpt: "scrcpy-mobile은 scrcpy2:// URL scheme을 지원해 접속 설정을 링크로 저장할 수 있습니다. 모드 전환(adb/vnc), 파라미터(bit-rate/max-size) 예시와 Shortcuts 활용법을 정리합니다."
---

## URL Scheme이 왜 중요한가

원격 제어 앱은 “매번 IP/포트/옵션을 입력”하는 순간부터 불편해집니다.  
README가 `scrcpy2://` URL scheme을 제공하는 이유는, 설정을 링크로 저장해서 **바로 접속**하기 위함입니다.

---

## 모드 전환 링크(README)

- ADB 모드: `scrcpy2://adb`
- VNC 모드: `scrcpy2://vnc`

---

## 접속 설정을 포함한 예시(README)

README의 예시:

```text
scrcpy2://example.com:5555?bit-rate=4M&max-size=1080
```

의미를 풀면:

- 호스트: `example.com`
- 포트: `5555`(ADB)
- 옵션: `bit-rate`, `max-size`

네트워크가 불안정하면 `bit-rate`를 낮추거나 `max-size`를 줄이는 방향이 직관적입니다.

---

## iOS Shortcuts(단축어)로 “원탭 연결” 만들기

README는 URL scheme을 복사해서 **Shortcuts.app**에 바로가기를 만들라고 권합니다.

실전 구성(개념):

1. 앱에서 옵션을 맞춘 뒤 **Copy URL Scheme**
2. Shortcuts에서 “URL 열기(Open URL)” 액션으로 해당 링크 실행
3. 홈 화면/위젯/백탭 등에 붙여서 빠르게 호출

다음 장에서는 scrcpy-mobile의 조작/입력 기능(제스처, 내비게이션 버튼, 키보드/클립보드, 오디오)을 정리합니다.

