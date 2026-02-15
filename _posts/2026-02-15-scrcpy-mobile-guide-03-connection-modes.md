---
layout: post
title: "scrcpy-mobile(Scrcpy Remote) 가이드 (03) - 연결 모드: ADB 모드 vs VNC 모드"
date: 2026-02-15
permalink: /scrcpy-mobile-guide-03-connection-modes/
author: wsvn53
categories: [개발 도구, 모바일]
tags: [scrcpy-mobile, ADB, VNC, noVNC, Connection]
original_url: "https://github.com/wsvn53/scrcpy-mobile"
excerpt: "scrcpy-mobile은 ADB Wi-Fi 모드와 VNC 모드를 제공합니다. 각각의 장단점과 전환 방법(Host/Port, URL Scheme)을 정리합니다."
---

## ADB 모드: 기본 추천 경로

README 기준으로 scrcpy-mobile의 주요 기능은 **ADB over Wi-Fi 기반 scrcpy**입니다.  
따라서 “성능/사용성”이 중요하면 ADB 모드를 먼저 시도하는 것이 합리적입니다.

### ADB 모드로 전환하는 방법(README)

README에 나온 전환 방법은 3가지입니다.

1. URL Scheme: `scrcpy2://adb`
2. Host 입력칸에 `adb` 입력 후 Connect
3. Port 입력칸에 `5555` 입력 후 Connect

그리고 Android 쪽에서 `adb tcpip 5555`가 활성화돼 있어야 한다고 안내합니다.

---

## VNC 모드: “대안”에 가깝다

README에 따르면 앱 설치 후 기본 모드는 VNC이며, VNC 모드로 돌아가는 방법도 3가지가 있습니다.

1. URL Scheme: `scrcpy2://vnc`
2. Host 입력칸에 `vnc` 입력 후 Connect
3. Port 입력칸에 `5900` 입력 후 Connect

다만 README는 VNC 모드에 대해 다음을 경고합니다.

- websockify로 프록시된 VNC 포트만 연결 가능
- noVNC 기반이라 성능/경험이 좋지 않을 수 있음

요약하면:

- **먼저 ADB 모드**로 해결을 시도하고
- 네트워크/환경 제약이 있으면 **VNC 모드**를 “차선”으로 고려하는 편이 낫습니다.

---

## 어떤 모드를 언제 쓰나(실전 기준)

- ADB 모드가 되는 환경이라면: ADB 모드
- ADB가 막히는 환경(포트/정책/기기 제한)에서 급한 확인만 필요: VNC 모드 검토

다음 장에서는 ADB 모드에서 핵심인 **`adb tcpip 5555`와 인증/연결 흐름**을 단계별로 정리합니다.

