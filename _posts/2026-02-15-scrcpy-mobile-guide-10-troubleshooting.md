---
layout: post
title: "scrcpy-mobile(Scrcpy Remote) 가이드 (10) - 트러블슈팅: 연결 실패, 인증/포트, 성능/오디오"
date: 2026-02-15
permalink: /scrcpy-mobile-guide-10-troubleshooting/
author: wsvn53
categories: [개발 도구, 모바일]
tags: [scrcpy-mobile, Troubleshooting, ADB, Wireless Debugging, Network, VNC]
original_url: "https://github.com/wsvn53/scrcpy-mobile"
excerpt: "ADB/VNC 모드 전환, 무선 디버깅 페어링, 5555 포트, 네트워크 격리, 오디오 조건(Android 11+) 등 scrcpy-mobile에서 흔한 이슈를 체크리스트로 정리합니다."
---

## 1) ADB 모드로 안 붙는다

체크리스트:

- Android에서 **무선 디버깅/ADB**가 켜져 있는지
- (필요한 경우) `adb tcpip 5555`가 적용돼 있는지
- iPhone과 Android가 **같은 네트워크**인지
- 공유기에서 **AP isolation(클라이언트 격리)** 같은 옵션이 켜져 있지 않은지
- VPN/프록시를 끄고 다시 시도했는지

---

## 2) 인증(Authorize)에서 막힌다

가능한 원인:

- Android에서 디버깅 승인 팝업을 놓침
- 이전에 저장된 adbkey/승인 정보가 꼬임

대응:

- Android의 무선 디버깅 화면에서 “페어링/승인 관리”를 확인
- 앱의 **adbkey 관리** 기능을 활용해 키를 새로 생성하거나(키 회전), 정상 키를 import

---

## 3) 페어링 코드가 안 된다(Android 10+)

체크리스트:

- Android가 정말 10+인지
- `Developer Options` → `Wireless debugging` → `Pair device with pairing code` 화면에서 코드를 보고 있는지
- 동일 네트워크인지(페어링도 네트워크 영향을 받음)

---

## 4) VNC 모드가 느리거나 불안정하다

README 자체가 VNC 모드의 한계를 설명합니다.

- websockify로 프록시된 VNC 포트에만 연결 가능
- noVNC 기반이라 성능/경험이 좋지 않을 수 있음

따라서 가능하면:

- VNC 대신 **ADB 모드로 전환**하는 쪽이 정석입니다.

---

## 5) 오디오가 안 나온다

README 조건:

- 오디오 포워딩은 **Android 11+**에서 지원

Android 버전을 먼저 확인하고, 네트워크 품질 문제(끊김/지연)도 같이 점검하는 편이 빠릅니다.

---

## 다음 단계

여기까지의 설정이 안정화되면, 실사용에서는 보통 다음 2가지를 추가로 다듬게 됩니다.

1. `scrcpy2://...` 링크를 만들어 “원탭 연결”로 고정
2. 네트워크에 맞게 비트레이트/해상도를 보수적으로 설정(끊김 최소화)

