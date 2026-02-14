---
layout: post
title: "Explain OpenClaw 완벽 가이드 (07) - 프라이버시/하드닝 체크리스트"
date: 2026-02-14
permalink: /explain-openclaw-guide-07-privacy-hardening/
author: centminmod
categories: [AI 에이전트, OpenClaw]
tags: [OpenClaw, Privacy, Threat Model, Hardening, Configuration, mDNS]
original_url: "https://github.com/centminmod/explain-openclaw"
excerpt: "DM/그룹 정책, loopback 바인딩, mDNS, trusted proxies, 최소 도구 표면 같은 핵심 하드닝 항목과 고프라이버시 설정의 요지를 정리합니다."
---
## 하드닝의 핵심: 누가 트리거하고, 무엇이 노출되고, 무엇을 할 수 있는가

Explain OpenClaw의 hardening-checklist/threat-model 문서는 보안을 크게 세 축으로 설명합니다.

- Trigger surface: DMs/그룹에서 누가 봇을 호출할 수 있는가
- Exposure surface: Gateway가 어디에 노출되는가(bind/프록시/mDNS)
- Action surface: 도구가 어떤 권한으로 동작하는가

---

## 추천 체크리스트(핵심만)

- DMs: `pairing` 또는 `allowlist`
- 그룹: 멘션 필요 + 허용 그룹만
- 네트워크: `gateway.bind: loopback` 유지, 원격은 SSH/Tailscale
- 비루프백 바인딩은 auth 필수
- `openclaw security audit --deep` 주기적으로
- 공유 네트워크에서는 mDNS discovery 끄기

```bash
openclaw config set discovery.mdns off
```

리버스 프록시를 쓴다면 trusted proxies를 검토합니다.
```bash
openclaw config set gateway.trustedProxies '["127.0.0.1"]'
```

---

## 고프라이버시 config 예시에서 뽑아야 할 것

Explain OpenClaw의 high-privacy-config 예시는 다음 방향으로 수렴합니다.

- loopback 바인딩 + 토큰 인증
- tools minimal
- plugins off
- sandbox를 더 강하게
- 민감 로그 마스킹

---

## 다음 글

마지막 글에서는 프롬프트 인젝션/공급망/인시던트 대응, 그리고 비용/리소스 최적화를 묶어 정리합니다.

- 다음: [Explain OpenClaw (08) - 최악의 시나리오와 운영 최적화](/blog-repo/explain-openclaw-guide-08-worst-case-ops/)
