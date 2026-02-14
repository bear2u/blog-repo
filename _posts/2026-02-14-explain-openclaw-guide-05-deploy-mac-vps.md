---
layout: post
title: "Explain OpenClaw 완벽 가이드 (05) - 배포 1: Mac mini/VPS"
date: 2026-02-14
permalink: /explain-openclaw-guide-05-deploy-mac-vps/
author: centminmod
categories: [AI 에이전트, OpenClaw]
tags: [OpenClaw, Deployment, Mac mini, VPS, Tailscale, SSH, Hardening]
original_url: "https://github.com/centminmod/explain-openclaw"
excerpt: "가장 현실적인 2가지 배포: 로컬 우선 Mac mini와 원격 Isolated VPS를 비교하고, loopback+터널 중심의 안전한 운영 패턴을 정리합니다."
---
## Mac mini(로컬 우선): 가장 안전한 기본값

Explain OpenClaw의 배포 런북은 Mac mini를 "가장 안전한 기본"으로 봅니다.

권장 포스처(요약):
- `gateway.bind: "loopback"`
- DM 정책: `pairing` 또는 `allowlist`
- 채널/도구는 필요한 것만
- 설정 변경 후 `openclaw security audit --deep`

설치/온보딩:
```bash
curl -fsSL https://openclaw.ai/install.sh | bash
openclaw onboard --install-daemon
openclaw security audit --deep
```

원격 접근은 포트 공개 대신 터널이 기본입니다.
```bash
ssh -N -L 18789:127.0.0.1:18789 user@mac-mini
```

---

## Isolated VPS(원격): 하드닝이 전제

VPS는 공인 IP를 가지므로, 문서가 반복 경고하는 포인트는 다음입니다.

- 18789를 공개 인터넷에 열지 말 것
- loopback 유지 + SSH/Tailscale로 접근
- 방화벽/업데이트/전용 유저 같은 기본 하드닝부터

베이스라인(UFW 예시):
```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw enable
sudo ufw status
```

---

## 다음 글

다음 글에서는 서버리스 Cloudflare Moltworker와 로컬 모델(Docker Model Runner) 배포를 정리합니다.

- 다음: [Explain OpenClaw (06) - 배포 2: Moltworker/로컬 모델](/blog-repo/explain-openclaw-guide-06-deploy-moltworker-local-models/)
