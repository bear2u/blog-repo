---
layout: post
title: "ClawRouter 완벽 가이드 (06) - 지갑과 결제(x402): 402 -> 서명 -> 재시도"
date: 2026-02-15
permalink: /clawrouter-guide-06-wallet-x402/
author: BlockRun
categories: [AI 에이전트, ClawRouter]
tags: [x402, USDC, Base, Wallet, EIP-712, Security]
original_url: "https://github.com/BlockRunAI/ClawRouter"
excerpt: "x402(402 Payment Required) 결제 플로우, 지갑 키 파일 위치/우선순위, 백업/복구, /wallet 사용법을 정리합니다."
---

## x402 결제 플로우(문서 요지)

`docs/architecture.md` 기준으로 결제는 대략 다음 흐름입니다.

```text
1) 요청 -> BlockRun API
2) 402 Payment Required (가격/수신자 정보 포함)
3) 지갑 키로 EIP-712 타입데이터 서명
4) 결제 헤더를 붙여 재요청
5) 200 OK 응답
```

---

## 지갑 키 저장/백업

`docs/configuration.md`는 지갑 키 파일 경로와 백업의 중요성을 강조합니다.

- 키 파일: `~/.openclaw/blockrun/wallet.key`
- VPS/머신을 종료하기 전에 키를 백업하지 않으면 잔액을 복구할 수 없음

백업 예시:

```bash
cp ~/.openclaw/blockrun/wallet.key ~/backup-wallet.key
chmod 600 ~/backup-wallet.key
```

---

## /wallet 커맨드

문서는 지갑 관리를 위한 커맨드를 소개합니다.

```text
/wallet
/wallet export
```

운영 관점에서는 `/wallet export`를 "백업 직전 마지막 수단"으로 취급하는 편이 안전합니다.

---

## OpenClaw 보안 스캐너 경고(문서 입장)

`openclaw.security.json`와 troubleshooting 문서는 다음을 강조합니다.

- 지갑 키는 서명에 로컬로만 사용
- 네트워크로 전송되는 것은 "키"가 아니라 "서명"(증명)

---

## 다음 글

다음 글에서는 /stats와 로그를 통해 "얼마나 아꼈는지"를 확인하는 방법을 정리합니다.

- 다음: [ClawRouter (07) - 관측과 비용 절감](/blog-repo/clawrouter-guide-07-observability-stats/)
