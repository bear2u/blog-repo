---
layout: post
title: "ClawRouter 완벽 가이드 (09) - 보안 모델: 키는 로컬, 전송은 서명"
date: 2026-02-15
permalink: /clawrouter-guide-09-security/
author: BlockRun
categories: [AI 에이전트, ClawRouter]
tags: [Security, Wallet Key, Signing, OpenClaw Security Scanner, Threat Model]
original_url: "https://github.com/BlockRunAI/ClawRouter"
excerpt: "ClawRouter가 지갑 개인키를 로컬에서만 사용한다는 주장(키 대신 서명 전송), OpenClaw 보안 스캐너 경고를 어떻게 이해해야 하는지 정리합니다."
---

## 핵심 주장(문서 기준)

`openclaw.security.json`와 troubleshooting 문서는 다음을 일관되게 말합니다.

- 지갑 개인키는 **로컬에서 서명 생성**에만 사용된다.
- 네트워크로 전송되는 것은 **개인키가 아니라 서명(증명)** 이다.

이 전제가 성립해야 "비수탁(non-custodial)" 결제 모델이 의미가 있습니다.

---

## 보안 스캐너 경고를 어떻게 볼까?

`docs/troubleshooting.md`는 OpenClaw 보안 스캐너가 다음 패턴을 경고할 수 있다고 설명합니다.

- 환경변수 읽기 + 네트워크 요청

ClawRouter는 결제 서명을 위해 지갑 키가 필요하므로 이 패턴이 나타나며,
문서는 이를 "의도된 동작"으로 설명합니다.

운영자 관점 체크리스트:

- 설치한 플러그인이 정확히 `@blockrun/clawrouter`인지 확인
- 소스가 공개되어 있으므로(레포) 필요하면 코드/릴리스를 직접 확인
- 키 파일(`wallet.key`)의 파일 권한(최소 600)과 백업 정책 확립

---

## 다음 글

마지막 글에서는 실제로 많이 마주치는 오류(unknown model, no api key, config validation, 포트 충돌)와 업데이트/재설치 경로를 정리합니다.

- 다음: [ClawRouter (10) - 트러블슈팅/운영](/blog-repo/clawrouter-guide-10-troubleshooting/)
