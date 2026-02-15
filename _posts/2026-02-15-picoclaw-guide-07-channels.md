---
layout: post
title: "PicoClaw 가이드 (07) - 채팅 채널: Telegram/Discord/LINE/Slack/OneBot 연결"
date: 2026-02-15
permalink: /picoclaw-guide-07-channels/
author: Sipeed
categories: [AI 에이전트, 개발 도구]
tags: ["PicoClaw", "Gateway", "Telegram", "Discord", "LINE", "Slack", "OneBot"]
original_url: "https://github.com/sipeed/picoclaw#-chat-apps"
excerpt: "PicoClaw gateway는 여러 채팅 앱으로 대화할 수 있습니다. README의 설정 예시를 기준으로 Telegram/Discord/LINE 중심으로 필요한 토큰과 allow list, webhook까지 정리합니다."
---

## gateway 모드가 하는 일

`picoclaw gateway`는 “메시지 수신 채널”을 붙여서 PicoClaw를 봇처럼 운영하는 모드입니다.

핵심은:

- 채널 토큰/자격 증명을 config에 넣고
- `gateway`를 실행하면
- 허용된 사용자(allow list)로부터 온 메시지를 에이전트가 처리한다

입니다.

---

## Telegram(README에서 추천)

README는 Telegram을 “Recommended”로 표시하고, 구성도 단순합니다.

- BotFather로 토큰 발급
- `channels.telegram.enabled=true`와 `token`, 그리고 `allow_from` 설정

운영에서는 allow list를 비워두기보다, 우선 자기 계정 ID만 넣는 편이 안전합니다.

---

## Discord

README는 Discord에 대해:

- bot token 발급
- MESSAGE CONTENT INTENT 등 intents 설정
- user id 확보

같은 “디스코드 쪽 준비”를 자세히 안내합니다.

권한은 최소한부터 시작합니다.

- `Send Messages`
- `Read Message History`

---

## LINE(Webhook)

README는 LINE의 경우 다음이 필요하다고 말합니다.

- Channel Secret / Channel Access Token
- webhook host/port/path 설정
- LINE은 HTTPS webhook을 요구하므로 reverse proxy 또는 ngrok 같은 터널 필요

Docker Compose로 gateway를 띄우는 경우, webhook 포트를 외부로 노출해야 할 수도 있습니다(README에 예시 언급).

---

## Slack/OneBot 등

`config.example.json`에는 Slack, OneBot 같은 항목도 포함돼 있습니다.

- Slack은 bot token/app token을 사용
- OneBot은 websocket URL과 재연결 간격 등을 가짐

이 채널들은 “환경 준비”가 필요한 편이므로, 먼저 Telegram/Discord로 기본 동작을 확보하고 확장하는 편이 낫습니다.

다음 장에서는 게이트웨이/에이전트가 사용하는 워크스페이스 레이아웃과 메모리 파일(IDENTITY/SOUL/MEMORY)을 정리합니다.

