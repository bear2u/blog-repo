---
layout: default
title: "OpenClaw 완벽 가이드"
permalink: /openclaw-guide/
---

<div class="guide-container">

# OpenClaw 완벽 가이드

**🦞 EXFOLIATE! EXFOLIATE!** — 직접 운영하는 개인 AI 어시스턴트 OpenClaw의 완벽 가이드입니다.

<div class="guide-meta">
<span class="author">원저자: Peter Steinberger & Community</span>
<span class="source"><a href="https://github.com/openclaw/openclaw">GitHub Repository</a></span>
</div>

---

## 목차

### Part 1: 기초
<div class="chapter-list">

<div class="chapter-item">
<span class="chapter-number">01</span>
<a href="{{ '/openclaw-guide-01-intro/' | relative_url }}">소개 및 개요</a>
<p>OpenClaw란? 핵심 기능, 빠른 시작, 지원 플랫폼</p>
</div>

<div class="chapter-item">
<span class="chapter-number">02</span>
<a href="{{ '/openclaw-guide-02-architecture/' | relative_url }}">Gateway 아키텍처</a>
<p>WebSocket 프로토콜, 컴포넌트, 클라이언트 흐름</p>
</div>

<div class="chapter-item">
<span class="chapter-number">03</span>
<a href="{{ '/openclaw-guide-03-installation/' | relative_url }}">설치 및 설정</a>
<p>npm/Docker 설치, 온보딩 마법사, 설정 파일</p>
</div>

</div>

### Part 2: 채널 & 통합
<div class="chapter-list">

<div class="chapter-item">
<span class="chapter-number">04</span>
<a href="{{ '/openclaw-guide-04-channels/' | relative_url }}">메시징 채널</a>
<p>WhatsApp, Telegram, Discord, Slack, Signal, iMessage 등</p>
</div>

<div class="chapter-item">
<span class="chapter-number">05</span>
<a href="{{ '/openclaw-guide-05-skills/' | relative_url }}">스킬 시스템</a>
<p>번들 스킬, 커스텀 스킬, ClawHub, 스킬 게이팅</p>
</div>

<div class="chapter-item">
<span class="chapter-number">06</span>
<a href="{{ '/openclaw-guide-06-tools/' | relative_url }}">도구 & 브라우저</a>
<p>브라우저 제어, 캔버스, 노드 명령어</p>
</div>

</div>

### Part 3: 고급 기능
<div class="chapter-list">

<div class="chapter-item">
<span class="chapter-number">07</span>
<a href="{{ '/openclaw-guide-07-voice/' | relative_url }}">음성 & Talk Mode</a>
<p>Voice Wake, ElevenLabs TTS, 음성 대화</p>
</div>

<div class="chapter-item">
<span class="chapter-number">08</span>
<a href="{{ '/openclaw-guide-08-hooks/' | relative_url }}">훅 & 자동화</a>
<p>이벤트 훅, Webhook, Cron 작업, Gmail 트리거</p>
</div>

<div class="chapter-item">
<span class="chapter-number">09</span>
<a href="{{ '/openclaw-guide-09-apps/' | relative_url }}">앱 & 노드</a>
<p>macOS 앱, iOS/Android 노드, 원격 게이트웨이</p>
</div>

<div class="chapter-item">
<span class="chapter-number">10</span>
<a href="{{ '/openclaw-guide-10-security/' | relative_url }}">보안 & 배포</a>
<p>보안 모델, 샌드박싱, Docker, Tailscale</p>
</div>

</div>

---

## 핵심 개념

| 개념 | 설명 |
|------|------|
| **Gateway** | 모든 채널과 클라이언트를 연결하는 WebSocket 컨트롤 플레인 |
| **채널** | WhatsApp, Telegram 등 메시징 플랫폼 연결 |
| **스킬** | 에이전트에게 도구 사용법을 가르치는 SKILL.md 파일 |
| **노드** | macOS/iOS/Android 디바이스 기능 노출 |
| **훅** | 이벤트 기반 자동화 스크립트 |

---

## 빠른 시작

```bash
# 설치 (Node ≥22 필요)
npm install -g openclaw@latest

# 온보딩 마법사 실행
openclaw onboard --install-daemon

# Gateway 시작
openclaw gateway --port 18789 --verbose

# 메시지 전송
openclaw agent --message "안녕하세요!" --thinking high
```

---

## 지원 채널

<div class="model-table">

| 채널 | 설명 | 설정 난이도 |
|------|------|-------------|
| WhatsApp | Baileys 기반 QR 페어링 | ⭐⭐ |
| Telegram | grammY 봇 API | ⭐ |
| Discord | discord.js 봇 | ⭐⭐ |
| Slack | Bolt SDK 앱 | ⭐⭐ |
| Signal | signal-cli | ⭐⭐⭐ |
| iMessage | BlueBubbles (권장) | ⭐⭐⭐ |
| WebChat | 내장 웹 UI | ⭐ |

</div>

---

<div class="guide-footer">
<p>이 가이드는 <a href="https://github.com/openclaw/openclaw">OpenClaw GitHub 저장소</a>를 분석하여 작성되었습니다.</p>
</div>

</div>

<style>
.guide-container {
  max-width: 800px;
  margin: 0 auto;
}

.guide-meta {
  display: flex;
  gap: 20px;
  color: #666;
  font-size: 0.9rem;
  margin-bottom: 20px;
}

.chapter-list {
  display: flex;
  flex-direction: column;
  gap: 15px;
  margin: 20px 0;
}

.chapter-item {
  display: flex;
  align-items: flex-start;
  gap: 15px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid #FF4500;
}

.chapter-number {
  font-size: 1.5rem;
  font-weight: bold;
  color: #FF4500;
  min-width: 40px;
}

.chapter-item a {
  font-size: 1.1rem;
  font-weight: 600;
  color: #333;
  text-decoration: none;
}

.chapter-item a:hover {
  color: #FF4500;
}

.chapter-item p {
  margin: 5px 0 0 0;
  color: #666;
  font-size: 0.9rem;
}

.guide-footer {
  margin-top: 40px;
  padding-top: 20px;
  border-top: 1px solid #eee;
  text-align: center;
  color: #666;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 20px 0;
}

th, td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #eee;
}

th {
  background: #f8f9fa;
  font-weight: 600;
}
</style>
