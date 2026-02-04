---
layout: default
title: "TrendRadar 완벽 가이드"
permalink: /trendradar-guide/
---

<div class="guide-container">

# TrendRadar 완벽 가이드

**30초 만에 배포 가능한** AI 기반 트렌드 모니터링 도구 TrendRadar의 아키텍처와 활용법을 완벽하게 분석합니다.

<div class="guide-meta">
<span class="author">원저자: sansan0</span>
<span class="source"><a href="https://github.com/sansan0/TrendRadar">GitHub Repository</a></span>
</div>

---

## 목차

### Part 1: 기초
<div class="chapter-list">

<div class="chapter-item">
<span class="chapter-number">01</span>
<a href="{{ '/trendradar-guide-01-intro/' | relative_url }}">소개 및 개요</a>
<p>TrendRadar란? 주요 기능, 빠른 시작, 프로젝트 구조</p>
</div>

<div class="chapter-item">
<span class="chapter-number">02</span>
<a href="{{ '/trendradar-guide-02-architecture/' | relative_url }}">아키텍처</a>
<p>모듈 구조, 데이터 흐름, 비동기 처리, 설정 시스템</p>
</div>

</div>

### Part 2: 핵심 기능
<div class="chapter-list">

<div class="chapter-item">
<span class="chapter-number">03</span>
<a href="{{ '/trendradar-guide-03-crawler/' | relative_url }}">크롤러 & 데이터 소스</a>
<p>NewsNow API, RSS 피드, 커스텀 크롤러, 필터링 및 중복 제거</p>
</div>

<div class="chapter-item">
<span class="chapter-number">04</span>
<a href="{{ '/trendradar-guide-04-notification/' | relative_url }}">알림 시스템</a>
<p>Telegram, WeChat, Slack, Email, Webhook 등 10개 이상의 알림 채널</p>
</div>

<div class="chapter-item">
<span class="chapter-number">05</span>
<a href="{{ '/trendradar-guide-05-ai-mcp/' | relative_url }}">AI 분석 & MCP</a>
<p>LLM 기반 뉴스 분석, MCP 서버 통합, Claude Desktop 연동</p>
</div>

</div>

### Part 3: 실전 활용
<div class="chapter-list">

<div class="chapter-item">
<span class="chapter-number">06</span>
<a href="{{ '/trendradar-guide-06-deployment/' | relative_url }}">배포 및 활용</a>
<p>GitHub Actions, Docker, 로컬 설치, 설정 최적화, 트러블슈팅</p>
</div>

</div>

---

## 핵심 개념

| 개념 | 설명 |
|------|------|
| **30초 배포** | GitHub Fork + Secrets 설정만으로 즉시 사용 |
| **다중 알림** | Telegram, WeChat, Slack, Email 등 10개+ 채널 |
| **AI 분석** | GPT/Claude로 뉴스 요약 및 번역 |
| **MCP 통합** | AI 에이전트(Claude Desktop)와 직접 연동 |
| **경량화** | 최소 리소스로 최대 효율 |

---

## 빠른 시작

```bash
# 1. Fork 레포지토리
# 2. Settings > Secrets에 환경 변수 추가
#    - TELEGRAM_BOT_TOKEN
#    - TELEGRAM_CHAT_ID
# 3. Actions 탭에서 워크플로우 활성화
# 4. 자동으로 1시간마다 뉴스 알림!
```

---

## 지원 알림 채널

<div class="channels-grid">
  <span class="channel-badge">📱 Telegram</span>
  <span class="channel-badge">💬 WeChat</span>
  <span class="channel-badge">🔔 DingTalk</span>
  <span class="channel-badge">🪶 Feishu</span>
  <span class="channel-badge">💼 Slack</span>
  <span class="channel-badge">📧 Email</span>
  <span class="channel-badge">🔔 ntfy</span>
  <span class="channel-badge">🐕 Bark</span>
  <span class="channel-badge">🔗 Webhook</span>
</div>

---

<div class="guide-footer">
<p>이 가이드는 <a href="https://github.com/sansan0/TrendRadar">TrendRadar GitHub 저장소</a>를 분석하여 작성되었습니다.</p>
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
  border-left: 4px solid #ff6b6b;
}

.chapter-number {
  font-size: 1.5rem;
  font-weight: bold;
  color: #ff6b6b;
  min-width: 40px;
}

.chapter-item a {
  font-size: 1.1rem;
  font-weight: 600;
  color: #333;
  text-decoration: none;
}

.chapter-item a:hover {
  color: #ff6b6b;
}

.chapter-item p {
  margin: 5px 0 0 0;
  color: #666;
  font-size: 0.9rem;
}

.channels-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 20px 0;
}

.channel-badge {
  padding: 8px 16px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border-radius: 20px;
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
