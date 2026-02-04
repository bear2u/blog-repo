---
layout: default
title: "Beads 완벽 가이드"
permalink: /beads-guide/
---

<div class="guide-container">

# Beads 완벽 가이드

AI 에이전트를 위한 분산 Git 기반 이슈 트래커 **Beads**의 아키텍처와 활용법을 완벽하게 분석합니다.

<div class="guide-meta">
<span class="author">원저자: Steve Yegge</span>
<span class="source"><a href="https://github.com/steveyegge/beads">GitHub Repository</a></span>
</div>

---

## 목차

### Part 1: 기초
<div class="chapter-list">

<div class="chapter-item">
<span class="chapter-number">01</span>
<a href="{{ '/beads-guide-01-intro/' | relative_url }}">소개 및 개요</a>
<p>Beads란 무엇인가? Git-first 이슈 트래커의 철학과 핵심 기능 소개</p>
</div>

<div class="chapter-item">
<span class="chapter-number">02</span>
<a href="{{ '/beads-guide-02-architecture/' | relative_url }}">아키텍처</a>
<p>3계층 데이터 모델 (CLI → SQLite → JSONL → Git)과 동기화 메커니즘</p>
</div>

<div class="chapter-item">
<span class="chapter-number">03</span>
<a href="{{ '/beads-guide-03-cli/' | relative_url }}">CLI 명령어</a>
<p>bd create, list, ready, show, update, close, dep, sync 등 전체 명령어 가이드</p>
</div>

</div>

### Part 2: 데이터 모델
<div class="chapter-list">

<div class="chapter-item">
<span class="chapter-number">04</span>
<a href="{{ '/beads-guide-04-data-model/' | relative_url }}">데이터 모델</a>
<p>Issue, Dependency, Comment, Event 타입과 JSONL/SQLite 스키마</p>
</div>

<div class="chapter-item">
<span class="chapter-number">05</span>
<a href="{{ '/beads-guide-05-daemon/' | relative_url }}">데몬 시스템</a>
<p>백그라운드 데몬 아키텍처, RPC 프로토콜, 자동 시작 메커니즘</p>
</div>

<div class="chapter-item">
<span class="chapter-number">06</span>
<a href="{{ '/beads-guide-06-sync/' | relative_url }}">동기화 메커니즘</a>
<p>SQLite↔JSONL 양방향 동기화, Export/Import 로직, 충돌 해결</p>
</div>

</div>

### Part 3: 고급 기능
<div class="chapter-list">

<div class="chapter-item">
<span class="chapter-number">07</span>
<a href="{{ '/beads-guide-07-dependency/' | relative_url }}">의존성 관리</a>
<p>4가지 의존성 타입, Ready 작업 감지, 그래프 알고리즘</p>
</div>

<div class="chapter-item">
<span class="chapter-number">08</span>
<a href="{{ '/beads-guide-08-molecules/' | relative_url }}">Molecules & Wisps</a>
<p>재사용 가능한 워크플로우 템플릿, YAML 기반 Molecule 정의</p>
</div>

<div class="chapter-item">
<span class="chapter-number">09</span>
<a href="{{ '/beads-guide-09-integration/' | relative_url }}">확장 및 통합</a>
<p>MCP 서버, GitHub/Jira 통합, Webhook, 플러그인 시스템</p>
</div>

</div>

### Part 4: 활용
<div class="chapter-list">

<div class="chapter-item">
<span class="chapter-number">10</span>
<a href="{{ '/beads-guide-10-best-practices/' | relative_url }}">활용 가이드 및 결론</a>
<p>AI 에이전트 워크플로우, 베스트 프랙티스, 문제 해결</p>
</div>

</div>

---

## 핵심 개념

| 개념 | 설명 |
|------|------|
| **Git-first** | 이슈가 JSONL로 Git 저장소에 저장, 코드와 함께 이동 |
| **3계층 모델** | CLI → SQLite(빠른 쿼리) → JSONL(Git 추적) → Remote |
| **해시 ID** | `bd-a1b2` 형태로 충돌 없는 분산 작업 |
| **Ready 감지** | 차단 없는 작업만 `bd ready`로 즉시 조회 |
| **자동 동기화** | 백그라운드 데몬이 5초 디바운스로 JSONL 업데이트 |

---

## 빠른 시작

```bash
# 설치
curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash

# 프로젝트에서 초기화
cd your-project
bd init

# Ready 작업 확인
bd ready

# 작업 생성
bd create "Add OAuth login" -p 1

# 작업 완료
bd close bd-a1b2
```

---

<div class="guide-footer">
<p>이 가이드는 <a href="https://github.com/steveyegge/beads">Beads GitHub 저장소</a>를 분석하여 작성되었습니다.</p>
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
  border-left: 4px solid #10a37f;
}

.chapter-number {
  font-size: 1.5rem;
  font-weight: bold;
  color: #10a37f;
  min-width: 40px;
}

.chapter-item a {
  font-size: 1.1rem;
  font-weight: 600;
  color: #333;
  text-decoration: none;
}

.chapter-item a:hover {
  color: #10a37f;
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
