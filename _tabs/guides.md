---
layout: page
title: Guides
permalink: /guides/
icon: fas fa-book
order: 1
---

<style>
.guide-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}
.guide-card {
  border: 1px solid var(--card-border-color, #e9ecef);
  border-radius: 0.5rem;
  padding: 1.2rem;
  background: var(--card-bg, #fff);
  transition: box-shadow 0.2s, transform 0.2s;
}
.guide-card:hover {
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  transform: translateY(-2px);
}
.guide-card h3 {
  margin: 0 0 0.5rem 0;
  font-size: 1.1rem;
}
.guide-card h3 a {
  text-decoration: none;
}
.guide-card p {
  margin: 0;
  font-size: 0.9rem;
  color: var(--text-muted-color, #6c757d);
}
.guide-card .badge {
  display: inline-block;
  font-size: 0.75rem;
  padding: 0.2rem 0.5rem;
  border-radius: 0.25rem;
  margin-bottom: 0.5rem;
}
.badge-coding { background: #d4edda; color: #155724; }
.badge-agent { background: #cce5ff; color: #004085; }
.badge-tool { background: #fff3cd; color: #856404; }
.badge-llm { background: #f8d7da; color: #721c24; }
.badge-blog { background: #e2e3e5; color: #383d41; }
.section-title {
  font-size: 1.3rem;
  margin: 1.5rem 0 1rem 0;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid var(--card-border-color, #e9ecef);
}
</style>

AI & 개발 관련 해외 콘텐츠를 한국어로 번역한 가이드 시리즈입니다.

<h2 class="section-title">AI 코딩 에이전트</h2>
<div class="guide-grid">
  <div class="guide-card">
    <span class="badge badge-coding">AI 코딩</span>
    <h3><a href="/blog-repo/claude-code-2-guide/">Claude Code 2.0</a></h3>
    <p>Anthropic의 CLI 코딩 에이전트 완벽 해설</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-coding">AI 코딩</span>
    <h3><a href="/blog-repo/opencode-guide/">OpenCode</a></h3>
    <p>오픈소스 AI 코딩 에이전트</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-coding">AI 코딩</span>
    <h3><a href="/blog-repo/ralph-guide/">Ralph</a></h3>
    <p>자율 AI 개발 루프 시스템</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-coding">AI 코딩</span>
    <h3><a href="/blog-repo/superset-guide/">Superset</a></h3>
    <p>CLI 코딩 에이전트를 위한 터보차지 터미널</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-tool">개발 도구</span>
    <h3><a href="/blog-repo/code-server-guide/">code-server</a></h3>
    <p>브라우저에서 실행하는 VS Code</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-coding">AI 코딩</span>
    <h3><a href="/blog-repo/specweave-guide/">SpecWeave</a></h3>
    <p>AI 코딩을 위한 엔터프라이즈 레이어</p>
  </div>
</div>

<h2 class="section-title">AI 에이전트</h2>
<div class="guide-grid">
  <div class="guide-card">
    <span class="badge badge-agent">AI 에이전트</span>
    <h3><a href="/blog-repo/claude-skills-guide/">Claude Skills</a></h3>
    <p>Anthropic 공식 스킬 시스템 완벽 가이드 (9챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-agent">AI 에이전트</span>
    <h3><a href="/blog-repo/wrenai-guide/">WrenAI</a></h3>
    <p>자연어를 SQL로 변환하는 GenBI 플랫폼</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-agent">AI 에이전트</span>
    <h3><a href="/blog-repo/ui-tars-guide/">UI-TARS</a></h3>
    <p>ByteDance의 멀티모달 AI 에이전트</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-agent">AI 에이전트</span>
    <h3><a href="/blog-repo/openclaw-guide/">OpenClaw</a></h3>
    <p>멀티채널 개인 AI 어시스턴트</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-agent">AI 에이전트</span>
    <h3><a href="/blog-repo/nanobot-guide/">Nanobot</a></h3>
    <p>4,000줄 초경량 개인 AI 어시스턴트</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-agent">AI 에이전트</span>
    <h3><a href="/blog-repo/beads-guide/">Beads</a></h3>
    <p>AI를 위한 분산 Git 기반 이슈 트래커</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-agent">AI 에이전트</span>
    <h3><a href="/blog-repo/context-engineering-guide/">Context Engineering</a></h3>
    <p>프롬프트를 넘어서는 컨텍스트 설계 기법</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-agent">AI 에이전트</span>
    <h3><a href="/blog-repo/acontext-guide/">Acontext</a></h3>
    <p>프로덕션 AI 에이전트용 컨텍스트 데이터 플랫폼</p>
  </div>
</div>

<h2 class="section-title">개발 도구</h2>
<div class="guide-grid">
  <div class="guide-card">
    <span class="badge badge-tool">개발 도구</span>
    <h3><a href="/blog-repo/maestro-guide/">Maestro</a></h3>
    <p>모바일 & 웹 UI 테스트 자동화 프레임워크</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-tool">개발 도구</span>
    <h3><a href="/blog-repo/trendradar-guide/">TrendRadar</a></h3>
    <p>기술 트렌드 모니터링 도구</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-tool">개발 도구</span>
    <h3><a href="/blog-repo/rs-sdk-guide/">RS-SDK</a></h3>
    <p>RuneScape 봇 개발 SDK</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-tool">개발 도구</span>
    <h3><a href="/blog-repo/scrapegraph-guide/">ScrapeGraphAI</a></h3>
    <p>LLM 기반 차세대 웹 스크래핑 라이브러리</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-tool">데스크톱</span>
    <h3><a href="/blog-repo/tauri-guide/">Tauri</a></h3>
    <p>Rust 기반 초경량 크로스 플랫폼 데스크톱 프레임워크</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-tool">CLI</span>
    <h3><a href="/blog-repo/ghostty-guide/">Ghostty</a></h3>
    <p>Zig 기반 GPU 가속 차세대 터미널 에뮬레이터</p>
  </div>
</div>

<h2 class="section-title">LLM 학습</h2>
<div class="guide-grid">
  <div class="guide-card">
    <span class="badge badge-llm">LLM</span>
    <h3><a href="/blog-repo/minimind-guide/">MiniMind</a></h3>
    <p>2시간 만에 LLM 훈련하기</p>
  </div>
</div>

<h2 class="section-title">일반 블로그</h2>
<div class="guide-grid">
  <div class="guide-card">
    <span class="badge badge-blog">블로그</span>
    <h3><a href="/blog-repo/my-ai-adoption-journey/">나의 AI 도입 여정</a></h3>
    <p>Mitchell Hashimoto의 AI 에이전트 도입 6단계</p>
  </div>
</div>
