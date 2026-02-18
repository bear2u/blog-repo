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
    <h3><a href="/blog-repo/cline-guide/">Cline</a></h3>
    <p>IDE 안에서 Plan/Act로 코드를 수정하고, 승인 기반으로 터미널/브라우저/MCP까지 사용하는 오픈소스 코딩 에이전트 (10챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-coding">AI 코딩</span>
    <h3><a href="/blog-repo/claude-code-system-prompts-guide/">Claude Code System Prompts</a></h3>
    <p>Claude Code v2.1.42(2026-02-13) 기준 110+ 시스템 프롬프트/리마인더/툴 설명을 추적하는 레포 분석 (8챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-coding">AI 코딩</span>
    <h3><a href="/blog-repo/aios-core-guide/">Synkra AIOS Core</a></h3>
    <p>CLI First 에이전트 애자일 프레임워크: 태스크/워크플로우/스쿼드로 개발을 표준화 (10챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-coding">AI 코딩</span>
    <h3><a href="/blog-repo/gsd-guide/">GSD (Get Shit Done)</a></h3>
    <p>Claude Code 컨텍스트 품질 저하 해결하는 메타 프롬프팅 시스템 (9챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-coding">AI 코딩</span>
    <h3><a href="/blog-repo/entire-cli-guide/">Entire CLI</a></h3>
    <p>Git 워크플로우에 통합된 AI 에이전트 세션 자동 캡처 도구 (25챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-coding">AI 코딩</span>
    <h3><a href="/blog-repo/oh-my-claudecode-guide/">oh-my-claudecode</a></h3>
    <p>Claude Code 멀티-에이전트 오케스트레이션 시스템 (5챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-coding">AI 코딩</span>
    <h3><a href="/blog-repo/claude-code-2-guide/">Claude Code 2.0</a></h3>
    <p>Anthropic의 CLI 코딩 에이전트 완벽 해설</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-coding">AI 코딩</span>
    <h3><a href="/blog-repo/mux-guide/">Mux</a></h3>
    <p>병렬 에이전트 개발을 위한 코딩 멀티플렉서 (8챕터)</p>
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
    <h3><a href="/blog-repo/ai-agent-automation-guide/">AI Agent Automation</a></h3>
    <p>로컬 우선 AI 워크플로우 실행 엔진: 워크플로우/태스크/스케줄/웹훅/인앱 어시스턴트까지 분석 (10챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-agent">RAG</span>
    <h3><a href="/blog-repo/ultrarag-guide/">UltraRAG</a></h3>
    <p>清华大学, NEUIR, OpenBMB가 만든 MCP 기반 경량 RAG 개발 프레임워크 (10챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-agent">AI 에이전트</span>
    <h3><a href="/blog-repo/picoclaw-guide/">PicoClaw</a></h3>
    <p>Go로 만든 초경량 개인 AI 어시스턴트: $10 하드웨어, 10MB 미만 RAM, 1초 부팅 목표 (10챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-agent">AI 에이전트</span>
    <h3><a href="/blog-repo/explain-openclaw-guide/">Explain OpenClaw</a></h3>
    <p>OpenClaw(구 Moltbot/Clawdbot) 운영/보안/배포 통합 지식베이스 (8챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-agent">AI 에이전트</span>
    <h3><a href="/blog-repo/tambo-guide/">Tambo</a></h3>
    <p>React 앱에 생성형 UI 에이전트를 추가하는 오픈소스 툴킷 (9챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-agent">AI 에이전트</span>
    <h3><a href="/blog-repo/goose-guide/">Goose</a></h3>
    <p>로컬 머신에서 동작하는 확장 가능한 오픈소스 AI 에이전트 (10챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-agent">AI 에이전트</span>
    <h3><a href="/blog-repo/effgen-guide/">effGen</a></h3>
    <p>Small Language Models을 강력한 AI 에이전트로 변환 (7챕터)</p>
  </div>
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
    <span class="badge badge-tool">웹 스크래핑</span>
    <h3><a href="/blog-repo/llm-reader-guide/">LLM Reader</a></h3>
    <p>웹페이지 HTML을 LLM 입력 텍스트로 정규화해 링크/이미지/표 추출 정확도를 높이는 Python 전처리 라이브러리 (10챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-tool">CLI</span>
    <h3><a href="/blog-repo/summarize-guide/">Summarize</a></h3>
    <p>Chrome/Firefox 사이드패널과 로컬 daemon을 결합한 URL·파일·미디어 요약 도구 (10챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-tool">개발 도구</span>
    <h3><a href="/blog-repo/nautilus-trader-guide/">NautilusTrader</a></h3>
    <p>Rust 코어 + Python API 기반 이벤트 구동 알고리즘 트레이딩 플랫폼 (10챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-tool">ベクターDB</span>
    <h3><a href="/blog-repo/zvec-guide/">zvec</a></h3>
    <p>Alibaba의 인프로세스 벡터 데이터베이스: 경량, 초고속, Proxima 기반 (7챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-tool">모바일</span>
    <h3><a href="/blog-repo/scrcpy-mobile-guide/">Scrcpy Remote (scrcpy-mobile)</a></h3>
    <p>아이폰에서 Wi-Fi ADB로 Android 기기를 원격 제어하는 scrcpy 모바일 포팅 (10챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-tool">개발 도구</span>
    <h3><a href="/blog-repo/gh-aw-guide/">GitHub Agentic Workflows (gh-aw)</a></h3>
    <p>자연어 마크다운 워크플로우를 컴파일해 GitHub Actions에서 안전하게 실행하는 gh CLI 확장 (10챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-llm">LLM</span>
    <h3><a href="/blog-repo/clawrouter-guide/">ClawRouter</a></h3>
    <p>OpenClaw용 로컬 LLM 라우터: x402(USDC)로 30+ 모델 요청을 자동 라우팅 (10챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-tool">모바일</span>
    <h3><a href="/blog-repo/divine-mobile-guide/">diVine(OpenVine)</a></h3>
    <p>Nostr 기반 Vine 스타일 Flutter 앱 divine-mobile 아키텍처/업로드/피드/배포 (12챕터)</p>
  </div>
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
  <div class="guide-card">
    <span class="badge badge-tool">개발 도구</span>
    <h3><a href="/blog-repo/check-if-email-exists-guide/">check-if-email-exists</a></h3>
    <p>이메일을 보내지 않고 이메일 주소 유효성 검증 (6챕터)</p>
  </div>
</div>

<h2 class="section-title">LLM 학습</h2>
<div class="guide-grid">
  <div class="guide-card">
    <span class="badge badge-llm">LLM</span>
    <h3><a href="/blog-repo/mlx-llm-tutorial-guide/">MLX LLM Tutorial</a></h3>
    <p>Apple Silicon에서 MLX로 LLM을 구현/학습/파인튜닝/웹 데모까지 따라가는 실습형 튜토리얼 분석 (10챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-llm">음성인식</span>
    <h3><a href="/blog-repo/tiny-audio-guide/">Tiny Audio</a></h3>
    <p>24시간/$12로 ASR 모델 훈련 - Frozen Encoder + Projector (6챕터)</p>
  </div>
  <div class="guide-card">
    <span class="badge badge-llm">LLM</span>
    <h3><a href="/blog-repo/minimind-guide/">MiniMind</a></h3>
    <p>2시간 만에 LLM 훈련하기</p>
  </div>
</div>

<h2 class="section-title">AI 음악</h2>
<div class="guide-grid">
  <div class="guide-card">
    <span class="badge badge-llm">AI 음악</span>
    <h3><a href="/blog-repo/ace-step-guide/">ACE-Step 1.5</a></h3>
    <p>오픈소스 AI 음악 생성 모델 (10챕터)</p>
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
