---
layout: default
title: RS-SDK 가이드
permalink: /rs-sdk-guide/
---

<section class="guide-header">
  <h1 class="guide-main-title">🎮 RS-SDK 완벽 가이드</h1>
  <p class="guide-subtitle">RuneScape 스타일 봇 개발을 위한 연구용 SDK</p>
</section>

<section class="guide-intro">
  <div class="intro-box">
    <p><strong>RS-SDK</strong>는 RuneScape 스타일의 MMO 게임에서 봇을 개발하고 연구할 수 있는 오픈소스 스타터 킷입니다. TypeScript SDK, 에이전트 문서, 서버 에뮬레이터를 포함하며, Claude Code와의 MCP 통합을 지원합니다.</p>
    <p>AI 에이전트 연구, 목표 지향적 프로그램 합성 기법(Ralph loops 등) 실험, 에이전트 간 협력/경쟁 연구를 위한 풍부한 테스트 환경을 제공합니다.</p>
  </div>
</section>

<section class="guide-toc">
  <h2 class="toc-title">📚 목차</h2>

  <div class="toc-grid">
    <a href="{{ '/rs-sdk-guide-01-intro/' | relative_url }}" class="toc-item">
      <span class="toc-number">01</span>
      <div class="toc-content">
        <h3>소개</h3>
        <p>RS-SDK란? 프로젝트 목표, LostCity 엔진 기반</p>
      </div>
    </a>

    <a href="{{ '/rs-sdk-guide-02-architecture/' | relative_url }}" class="toc-item">
      <span class="toc-number">02</span>
      <div class="toc-content">
        <h3>아키텍처</h3>
        <p>Engine, WebClient, Gateway, SDK 구조</p>
      </div>
    </a>

    <a href="{{ '/rs-sdk-guide-03-getting-started/' | relative_url }}" class="toc-item">
      <span class="toc-number">03</span>
      <div class="toc-content">
        <h3>시작하기</h3>
        <p>설치, 봇 생성, 첫 스크립트 작성</p>
      </div>
    </a>

    <a href="{{ '/rs-sdk-guide-04-sdk-api/' | relative_url }}" class="toc-item">
      <span class="toc-number">04</span>
      <div class="toc-content">
        <h3>SDK API</h3>
        <p>BotSDK (저수준)와 BotActions (고수준) API</p>
      </div>
    </a>

    <a href="{{ '/rs-sdk-guide-05-mcp/' | relative_url }}" class="toc-item">
      <span class="toc-number">05</span>
      <div class="toc-content">
        <h3>MCP 통합</h3>
        <p>Claude Code 인터랙티브 봇 제어</p>
      </div>
    </a>

    <a href="{{ '/rs-sdk-guide-06-skills/' | relative_url }}" class="toc-item">
      <span class="toc-number">06</span>
      <div class="toc-content">
        <h3>스킬 자동화</h3>
        <p>Woodcutting, Mining, Fishing, Combat 등</p>
      </div>
    </a>

    <a href="{{ '/rs-sdk-guide-07-economy/' | relative_url }}" class="toc-item">
      <span class="toc-number">07</span>
      <div class="toc-content">
        <h3>경제 시스템</h3>
        <p>뱅킹, 쇼핑, 아이템 관리</p>
      </div>
    </a>

    <a href="{{ '/rs-sdk-guide-08-navigation/' | relative_url }}" class="toc-item">
      <span class="toc-number">08</span>
      <div class="toc-content">
        <h3>이동 & 경로</h3>
        <p>Pathfinding, 게이트, 문 열기</p>
      </div>
    </a>

    <a href="{{ '/rs-sdk-guide-09-best-practices/' | relative_url }}" class="toc-item">
      <span class="toc-number">09</span>
      <div class="toc-content">
        <h3>베스트 프랙티스</h3>
        <p>에러 처리, 검증 패턴, 팁</p>
      </div>
    </a>

    <a href="{{ '/rs-sdk-guide-10-hosting/' | relative_url }}" class="toc-item">
      <span class="toc-number">10</span>
      <div class="toc-content">
        <h3>서버 호스팅</h3>
        <p>로컬 서버 설정, 배포</p>
      </div>
    </a>
  </div>
</section>

<section class="guide-features">
  <h2>✨ 주요 특징</h2>
  <div class="features-grid">
    <div class="feature-item">
      <span class="feature-icon">🔬</span>
      <h4>연구 지향</h4>
      <p>AI 에이전트 연구를 위한 안전한 테스트 환경</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">🤖</span>
      <h4>Claude 통합</h4>
      <p>MCP를 통한 Claude Code 인터랙티브 제어</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">📝</span>
      <h4>TypeScript SDK</h4>
      <p>강력한 타입 지원의 봇 자동화 라이브러리</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">🏆</span>
      <h4>리더보드</h4>
      <p>봇 순위 시스템 및 경쟁</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">🎯</span>
      <h4>목표 지향</h4>
      <p>Ralph loops 등 프로그램 합성 기법 연구</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">🌐</span>
      <h4>완전 오픈소스</h4>
      <p>서버, 클라이언트, SDK 모두 공개</p>
    </div>
  </div>
</section>

<section class="guide-quickstart">
  <h2>🚀 빠른 시작</h2>
  <div class="quickstart-box">
    <h4>설치 및 봇 생성</h4>
    <pre><code># 저장소 클론
git clone https://github.com/MaxBittker/rs-sdk.git
cd rs-sdk

# 의존성 설치
bun install

# 봇 생성
bun scripts/create-bot.ts mybot

# 봇 실행
bun bots/mybot/script.ts</code></pre>

    <h4>Claude Code와 함께 사용</h4>
    <pre><code># Claude Code에서 자동 MCP 연동
claude "start a new bot with name: mybot"</code></pre>
  </div>
</section>

<section class="guide-links">
  <h2>🔗 관련 링크</h2>
  <div class="links-grid">
    <a href="https://github.com/MaxBittker/rs-sdk" target="_blank" class="link-item">
      <span>📦</span> GitHub 저장소
    </a>
    <a href="https://rs-sdk-demo.fly.dev/hiscores" target="_blank" class="link-item">
      <span>🏆</span> 리더보드
    </a>
    <a href="https://discord.gg/3DcuU5cMJN" target="_blank" class="link-item">
      <span>💬</span> Discord
    </a>
    <a href="https://lostcity.rs" target="_blank" class="link-item">
      <span>🏛️</span> LostCity
    </a>
  </div>
</section>

<style>
.guide-header {
  text-align: center;
  padding: 3rem 1rem;
  background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
  color: white;
  border-radius: 16px;
  margin-bottom: 2rem;
}

.guide-main-title {
  font-size: 2.5rem;
  margin: 0 0 0.5rem 0;
}

.guide-subtitle {
  font-size: 1.2rem;
  opacity: 0.9;
  margin: 0;
}

.guide-intro {
  max-width: 800px;
  margin: 0 auto 2rem;
}

.intro-box {
  background: var(--card-bg, #f8f9fa);
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 12px;
  padding: 1.5rem;
  line-height: 1.7;
}

.intro-box p {
  margin: 0 0 1rem 0;
}

.intro-box p:last-child {
  margin-bottom: 0;
}

.guide-toc {
  max-width: 900px;
  margin: 0 auto 3rem;
}

.toc-title {
  font-size: 1.5rem;
  margin-bottom: 1.5rem;
  text-align: center;
}

.toc-grid {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.toc-item {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  padding: 1.25rem;
  background: var(--card-bg, #fff);
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 12px;
  text-decoration: none;
  color: inherit;
  transition: all 0.2s;
}

.toc-item:hover {
  transform: translateX(8px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  border-color: #7c3aed;
}

.toc-number {
  font-size: 1.5rem;
  font-weight: 700;
  color: #7c3aed;
  min-width: 50px;
  text-align: center;
}

.toc-content h3 {
  margin: 0 0 0.25rem 0;
  font-size: 1.1rem;
  color: var(--heading-color, #1a1a2e);
}

.toc-content p {
  margin: 0;
  font-size: 0.9rem;
  color: var(--text-muted, #666);
}

.guide-features {
  max-width: 900px;
  margin: 0 auto 3rem;
}

.guide-features h2 {
  text-align: center;
  margin-bottom: 1.5rem;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
}

.feature-item {
  background: var(--card-bg, #fff);
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 12px;
  padding: 1.5rem;
  text-align: center;
}

.feature-icon {
  font-size: 2rem;
  display: block;
  margin-bottom: 0.75rem;
}

.feature-item h4 {
  margin: 0 0 0.5rem 0;
  color: var(--heading-color, #1a1a2e);
}

.feature-item p {
  margin: 0;
  font-size: 0.9rem;
  color: var(--text-muted, #666);
}

.guide-quickstart {
  max-width: 800px;
  margin: 0 auto 3rem;
}

.guide-quickstart h2 {
  text-align: center;
  margin-bottom: 1.5rem;
}

.quickstart-box {
  background: var(--card-bg, #1e1e2e);
  border: 1px solid var(--border-color, #333);
  border-radius: 12px;
  padding: 1.5rem;
}

.quickstart-box h4 {
  margin: 0 0 0.75rem 0;
  color: #7c3aed;
}

.quickstart-box pre {
  background: #0d1117;
  border-radius: 8px;
  padding: 1rem;
  overflow-x: auto;
  margin: 0 0 1.5rem 0;
}

.quickstart-box pre:last-child {
  margin-bottom: 0;
}

.quickstart-box code {
  color: #e6edf3;
  font-size: 0.9rem;
}

.guide-links {
  max-width: 800px;
  margin: 0 auto;
}

.guide-links h2 {
  text-align: center;
  margin-bottom: 1.5rem;
}

.links-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
}

.link-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  background: var(--card-bg, #fff);
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 8px;
  text-decoration: none;
  color: var(--text-color, #333);
  transition: all 0.2s;
}

.link-item:hover {
  border-color: #7c3aed;
  color: #7c3aed;
}

@media (prefers-color-scheme: dark) {
  .guide-header {
    background: linear-gradient(135deg, #6d28d9 0%, #7c3aed 100%);
  }

  .intro-box, .toc-item, .feature-item, .link-item {
    --card-bg: #1e1e2e;
    --border-color: #333;
  }
}

@media (max-width: 600px) {
  .guide-main-title {
    font-size: 1.8rem;
  }

  .toc-item {
    flex-direction: column;
    text-align: center;
    gap: 0.75rem;
  }

  .toc-number {
    min-width: auto;
  }
}
</style>
