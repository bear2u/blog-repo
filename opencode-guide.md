---
layout: default
title: OpenCode 가이드
permalink: /opencode-guide/
---

<section class="guide-header">
  <h1 class="guide-main-title">🚀 OpenCode 완벽 가이드</h1>
  <p class="guide-subtitle">오픈소스 AI 코딩 에이전트의 모든 것</p>
</section>

<section class="guide-intro">
  <div class="intro-box">
    <p><strong>OpenCode</strong>는 Claude Code와 유사한 기능을 제공하는 100% 오픈소스 AI 코딩 에이전트입니다. 특정 AI 프로바이더에 종속되지 않으며, 강력한 TUI(Terminal UI), LSP 지원, 클라이언트-서버 아키텍처를 제공합니다.</p>
    <p>이 가이드는 OpenCode의 설치부터 고급 활용까지 상세하게 다룹니다.</p>
  </div>
</section>

<section class="guide-toc">
  <h2 class="toc-title">📚 목차</h2>

  <div class="toc-grid">
    <a href="{{ '/2025/02/04/opencode-guide-01-intro' | relative_url }}" class="toc-item">
      <span class="toc-number">01</span>
      <div class="toc-content">
        <h3>소개 및 주요 특징</h3>
        <p>OpenCode란? Claude Code와의 차이점, 핵심 기능 개요</p>
      </div>
    </a>

    <a href="{{ '/2025/02/04/opencode-guide-02-installation' | relative_url }}" class="toc-item">
      <span class="toc-number">02</span>
      <div class="toc-content">
        <h3>설치 가이드</h3>
        <p>npm, Homebrew, Scoop, 데스크톱 앱 설치 방법</p>
      </div>
    </a>

    <a href="{{ '/2025/02/04/opencode-guide-03-architecture' | relative_url }}" class="toc-item">
      <span class="toc-number">03</span>
      <div class="toc-content">
        <h3>아키텍처</h3>
        <p>모노레포 구조, 클라이언트-서버 모델, 패키지 구성</p>
      </div>
    </a>

    <a href="{{ '/2025/02/04/opencode-guide-04-agents' | relative_url }}" class="toc-item">
      <span class="toc-number">04</span>
      <div class="toc-content">
        <h3>에이전트 시스템</h3>
        <p>Build, Plan, Explore 에이전트 및 커스텀 에이전트 생성</p>
      </div>
    </a>

    <a href="{{ '/2025/02/04/opencode-guide-05-tools' | relative_url }}" class="toc-item">
      <span class="toc-number">05</span>
      <div class="toc-content">
        <h3>내장 도구</h3>
        <p>Edit, Bash, Read, Grep, WebFetch 등 AI 도구 상세 설명</p>
      </div>
    </a>

    <a href="{{ '/2025/02/04/opencode-guide-06-providers' | relative_url }}" class="toc-item">
      <span class="toc-number">06</span>
      <div class="toc-content">
        <h3>AI 프로바이더</h3>
        <p>Anthropic, OpenAI, Google, Azure 등 멀티 프로바이더 지원</p>
      </div>
    </a>

    <a href="{{ '/2025/02/04/opencode-guide-07-configuration' | relative_url }}" class="toc-item">
      <span class="toc-number">07</span>
      <div class="toc-content">
        <h3>설정 및 권한</h3>
        <p>opencode.json 설정, 권한 시스템, 환경 변수</p>
      </div>
    </a>

    <a href="{{ '/2025/02/04/opencode-guide-08-mcp' | relative_url }}" class="toc-item">
      <span class="toc-number">08</span>
      <div class="toc-content">
        <h3>MCP 통합</h3>
        <p>Model Context Protocol 서버 연동 및 도구 확장</p>
      </div>
    </a>

    <a href="{{ '/2025/02/04/opencode-guide-09-tui-desktop' | relative_url }}" class="toc-item">
      <span class="toc-number">09</span>
      <div class="toc-content">
        <h3>TUI & 데스크톱 앱</h3>
        <p>터미널 UI 사용법, Tauri 기반 데스크톱 앱</p>
      </div>
    </a>

    <a href="{{ '/2025/02/04/opencode-guide-10-lsp-skills' | relative_url }}" class="toc-item">
      <span class="toc-number">10</span>
      <div class="toc-content">
        <h3>LSP & 스킬 시스템</h3>
        <p>Language Server Protocol 지원, 커스텀 스킬 작성</p>
      </div>
    </a>
  </div>
</section>

<section class="guide-features">
  <h2>✨ 주요 특징</h2>
  <div class="features-grid">
    <div class="feature-item">
      <span class="feature-icon">🔓</span>
      <h4>100% 오픈소스</h4>
      <p>MIT 라이선스로 완전 공개된 소스 코드</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">🔀</span>
      <h4>프로바이더 독립</h4>
      <p>Claude, OpenAI, Google, 로컬 모델 모두 지원</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">📟</span>
      <h4>TUI 중심 설계</h4>
      <p>Neovim 사용자를 위한 터미널 네이티브 경험</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">🖥️</span>
      <h4>데스크톱 앱</h4>
      <p>Tauri v2 기반 네이티브 데스크톱 애플리케이션</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">🔌</span>
      <h4>LSP 통합</h4>
      <p>기본 내장된 Language Server Protocol 지원</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">🌐</span>
      <h4>클라이언트-서버</h4>
      <p>원격 제어 가능한 분리된 아키텍처</p>
    </div>
  </div>
</section>

<section class="guide-quickstart">
  <h2>🚀 빠른 시작</h2>
  <div class="quickstart-box">
    <h4>설치</h4>
    <pre><code># npm으로 설치
npm i -g opencode-ai@latest

# Homebrew (macOS/Linux)
brew install anomalyco/tap/opencode

# 또는 curl로 직접 설치
curl -fsSL https://opencode.ai/install | bash</code></pre>

    <h4>실행</h4>
    <pre><code># 프로젝트 디렉토리에서 실행
cd your-project
opencode</code></pre>
  </div>
</section>

<section class="guide-links">
  <h2>🔗 관련 링크</h2>
  <div class="links-grid">
    <a href="https://github.com/anomalyco/opencode" target="_blank" class="link-item">
      <span>📦</span> GitHub 저장소
    </a>
    <a href="https://opencode.ai/docs" target="_blank" class="link-item">
      <span>📚</span> 공식 문서
    </a>
    <a href="https://opencode.ai/discord" target="_blank" class="link-item">
      <span>💬</span> Discord 커뮤니티
    </a>
    <a href="https://opencode.ai/zen" target="_blank" class="link-item">
      <span>☯️</span> OpenCode Zen
    </a>
  </div>
</section>

<style>
.guide-header {
  text-align: center;
  padding: 3rem 1rem;
  background: linear-gradient(135deg, #059669 0%, #10b981 100%);
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
  border-color: #059669;
}

.toc-number {
  font-size: 1.5rem;
  font-weight: 700;
  color: #059669;
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
  color: #059669;
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
  border-color: #059669;
  color: #059669;
}

@media (prefers-color-scheme: dark) {
  .guide-header {
    background: linear-gradient(135deg, #047857 0%, #059669 100%);
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
