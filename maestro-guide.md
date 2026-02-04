---
layout: default
title: Maestro 가이드
permalink: /maestro-guide/
---

<section class="guide-header">
  <h1 class="guide-main-title">🎭 Maestro 완벽 가이드</h1>
  <p class="guide-subtitle">모바일 & 웹 UI 테스트 자동화 프레임워크</p>
</section>

<section class="guide-intro">
  <div class="intro-box">
    <p><strong>Maestro</strong>는 Android, iOS, 웹 앱을 위한 오픈소스 UI/E2E 테스트 프레임워크입니다. 인간이 읽을 수 있는 YAML 문법으로 테스트를 작성하고, 에뮬레이터, 시뮬레이터, 실제 디바이스에서 실행할 수 있습니다.</p>
    <p>Appium, Espresso, XCTest 등 기존 도구의 학습을 바탕으로, 플레이크(flakiness) 문제를 해결하고 빠른 반복 개발을 가능하게 합니다. 첫 테스트를 5분 이내에 작성할 수 있습니다.</p>
  </div>
</section>

<section class="guide-toc">
  <h2 class="toc-title">📚 목차</h2>

  <div class="toc-grid">
    <a href="{{ '/maestro-guide-01-intro/' | relative_url }}" class="toc-item">
      <span class="toc-number">01</span>
      <div class="toc-content">
        <h3>소개</h3>
        <p>Maestro란? 특징, 장점, 기존 도구와 비교</p>
      </div>
    </a>

    <a href="{{ '/maestro-guide-02-installation/' | relative_url }}" class="toc-item">
      <span class="toc-number">02</span>
      <div class="toc-content">
        <h3>설치 및 설정</h3>
        <p>CLI 설치, 요구사항, 환경 설정</p>
      </div>
    </a>

    <a href="{{ '/maestro-guide-03-yaml-flows/' | relative_url }}" class="toc-item">
      <span class="toc-number">03</span>
      <div class="toc-content">
        <h3>YAML 플로우</h3>
        <p>기본 문법, 명령어, 플로우 작성법</p>
      </div>
    </a>

    <a href="{{ '/maestro-guide-04-commands/' | relative_url }}" class="toc-item">
      <span class="toc-number">04</span>
      <div class="toc-content">
        <h3>핵심 명령어</h3>
        <p>tapOn, inputText, assertVisible, swipe 등</p>
      </div>
    </a>

    <a href="{{ '/maestro-guide-05-platforms/' | relative_url }}" class="toc-item">
      <span class="toc-number">05</span>
      <div class="toc-content">
        <h3>플랫폼별 테스트</h3>
        <p>Android, iOS, Web 앱 테스트</p>
      </div>
    </a>

    <a href="{{ '/maestro-guide-06-advanced/' | relative_url }}" class="toc-item">
      <span class="toc-number">06</span>
      <div class="toc-content">
        <h3>고급 기능</h3>
        <p>조건부 로직, 변수, 반복, 서브플로우</p>
      </div>
    </a>

    <a href="{{ '/maestro-guide-07-ai/' | relative_url }}" class="toc-item">
      <span class="toc-number">07</span>
      <div class="toc-content">
        <h3>AI 통합</h3>
        <p>assertWithAI, extractTextWithAI, MaestroGPT</p>
      </div>
    </a>

    <a href="{{ '/maestro-guide-08-studio/' | relative_url }}" class="toc-item">
      <span class="toc-number">08</span>
      <div class="toc-content">
        <h3>Maestro Studio</h3>
        <p>비주얼 테스트 IDE, 레코딩, 인스펙터</p>
      </div>
    </a>

    <a href="{{ '/maestro-guide-09-cloud/' | relative_url }}" class="toc-item">
      <span class="toc-number">09</span>
      <div class="toc-content">
        <h3>Maestro Cloud</h3>
        <p>병렬 실행, CI/CD 통합, 스케일링</p>
      </div>
    </a>

    <a href="{{ '/maestro-guide-10-architecture/' | relative_url }}" class="toc-item">
      <span class="toc-number">10</span>
      <div class="toc-content">
        <h3>아키텍처 & MCP</h3>
        <p>내부 구조, 모듈, MCP 서버 통합</p>
      </div>
    </a>
  </div>
</section>

<section class="guide-features">
  <h2>✨ 주요 특징</h2>
  <div class="features-grid">
    <div class="feature-item">
      <span class="feature-icon">📱</span>
      <h4>크로스 플랫폼</h4>
      <p>Android, iOS, Web 앱 모두 지원</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">📝</span>
      <h4>YAML 문법</h4>
      <p>인간이 읽기 쉬운 선언적 테스트 정의</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">🛡️</span>
      <h4>플레이크 방지</h4>
      <p>내장된 스마트 대기와 재시도</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">🤖</span>
      <h4>AI 지원</h4>
      <p>GPT 기반 어서션과 텍스트 추출</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">🎨</span>
      <h4>Maestro Studio</h4>
      <p>비주얼 테스트 빌더 IDE</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">☁️</span>
      <h4>클라우드 실행</h4>
      <p>병렬 테스트로 90% 시간 단축</p>
    </div>
  </div>
</section>

<section class="guide-quickstart">
  <h2>🚀 빠른 시작</h2>
  <div class="quickstart-box">
    <h4>설치</h4>
    <pre><code># macOS, Linux, Windows (WSL)
curl -fsSL "https://get.maestro.mobile.dev" | bash</code></pre>

    <h4>첫 번째 플로우</h4>
    <pre><code># flow.yaml
appId: com.android.contacts
---
- launchApp
- tapOn: "Create new contact"
- tapOn: "First Name"
- inputText: "John"
- tapOn: "Save"</code></pre>

    <h4>실행</h4>
    <pre><code>maestro test flow.yaml</code></pre>
  </div>
</section>

<section class="guide-links">
  <h2>🔗 관련 링크</h2>
  <div class="links-grid">
    <a href="https://github.com/mobile-dev-inc/Maestro" target="_blank" class="link-item">
      <span>📦</span> GitHub 저장소
    </a>
    <a href="https://docs.maestro.dev" target="_blank" class="link-item">
      <span>📘</span> 공식 문서
    </a>
    <a href="https://maestro.dev" target="_blank" class="link-item">
      <span>🌐</span> 공식 웹사이트
    </a>
    <a href="https://maestrodev.typeform.com/to/FelIEe8A" target="_blank" class="link-item">
      <span>💬</span> Slack 커뮤니티
    </a>
  </div>
</section>

<style>
.guide-header {
  text-align: center;
  padding: 3rem 1rem;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
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
  border-color: #6366f1;
}

.toc-number {
  font-size: 1.5rem;
  font-weight: 700;
  color: #6366f1;
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
  color: #6366f1;
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
  border-color: #6366f1;
  color: #6366f1;
}

@media (prefers-color-scheme: dark) {
  .guide-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
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
