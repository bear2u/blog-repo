---
layout: default
title: Ralph 가이드
permalink: /ralph-guide/
---

<section class="guide-header">
  <h1 class="guide-main-title">🔄 Ralph for Claude Code 완벽 가이드</h1>
  <p class="guide-subtitle">자율 AI 개발 루프 시스템</p>
</section>

<section class="guide-intro">
  <div class="intro-box">
    <p><strong>Ralph</strong>는 Geoffrey Huntley의 "Ralph 기법"을 구현한 오픈소스 도구로, Claude Code를 활용한 자율적인 연속 개발 사이클을 가능하게 합니다. AI가 프로젝트를 완료할 때까지 지속적으로 개선하며, 무한 루프와 API 남용을 방지하는 안전장치가 내장되어 있습니다.</p>
    <p>한 번 설치하면 어디서든 사용할 수 있는 글로벌 명령어로, 프로젝트 요구사항을 정의하고 Ralph를 실행하면 자동으로 코드를 작성하고 테스트합니다.</p>
  </div>
</section>

<section class="guide-toc">
  <h2 class="toc-title">📚 목차</h2>

  <div class="toc-grid">
    <a href="{{ '/ralph-guide-01-intro/' | relative_url }}" class="toc-item">
      <span class="toc-number">01</span>
      <div class="toc-content">
        <h3>소개</h3>
        <p>Ralph란? 특징, Geoffrey Huntley의 기법</p>
      </div>
    </a>

    <a href="{{ '/ralph-guide-02-installation/' | relative_url }}" class="toc-item">
      <span class="toc-number">02</span>
      <div class="toc-content">
        <h3>설치 및 시작</h3>
        <p>글로벌 설치, 프로젝트 초기화, Quick Start</p>
      </div>
    </a>

    <a href="{{ '/ralph-guide-03-files/' | relative_url }}" class="toc-item">
      <span class="toc-number">03</span>
      <div class="toc-content">
        <h3>파일 구조</h3>
        <p>.ralph/ 디렉토리, PROMPT.md, fix_plan.md</p>
      </div>
    </a>

    <a href="{{ '/ralph-guide-04-concepts/' | relative_url }}" class="toc-item">
      <span class="toc-number">04</span>
      <div class="toc-content">
        <h3>핵심 개념</h3>
        <p>자율 루프, 종료 감지, EXIT_SIGNAL</p>
      </div>
    </a>

    <a href="{{ '/ralph-guide-05-commands/' | relative_url }}" class="toc-item">
      <span class="toc-number">05</span>
      <div class="toc-content">
        <h3>CLI 명령어</h3>
        <p>ralph, ralph-enable, ralph-import 등</p>
      </div>
    </a>

    <a href="{{ '/ralph-guide-06-configuration/' | relative_url }}" class="toc-item">
      <span class="toc-number">06</span>
      <div class="toc-content">
        <h3>구성 및 설정</h3>
        <p>.ralphrc, 속도 제한, 타임아웃</p>
      </div>
    </a>

    <a href="{{ '/ralph-guide-07-circuit-breaker/' | relative_url }}" class="toc-item">
      <span class="toc-number">07</span>
      <div class="toc-content">
        <h3>서킷 브레이커</h3>
        <p>에러 감지, 상태 전환, 자동 복구</p>
      </div>
    </a>

    <a href="{{ '/ralph-guide-08-session/' | relative_url }}" class="toc-item">
      <span class="toc-number">08</span>
      <div class="toc-content">
        <h3>세션 관리</h3>
        <p>세션 연속성, 만료, 리셋 트리거</p>
      </div>
    </a>

    <a href="{{ '/ralph-guide-09-monitoring/' | relative_url }}" class="toc-item">
      <span class="toc-number">09</span>
      <div class="toc-content">
        <h3>모니터링</h3>
        <p>tmux 통합, 라이브 대시보드, 로그</p>
      </div>
    </a>

    <a href="{{ '/ralph-guide-10-best-practices/' | relative_url }}" class="toc-item">
      <span class="toc-number">10</span>
      <div class="toc-content">
        <h3>베스트 프랙티스</h3>
        <p>효과적인 프롬프트, 문제 해결, 팁</p>
      </div>
    </a>
  </div>
</section>

<section class="guide-features">
  <h2>✨ 주요 특징</h2>
  <div class="features-grid">
    <div class="feature-item">
      <span class="feature-icon">🔄</span>
      <h4>자율 개발 루프</h4>
      <p>Claude Code가 자동으로 반복하며 프로젝트 완성</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">🧠</span>
      <h4>지능형 종료 감지</h4>
      <p>Dual-condition 체크로 조기 종료 방지</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">⚡</span>
      <h4>서킷 브레이커</h4>
      <p>무한 루프와 에러 상황 자동 감지 및 복구</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">📊</span>
      <h4>라이브 모니터링</h4>
      <p>tmux 기반 실시간 진행 상황 대시보드</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">🔒</span>
      <h4>속도 제한</h4>
      <p>시간당 API 호출 제한으로 비용 관리</p>
    </div>
    <div class="feature-item">
      <span class="feature-icon">📋</span>
      <h4>PRD 가져오기</h4>
      <p>기존 요구사항 문서를 Ralph 형식으로 변환</p>
    </div>
  </div>
</section>

<section class="guide-quickstart">
  <h2>🚀 빠른 시작</h2>
  <div class="quickstart-box">
    <h4>설치 (한 번만)</h4>
    <pre><code>git clone https://github.com/frankbria/ralph-claude-code.git
cd ralph-claude-code
./install.sh</code></pre>

    <h4>프로젝트에서 Ralph 활성화</h4>
    <pre><code>cd my-project
ralph-enable          # 인터랙티브 위저드
ralph --monitor       # 자율 개발 시작</code></pre>

    <h4>기존 PRD로 새 프로젝트 생성</h4>
    <pre><code>ralph-import requirements.md my-app
cd my-app
ralph --monitor</code></pre>
  </div>
</section>

<section class="guide-links">
  <h2>🔗 관련 링크</h2>
  <div class="links-grid">
    <a href="https://github.com/frankbria/ralph-claude-code" target="_blank" class="link-item">
      <span>📦</span> GitHub 저장소
    </a>
    <a href="https://ghuntley.com/ralph/" target="_blank" class="link-item">
      <span>📖</span> Ralph 기법 원문
    </a>
    <a href="https://claude.ai/code" target="_blank" class="link-item">
      <span>🤖</span> Claude Code
    </a>
    <a href="https://github.com/hesreallyhim/awesome-claude-code" target="_blank" class="link-item">
      <span>⭐</span> Awesome Claude Code
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
  color: #10b981;
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
