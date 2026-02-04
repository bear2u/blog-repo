---
layout: default
title: UI-TARS 완벽 가이드
permalink: /ui-tars-guide/
---

<section class="guide-header">
  <h1 class="guide-title">UI-TARS 완벽 가이드</h1>
  <p class="guide-desc">ByteDance의 멀티모달 AI 에이전트 스택 소스 분석</p>
  <p class="guide-author">원본: <a href="https://github.com/bytedance/UI-TARS-desktop" target="_blank">GitHub Repository</a></p>
</section>

<section class="guide-container">
  <div class="guide-intro">
    <p>이 시리즈는 ByteDance의 UI-TARS 오픈소스 프로젝트를 상세히 분석한 가이드입니다. Vision-Language Model 기반 GUI 자동화, MCP 프로토콜 통합, 이벤트 스트림 아키텍처 등 핵심 기술을 다룹니다.</p>
  </div>

  <div class="guide-toc">
    <h2>목차</h2>

    <div class="toc-section">
      <h3>Part 1: 개요</h3>
      <ul class="toc-list">
        <li class="toc-item">
          <a href="{{ '/ui-tars-guide-01-intro/' | relative_url }}">
            <span class="toc-part">1.</span>
            <span class="toc-title">소개 및 개요</span>
          </a>
        </li>
        <li class="toc-item">
          <a href="{{ '/ui-tars-guide-02-architecture/' | relative_url }}">
            <span class="toc-part">2.</span>
            <span class="toc-title">전체 아키텍처</span>
          </a>
        </li>
      </ul>
    </div>

    <div class="toc-section">
      <h3>Part 2: 애플리케이션</h3>
      <ul class="toc-list">
        <li class="toc-item">
          <a href="{{ '/ui-tars-guide-03-desktop-app/' | relative_url }}">
            <span class="toc-part">3.</span>
            <span class="toc-title">Desktop 앱 분석</span>
          </a>
        </li>
        <li class="toc-item">
          <a href="{{ '/ui-tars-guide-04-agent-tars/' | relative_url }}">
            <span class="toc-part">4.</span>
            <span class="toc-title">Agent TARS Core</span>
          </a>
        </li>
      </ul>
    </div>

    <div class="toc-section">
      <h3>Part 3: 핵심 모듈</h3>
      <ul class="toc-list">
        <li class="toc-item">
          <a href="{{ '/ui-tars-guide-05-gui-agent/' | relative_url }}">
            <span class="toc-part">5.</span>
            <span class="toc-title">GUI Agent SDK</span>
          </a>
        </li>
        <li class="toc-item">
          <a href="{{ '/ui-tars-guide-06-operators/' | relative_url }}">
            <span class="toc-part">6.</span>
            <span class="toc-title">Operators</span>
          </a>
        </li>
        <li class="toc-item">
          <a href="{{ '/ui-tars-guide-07-tarko/' | relative_url }}">
            <span class="toc-part">7.</span>
            <span class="toc-title">Tarko 프레임워크</span>
          </a>
        </li>
      </ul>
    </div>

    <div class="toc-section">
      <h3>Part 4: 인프라 & 활용</h3>
      <ul class="toc-list">
        <li class="toc-item">
          <a href="{{ '/ui-tars-guide-08-mcp/' | relative_url }}">
            <span class="toc-part">8.</span>
            <span class="toc-title">MCP 인프라</span>
          </a>
        </li>
        <li class="toc-item">
          <a href="{{ '/ui-tars-guide-09-context/' | relative_url }}">
            <span class="toc-part">9.</span>
            <span class="toc-title">Context Engineering</span>
          </a>
        </li>
        <li class="toc-item">
          <a href="{{ '/ui-tars-guide-10-conclusion/' | relative_url }}">
            <span class="toc-part">10.</span>
            <span class="toc-title">활용 가이드 및 결론</span>
          </a>
        </li>
      </ul>
    </div>
  </div>

  <div class="guide-quick-links">
    <h2>빠른 참조</h2>
    <div class="quick-links-grid">
      <a href="{{ '/ui-tars-guide-02-architecture/' | relative_url }}" class="quick-link-card">
        <span class="quick-link-icon">🏗️</span>
        <span class="quick-link-title">아키텍처</span>
        <span class="quick-link-desc">계층화된 모듈식 설계</span>
      </a>
      <a href="{{ '/ui-tars-guide-05-gui-agent/' | relative_url }}" class="quick-link-card">
        <span class="quick-link-icon">🤖</span>
        <span class="quick-link-title">GUI Agent</span>
        <span class="quick-link-desc">Action Parser & SDK</span>
      </a>
      <a href="{{ '/ui-tars-guide-06-operators/' | relative_url }}" class="quick-link-card">
        <span class="quick-link-icon">⚙️</span>
        <span class="quick-link-title">Operators</span>
        <span class="quick-link-desc">Browser, NutJS, ADB</span>
      </a>
      <a href="{{ '/ui-tars-guide-08-mcp/' | relative_url }}" class="quick-link-card">
        <span class="quick-link-icon">🔧</span>
        <span class="quick-link-title">MCP 인프라</span>
        <span class="quick-link-desc">서버 & 클라이언트</span>
      </a>
    </div>
  </div>
</section>
