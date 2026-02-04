---
layout: default
title: Claude Code 2.0 가이드
permalink: /claude-code-2-guide/
---

<section class="guide-header">
  <h1 class="guide-title">Claude Code 2.0 완벽 가이드</h1>
  <p class="guide-desc">코딩 에이전트 활용법에 대한 종합 가이드 시리즈</p>
  <p class="guide-author">원문: <a href="https://sankalp.bearblog.dev/my-experience-with-claude-code-20-and-how-to-get-better-at-using-coding-agents/" target="_blank">Sankalp's Blog</a></p>
</section>

<section class="guide-container">
  <div class="guide-intro">
    <p>이 시리즈는 Sankalp의 Claude Code 2.0 심층 가이드를 번역한 것입니다. 코딩 에이전트를 효과적으로 사용하는 방법, 컨텍스트 엔지니어링, 그리고 실제 워크플로우 전략을 다룹니다.</p>
  </div>

  <div class="guide-toc">
    <h2>목차</h2>

    <div class="toc-section">
      <h3>Part 1: 소개</h3>
      <ul class="toc-list">
        {% for post in site.posts reversed %}
          {% if post.series == 'claude-code-2-guide' and post.part >= 1 and post.part <= 3 %}
            <li class="toc-item">
              <a href="{{ post.url | relative_url }}">
                <span class="toc-part">{{ post.part }}.</span>
                <span class="toc-title">{{ post.title | remove: "Claude Code 2.0 가이드 (" | remove: ") - " | split: " - " | last }}</span>
              </a>
            </li>
          {% endif %}
        {% endfor %}
      </ul>
    </div>

    <div class="toc-section">
      <h3>Part 2: 기본 개념</h3>
      <ul class="toc-list">
        {% for post in site.posts reversed %}
          {% if post.series == 'claude-code-2-guide' and post.part >= 4 and post.part <= 5 %}
            <li class="toc-item">
              <a href="{{ post.url | relative_url }}">
                <span class="toc-part">{{ post.part }}.</span>
                <span class="toc-title">{{ post.title | remove: "Claude Code 2.0 가이드 (" | remove: ") - " | split: " - " | last }}</span>
              </a>
            </li>
          {% endif %}
        {% endfor %}
      </ul>
    </div>

    <div class="toc-section">
      <h3>Part 3: 기능 심층 탐구</h3>
      <ul class="toc-list">
        {% for post in site.posts reversed %}
          {% if post.series == 'claude-code-2-guide' and post.part >= 6 and post.part <= 8 %}
            <li class="toc-item">
              <a href="{{ post.url | relative_url }}">
                <span class="toc-part">{{ post.part }}.</span>
                <span class="toc-title">{{ post.title | remove: "Claude Code 2.0 가이드 (" | remove: ") - " | split: " - " | last }}</span>
              </a>
            </li>
          {% endif %}
        {% endfor %}
      </ul>
    </div>

    <div class="toc-section">
      <h3>Part 4: 고급 주제</h3>
      <ul class="toc-list">
        {% for post in site.posts reversed %}
          {% if post.series == 'claude-code-2-guide' and post.part >= 9 and post.part <= 12 %}
            <li class="toc-item">
              <a href="{{ post.url | relative_url }}">
                <span class="toc-part">{{ post.part }}.</span>
                <span class="toc-title">{{ post.title | remove: "Claude Code 2.0 가이드 (" | remove: ") - " | split: " - " | last }}</span>
              </a>
            </li>
          {% endif %}
        {% endfor %}
      </ul>
    </div>
  </div>

  <div class="guide-quick-links">
    <h2>빠른 참조</h2>
    <div class="quick-links-grid">
      <a href="{{ '/claude-code-2-guide-04-concepts/' | relative_url }}" class="quick-link-card">
        <span class="quick-link-icon">📚</span>
        <span class="quick-link-title">핵심 개념</span>
        <span class="quick-link-desc">컨텍스트, 도구 호출, 에이전트 등</span>
      </a>
      <a href="{{ '/claude-code-2-guide-07-subagents/' | relative_url }}" class="quick-link-card">
        <span class="quick-link-icon">🤖</span>
        <span class="quick-link-title">서브 에이전트</span>
        <span class="quick-link-desc">Explore, Plan, Task 도구</span>
      </a>
      <a href="{{ '/claude-code-2-guide-09-context-engineering/' | relative_url }}" class="quick-link-card">
        <span class="quick-link-icon">⚙️</span>
        <span class="quick-link-title">컨텍스트 엔지니어링</span>
        <span class="quick-link-desc">토큰 관리와 최적화</span>
      </a>
      <a href="{{ '/claude-code-2-guide-11-skills-hooks/' | relative_url }}" class="quick-link-card">
        <span class="quick-link-icon">🔧</span>
        <span class="quick-link-title">스킬 & 훅</span>
        <span class="quick-link-desc">워크플로우 자동화</span>
      </a>
    </div>
  </div>
</section>
