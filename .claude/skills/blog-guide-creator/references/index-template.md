# 인덱스 페이지 템플릿

## 파일 위치
블로그 루트: `{series-slug}.md`

## Front Matter

```yaml
---
layout: default
title: {프로젝트명} 완벽 가이드
permalink: /{series-slug}/
---
```

## HTML 구조

```html
<section class="guide-header">
  <h1 class="guide-title">{프로젝트명} 완벽 가이드</h1>
  <p class="guide-desc">{한 줄 설명}</p>
  <p class="guide-author">원본: <a href="{원본URL}" target="_blank">{링크텍스트}</a></p>
</section>

<section class="guide-container">
  <div class="guide-intro">
    <p>{시리즈 소개 문구}</p>
  </div>

  <div class="guide-toc">
    <h2>목차</h2>

    <div class="toc-section">
      <h3>Part 1: {파트명}</h3>
      <ul class="toc-list">
        <li class="toc-item">
          <a href="{{ '/{series}-{part}-{slug}/' | relative_url }}">
            <span class="toc-part">{번호}.</span>
            <span class="toc-title">{제목}</span>
          </a>
        </li>
        <!-- 반복 -->
      </ul>
    </div>

    <!-- Part 2, 3, 4 반복 -->
  </div>

  <div class="guide-quick-links">
    <h2>빠른 참조</h2>
    <div class="quick-links-grid">
      <a href="{{ '/{permalink}/' | relative_url }}" class="quick-link-card">
        <span class="quick-link-icon">{이모지}</span>
        <span class="quick-link-title">{제목}</span>
        <span class="quick-link-desc">{설명}</span>
      </a>
      <!-- 3-4개 반복 -->
    </div>
  </div>
</section>
```

## 빠른 참조 아이콘 예시
- 🏗️ 아키텍처
- 🤖 AI/에이전트
- ⚙️ 설정/Operators
- 🔧 도구/인프라
- 📚 개념/가이드
- 💻 코드/개발
- 🔌 API/통합
- 🚀 시작하기
