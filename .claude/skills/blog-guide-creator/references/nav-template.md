# 네비게이션 메뉴 템플릿

## 드롭다운 메뉴 추가 위치
`_includes/nav.html` 파일의 `<ul class="nav-menu">` 내부

## 드롭다운 HTML 구조

```html
<li class="nav-dropdown">
  <a href="{{ '/{series-slug}/' | relative_url }}" class="nav-link nav-dropdown-toggle">
    {메뉴명} <span class="dropdown-arrow">▼</span>
  </a>
  <ul class="dropdown-menu">
    <li><a href="{{ '/{series-slug}/' | relative_url }}">전체 목차</a></li>
    <li class="dropdown-divider"></li>
    <li><a href="{{ '/{series}-01-{slug}/' | relative_url }}">1. {제목}</a></li>
    <li><a href="{{ '/{series}-02-{slug}/' | relative_url }}">2. {제목}</a></li>
    <li><a href="{{ '/{series}-03-{slug}/' | relative_url }}">3. {제목}</a></li>
    <!-- 모든 챕터 반복 -->
  </ul>
</li>
```

## 삽입 위치 예시

```html
<ul class="nav-menu" id="navMenu">
  <li><a href="{{ '/' | relative_url }}" class="nav-link">홈</a></li>

  <!-- 기존 드롭다운 메뉴들 -->
  <li class="nav-dropdown">
    <a href="{{ '/claude-code-2-guide/' | relative_url }}" ...>
    ...
  </li>

  <!-- 새로운 드롭다운 메뉴 추가 -->
  <li class="nav-dropdown">
    <a href="{{ '/new-guide/' | relative_url }}" ...>
    ...
  </li>

  <li><a href="..." class="nav-link">GitHub</a></li>
</ul>
```

## post.html 시리즈 링크 추가

`_layouts/post.html`에서 시리즈별 목차 링크 조건 추가:

```liquid
{% if page.series == 'new-series-slug' %}
<a href="{{ '/new-series-slug/' | relative_url }}" class="series-toc-link">전체 목차 →</a>
{% endif %}
```

## index.html 시리즈 가이드 섹션

홈페이지 하단 시리즈 가이드 섹션에 새 가이드 카드 추가:

```html
<a href="{{ '/{series-slug}/' | relative_url }}" class="guide-card">
  <span class="guide-icon">{이모지}</span>
  <h3 class="guide-title">{가이드 제목}</h3>
  <p class="guide-desc">{설명}</p>
  <span class="guide-count">{N}개 챕터</span>
</a>
```
