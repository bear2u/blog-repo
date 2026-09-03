---
layout: page
title: Guides
permalink: /guides/
icon: fas fa-book
order: 1
---

{% include browse-styles.html %}

<p class="browse-intro">루트 목차 페이지와 시리즈 글에서 자동으로 모은 가이드입니다. 검색으로 제목·설명을 걸러 보세요.</p>

<div class="browse-actions">
  <a href="{{ "/" | relative_url }}">최근 글</a>
  <a href="{{ "/trends/" | relative_url }}">GitHub Trending</a>
</div>

<label class="visually-hidden" for="guide-search">시리즈 검색</label>
<input type="search" id="guide-search" class="browse-search" placeholder="시리즈 검색…" autocomplete="off">
<p class="browse-count" id="guide-count"></p>

{% include series-cards.html limit=0 include_pages=true searchable=true grid_id="series-grid" %}

<p class="browse-empty" id="guide-empty">검색과 맞는 시리즈가 없습니다.</p>

<script>
(function () {
  var input = document.getElementById("guide-search");
  var grid = document.getElementById("series-grid");
  var empty = document.getElementById("guide-empty");
  var count = document.getElementById("guide-count");
  if (!input || !grid) return;
  var cards = Array.prototype.slice.call(grid.querySelectorAll(".browse-card"));
  var total = cards.length;
  function apply() {
    var q = (input.value || "").trim().toLowerCase();
    var shown = 0;
    cards.forEach(function (card) {
      var hay = card.getAttribute("data-search") || "";
      var match = !q || hay.indexOf(q) !== -1;
      card.hidden = !match;
      if (match) shown += 1;
    });
    if (count) {
      count.textContent = q ? (shown + " / " + total + "개 시리즈") : (total + "개 시리즈");
    }
    if (empty) empty.style.display = shown ? "none" : "block";
  }
  input.addEventListener("input", apply);
  apply();
})();
</script>
