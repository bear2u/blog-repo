---
layout: page
title: Calendar
permalink: /calendar/
icon: fas fa-calendar-alt
order: 3
---

# GitHub Trending 캘린더

GitHub Trending(daily) 스냅샷을 **날짜별로 저장**하고, 월간 캘린더 형태로 모아봅니다.

- 소스: `https://github.com/trending?since=daily`
- 기준: 이 블로그에 저장된 스냅샷 포스트(permaliink가 `/github-trending-YYYY-MM-DD/`인 글)

<div class="calendar-controls">
  <button id="calPrev" type="button" class="btn btn-sm btn-outline-primary">◀</button>
  <div id="calTitle" class="calendar-title"></div>
  <button id="calNext" type="button" class="btn btn-sm btn-outline-primary">▶</button>
</div>

<div class="calendar-grid" id="calGrid" aria-label="GitHub Trending calendar"></div>

<div class="calendar-detail" id="calDetail" style="display:none;"></div>

<noscript>
  <p>자바스크립트가 비활성화되어 캘린더 UI를 표시할 수 없습니다. 대신 목록으로 봅니다.</p>
  <ul>
    {% assign daily_posts = site.posts | where_exp: "p", "p.url contains '/github-trending-'" | sort: "date" | reverse %}
    {% for p in daily_posts %}
      <li><a href="{{ p.url | relative_url }}">{{ p.title }}</a> ({{ p.date | date: "%Y-%m-%d" }})</li>
    {% endfor %}
  </ul>
</noscript>

<style>
.calendar-controls{
  display:flex;
  align-items:center;
  gap: .75rem;
  margin: 1rem 0;
}
.calendar-title{
  font-weight: 700;
  flex: 1;
  text-align:center;
}
.calendar-grid{
  display:grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: .5rem;
}
.cal-cell{
  border: 1px solid var(--card-border-color, #e9ecef);
  border-radius: .5rem;
  padding: .5rem;
  min-height: 5.2rem;
  background: var(--card-bg, #fff);
}
.cal-dow{
  font-size: .85rem;
  font-weight: 700;
  opacity: .75;
  text-align:center;
  padding: .25rem 0;
}
.cal-day{
  display:flex;
  justify-content: space-between;
  align-items: baseline;
  font-weight: 700;
  margin-bottom: .25rem;
}
.cal-day .muted{
  font-weight: 600;
  font-size: .8rem;
  opacity: .6;
}
.cal-empty{
  opacity: .35;
}
.cal-item{
  margin-top: .25rem;
  font-size: .85rem;
  line-height: 1.25;
  overflow:hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}
.cal-item a{ text-decoration:none; }
.calendar-detail{
  margin-top: 1rem;
  padding: 1rem;
  border: 1px solid var(--card-border-color, #e9ecef);
  border-radius: .75rem;
  background: var(--card-bg, #fff);
}
.calendar-detail h2{
  margin: 0 0 .5rem 0;
  font-size: 1.2rem;
}
</style>

<script>
(() => {
  const entries = [
    {% assign daily_posts = site.posts | where_exp: "p", "p.url contains '/github-trending-'" | sort: "date" %}
    {% for p in daily_posts %}
      {
        date: "{{ p.date | date: "%Y-%m-%d" }}",
        title: {{ p.title | jsonify }},
        url: "{{ p.url | relative_url }}"
      }{% unless forloop.last %},{% endunless %}
    {% endfor %}
  ];

  const byDate = new Map(entries.map(e => [e.date, e]));
  const qs = new URLSearchParams(window.location.search);
  const today = new Date();

  function pad2(n){ return String(n).padStart(2,'0'); }
  function ymToStr(y,m){ return `${y}-${pad2(m)}`; }
  function ymdToStr(y,m,d){ return `${y}-${pad2(m)}-${pad2(d)}`; }

  let viewY = Number(qs.get('y')) || today.getFullYear();
  let viewM = Number(qs.get('m')) || (today.getMonth() + 1);

  function setQuery(y, m){
    const next = new URL(window.location.href);
    next.searchParams.set('y', String(y));
    next.searchParams.set('m', String(m));
    window.history.replaceState({}, '', next);
  }

  function render(){
    setQuery(viewY, viewM);

    const titleEl = document.getElementById('calTitle');
    const gridEl = document.getElementById('calGrid');
    const detailEl = document.getElementById('calDetail');
    if (!titleEl || !gridEl || !detailEl) return;

    titleEl.textContent = `${viewY}년 ${viewM}월`;
    gridEl.innerHTML = '';

    const dows = ['일','월','화','수','목','금','토'];
    for (const dow of dows){
      const el = document.createElement('div');
      el.className = 'cal-dow';
      el.textContent = dow;
      gridEl.appendChild(el);
    }

    const first = new Date(viewY, viewM - 1, 1);
    const last = new Date(viewY, viewM, 0);
    const startDow = first.getDay();
    const daysInMonth = last.getDate();

    // leading blanks
    for (let i=0; i<startDow; i++){
      const cell = document.createElement('div');
      cell.className = 'cal-cell cal-empty';
      gridEl.appendChild(cell);
    }

    for (let day=1; day<=daysInMonth; day++){
      const dateStr = ymdToStr(viewY, viewM, day);
      const entry = byDate.get(dateStr);

      const cell = document.createElement('div');
      cell.className = 'cal-cell';

      const dayRow = document.createElement('div');
      dayRow.className = 'cal-day';
      dayRow.innerHTML = `<span>${day}</span><span class="muted">${ymToStr(viewY,viewM)}</span>`;
      cell.appendChild(dayRow);

      if (entry){
        const item = document.createElement('div');
        item.className = 'cal-item';
        item.innerHTML = `<a href=\"${entry.url}\">${entry.title}</a>`;
        cell.appendChild(item);

        cell.style.cursor = 'pointer';
        cell.addEventListener('click', (ev) => {
          // link 클릭은 기본 동작 유지
          if (ev.target && ev.target.closest && ev.target.closest('a')) return;
          detailEl.style.display = 'block';
          detailEl.innerHTML = `<h2>${dateStr}</h2><p><a href=\"${entry.url}\">${entry.title}</a>로 이동</p>`;
          detailEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
      }

      gridEl.appendChild(cell);
    }
  }

  document.getElementById('calPrev')?.addEventListener('click', () => {
    viewM -= 1;
    if (viewM < 1){ viewM = 12; viewY -= 1; }
    render();
  });
  document.getElementById('calNext')?.addEventListener('click', () => {
    viewM += 1;
    if (viewM > 12){ viewM = 1; viewY += 1; }
    render();
  });

  render();
})();
</script>
