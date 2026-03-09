---
layout: page
title: Trends
permalink: /trends/
icon: fas fa-chart-line
order: 2
---

# Trends

자동으로 수집/등록된 트렌드 스냅샷을 모아봅니다.

## GitHub Trending

{% assign trending_posts = site.posts | where_exp: "post", "post.tags contains 'Trending'" %}
{% assign trending_posts = trending_posts | sort: "date" | reverse %}

{% if trending_posts.size == 0 %}
아직 등록된 GitHub Trending 포스트가 없습니다.
{% else %}
<ul>
  {% for post in trending_posts %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <span style="opacity: 0.7;">({{ post.date | date: "%Y-%m-%d" }})</span>
      {% if post.excerpt %} — {{ post.excerpt | strip_html }}{% endif %}
    </li>
  {% endfor %}
</ul>
{% endif %}

