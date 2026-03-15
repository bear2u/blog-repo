---
layout: page
title: Project N.O.M.A.D. 가이드
permalink: /project-nomad-guide/
icon: fas fa-globe
---

# Project N.O.M.A.D. 완벽 가이드

> **Knowledge That Never Goes Offline** (`README.md`)

**Project N.O.M.A.D.**는 오프라인-우선(offline-first) 지식/교육 서버를 목표로, 브라우저로 접속하는 **Command Center(UI) + API**가 Docker 기반 도구/리소스를 설치·구성·업데이트로 오케스트레이션하는 프로젝트입니다. (`README.md`, `admin/start/routes.ts`)

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/blog-repo/project-nomad-guide-01-intro/) | 프로젝트 목표, 포함 기능, 구성요소 지도 |
| 02 | [설치 및 빠른 시작](/blog-repo/project-nomad-guide-02-install-and-quickstart/) | Debian 기반 설치, 포트/접속, 기본 점검 |
| 03 | [아키텍처](/blog-repo/project-nomad-guide-03-architecture/) | Docker 오케스트레이션, Command Center 구조 |
| 04 | [API/운영](/blog-repo/project-nomad-guide-04-api-and-operations/) | 주요 API 그룹, 서비스 설치/업데이트 흐름 |
| 05 | [모범사례/문제해결/자동화](/blog-repo/project-nomad-guide-05-best-practices-and-doc-automation/) | 운영 체크리스트, 트러블슈팅, 점검 자동화 |

---

## 빠른 시작(README 기반)

```bash
sudo apt-get update && sudo apt-get install -y curl \
  && curl -fsSL https://raw.githubusercontent.com/Crosstalk-Solutions/project-nomad/refs/heads/main/install/install_nomad.sh -o install_nomad.sh \
  && sudo bash install_nomad.sh
```

설치 후 브라우저에서 `http://localhost:8080` (또는 `http://DEVICE_IP:8080`)로 접속합니다. (`README.md`)

---

## 관련 링크

- GitHub 저장소: https://github.com/Crosstalk-Solutions/project-nomad

