---
layout: post
title: "Project N.O.M.A.D. 완벽 가이드 (01) - 소개 및 개요"
date: 2026-03-15
permalink: /project-nomad-guide-01-intro/
author: Crosstalk-Solutions
categories: [GitHub Trending, project-nomad]
tags: [Trending, GitHub, project-nomad, Offline, Docker, AdonisJS]
original_url: "https://github.com/Crosstalk-Solutions/project-nomad"
excerpt: "오프라인-우선 지식/교육 서버 Project N.O.M.A.D.의 목표, 포함 기능, Command Center(UI/API)+Docker 오케스트레이션 구조를 레포 근거로 개관합니다."
---

## 이 문서의 목적

- Project N.O.M.A.D.가 “무엇을 해결하는지”와 “무엇이 포함되는지”를 README 근거로 요약합니다. (`README.md`)
- 이후 챕터에서 볼 **구성요소(install/ · admin/ · collections/)** 와 **API 영역**을 미리 지도처럼 잡습니다. (`admin/start/routes.ts`)

---

## 빠른 요약(README/코드 구조 기반)

- **한 줄 요약**: 오프라인-우선(offline-first) 지식/교육 서버로, 브라우저에서 접근하는 **Command Center(UI) + API**가 **Docker 기반 도구/리소스**를 설치·구성·업데이트로 오케스트레이션합니다. (`README.md`, `admin/start/routes.ts`)
- **접속**: 설치 후 `http://localhost:8080` (또는 `http://DEVICE_IP:8080`) (`README.md`)
- **레포 구조(상위)**:
  - `install/` : 설치/업데이트 관련 스크립트/사이드카
  - `admin/` : Command Center(AdonisJS) + Inertia/React UI + API
  - `collections/` : 큐레이션된 콘텐츠 컬렉션

---

## “무엇이 포함되나?”(README의 Built-in capabilities)

README가 명시하는 내장 기능(“Powered by” 포함)은 다음 범주로 정리됩니다. (`README.md`)

- **AI Chat + Knowledge Base**: Ollama + Qdrant 기반 로컬 챗/문서 업로드/RAG
- **Information Library**: Kiwix 기반 오프라인 위키/문서
- **Education Platform**: Kolibri 기반 교육 콘텐츠/진도
- **Offline Maps**: ProtoMaps 기반 지역 맵 다운로드
- **Data Tools**: CyberChef 기반 데이터 유틸리티
- **Notes**: FlatNotes 기반 로컬 노트
- **Benchmark**: 하드웨어 점수/리더보드(내장)

이 항목들은 “도구를 개별 설치”가 아니라, Command Center가 컨테이너로 설치/관리하는 모델로 설명됩니다. (`README.md`)

---

## 상위 아키텍처(개념도)

README의 “How It Works” 설명과 `admin/start/routes.ts`의 API 표면을 합치면, 핵심은 **Command Center(UI/API)** 가 Docker 컨테이너(도구/리소스)를 **오케스트레이션**하는 것입니다. (`README.md`, `admin/start/routes.ts`)

```mermaid
flowchart LR
  U[User Browser] -->|HTTP :8080| CC[Command Center\nadmin/ (AdonisJS)]
  CC -->|exposes| API[/HTTP API\nadmin/start/routes.ts/]
  CC -->|Docker orchestration| D[(Docker Engine)]
  D --> T1[Tool Containers\n(Ollama/Qdrant/Kiwix/...)]
  CC -->|manages| C[collections/\ncurated content]
```

---

## “어디를 보면 되는가?”(근거 파일 지도)

- 개요/설치/운영: `README.md`
- 설치 스크립트(README에서 raw URL로 호출): `install/install_nomad.sh`
- 제거 스크립트(README에서 raw URL로 호출): `install/uninstall_nomad.sh`
- Command Center 라우트(모든 API 엔드포인트): `admin/start/routes.ts`

---

## 주의사항/함정(README 기반)

- 설치/업데이트는 기본적으로 **sudo/root 권한**이 필요합니다. (`README.md`)
- README는 “인터넷에 직접 노출”을 강하게 경고합니다(로컬/LAN 사용 전제). (`README.md`)

---

## TODO/확인 필요

- `install/install_nomad.sh`가 실제로 설치하는 구성요소(패키지, docker, systemd 유닛, 데이터 경로)를 단계별로 표로 정리
- “도구 컨테이너 목록”을 코드에서(예: `admin/app/services/*` 또는 설정 파일) 근거로 추출해 맵핑

---

## 위키 링크

- `[[Project NOMAD Guide - Index]]` → [가이드 목차](/blog-repo/project-nomad-guide/)
- `[[Project NOMAD Guide - Install]]` → [02. 설치 및 빠른 시작](/blog-repo/project-nomad-guide-02-install-and-quickstart/)

---

*다음 글에서는 README의 Quick Install 흐름을 그대로 재현 가능한 형태로 정리하고, 설치 후 “살아있는지” 확인하는 최소 점검을 추가합니다.*

