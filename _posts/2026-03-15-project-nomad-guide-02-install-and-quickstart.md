---
layout: post
title: "Project N.O.M.A.D. 완벽 가이드 (02) - 설치 및 빠른 시작"
date: 2026-03-15
permalink: /project-nomad-guide-02-install-and-quickstart/
author: Crosstalk-Solutions
categories: [GitHub Trending, project-nomad]
tags: [Trending, GitHub, project-nomad, Installation, Docker, Ubuntu]
original_url: "https://github.com/Crosstalk-Solutions/project-nomad"
excerpt: "README의 터미널 기반 설치 스크립트 흐름을 기준으로, 설치→브라우저 접속(:8080)→기본 헬스체크(/api/health)→기초 운영 스크립트(start/stop/update/uninstall)까지 정리합니다."
---

## 이 문서의 목적

- README의 “Installation & Quickstart” 경로를 **재현 가능한 체크리스트**로 정리합니다. (`README.md`)
- 설치 후 최소 확인(헬스체크/API 응답)을 추가해 “정상 기동” 여부를 빠르게 판단합니다. (`admin/start/routes.ts`)

---

## 빠른 요약(README/라우트 기반)

- 설치는 Debian 계열(예: Ubuntu)에서 **sudo로 install 스크립트 실행**이 기본 경로입니다. (`README.md`, `install/install_nomad.sh`)
- 설치 후 접속: `http://localhost:8080` (또는 `http://DEVICE_IP:8080`) (`README.md`)
- 서버 헬스체크(코드 근거): `GET /api/health` → `{ status: 'ok' }` (`admin/start/routes.ts`)
- 설치 후 보조 스크립트 경로(README): `/opt/project-nomad/start_nomad.sh`, `/opt/project-nomad/stop_nomad.sh`, `/opt/project-nomad/update_nomad.sh` (`README.md`)

---

## 1) 설치(README의 Quick Install)

```bash
sudo apt-get update && sudo apt-get install -y curl \
  && curl -fsSL https://raw.githubusercontent.com/Crosstalk-Solutions/project-nomad/refs/heads/main/install/install_nomad.sh -o install_nomad.sh \
  && sudo bash install_nomad.sh
```

근거:
- 스크립트 본체: `install/install_nomad.sh`
- 안내 문서: `README.md`

---

## 2) 브라우저 접속(:8080)

README에 따르면 설치 완료 후 아래로 접속합니다. (`README.md`)

- 로컬 장치: `http://localhost:8080`
- LAN 접근: `http://DEVICE_IP:8080`

---

## 3) 최소 헬스체크(/api/health)

라우트 정의에 따르면 다음 엔드포인트가 존재합니다. (`admin/start/routes.ts`)

```bash
curl -sS http://localhost:8080/api/health
```

성공 시 기대 응답(코드 근거): `{ status: 'ok' }` (`admin/start/routes.ts`)

---

## 4) (README) 설치 후 운영 스크립트: `/opt/project-nomad/*`

README는 설치 후 운영/문제해결을 위한 스크립트가 `/opt/project-nomad` 아래에 있다고 안내합니다. (`README.md`)

- 시작: `sudo bash /opt/project-nomad/start_nomad.sh`
- 중지: `sudo bash /opt/project-nomad/stop_nomad.sh`
- 업데이트(컨트롤 플레인 위주): `sudo bash /opt/project-nomad/update_nomad.sh`

---

## 5) 제거(README의 Uninstall Script)

```bash
curl -fsSL https://raw.githubusercontent.com/Crosstalk-Solutions/project-nomad/refs/heads/main/install/uninstall_nomad.sh -o uninstall_nomad.sh \
  && sudo bash uninstall_nomad.sh
```

근거:
- 스크립트 본체: `install/uninstall_nomad.sh`
- 안내 문서: `README.md`

---

## 근거(파일/경로)

- 설치/접속/운영 스크립트 안내: `README.md`
- 설치 스크립트: `install/install_nomad.sh`
- 제거 스크립트: `install/uninstall_nomad.sh`
- 헬스체크 라우트: `admin/start/routes.ts` (`/api/health`)

---

## 주의사항/함정(README 기반)

- 설치 스크립트 실행에는 sudo/root 권한이 필요합니다. (`README.md`)
- README는 **인터넷 직접 노출**을 권장하지 않습니다(로컬/LAN 위주). (`README.md`)

---

## TODO/확인 필요

- `install/install_nomad.sh`가 설정하는 데이터 디렉토리/포트/컨테이너 목록을 실제 스크립트 근거로 정리
- `/opt/project-nomad` 아래 스크립트가 어떤 컨테이너(또는 systemd 유닛)를 제어하는지 추적

---

## 위키 링크

- `[[Project NOMAD Guide - Index]]` → [가이드 목차](/blog-repo/project-nomad-guide/)
- `[[Project NOMAD Guide - Architecture]]` → [03. 아키텍처](/blog-repo/project-nomad-guide-03-architecture/)

---

*다음 글에서는 `admin/`(AdonisJS) 기준으로 Command Center가 어떤 API 그룹을 노출하고, Docker 기반 서비스 관리를 어떻게 수행하는지 아키텍처로 정리합니다.*

