---
layout: post
title: "Acontext 완벽 가이드 (4) - 스토리지 계층"
date: 2026-02-06
permalink: /acontext-guide-04-storage/
author: memodb-io
categories: [AI 에이전트, Acontext]
tags: [Acontext, Disk, Skill, Sandbox, Multi-tenant]
original_url: "https://docs.acontext.io/store/disk"
excerpt: "Disk/Skill/Sandbox와 사용자 단위 격리 모델을 설명합니다."
---

## Disk

Disk는 에이전트를 위한 **영구 파일 저장소**입니다.

- 파일 업로드/다운로드 URL
- 경로/메타데이터 관리
- 툴 호출로 읽기/쓰기/검색 연동

## Skill

Skill은 `SKILL.md`를 포함한 ZIP 패키지로 업로드합니다.

- 지침 + 스크립트 + 리소스를 묶어 재사용
- 서버 측에서 보관/조회
- 샌드박스 마운트 가능

## Sandbox

Sandbox는 격리된 컨테이너 환경에서 명령 실행을 제공합니다.

- 코드 실행
- 파일 편집
- 결과 산출물(export)

## 사용자 단위 리소스 격리

Session, Disk, Skill 생성 시 `user` 식별자를 붙일 수 있습니다.

- 멀티테넌시 분리
- 사용자 기준 조회
- 사용자 삭제 시 연관 리소스 정리(캐스케이드)
