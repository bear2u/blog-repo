---
layout: post
title: "Acontext 완벽 가이드 (5) - 에이전트 툴"
date: 2026-02-06
permalink: /acontext-guide-05-agent-tools/
author: memodb-io
categories: [AI 에이전트, Acontext]
tags: [Acontext, Tool Calling, Sandbox Tools, Disk Tools, Skill Tools]
original_url: "https://docs.acontext.io/tool/whatis"
excerpt: "LLM 함수 호출로 연결하는 Acontext Agent Tool 세트를 정리합니다."
---

## Agent Tool 개념

Acontext는 저수준 SDK를 직접 조합하지 않아도 되도록, LLM 함수 호출용 툴 세트를 제공합니다.

## 툴 세트

1. **Sandbox Tools**
2. **Disk Tools**
3. **Skill Content Tools**

## Sandbox Tools

대표 도구:

- `bash_execution_sandbox`
- `text_editor_sandbox`
- `export_file_sandbox`

용도:

- 코드 실행
- 파일 생성/수정
- 결과 파일 내보내기

## Disk Tools

대표 도구:

- `write_file_disk`, `read_file_disk`, `replace_string_disk`
- `list_disk`, `glob_disk`, `grep_disk`, `download_file_disk`

용도:

- 영구 파일 저장소를 LLM이 직접 조작

## Skill Content Tools

대표 도구:

- `get_skill`
- `get_skill_file`

용도:

- 샌드박스 없이 스킬 문서형 지식 열람

실행 스크립트가 필요한 스킬은 Skill Tools보다 Sandbox 마운트 방식이 적합합니다.
