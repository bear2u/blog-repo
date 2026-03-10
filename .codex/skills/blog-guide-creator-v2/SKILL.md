---
name: blog-guide-creator-v2
description: |
  GitHub 레포지토리(또는 외부 문서/블로그 URL)를 분석해 /home/blog Jekyll 블로그에
  "한국어 위키형 가이드 시리즈"를 생성/업데이트하는 v2 스킬.
  긴 작업을 한 턴에 몰지 않고, 단계형 에이전트 워크플로우와 체크포인트를 사용해
  여러 턴에 걸쳐 재개 가능하게 진행한다.

  사용 시점:
  - "이 레포를 위키 형태로 정리해줘"
  - "옵시디언 링크로 연결된 블로그 문서로 만들어줘"
  - "Mermaid 포함해서 체계적으로 문서화해줘"
  - 기존 가이드 시리즈를 위키 구조로 재정리할 때
---

# Blog Guide Creator v2 (Wiki Form)

## 목표

- 입력(레포 URL/문서 URL)만으로 `_posts/`에 위키형 챕터 문서를 만들고, 시리즈 인덱스(루트) + guides 탭을 갱신한다.
- 결과물은 "블로그 포스트"이면서 동시에 "위키 노드"처럼 탐색 가능해야 한다.
- 모든 시리즈는 확장 가능성을 전제로 작성한다(새 노드/챕터를 쉽게 추가 가능).
- 긴 분석/생성 작업은 **분석 -> 챕터 배치 생성 -> 마무리**의 단계로 나눠 여러 턴에 걸쳐 수행한다.
- 각 턴이 끝날 때마다 현재 상태와 다음 작업을 요약하고, 다음 턴은 기존 체크포인트에서 재개한다.

## 핵심 운영 원칙

- 한 턴에 모든 문서를 만들려고 하지 말고, 반드시 **작은 배치**로 끊는다.
- 분석 결과와 진행 상태는 디스크에 남긴다. 메모리에만 유지하지 않는다.
- 이미 완료한 챕터는 다시 만들지 않는다. 수정 근거가 있거나 사용자가 요청한 경우만 다시 쓴다.
- 진행 메시지는 사용자에게 보여주되, 실제 진척은 파일과 상태 파일로 보존한다.
- 타임아웃 회피가 목적이므로, `jekyll build`와 `git commit`은 **기본 비활성화**하고 마지막 턴 또는 사용자 요청 시에만 수행한다.

## 산출물(이 repo 기준)

- 챕터 포스트: `/home/blog/_posts/YYYY-MM-DD-{series}-guide-XX-{slug}.md`
- 시리즈 인덱스(루트): `/home/blog/{series}-guide.md`
- 가이드 목록 카드: `/home/blog/_tabs/guides.md` 해당 섹션의 맨 위
- 진행 상태 파일: `/home/blog/.codex/state/blog-guide-creator-v2/{series}.json`
- 분석 메모: `/home/blog/.codex/state/blog-guide-creator-v2/{series}.analysis.md`
- 위키 링크 규칙:
  - 각 문서 상단/하단에 관련 노드 링크를 명시
  - Obsidian 스타일 `[[문서명]]` + 블로그 URL 링크를 함께 적는다

## 입력(사용자에게 확인할 것)

- 대상: `REPO_URL` 또는 `DOC_URL` (둘 중 하나)
- 표기: `PROJECT_NAME` (표지/타이틀용), `AUTHOR` (front matter)
- 분류: `categories`, `tags` (애매하면 "프로젝트 성격" 1개를 우선)
- 시리즈 슬러그: `{series}` (기본: `project-name` 소문자/하이픈)
- 챕터 수: 자동(권장 8~12) 또는 사용자 지정
- `_tabs/guides.md` 에서 들어갈 섹션(예: AI 에이전트, 개발 도구 등)

## 필수 실행 시나리오(v2)

아래 3단계는 반드시 반영한다.

1) GitHub 프로젝트를 클론한다.

```bash
git clone [프로젝트 주소]
```

2) 분석/문서 생성은 "코드베이스 기반 위키 작성 프롬프트"를 기준으로 한다.

- 기본 프롬프트 파일: `references/wiki-generation-prompt.md`
- 핵심 요구: **추측 금지**, **근거(파일/클래스/함수/설정 키) 제시**, **Mermaid 필수**, **실무형(명령어/경로/주의사항)**

3) 위키를 읽으면서 설명이 필요한 부분은 Mermaid 도표를 만든다.

- 아키텍처
- 데이터/이벤트 흐름
- 실행 파이프라인
- 상태 전이(있을 경우)

## 에이전트형 턴 운영 방식

이 스킬은 긴 단일 턴 실행이 아니라, 아래와 같은 **다단계 에이전트 워크플로우**로 사용한다.

### Phase 1. 분석 턴

- 레포를 클론하고 핵심 파일을 분석한다.
- 시리즈 메타데이터와 노드 구조를 확정한다.
- 결과를 상태 파일에 저장한다.
- 이 턴에서는 보통 문서를 0~1개만 만든다.

상태 파일 예시 필드:

```json
{
  "series": "notebooklm-py",
  "repo_url": "https://github.com/teng-lin/notebooklm-py",
  "project_name": "notebooklm-py",
  "author": "teng-lin",
  "section": "개발 도구",
  "status": "writing",
  "scratch_dir": "/tmp/codex-blog-guide-creator.xxxxxx/notebooklm-py",
  "chapters": [
    { "num": "01", "slug": "intro", "title": "소개 및 개요", "status": "done" },
    { "num": "02", "slug": "setup", "title": "설치와 실행", "status": "pending" }
  ],
  "completed_files": [],
  "next_batch": ["02", "03"],
  "last_updated": "2026-03-10"
}
```

### Phase 2. 챕터 생성 턴

- 한 턴에 챕터를 **2~3개만** 생성한다.
- 각 챕터는 완성 즉시 `_posts/`에 저장한다.
- 저장 직후 상태 파일의 `chapters[].status`, `completed_files`, `next_batch`를 갱신한다.
- 턴 종료 시:
  - 이번 턴에 생성한 파일 목록
  - 남은 챕터 수
  - 다음 턴 추천 배치
  를 사용자에게 짧게 알린다.

### Phase 3. 마무리 턴

- 모든 챕터가 생성되면 시리즈 인덱스(루트)와 `_tabs/guides.md`를 갱신한다.
- 링크, 위키 링크, 챕터 개수, 카드 설명을 정리한다.
- 필요한 경우에만 최소 검증을 수행한다.

### Phase 4. 선택적 검증/커밋 턴

- `git status`는 기본 검증으로 허용한다.
- `bundle exec jekyll build`는 사용자가 요청하거나 시간이 충분한 마지막 턴에만 실행한다.
- `git commit`도 기본 자동 실행하지 않는다. 사용자가 원할 때만 수행한다.

### 재개 규칙

- 새 요청이 들어오면 먼저 `/home/blog/.codex/state/blog-guide-creator-v2/{series}.json` 존재 여부를 확인한다.
- 상태 파일이 있으면 **처음부터 다시 분석하지 말고** 거기서 재개한다.
- 상태 파일이 없거나 소스 대상이 바뀌었을 때만 새 작업을 시작한다.

## 문서 생성 프롬프트(기본)

- 파일: `references/wiki-generation-prompt.md`
- 사용법: 레포 분석 결과(파일/경로/코드 스니펫 근거)와 함께 위 파일 내용을 "그대로" 베이스 프롬프트로 사용한다.
- 단, 긴 한 번의 출력으로 전체 `wiki/*.md`를 모두 생성하지 말고, **현재 턴에 필요한 노드/챕터만** 선택해서 작성한다.
- v2 출력 매핑: 프롬프트의 `wiki/*.md` 파일 단위 산출물은, 이 블로그에서는 **위키 노드(챕터 포스트)** 단위로 쪼개어 `_posts/`에 저장한다.

## 워크플로우

### 1) 임시 작업 디렉토리 준비

```bash
SCRATCH="$(mktemp -d /tmp/codex-blog-guide-creator.XXXXXX)"
echo "scratch: $SCRATCH"
STATE_DIR="/home/blog/.codex/state/blog-guide-creator-v2"
mkdir -p "$STATE_DIR"
```

### 2) 소스 수집(레포/문서)

- GitHub 레포인 경우:

```bash
cd "$SCRATCH"
git clone --depth 1 "$REPO_URL"
REPO_NAME="$(basename "$REPO_URL" .git)"
cd "$REPO_NAME"
```

### 3) 핵심 파일 분석(우선순위)

- `README.md`
- `AGENTS.md` / `CLAUDE.md` / `CONTRIBUTING.md` (있으면)
- 언어/빌드 설정: `package.json`, `Cargo.toml`, `go.mod`, `pyproject.toml`, `requirements.txt` 등
- 핵심 소스 디렉토리: `src/`, `crates/`, `packages/` 등

권장 커맨드:

```bash
ls -la
find . -maxdepth 2 -type d | head -50
find . -maxdepth 3 -name "*.md" | head -50
```

### 4) 위키 노드 구조 설계

- 원칙: 내용 기준 5~15개 노드로 분할한다.
- 인덱스(허브) + 핵심 노드 + 운영/확장 노드 구조를 기본으로 한다.
- 문서 파일이 명확히 N개면 1:1 매핑을 우선하되, 탐색성을 위해 상위 허브 노드는 유지한다.
- 반드시 "문서 점검 자동화" 노드를 1개 포함한다.
- 기본 권장값은 **8개 노드**다. 큰 레포만 10개 이상으로 늘린다.
- 한 턴에서 처리할 배치는 기본 **2~3개 노드**다.

노드 목록 확정 예시:

```text
01:intro
02:setup
03:architecture
04:core-modules
05:workflow
06:state-and-data
07:operations
08:doc-automation
09:extensions
10:troubleshooting
```

### 5) 파일 생성(위키형 블로그)

- 날짜는 항상 시스템 날짜 사용: `CURRENT_DATE=$(date +%Y-%m-%d)`
- 파일명: `_posts/${CURRENT_DATE}-${series}-guide-${num}-${slug}.md`
- permalink: `/{series}-guide-{num}-{slug}/`

템플릿은 필요할 때만 로드:
- 포스트 front matter/본문 틀: `references/post-template.md`
- 시리즈 인덱스 페이지 틀: `references/index-template.md`
- 네비/목차 스타일: `references/nav-template.md`
- 챕터 구성 가이드: `references/chapter-structure.md`

문서 본문 규칙(v2):

- **추측(가정) 금지**: 근거를 못 찾으면 "확인 필요"로 표시
- 각 챕터에 최소 포함 섹션:
  - "이 문서의 목적", "빠른 요약", "근거(파일/경로)", "주의사항/함정", "TODO/확인 필요"
- 핵심 흐름/아키텍처는 Mermaid로 시각화(가능한 범위에서):
  - Context / Component(or Container) / Sequence / (가능하면) Deployment(or Runtime Topology)
- 각 챕터에 `## 위키 링크` 섹션을 넣고 관련 노드를 연결
- Obsidian 문법 + URL 병기
  - 예: `[[Project Guide - Architecture]]`
  - 예: `[Architecture](/blog-repo/{series}-guide-03-architecture/)`
- "문서관리/점검 자동화" 챕터에는 체크리스트 + 자동화 스크립트(예시) + 확장 포인트 포함
- 챕터를 저장할 때마다 상태 파일을 즉시 갱신한다.
- 이미 생성된 파일이 있으면 덮어쓰기 전에 상태 파일과 기존 문서를 검토한다.

턴 종료 시 사용자 보고 형식:

```text
이번 턴 완료:
- 01 intro
- 02 setup

현재 상태:
- 완료 2 / 8
- 다음 배치: 03 architecture, 04 core-modules
- 상태 파일: /home/blog/.codex/state/blog-guide-creator-v2/{series}.json
```

### 6) guides 탭 카드 업데이트

- 파일: `/home/blog/_tabs/guides.md`
- 규칙: 해당 섹션 `guide-grid` 맨 위에 새 카드 추가

### 7) 검증(최소)

```bash
cd /home/blog
git status
```

필요할 때만(환경이 갖춰져 있고, 마지막 턴이며, 사용자가 원할 때만):

```bash
bundle exec jekyll build
```

### 8) 정리

```bash
rm -rf "$SCRATCH"
```

## Git 작업

- 기본: 파일까지만 만들고 종료한다
- `git add` / `commit`은 사용자가 요청한 경우에만 수행
- `push`는 사용자가 명시적으로 요청한 경우에만 수행

```bash
cd /home/blog
git add _posts/{date}-{series}-guide-*.md {series}-guide.md _tabs/guides.md
git commit -m "{PROJECT_NAME} 위키형 가이드 시리즈 추가"
# git push
```
