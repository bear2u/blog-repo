---
name: blog-guide-creator-v2
description: |
  GitHub 레포지토리(또는 외부 문서/블로그 URL)를 분석해 /home/blog Jekyll 블로그에
  "한국어 위키형 가이드 시리즈"를 생성/업데이트하는 v2 스킬.

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

## 산출물(이 repo 기준)

- 챕터 포스트: `/home/blog/_posts/YYYY-MM-DD-{series}-guide-XX-{slug}.md`
- 시리즈 인덱스(루트): `/home/blog/{series}-guide.md`
- 가이드 목록 카드: `/home/blog/_tabs/guides.md` 해당 섹션의 맨 위
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

2) 분석 단계에서 아래 지시를 기준으로 문서 구조를 설계한다.

```text
해당 레포지토리의 위키를 만들어줘.
옵시디언 문법으로 각 체계를 연결하도록 하고, 확장가능성이 있어야 해.
문서관리 도구를 만들어서 점검하는 자동화 체계도 만들고.
```

3) 위키를 읽으면서 설명이 필요한 부분은 Mermaid 도표를 만든다.

- 아키텍처
- 데이터/이벤트 흐름
- 실행 파이프라인
- 상태 전이(있을 경우)

## 워크플로우

### 1) 임시 작업 디렉토리 준비

```bash
SCRATCH="$(mktemp -d /tmp/codex-blog-guide-creator.XXXXXX)"
echo "scratch: $SCRATCH"
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

- 각 챕터에 `## 위키 링크` 섹션을 넣고 관련 노드를 연결
- Obsidian 문법 + URL 병기
  - 예: `[[Project Guide - Architecture]]`
  - 예: `[Architecture](/blog-repo/{series}-guide-03-architecture/)`
- 설명이 필요한 섹션마다 Mermaid 다이어그램을 우선 제공
- "문서관리/점검 자동화" 챕터에는 체크리스트 + 자동화 스크립트(예시) + 확장 포인트 포함

### 6) guides 탭 카드 업데이트

- 파일: `/home/blog/_tabs/guides.md`
- 규칙: 해당 섹션 `guide-grid` 맨 위에 새 카드 추가

### 7) 검증(최소)

```bash
cd /home/blog
git status
```

가능하면(환경이 갖춰져 있으면):

```bash
bundle exec jekyll build
```

### 8) 정리

```bash
rm -rf "$SCRATCH"
```

## Git 작업(필수)

기본 동작은 **항상** `commit` 후 `push`까지 완료한다.
사용자가 명시적으로 "push는 하지 마"라고 한 경우에만 생략한다.

```bash
cd /home/blog
git add _posts/{date}-{series}-guide-*.md {series}-guide.md _tabs/guides.md
git commit -m "{PROJECT_NAME} 위키형 가이드 시리즈 추가"
git push
```
