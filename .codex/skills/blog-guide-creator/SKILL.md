---
name: blog-guide-creator
description: |
  GitHub 레포지토리(또는 외부 문서/블로그 URL)를 분석해서 이 저장소(/home/blog)의 Jekyll 블로그에
  "한국어 챕터형 가이드 시리즈"를 생성/업데이트하는 스킬.

  사용 시점:
  - "이 레포 분석해서 블로그 가이드(챕터별)로 써줘"
  - "문서를 번역해서 한국어 시리즈로 만들어줘"
  - 기존 가이드 시리즈에 새 챕터를 추가하거나 구조를 재정리할 때
---

# Blog Guide Creator

## 목표

- 입력(레포 URL/문서 URL)만으로, `_posts/`에 챕터별 글을 만들고 시리즈 인덱스 페이지(루트)와 가이드 목록을 갱신한다.
- 결과물은 이 블로그의 관례(파일명, permalink, guides 탭 카드)와 맞춰 생성한다.

## 산출물(이 repo 기준)

- 챕터 포스트: `/home/blog/_posts/YYYY-MM-DD-{series}-guide-XX-{slug}.md`
- 시리즈 인덱스(루트 페이지): `/home/blog/{series}-guide.md` (예: `superset-guide.md`)
- 가이드 목록 카드: `/home/blog/_tabs/guides.md` 에 최신 가이드를 섹션 맨 위에 추가

## 입력(사용자에게 확인할 것)

- 대상: `REPO_URL` 또는 `DOC_URL` (둘 중 하나)
- 표기: `PROJECT_NAME` (표지/타이틀용), `AUTHOR` (front matter)
- 분류: `categories`, `tags` (애매하면 "프로젝트 성격" 1개를 우선)
- 시리즈 슬러그: `{series}` (기본: `project-name` 소문자/하이픈)
- 챕터 수: "자동(권장)" 또는 사용자 지정(N개)
- `_tabs/guides.md` 에서 들어갈 섹션(예: AI 에이전트, 개발 도구 등)

## 워크플로우

### 1) 임시 작업 디렉토리(클론용) 준비

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

### 4) 챕터 구조 설계

- 원칙: "내용 기준"으로 5-15개 범위에서 자연스럽게 쪼개거나 합친다.
- 문서 파일이 명확히 N개면, 1:1 매핑(N챕터)이 우선.
- 사용자가 챕터 수를 지정하면 그 제약을 우선한다.

이 단계에서 "챕터 목록"을 확정한다:

```text
01:intro:소개 및 개요
02:installation:설치 및 시작
03:architecture:아키텍처
...
```

### 5) 파일 생성(블로그)

- 날짜는 항상 시스템 날짜 사용: `CURRENT_DATE=$(date +%Y-%m-%d)`
- 파일명 패턴:
  - `_posts/${CURRENT_DATE}-${series}-guide-${num}-${slug}.md`
- permalink 패턴(템플릿 참고): `/{series}-guide-{num}-{slug}/`

템플릿은 필요할 때만 로드:
- 포스트 front matter/본문 틀: `references/post-template.md`
- 시리즈 인덱스 페이지 틀: `references/index-template.md`
- 네비/목차 스타일: `references/nav-template.md`
- 챕터 구성 가이드: `references/chapter-structure.md`

### 6) guides 탭 카드 업데이트

- 파일: `/home/blog/_tabs/guides.md`
- 규칙: 해당 섹션의 `guide-grid` 맨 위에 새 카드(최신)를 추가한다.

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
사용자가 명시적으로 \"push는 하지 마\"라고 한 경우에만 `push`를 생략한다.

```bash
cd /home/blog
git add _posts/{date}-{series}-guide-*.md {series}-guide.md _tabs/guides.md
git commit -m "{PROJECT_NAME} 가이드 시리즈 추가"
git push
```

**Appropriate for:** Templates, boilerplate code, document templates, images, icons, fonts, or any files meant to be copied or used in the final output.

---

**Not every skill requires all three types of resources.**
