---
name: blog-guide-creator
description: |
  GitHub 레포지토리나 블로그 URL을 분석하여 한국어 챕터별 가이드 시리즈를 자동 생성하는 스킬.

  사용 시점:
  - 사용자가 GitHub 레포지토리 URL을 주고 "가이드 만들어줘", "분석해서 블로그 작성해줘" 요청 시
  - "소스 분석해서 챕터별로 작성해줘" 요청 시
  - 외부 블로그/문서 URL을 주고 "번역해서 시리즈로 만들어줘" 요청 시
  - 기존 블로그에 새로운 가이드 시리즈 추가 요청 시

  생성 결과물:
  - 챕터별 개별 포스트 파일들 (_posts/)
  - 가이드 인덱스 페이지 (루트)
  - _tabs/guides.md와 index.html 자동 업데이트
  - Git commit & push
---

# Blog Guide Creator - GitHub 레포지토리 자동 가이드 생성기

GitHub 레포지토리 URL을 받아 자동으로 클론, 분석하여 한국어 블로그 가이드 시리즈를 생성합니다.

---

## 🎯 워크플로우 개요

```
GitHub URL 입력
    ↓
레포지토리 클론 (Scratchpad)
    ↓
핵심 파일 분석 (README, 소스코드, 문서)
    ↓
챕터 구조 자동 설계 (8-12개)
    ↓
각 챕터별 포스트 생성 (_posts/)
    ↓
인덱스 페이지 생성 (루트)
    ↓
가이드 목록 업데이트 (_tabs/guides.md, index.html)
    ↓
Git commit & push
    ↓
Cleanup (클론한 레포지토리 삭제)
```

---

## 📋 Phase 1: 레포지토리 클론 및 초기 분석

### 1.1 저장소 클론

```bash
# 현재 날짜 확인 (파일명에 사용)
CURRENT_DATE=$(date +%Y-%m-%d)
echo "현재 날짜: $CURRENT_DATE"

# 레포지토리 이름 추출
REPO_URL="https://github.com/user/project"
REPO_NAME=$(basename "$REPO_URL" .git)
echo "레포지토리: $REPO_NAME"

# Scratchpad에 클론 (임시 파일용 전용 디렉토리)
cd /tmp/claude-0/-home-blog/*/scratchpad
rm -rf "$REPO_NAME" 2>/dev/null
git clone --depth 1 "$REPO_URL"
cd "$REPO_NAME"
```

### 1.2 디렉토리 구조 파악

```bash
# 프로젝트 루트 파일 목록
echo "=== 루트 파일 ==="
ls -la

# 주요 디렉토리 구조 (최대 2단계)
echo "=== 디렉토리 구조 ==="
find . -maxdepth 2 -type d | head -30

# 마크다운 문서 찾기
echo "=== 문서 파일 ==="
find . -maxdepth 3 -name "*.md" | head -20
```

### 1.3 프로젝트 메타데이터 확인

```bash
# 언어/프레임워크 확인
if [ -f "package.json" ]; then
    echo "Node.js/TypeScript 프로젝트"
    cat package.json | head -30
elif [ -f "Cargo.toml" ]; then
    echo "Rust 프로젝트"
    cat Cargo.toml
elif [ -f "go.mod" ]; then
    echo "Go 프로젝트"
    cat go.mod
elif [ -f "pyproject.toml" ]; then
    echo "Python 프로젝트"
    cat pyproject.toml
elif [ -f "requirements.txt" ]; then
    echo "Python 프로젝트"
    cat requirements.txt
fi
```

---

## 📖 Phase 2: 핵심 파일 분석 및 내용 추출

### 2.1 필수 분석 파일 순서

1. **README.md** (최우선)
   - 프로젝트 설명
   - 주요 기능
   - 설치/실행 방법
   - 라이선스

2. **CLAUDE.md / AGENTS.md / CONTRIBUTING.md**
   - 개발 가이드
   - 아키텍처 설명
   - 코딩 컨벤션

3. **패키지 설정 파일**
   - package.json / Cargo.toml / go.mod
   - 의존성 목록
   - 스크립트/명령어

4. **핵심 소스 디렉토리**
   - src/ 또는 주요 코드 디렉토리
   - 모듈/패키지 구조

### 2.2 분석 체크리스트

각 파일을 읽으면서 다음 정보를 추출:

```markdown
## 프로젝트 정보
- **이름**: {프로젝트명}
- **한 줄 설명**: {간단한 설명}
- **주요 목적**: {무엇을 하는 도구/라이브러리인가?}
- **타겟 사용자**: {누가 사용하는가?}

## 기술 스택
- **언어**: {주 언어}
- **프레임워크**: {사용 프레임워크}
- **주요 라이브러리**: {핵심 의존성 3-5개}
- **플랫폼**: {OS, 환경}

## 아키텍처
- **디렉토리 구조**: {주요 폴더 설명}
- **모듈 분리**: {어떻게 나뉘는가?}
- **설계 패턴**: {사용된 패턴}

## 핵심 기능 (5-10개)
1. {기능1}
2. {기능2}
...

## 설치/실행
- **요구사항**: {필요한 것들}
- **설치 명령**: {설치 커맨드}
- **실행 명령**: {실행 커맨드}
```

---

## 🏗️ Phase 3: 챕터 구조 자동 설계

### 3.1 기본 템플릿 (10개 챕터)

| # | 제목 패턴 | 내용 | 조건 |
|---|----------|------|------|
| 01 | 소개 및 개요 | 프로젝트란? 주요 특징, 왜 사용하는가 | **필수** |
| 02 | 설치 및 시작 | 요구사항, 설치 방법, 빠른 시작 | **필수** |
| 03 | 아키텍처 분석 | 전체 구조, 디렉토리, 설계 원칙 | **필수** |
| 04 | 핵심 모듈 1 | 가장 중요한 모듈 상세 분석 | 프로젝트에 따라 동적 |
| 05 | 핵심 모듈 2 | 두 번째 주요 모듈 | 프로젝트에 따라 동적 |
| 06 | 핵심 모듈 3 | 세 번째 주요 모듈 | 프로젝트에 따라 동적 |
| 07 | API/라우터 | API 구조, 엔드포인트 | 백엔드가 있으면 |
| 08 | 외부 연동 | 외부 서비스, 플러그인 | 통합 기능이 있으면 |
| 09 | UI/프론트엔드 | UI 컴포넌트, 상태 관리 | 프론트엔드가 있으면 |
| 10 | 확장 및 커스터마이징 | 설정, 확장 방법, 결론 | **필수** |

### 3.2 프로젝트 타입별 커스터마이징

#### CLI 도구
- 04: CLI 명령어
- 05: 설정 시스템
- 06: 플러그인 아키텍처

#### 웹 애플리케이션
- 04: 백엔드 API
- 05: 프론트엔드 구조
- 06: 데이터베이스 설계

#### AI 에이전트
- 04: 에이전트 코어
- 05: LLM 통합
- 06: 도구/함수 시스템

#### 라이브러리
- 04: 핵심 API
- 05: 내부 구현
- 06: 확장 포인트

### 3.3 챕터 제목 한국어 네이밍 규칙

```
{프로젝트명} 완벽 가이드 ({번호:02d}) - {챕터 제목}
```

예시:
- Superset 완벽 가이드 (01) - 소개 및 개요
- WrenAI 완벽 가이드 (05) - RAG 파이프라인
- UI-TARS 완벽 가이드 (08) - MCP 통합

---

## 📝 Phase 4: 포스트 파일 자동 생성

### 4.1 날짜 및 파일명 규칙

**중요**: 항상 현재 시스템 날짜 사용

```bash
# 현재 날짜 가져오기
CURRENT_DATE=$(date +%Y-%m-%d)

# 시리즈 slug 생성 (소문자, 하이픈)
SERIES_SLUG=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')

# 파일명 패턴
FILE_PATTERN="${CURRENT_DATE}-${SERIES_SLUG}-guide-{part:02d}-{slug}.md"

# 예시
# 2026-02-06-superset-guide-01-intro.md
# 2026-02-06-superset-guide-02-installation.md
```

### 4.2 Front Matter 템플릿

```yaml
---
layout: post
title: "{프로젝트명} 완벽 가이드 ({번호:02d}) - {챕터 제목}"
date: {현재날짜}
permalink: /{series-slug}-guide-{part:02d}-{slug}/
author: {원저자/팀}
categories: [{카테고리1}, {카테고리2}]
tags: [{태그1}, {태그2}, {태그3}, ...]
original_url: "{GitHub URL}"
excerpt: "{한 줄 요약 (50자 이내)}"
---
```

### 4.3 카테고리 및 태그 매핑

| 프로젝트 타입 | categories | 예시 tags |
|--------------|------------|-----------|
| CLI 도구 | [AI 코딩, CLI] | CLI, Terminal, Agent |
| 웹 앱 | [웹 개발, AI] | React, API, Backend |
| AI 에이전트 | [AI 에이전트] | LLM, Agent, Automation |
| 라이브러리 | [개발 도구] | Library, SDK, API |
| LLM 관련 | [LLM, 머신러닝] | LLM, Training, Inference |

### 4.4 본문 구조 템플릿

```markdown
---
{front matter}
---

## {주제1}

{설명}

```{언어}
// 코드 예제
```

---

## {주제2}

### {서브토픽}

{내용}

| 항목 | 설명 |
|------|------|
| 값1 | 설명1 |

---

## {주제3}

```
┌─────────────────────────────────┐
│         ASCII 다이어그램          │
└─────────────────────────────────┘
```

---

*다음 글에서는 {다음 챕터 주제}를 살펴봅니다.*
```

### 4.5 포스트 생성 스크립트 (예시)

```bash
# 블로그 디렉토리로 이동
cd /home/blog

# 현재 날짜
CURRENT_DATE=$(date +%Y-%m-%d)
SERIES="superset"

# 챕터 목록 (배열)
declare -a CHAPTERS=(
    "01:intro:소개 및 개요"
    "02:installation:설치 및 시작"
    "03:architecture:아키텍처 분석"
    # ... 더 많은 챕터
)

# 각 챕터 파일 생성
for chapter in "${CHAPTERS[@]}"; do
    IFS=':' read -r num slug title <<< "$chapter"

    FILE="_posts/${CURRENT_DATE}-${SERIES}-guide-${num}-${slug}.md"

    # Write 도구로 파일 생성 (실제로는 Claude에서 Write 도구 사용)
    echo "생성: $FILE"
done
```

---

## 📑 Phase 5: 인덱스 페이지 생성

### 5.1 파일 위치 및 이름

```
/home/blog/{series-slug}-guide.md
```

예: `/home/blog/superset-guide.md`

### 5.2 인덱스 페이지 템플릿

```markdown
---
layout: page
title: {프로젝트명} 가이드
permalink: /{series-slug}-guide/
icon: fas fa-{아이콘}
---

# {프로젝트명} 완벽 가이드

> **{한 줄 설명}**

**{프로젝트명}**은 {2-3문장 소개}.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/{series}-guide-01-intro/) | 프로젝트 소개, 주요 특징 |
| 02 | [설치](/blog-repo/{series}-guide-02-installation/) | 요구사항, 설치 방법 |
...

---

## 주요 특징

- **{특징1}** - {설명}
- **{특징2}** - {설명}
...

---

## 빠른 시작

```bash
# 설치
{설치 명령어}

# 실행
{실행 명령어}
```

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    {프로젝트명}                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   {ASCII 다이어그램}                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 기술 | 용도 |
|------|------|
| {기술1} | {용도} |
...

---

## 관련 링크

- [GitHub 저장소]({github-url})
- [공식 문서]({docs-url})
```

### 5.3 아이콘 선택 가이드

| 프로젝트 타입 | 아이콘 |
|--------------|--------|
| CLI/터미널 | `fa-terminal` |
| AI/에이전트 | `fa-robot` |
| 데이터베이스 | `fa-database` |
| 웹 애플리케이션 | `fa-globe` |
| 라이브러리/SDK | `fa-code` |
| 개발 도구 | `fa-tools` |
| 문서/가이드 | `fa-book` |

---

## 🔗 Phase 6: 가이드 목록 업데이트

### 6.1 _tabs/guides.md 업데이트

파일 위치: `/home/blog/_tabs/guides.md`

적절한 섹션에 새 가이드 카드 추가:

```html
<h2 class="section-title">{적절한 섹션 제목}</h2>
<div class="guide-grid">
  <!-- 기존 카드들 -->

  <!-- 새 가이드 추가 -->
  <div class="guide-card">
    <span class="badge badge-{타입}">{카테고리}</span>
    <h3><a href="/blog-repo/{series}-guide/">{프로젝트명}</a></h3>
    <p>{한 줄 설명}</p>
  </div>
</div>
```

#### 섹션 매핑

| 프로젝트 타입 | 섹션 | badge class |
|--------------|------|-------------|
| Claude Code, Cursor 등 | AI 코딩 에이전트 | `badge-coding` |
| Agent, Bot | AI 에이전트 | `badge-agent` |
| Tool, Framework | 개발 도구 | `badge-tool` |
| LLM, ML | LLM 학습 | `badge-llm` |

### 6.2 index.html 업데이트

파일 위치: `/home/blog/index.html`

동일한 방식으로 홈페이지에도 추가:

```html
<!-- 동일한 HTML 구조 -->
```

---

## 📤 Phase 7: Git Commit & Push

### 7.1 변경사항 확인

```bash
cd /home/blog

# 생성된 파일들 확인
git status

# 새로 추가된 파일들:
# - _posts/{date}-{series}-guide-*.md (10개)
# - {series}-guide.md (1개)
# - _tabs/guides.md (수정)
# - index.html (수정)
```

### 7.2 Commit 메시지 템플릿

```bash
git add _posts/{series}-guide-*.md {series}-guide.md _tabs/guides.md index.html

git commit -m "$(cat <<'EOF'
{프로젝트명} 가이드 추가 - {한 줄 설명}

{프로젝트명}({GitHub URL})을 분석하여 작성한 완벽 가이드 시리즈입니다.

## 생성 내용
- {챕터수}개 챕터로 구성된 완벽 가이드
- 인덱스 페이지: {series}-guide.md
- 가이드 목록 업데이트: guides.md, index.html

## 주요 챕터
- 01. 소개 및 개요
- 02. 설치 및 시작
- 03. 아키텍처 분석
...

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

### 7.3 Push

```bash
git push origin main
```

---

## 🧹 Phase 8: Cleanup (정리)

### 8.1 클론한 레포지토리 삭제

**중요**: 작업이 완료되면 Scratchpad에 클론한 레포지토리를 삭제하여 디스크 공간 확보

```bash
# 클론한 레포지토리 위치
SCRATCHPAD_DIR="/tmp/claude-0/-home-blog/*/scratchpad"
REPO_NAME="project-name"

# 레포지토리 삭제
cd "$SCRATCHPAD_DIR"
rm -rf "$REPO_NAME"

echo "✅ Cleanup 완료: $REPO_NAME 삭제됨"
```

### 8.2 전체 Cleanup 스크립트

```bash
# 작업 완료 후 자동 정리
cleanup_repo() {
    local repo_name=$1
    local scratchpad_dir="/tmp/claude-0/-home-blog/*/scratchpad"

    if [ -d "$scratchpad_dir/$repo_name" ]; then
        rm -rf "$scratchpad_dir/$repo_name"
        echo "✅ $repo_name 삭제 완료"
    else
        echo "⚠️  $repo_name 디렉토리를 찾을 수 없음"
    fi
}

# 사용 예시
cleanup_repo "superset"
```

### 8.3 최종 확인

```bash
# 1. Git push 성공 확인
git log --oneline -1

# 2. 생성된 파일들 확인
echo "생성된 포스트 파일:"
ls -1 _posts/{series}-guide-*.md

echo "인덱스 페이지:"
ls -l {series}-guide.md

# 3. Scratchpad 정리 확인
echo "Scratchpad 상태:"
ls -la /tmp/claude-0/-home-blog/*/scratchpad/

# 4. 최종 성공 메시지
echo "
✅ 가이드 생성 완료!
✅ Git push 완료
✅ Cleanup 완료
🎉 모든 작업이 성공적으로 완료되었습니다!
"
```

---

## 🎨 명명 규칙 요약

| 항목 | 형식 | 예시 |
|-----|------|------|
| 시리즈 slug | `{project}-guide` | `superset-guide` |
| 포스트 파일 | `{date}-{series}-{part}-{slug}.md` | `2026-02-06-superset-guide-01-intro.md` |
| permalink | `/{series}-{part}-{slug}/` | `/superset-guide-01-intro/` |
| 인덱스 파일 | `{series}-guide.md` | `superset-guide.md` |

**중요**: 날짜는 항상 `date +%Y-%m-%d` 명령으로 현재 시스템 날짜 사용

---

## ✅ 완료 체크리스트

생성 완료 후 확인:

```bash
# 1. 포스트 파일 개수 확인
ls -1 _posts/{series}-guide-*.md | wc -l
# 예상: 8-12개

# 2. 인덱스 페이지 존재 확인
ls -l {series}-guide.md

# 3. guides.md에 추가되었는지 확인
grep "{프로젝트명}" _tabs/guides.md

# 4. index.html에 추가되었는지 확인
grep "{프로젝트명}" index.html

# 5. Git 상태 확인
git status

# 6. Cleanup 확인 (클론한 레포지토리 삭제됨)
SCRATCHPAD="/tmp/claude-0/-home-blog/*/scratchpad"
ls -la "$SCRATCHPAD" | grep -v "{repo-name}" && echo "✅ Cleanup 완료"
```

---

## 🚀 실행 예시

### 입력

```
https://github.com/superset-sh/superset
```

### 실행 과정

```
✅ 현재 날짜: 2026-02-06
✅ 레포지토리 클론 완료: /tmp/.../scratchpad/superset
✅ 분석 완료: CLI 코딩 에이전트를 위한 터보차지 터미널
✅ 10개 챕터 구조 설계:
   01. 소개 및 개요
   02. 설치 및 시작
   03. 아키텍처 분석
   04. Electron 앱
   05. Workspace & Worktree
   06. 터미널 관리
   07. tRPC 라우터
   08. MCP 서버
   09. UI 컴포넌트
   10. 확장 및 커스터마이징

✅ 포스트 파일 10개 생성 완료:
   _posts/2026-02-06-superset-guide-01-intro.md
   _posts/2026-02-06-superset-guide-02-installation.md
   ...

✅ 인덱스 페이지 생성: superset-guide.md
✅ _tabs/guides.md 업데이트 완료
✅ index.html 업데이트 완료
✅ Git commit & push 완료: a3f72b9
✅ Cleanup 완료: superset 디렉토리 삭제됨

🎉 모든 작업이 성공적으로 완료되었습니다!
```

---

## 📚 참조 파일

- `references/chapter-structure.md` - 챕터 구성 가이드
- `references/post-template.md` - 포스트 템플릿
- `references/index-template.md` - 인덱스 템플릿
- `references/nav-template.md` - 네비게이션 템플릿

---

## 🔧 고급 기능

### 다중 레포지토리 처리

여러 관련 레포지토리가 있는 경우:

```bash
# 메인 레포
git clone https://github.com/user/main-project

# 관련 레포들
git clone https://github.com/user/main-project-cli
git clone https://github.com/user/main-project-web

# 모든 소스를 통합 분석
```

### 기존 문서 통합

외부 문서 사이트가 있는 경우:

```bash
# 공식 문서 크롤링 (WebFetch 사용)
# https://docs.project.com/guide/introduction
# https://docs.project.com/guide/getting-started
```

---

## 🐛 문제 해결

### 문제 1: 날짜가 잘못된 경우

```bash
# 올바른 날짜 확인
date +%Y-%m-%d

# 파일명이 과거 날짜로 생성되었다면 재생성 필요
```

### 문제 2: 카테고리가 애매한 경우

프로젝트가 여러 카테고리에 걸치면 주요 용도 기준으로 분류:
- AI 기능이 핵심이면 → AI 에이전트
- 개발 도구로 사용되면 → 개발 도구

### 문제 3: 챕터 수가 너무 많거나 적은 경우

- 너무 많음 (15개 이상): 유사한 챕터 병합
- 너무 적음 (5개 이하): 주요 모듈을 더 세분화

---

## 📖 한국어 번역 가이드라인

### 번역 원칙

1. **정확성**: 기술 용어는 원문 유지, 설명은 자연스러운 한국어로
2. **일관성**: 동일 용어는 항상 동일하게 번역
3. **가독성**: 너무 직역하지 말고 자연스럽게

### 용어 번역표

| English | 한국어 | 비고 |
|---------|--------|------|
| Agent | 에이전트 | 그대로 사용 |
| CLI | CLI | 약어 유지 |
| Backend | 백엔드 | |
| Frontend | 프론트엔드 | |
| Framework | 프레임워크 | |
| Library | 라이브러리 | |
| Module | 모듈 | |
| Component | 컴포넌트 | |
| Repository | 레포지토리 / 저장소 | 문맥에 따라 |
| Issue | 이슈 | |
| Pull Request | 풀 리퀘스트 / PR | |

### 문장 스타일

```markdown
❌ 나쁜 예: "이 프로젝트는 CLI 에이전트를 제공합니다."
✅ 좋은 예: "CLI 에이전트를 제공하는 프로젝트입니다."

❌ 나쁜 예: "당신은 설치할 수 있습니다."
✅ 좋은 예: "설치할 수 있습니다." 또는 "설치하세요."

❌ 나쁜 예: "이것은 매우 강력합니다."
✅ 좋은 예: "강력한 기능을 제공합니다."
```

---

## 🎯 사용 시나리오 예시

### 시나리오 1: 새 AI 도구 발견

```
사용자: https://github.com/anthropics/claude-cli 분석해서 가이드 만들어줘
→ 스킬 실행 → 10개 챕터 가이드 생성 → Git push
```

### 시나리오 2: 기존 문서 번역

```
사용자: https://docs.langchain.com/ 보고 한국어 가이드 시리즈 만들어줘
→ 문서 크롤링 → 챕터 구성 → 번역 → 가이드 생성
```

### 시나리오 3: 다중 레포 통합

```
사용자: superset-sh의 모든 레포 분석해서 통합 가이드 만들어줘
→ 여러 레포 클론 → 통합 분석 → 하나의 가이드 시리즈 생성
```

---

## 끝

이 스킬을 사용하면 GitHub 레포지토리 URL만 제공하면 자동으로:
1. 레포지토리 클론 및 분석 (Scratchpad 사용)
2. 한국어 가이드 시리즈 생성 (8-12개 챕터)
3. 블로그 인덱스 페이지 생성
4. 홈/가이드 목록 업데이트
5. Git commit & push
6. **Cleanup: 클론한 레포지토리 자동 삭제**

**모든 과정이 자동화되어 5-10분 내에 완성됩니다!** 🚀

### 디스크 공간 절약
Cleanup 단계에서 클론한 레포지토리를 자동으로 삭제하여 디스크 공간을 절약합니다.
