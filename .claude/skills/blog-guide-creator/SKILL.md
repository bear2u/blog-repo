---
name: blog-guide-creator
description: |
  GitHub 레포지토리나 블로그 URL을 분석하여 챕터별 가이드 시리즈를 생성하는 스킬.

  사용 시점:
  - 사용자가 GitHub 레포지토리 URL을 주고 "가이드 만들어줘", "분석해서 블로그 작성해줘" 요청 시
  - "소스 분석해서 챕터별로 작성해줘" 요청 시
  - 외부 블로그/문서 URL을 주고 "번역해서 시리즈로 만들어줘" 요청 시
  - 기존 블로그에 새로운 가이드 시리즈 추가 요청 시

  생성 결과물:
  - 메인 네비게이션 드롭다운 메뉴
  - 가이드 인덱스 페이지 (전체 목차)
  - 챕터별 개별 포스트 파일들
  - 시리즈 이전/다음 네비게이션
---

# Blog Guide Creator

GitHub 레포지토리나 외부 콘텐츠를 분석하여 Jekyll 블로그용 가이드 시리즈를 생성한다.

## 워크플로우

### 1단계: 소스 분석
- GitHub 레포지토리인 경우: 클론 후 구조 분석
- 블로그/문서인 경우: WebFetch로 내용 수집
- 핵심 모듈, 아키텍처, 주요 기능 파악

### 2단계: 챕터 구조 설계
references/chapter-structure.md 참조하여:
- 8-12개 챕터로 구성
- 논리적 순서: 개요 → 아키텍처 → 핵심 모듈 → 활용 → 결론
- 각 챕터는 독립적으로 이해 가능하게

### 3단계: 포스트 파일 생성
references/post-template.md 형식으로 각 챕터 작성:
- `_posts/YYYY-MM-DD-{series}-{part}-{slug}.md`
- front matter에 `series`, `part`, `permalink` 필수
- 코드 예제와 다이어그램 포함

### 4단계: 인덱스 페이지 생성
references/index-template.md 형식으로:
- `{series-name}.md` 루트에 생성
- Part별 목차 구성
- 빠른 참조 섹션

### 5단계: 네비게이션 메뉴 추가
`_includes/nav.html`에 드롭다운 메뉴 추가:
- 전체 목차 링크
- 각 챕터 링크

### 6단계: 시리즈 네비게이션 확인
`_layouts/post.html`에서 새 시리즈 지원 확인

## 명명 규칙

| 항목 | 형식 | 예시 |
|-----|------|-----|
| 시리즈 slug | `{project}-guide` | `ui-tars-guide` |
| 포스트 파일 | `YYYY-MM-DD-{series}-{part}-{slug}.md` | `2025-02-04-ui-tars-guide-01-intro.md` |
| permalink | `/{series}-{part}-{slug}/` | `/ui-tars-guide-01-intro/` |
| 인덱스 파일 | `{series}.md` | `ui-tars-guide.md` |

## 체크리스트

생성 완료 후 확인:
- [ ] 모든 포스트에 `series`, `part`, `permalink` 있음
- [ ] 인덱스 페이지 링크 모두 작동
- [ ] nav.html에 드롭다운 추가됨
- [ ] post.html에 시리즈 목차 링크 추가됨
- [ ] 홈 index.html의 시리즈 가이드 섹션 업데이트
- [ ] git commit & push 완료
