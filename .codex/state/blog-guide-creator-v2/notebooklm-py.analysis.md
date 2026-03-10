마지막 업데이트: 2026-03-10

# notebooklm-py 분석 메모

## 시리즈 목표

- 기존 5챕터 요약형 시리즈를 v2 위키형 8노드 구조로 재작성
- 설치/인증, 아키텍처, 사용 흐름, 운영, 문서 점검 자동화를 분리
- 각 포스트에 최소 목적, 빠른 요약, 근거, 주의사항, TODO, 위키 링크 포함

## 핵심 근거 파일

- `README.md`
- `pyproject.toml`
- `docs/cli-reference.md`
- `docs/configuration.md`
- `docs/development.md`
- `docs/troubleshooting.md`
- `src/notebooklm/client.py`
- `src/notebooklm/_core.py`
- `src/notebooklm/auth.py`
- `src/notebooklm/paths.py`
- `src/notebooklm/notebooklm_cli.py`
- `src/notebooklm/_sources.py`
- `src/notebooklm/_research.py`
- `src/notebooklm/_chat.py`
- `src/notebooklm/_artifacts.py`
- `src/notebooklm/data/SKILL.md`
- `.github/workflows/*.yml`
- `scripts/check_rpc_health.py`
- `scripts/diagnose_get_notebook.py`

## 확정한 노드 구조

1. 소개 및 범위
2. 설치와 인증
3. 아키텍처와 호출 계층
4. CLI와 Python API
5. 소스, 리서치, 채팅
6. 생성물과 다운로드
7. 테스트, CI, 보안, 운영
8. 문서 점검 자동화와 트러블슈팅

## 수정 대상

- 기존 2026-03-09 `notebooklm-py-guide-*` 5개 포스트 삭제
- `/home/blog/notebooklm-py-guide.md` 허브 페이지 교체
- `_tabs/guides.md` 카드 설명과 챕터 수 업데이트 필요

