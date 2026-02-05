---
layout: post
title: "Superset 완벽 가이드 (5) - Workspace & Worktree"
date: 2025-02-05
permalink: /superset-guide-05-workspace/
author: Superset Team
categories: [AI 에이전트, Superset]
tags: [Superset, Workspace, Git Worktree, Isolation, Branch]
original_url: "https://github.com/superset-sh/superset"
excerpt: "Superset의 Git Worktree 기반 워크스페이스 격리 시스템을 분석합니다."
---

## 워크스페이스 격리의 필요성

코딩 에이전트를 병렬로 실행할 때 가장 큰 문제는 **작업 충돌**입니다.

```
┌─────────────────────────────────────────────────────────────┐
│              병렬 실행 시 발생할 수 있는 문제                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Agent A: feature-a 작업 중                                  │
│      │                                                       │
│      └─→ src/app.tsx 수정                                   │
│                 ↑                                            │
│  Agent B: feature-b 작업 중                                  │
│      │                                                       │
│      └─→ src/app.tsx 수정  ← 충돌!                          │
│                                                              │
│  결과: 두 에이전트가 같은 파일을 동시에 수정하여 충돌        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Superset의 해결책: Git Worktree

```
┌─────────────────────────────────────────────────────────────┐
│              Git Worktree 기반 격리                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Main Repo: /projects/my-app/                               │
│      │                                                       │
│      ├─→ Worktree A: /projects/my-app-wt/feature-a/         │
│      │       └─→ Agent A 작업                               │
│      │       └─→ 독립적인 src/app.tsx                       │
│      │                                                       │
│      └─→ Worktree B: /projects/my-app-wt/feature-b/         │
│              └─→ Agent B 작업                               │
│              └─→ 독립적인 src/app.tsx                       │
│                                                              │
│  결과: 각 에이전트가 독립된 파일 시스템에서 작업             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Git Worktree 이해하기

### 기본 개념

Git Worktree는 하나의 레포지토리에서 여러 작업 디렉토리를 생성하는 기능입니다.

```bash
# 기존 방식: 레포 복제 (느림, 공간 낭비)
git clone repo.git feature-a/
git clone repo.git feature-b/

# Worktree 방식: 작업 디렉토리만 추가 (빠름, 공간 효율)
git worktree add ../feature-a -b feature-a
git worktree add ../feature-b -b feature-b
```

### 장점

| 특성 | 설명 |
|------|------|
| **빠른 생성** | .git 객체 공유로 즉시 생성 |
| **공간 효율** | 전체 복제 없이 작업 디렉토리만 추가 |
| **브랜치 연동** | 각 worktree가 다른 브랜치에서 작업 |
| **충돌 방지** | 같은 브랜치는 하나의 worktree에서만 체크아웃 |

---

## Superset의 워크스페이스 구조

```
/projects/
└── my-app/                      # 메인 레포지토리
    ├── .git/                    # Git 데이터 (공유됨)
    ├── .superset/               # Superset 설정
    │   ├── config.json          # 설정/정리 스크립트
    │   ├── setup.sh             # 설정 스크립트
    │   └── teardown.sh          # 정리 스크립트
    └── src/                     # 소스 코드

/projects/my-app-wt/             # Worktree 루트
├── feature-login/               # Workspace 1
│   ├── .git                     # worktree 링크 (파일)
│   └── src/                     # 독립된 소스 코드
│
├── feature-payment/             # Workspace 2
│   ├── .git
│   └── src/
│
└── bugfix-header/               # Workspace 3
    ├── .git
    └── src/
```

---

## 워크스페이스 생성 흐름

```typescript
// lib/trpc/routers/workspaces/procedures/create.ts

export const createWorkspace = async ({
  projectId,
  name,
  branchName,
  baseBranch = "main",
}: CreateInput) => {
  // 1. 워크스페이스 이름 생성
  const workspaceName = name || generateWorkspaceName();

  // 2. 경로 결정
  const worktreePath = path.join(
    projectPath,
    "..",
    `${projectName}-wt`,
    workspaceName
  );

  // 3. Git worktree 생성
  await createGitWorktree({
    repoPath: projectPath,
    worktreePath,
    branchName: branchName || workspaceName,
    baseBranch,
  });

  // 4. 설정 스크립트 실행
  await runSetupScripts(worktreePath);

  // 5. DB에 워크스페이스 저장
  const workspace = await db.insert(workspaces).values({
    id: uuid(),
    projectId,
    name: workspaceName,
    path: worktreePath,
    branch: branchName,
  });

  return workspace;
};
```

### Git Worktree 생성

```typescript
// lib/trpc/routers/workspaces/utils/worktree.ts

export async function createGitWorktree({
  repoPath,
  worktreePath,
  branchName,
  baseBranch,
}: WorktreeInput) {
  // 브랜치가 이미 존재하는지 확인
  const branchExists = await checkBranchExists(repoPath, branchName);

  if (branchExists) {
    // 기존 브랜치로 worktree 생성
    await execAsync(
      `git worktree add "${worktreePath}" "${branchName}"`,
      { cwd: repoPath }
    );
  } else {
    // 새 브랜치 생성과 함께 worktree 생성
    await execAsync(
      `git worktree add -b "${branchName}" "${worktreePath}" "${baseBranch}"`,
      { cwd: repoPath }
    );
  }
}
```

---

## 설정/정리 스크립트

### 설정 파일 (`.superset/config.json`)

```json
{
  "setup": ["./.superset/setup.sh"],
  "teardown": ["./.superset/teardown.sh"]
}
```

### 설정 스크립트 예시 (`.superset/setup.sh`)

```bash
#!/bin/bash
# .superset/setup.sh

# 환경 변수 사용 가능
# - SUPERSET_WORKSPACE_NAME: 워크스페이스 이름
# - SUPERSET_ROOT_PATH: 메인 레포지토리 경로

echo "Setting up workspace: $SUPERSET_WORKSPACE_NAME"

# 환경 파일 복사
cp "$SUPERSET_ROOT_PATH/.env" .env

# 의존성 설치
bun install

# 데이터베이스 마이그레이션
bun run db:push

echo "Workspace ready!"
```

### 정리 스크립트 예시 (`.superset/teardown.sh`)

```bash
#!/bin/bash
# .superset/teardown.sh

echo "Cleaning up workspace: $SUPERSET_WORKSPACE_NAME"

# 임시 파일 정리
rm -rf node_modules
rm -rf .next
rm -rf dist

echo "Cleanup complete!"
```

### 스크립트 실행 로직

```typescript
// lib/trpc/routers/workspaces/utils/setup.ts

export async function runSetupScripts(workspacePath: string) {
  const configPath = path.join(workspacePath, ".superset", "config.json");

  if (!fs.existsSync(configPath)) {
    console.log("[workspace/setup] No config.json found, skipping setup");
    return;
  }

  const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));

  for (const script of config.setup || []) {
    const scriptPath = path.join(workspacePath, script);

    await execAsync(`bash "${scriptPath}"`, {
      cwd: workspacePath,
      env: {
        ...process.env,
        SUPERSET_WORKSPACE_NAME: path.basename(workspacePath),
        SUPERSET_ROOT_PATH: path.resolve(workspacePath, "../.."),
      },
    });
  }
}
```

---

## 워크스페이스 삭제

```typescript
// lib/trpc/routers/workspaces/procedures/delete.ts

export const deleteWorkspace = async ({ workspaceId }: DeleteInput) => {
  const workspace = await db.query.workspaces.findFirst({
    where: eq(workspaces.id, workspaceId),
  });

  // 1. 정리 스크립트 실행
  await runTeardownScripts(workspace.path);

  // 2. Git worktree 제거
  await removeGitWorktree(workspace.path);

  // 3. 디렉토리 삭제
  await fs.rm(workspace.path, { recursive: true, force: true });

  // 4. DB에서 삭제
  await db.delete(workspaces).where(eq(workspaces.id, workspaceId));
};
```

### Worktree 제거

```typescript
// lib/trpc/routers/workspaces/utils/worktree.ts

export async function removeGitWorktree(worktreePath: string) {
  // worktree 링크 파일에서 메인 레포 경로 추출
  const gitPath = path.join(worktreePath, ".git");
  const gitContent = fs.readFileSync(gitPath, "utf-8");
  const mainGitDir = gitContent.match(/gitdir: (.+)/)?.[1];

  if (mainGitDir) {
    const repoPath = path.resolve(mainGitDir, "../..");

    // worktree 제거
    await execAsync(`git worktree remove "${worktreePath}" --force`, {
      cwd: repoPath,
    });

    // prune으로 정리
    await execAsync(`git worktree prune`, { cwd: repoPath });
  }
}
```

---

## 워크스페이스 상태 관리

### 상태 조회

```typescript
// lib/trpc/routers/workspaces/procedures/status.ts

export const getWorkspaceStatus = async ({ workspaceId }: StatusInput) => {
  const workspace = await db.query.workspaces.findFirst({
    where: eq(workspaces.id, workspaceId),
  });

  // Git 상태 조회
  const gitStatus = await getGitStatus(workspace.path);

  return {
    id: workspace.id,
    name: workspace.name,
    branch: workspace.branch,
    path: workspace.path,

    // Git 상태
    uncommittedChanges: gitStatus.modified.length + gitStatus.untracked.length,
    aheadCount: gitStatus.ahead,
    behindCount: gitStatus.behind,

    // 변경된 파일 목록
    modifiedFiles: gitStatus.modified,
    untrackedFiles: gitStatus.untracked,
    stagedFiles: gitStatus.staged,
  };
};
```

### Git 상태 파싱

```typescript
// lib/trpc/routers/workspaces/utils/git.ts

export async function getGitStatus(repoPath: string) {
  const { stdout } = await execAsync(
    "git status --porcelain=v2 --branch",
    { cwd: repoPath }
  );

  const lines = stdout.split("\n");
  const result = {
    branch: "",
    ahead: 0,
    behind: 0,
    modified: [] as string[],
    untracked: [] as string[],
    staged: [] as string[],
  };

  for (const line of lines) {
    if (line.startsWith("# branch.head")) {
      result.branch = line.split(" ")[2];
    } else if (line.startsWith("# branch.ab")) {
      const match = line.match(/\+(\d+) -(\d+)/);
      if (match) {
        result.ahead = parseInt(match[1]);
        result.behind = parseInt(match[2]);
      }
    } else if (line.startsWith("1 ")) {
      // 변경된 파일
      const parts = line.split("\t");
      const status = parts[0].split(" ")[1];
      const file = parts[1];

      if (status[0] !== ".") result.staged.push(file);
      if (status[1] !== ".") result.modified.push(file);
    } else if (line.startsWith("? ")) {
      // 추적되지 않는 파일
      result.untracked.push(line.slice(2));
    }
  }

  return result;
}
```

---

## 브랜치 전환

```typescript
// lib/trpc/routers/workspaces/procedures/branch.ts

export const switchBranch = async ({
  workspaceId,
  branchName,
  createNew,
}: SwitchInput) => {
  const workspace = await db.query.workspaces.findFirst({
    where: eq(workspaces.id, workspaceId),
  });

  if (createNew) {
    // 새 브랜치 생성 후 체크아웃
    await execAsync(
      `git checkout -b "${branchName}"`,
      { cwd: workspace.path }
    );
  } else {
    // 기존 브랜치로 체크아웃
    await execAsync(
      `git checkout "${branchName}"`,
      { cwd: workspace.path }
    );
  }

  // DB 업데이트
  await db.update(workspaces)
    .set({ branch: branchName })
    .where(eq(workspaces.id, workspaceId));
};
```

---

## 워크스페이스 UI (Renderer)

### 사이드바 상태

```typescript
// renderer/stores/workspace-sidebar-state.ts

interface WorkspaceSidebarState {
  // 선택된 워크스페이스
  activeWorkspaceId: string | null;

  // 워크스페이스 목록
  workspaces: Workspace[];

  // 액션
  setActiveWorkspace: (id: string) => void;
  refreshWorkspaces: () => Promise<void>;
}

export const useWorkspaceSidebar = create<WorkspaceSidebarState>((set) => ({
  activeWorkspaceId: null,
  workspaces: [],

  setActiveWorkspace: (id) => set({ activeWorkspaceId: id }),

  refreshWorkspaces: async () => {
    const workspaces = await trpc.workspaces.list.query();
    set({ workspaces });
  },
}));
```

### 새 워크스페이스 모달

```typescript
// renderer/stores/new-workspace-modal.ts

interface NewWorkspaceModalState {
  isOpen: boolean;
  projectId: string | null;

  open: (projectId: string) => void;
  close: () => void;
}

export const useNewWorkspaceModal = create<NewWorkspaceModalState>((set) => ({
  isOpen: false,
  projectId: null,

  open: (projectId) => set({ isOpen: true, projectId }),
  close: () => set({ isOpen: false, projectId: null }),
}));
```

---

## 베스트 프랙티스

### 1. 워크스페이스 이름 규칙

```
feature-{feature-name}    # 새 기능
bugfix-{issue-number}     # 버그 수정
refactor-{module-name}    # 리팩토링
experiment-{description}  # 실험
```

### 2. 설정 스크립트 최적화

```bash
#!/bin/bash
# 빠른 설정을 위한 팁

# 1. 캐시 활용
if [ -d "../.cache/node_modules" ]; then
  cp -r "../.cache/node_modules" ./node_modules
fi

# 2. 병렬 설치
bun install &
cp "$SUPERSET_ROOT_PATH/.env" .env &
wait

# 3. 불필요한 작업 건너뛰기
if [ -f ".setup-complete" ]; then
  echo "Setup already complete, skipping"
  exit 0
fi
```

### 3. 정리 자동화

```bash
#!/bin/bash
# 디스크 공간 관리

# 오래된 워크스페이스 정리 (30일 이상)
find /projects/*-wt -maxdepth 1 -type d -mtime +30 -exec rm -rf {} \;

# Git 가비지 컬렉션
git gc --prune=now
```

---

*다음 글에서는 터미널 관리와 node-pty 통합을 분석합니다.*
