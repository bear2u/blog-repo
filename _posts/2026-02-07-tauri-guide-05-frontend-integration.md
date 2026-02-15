---
layout: post
title: "Tauri 완벽 가이드 (05) - 프론트엔드 통합"
date: 2026-02-07
permalink: /tauri-guide-05-frontend-integration/
author: Tauri Programme
categories: [웹 개발, 데스크톱]
tags: [Tauri, Frontend, TypeScript, React, Vue, "@tauri-apps/api"]
original_url: "https://github.com/tauri-apps/tauri"
excerpt: "@tauri-apps/api를 사용한 프론트엔드-백엔드 통신"
---

## @tauri-apps/api 소개

`@tauri-apps/api`는 TypeScript로 작성된 프론트엔드 JavaScript 라이브러리로, Tauri의 Rust 백엔드와 통신할 수 있게 해줍니다.

---

## 설치

```bash
npm install @tauri-apps/api
# 또는
pnpm add @tauri-apps/api
# 또는
yarn add @tauri-apps/api
```

---

## 핵심 모듈

### 1. tauri - 명령어 호출

```typescript
import { invoke } from '@tauri-apps/api/tauri';

// 간단한 호출
const result = await invoke('my_command');

// 매개변수 전달
const result = await invoke('greet', {
    name: 'Alice',
    age: 30
});

// 타입 안전성
interface User {
    id: number;
    name: string;
}

const user = await invoke<User>('get_user', { id: 1 });
```

### 2. event - 이벤트 시스템

```typescript
import { listen, emit } from '@tauri-apps/api/event';

// 이벤트 리스닝
const unlisten = await listen('my-event', (event) => {
    console.log('Received:', event.payload);
});

// 이벤트 발생
await emit('my-event', { message: 'Hello' });

// 리스너 제거
unlisten();
```

### 3. window - 윈도우 관리

```typescript
import { appWindow } from '@tauri-apps/api/window';

// 윈도우 제목 변경
await appWindow.setTitle('New Title');

// 윈도우 크기 조정
await appWindow.setSize(new LogicalSize(800, 600));

// 윈도우 최소화/최대화
await appWindow.minimize();
await appWindow.maximize();
await appWindow.toggleMaximize();

// 윈도우 닫기
await appWindow.close();

// 전체 화면
await appWindow.setFullscreen(true);
```

### 4. dialog - 대화상자

```typescript
import { open, save, message, ask, confirm } from '@tauri-apps/api/dialog';

// 파일 열기
const selected = await open({
    multiple: false,
    filters: [{
        name: 'Image',
        extensions: ['png', 'jpeg', 'jpg']
    }]
});

// 파일 저장
const path = await save({
    defaultPath: 'document.txt',
    filters: [{
        name: 'Text',
        extensions: ['txt']
    }]
});

// 메시지 박스
await message('Operation completed!', 'Success');

// 확인 대화상자
const yes = await ask('Are you sure?', 'Confirm');

// Yes/No 대화상자
const confirmed = await confirm('Delete file?', 'Warning');
```

### 5. fs - 파일 시스템

```typescript
import {
    readTextFile,
    writeTextFile,
    readBinaryFile,
    writeBinaryFile,
    createDir,
    removeFile,
    renameFile,
    copyFile,
    exists,
    BaseDirectory
} from '@tauri-apps/api/fs';

// 텍스트 파일 읽기
const content = await readTextFile('myfile.txt', {
    dir: BaseDirectory.App
});

// 텍스트 파일 쓰기
await writeTextFile('myfile.txt', 'Hello World', {
    dir: BaseDirectory.App
});

// 바이너리 파일
const bytes = await readBinaryFile('image.png', {
    dir: BaseDirectory.Resource
});

// 디렉토리 생성
await createDir('my-folder', {
    dir: BaseDirectory.App,
    recursive: true
});

// 파일 존재 확인
const fileExists = await exists('myfile.txt', {
    dir: BaseDirectory.App
});
```

### 6. path - 경로 관리

```typescript
import {
    appDataDir,
    appLogDir,
    appConfigDir,
    audioDir,
    cacheDir,
    dataDir,
    desktopDir,
    documentDir,
    downloadDir,
    homeDir,
    pictureDir,
    publicDir,
    runtimeDir,
    tempDir,
    videoDir
} from '@tauri-apps/api/path';

const appData = await appDataDir();
const downloads = await downloadDir();
const documents = await documentDir();
```

### 7. http - HTTP 클라이언트

```typescript
import { fetch, Body, ResponseType } from '@tauri-apps/api/http';

// GET 요청
const response = await fetch('https://api.example.com/data', {
    method: 'GET',
    headers: {
        'Content-Type': 'application/json'
    }
});

// POST 요청
const response = await fetch('https://api.example.com/users', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: Body.json({
        name: 'Alice',
        email: 'alice@example.com'
    })
});

// 파일 업로드
const response = await fetch('https://api.example.com/upload', {
    method: 'POST',
    body: Body.form({
        file: {
            file: await readBinaryFile('image.png'),
            mime: 'image/png',
            fileName: 'image.png'
        }
    })
});
```

### 8. notification - 알림

```typescript
import { sendNotification, isPermissionGranted, requestPermission } from '@tauri-apps/api/notification';

// 권한 확인
let permissionGranted = await isPermissionGranted();

if (!permissionGranted) {
    const permission = await requestPermission();
    permissionGranted = permission === 'granted';
}

// 알림 전송
if (permissionGranted) {
    sendNotification({
        title: 'Tauri App',
        body: 'Operation completed successfully!',
        icon: 'icon.png'
    });
}
```

### 9. clipboard - 클립보드

```typescript
import { writeText, readText } from '@tauri-apps/api/clipboard';

// 클립보드에 쓰기
await writeText('Hello Clipboard!');

// 클립보드 읽기
const text = await readText();
console.log(text); // "Hello Clipboard!"
```

### 10. globalShortcut - 글로벌 단축키

```typescript
import { register, unregister } from '@tauri-apps/api/globalShortcut';

// 단축키 등록
await register('CommandOrControl+Shift+C', () => {
    console.log('Shortcut triggered!');
});

// 단축키 해제
await unregister('CommandOrControl+Shift+C');
```

---

## React 통합 예제

```typescript
import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';

interface User {
    id: number;
    name: string;
}

function App() {
    const [users, setUsers] = useState<User[]>([]);
    const [loading, setLoading] = useState(false);

    // 사용자 목록 가져오기
    const fetchUsers = async () => {
        setLoading(true);
        try {
            const data = await invoke<User[]>('get_users');
            setUsers(data);
        } catch (error) {
            console.error('Failed to fetch users:', error);
        } finally {
            setLoading(false);
        }
    };

    // 이벤트 리스닝
    useEffect(() => {
        const unlisten = listen<User>('user-added', (event) => {
            setUsers(prev => [...prev, event.payload]);
        });

        return () => {
            unlisten.then(fn => fn());
        };
    }, []);

    return (
        <div>
            <h1>Users</h1>
            <button onClick={fetchUsers} disabled={loading}>
                {loading ? 'Loading...' : 'Refresh'}
            </button>
            <ul>
                {users.map(user => (
                    <li key={user.id}>{user.name}</li>
                ))}
            </ul>
        </div>
    );
}
```

---

## Vue 통합 예제

```vue
<template>
  <div>
    <h1>Users</h1>
    <button @click="fetchUsers" :disabled="loading">
      {{ loading ? 'Loading...' : 'Refresh' }}
    </button>
    <ul>
      <li v-for="user in users" :key="user.id">
        {{ user.name }}
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue';
import { invoke } from '@tauri-apps/api/tauri';
import { listen, UnlistenFn } from '@tauri-apps/api/event';

interface User {
    id: number;
    name: string;
}

const users = ref<User[]>([]);
const loading = ref(false);
let unlisten: UnlistenFn | null = null;

const fetchUsers = async () => {
    loading.value = true;
    try {
        users.value = await invoke<User[]>('get_users');
    } catch (error) {
        console.error('Failed to fetch users:', error);
    } finally {
        loading.value = false;
    }
};

onMounted(async () => {
    unlisten = await listen<User>('user-added', (event) => {
        users.value.push(event.payload);
    });
});

onUnmounted(() => {
    if (unlisten) unlisten();
});
</script>
```

---

## 커스텀 훅 (React)

```typescript
// hooks/useTauriCommand.ts
import { useState, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/tauri';

export function useTauriCommand<T, P = Record<string, unknown>>(
    command: string
) {
    const [data, setData] = useState<T | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const execute = useCallback(async (params?: P) => {
        setLoading(true);
        setError(null);

        try {
            const result = await invoke<T>(command, params);
            setData(result);
            return result;
        } catch (err) {
            const errorMessage = err as string;
            setError(errorMessage);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [command]);

    return { data, loading, error, execute };
}

// 사용
function MyComponent() {
    const { data, loading, error, execute } = useTauriCommand<User[]>('get_users');

    return (
        <button onClick={() => execute()}>
            Load Users
        </button>
    );
}
```

---

## 타입 안전성

### 공유 타입 정의

```typescript
// types/api.ts
export interface User {
    id: number;
    name: string;
    email: string;
}

export interface CreateUserRequest {
    name: string;
    email: string;
}

export interface UpdateUserRequest {
    id: number;
    name?: string;
    email?: string;
}
```

### Rust와 TypeScript 타입 동기화

```rust
// src-tauri/src/types.rs
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct User {
    pub id: i32,
    pub name: String,
    pub email: String,
}
```

```typescript
// 자동 생성된 타입 (typescript-generator 등 사용)
export interface User {
    id: number;
    name: string;
    email: string;
}
```

---

## 에러 처리 패턴

```typescript
async function safeInvoke<T>(
    command: string,
    params?: Record<string, unknown>
): Promise<T | null> {
    try {
        return await invoke<T>(command, params);
    } catch (error) {
        console.error(`Command ${command} failed:`, error);
        return null;
    }
}

// 사용
const user = await safeInvoke<User>('get_user', { id: 1 });
if (user) {
    console.log(user.name);
}
```

---

*다음 글에서는 Tauri 플러그인 시스템을 살펴봅니다.*
