---
layout: post
title: "UI-TARS 완벽 가이드 (10) - 활용 가이드 및 결론"
date: 2025-02-04
permalink: /ui-tars-guide-10-conclusion/
author: ByteDance
categories: [AI 에이전트, UI-TARS]
tags: [UI-TARS, 활용 가이드, 확장, 커스터마이징, 결론]
original_url: "https://github.com/bytedance/UI-TARS-desktop"
excerpt: "UI-TARS의 실제 활용 방법과 확장 가이드를 소개합니다. 커스텀 Operator, MCP 서버 개발, 프롬프트 튜닝 방법을 알아봅니다."
---

## 실제 활용 예제

### Agent TARS CLI 사용

```bash
# 기본 실행 (Web UI와 함께)
npx @agent-tars/cli@latest

# 특정 모델로 실행
agent-tars --provider anthropic --model claude-3-5-sonnet-20241022

# 헤드리스 모드 (서버 환경)
agent-tars --headless --provider openai --model gpt-4o

# 설정 파일 사용
agent-tars --config ./agent-tars.config.js
```

### 설정 파일 예제

```javascript
// agent-tars.config.js
module.exports = {
  // LLM 설정
  provider: 'openai',
  model: 'gpt-4o',
  apiKey: process.env.OPENAI_API_KEY,

  // 브라우저 설정
  browser: {
    headless: false,
    control: 'hybrid',
    viewport: { width: 1920, height: 1080 }
  },

  // 작업 공간
  workspace: {
    dir: './workspace',
    allowedPaths: ['./workspace', './data'],
    blockedPaths: ['./secrets']
  },

  // MCP 서버
  mcpServers: [
    {
      name: 'filesystem',
      module: '@anthropic-ai/mcp-filesystem'
    },
    {
      name: 'browser',
      module: '@anthropic-ai/mcp-browser'
    },
    {
      name: 'custom',
      module: './my-mcp-server.js'
    }
  ],

  // 에이전트 동작
  maxIterations: 50,
  timeout: 600000,
  verbose: true
};
```

---

## 커스텀 Operator 개발

### 새로운 Operator 구현

```typescript
// my-custom-operator.ts

import { BaseOperator, OperatorOptions } from '@anthropic-ai/gui-agent-operators';

export interface MyCustomOperatorOptions extends OperatorOptions {
  customSetting?: string;
}

export class MyCustomOperator extends BaseOperator {
  private customSetting: string;

  constructor(options: MyCustomOperatorOptions = {}) {
    super(options);
    this.customSetting = options.customSetting || 'default';
  }

  async initialize(): Promise<void> {
    // 초기화 로직
    console.log('Custom operator initialized');

    // 화면 크기 설정
    this.screenSize = { width: 1920, height: 1080 };
  }

  async screenshot(): Promise<Buffer> {
    // 스크린샷 캡처 구현
    // 예: 외부 API 호출, 특수 하드웨어 연동 등

    const response = await fetch('http://my-screenshot-service/capture');
    return Buffer.from(await response.arrayBuffer());
  }

  async click(coordinate: [number, number]): Promise<void> {
    const [x, y] = this.toPixelCoordinate(coordinate);

    // 클릭 동작 구현
    await fetch('http://my-control-service/click', {
      method: 'POST',
      body: JSON.stringify({ x, y })
    });
  }

  async type(text: string): Promise<void> {
    // 텍스트 입력 구현
    await fetch('http://my-control-service/type', {
      method: 'POST',
      body: JSON.stringify({ text })
    });
  }

  async key(key: string): Promise<void> {
    // 키 입력 구현
    await fetch('http://my-control-service/key', {
      method: 'POST',
      body: JSON.stringify({ key })
    });
  }

  async cleanup(): Promise<void> {
    // 정리 작업
    console.log('Custom operator cleanup');
  }
}
```

### Operator 등록

```typescript
// GUI Agent에서 사용
import { GUIAgent } from '@anthropic-ai/gui-agent-sdk';
import { MyCustomOperator } from './my-custom-operator';

const agent = new GUIAgent({
  operator: new MyCustomOperator({
    customSetting: 'my-value'
  }),
  model: llmClient,
  maxLoopCount: 30
});

await agent.run('Open the application and click Start');
```

---

## 커스텀 MCP 서버 개발

### 간단한 MCP 서버

```typescript
// my-mcp-server.ts

import { BaseMCPServer } from '@anthropic-ai/mcp-shared';

export class MyMCPServer extends BaseMCPServer {
  constructor() {
    super('my-server', '1.0.0');
  }

  protected registerHandlers(): void {
    // 커스텀 도구 1: 데이터베이스 쿼리
    this.registerTool(
      'db_query',
      'Execute a database query',
      {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'SQL query' },
          database: { type: 'string', description: 'Database name' }
        },
        required: ['query', 'database']
      },
      async (args) => {
        // 실제 데이터베이스 연결 및 쿼리 실행
        const result = await this.executeQuery(args.database, args.query);
        return { success: true, data: result };
      }
    );

    // 커스텀 도구 2: 외부 API 호출
    this.registerTool(
      'api_call',
      'Call an external API',
      {
        type: 'object',
        properties: {
          url: { type: 'string' },
          method: { type: 'string', enum: ['GET', 'POST', 'PUT', 'DELETE'] },
          body: { type: 'object' }
        },
        required: ['url', 'method']
      },
      async (args) => {
        const response = await fetch(args.url, {
          method: args.method,
          body: args.body ? JSON.stringify(args.body) : undefined,
          headers: { 'Content-Type': 'application/json' }
        });

        return {
          success: response.ok,
          status: response.status,
          data: await response.json()
        };
      }
    );

    // 커스텀 도구 3: 파일 변환
    this.registerTool(
      'convert_file',
      'Convert file format',
      {
        type: 'object',
        properties: {
          inputPath: { type: 'string' },
          outputPath: { type: 'string' },
          format: { type: 'string', enum: ['pdf', 'docx', 'html'] }
        },
        required: ['inputPath', 'outputPath', 'format']
      },
      async (args) => {
        // 파일 변환 로직
        await this.convertFile(args.inputPath, args.outputPath, args.format);
        return { success: true, output: args.outputPath };
      }
    );
  }

  private async executeQuery(database: string, query: string): Promise<any> {
    // 데이터베이스 쿼리 실행 구현
    return [];
  }

  private async convertFile(
    input: string,
    output: string,
    format: string
  ): Promise<void> {
    // 파일 변환 구현
  }
}

// 서버 실행
const server = new MyMCPServer();
server.run();
```

---

## 프롬프트 튜닝

### 시스템 프롬프트 커스터마이징

```typescript
const customSystemPrompt = `
You are an AI assistant specialized in web automation tasks.

## Your Capabilities
- Navigate websites
- Fill forms and submit data
- Extract information from web pages
- Take screenshots and analyze them

## Guidelines
1. Always verify the current page before taking actions
2. Use precise coordinates based on visible UI elements
3. Wait for page loads to complete before proceeding
4. Report errors clearly and suggest alternatives

## Action Format
Thought: [Your reasoning]
Action: [action_name](args)

## Available Actions
- click(x, y): Click at normalized coordinates (0-1)
- type("text"): Type text
- scroll(x, y, direction): Scroll the page
- key("key"): Press a key
- finished("summary"): Task complete

## Error Handling
If an action fails:
1. Take a new screenshot
2. Analyze what went wrong
3. Try an alternative approach

Remember: Be precise, be patient, verify before acting.
`;

const agent = new GUIAgent({
  operator,
  model: llmClient,
  systemPrompt: customSystemPrompt
});
```

### 작업별 프롬프트 템플릿

```typescript
const promptTemplates = {
  webSearch: `
Search for "{{query}}" using the browser.
1. Go to the search engine
2. Enter the search query
3. Click search
4. Report the top 3 results
`,

  formFill: `
Fill out the form with the following data:
{{#each fields}}
- {{name}}: {{value}}
{{/each}}

After filling, click Submit and verify success.
`,

  dataExtraction: `
Extract the following information from the page:
{{fields}}

Return the data in JSON format.
`
};

// 템플릿 사용
const instruction = promptTemplates.formFill
  .replace('{{#each fields}}', fields.map(f => `- ${f.name}: ${f.value}`).join('\n'))
  .replace('{{/each}}', '');
```

---

## 에러 핸들링 및 복구

### 재시도 로직

```typescript
async function runWithRetry(
  agent: GUIAgent,
  instruction: string,
  maxRetries: number = 3
): Promise<GUIAgentResult> {
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`Attempt ${attempt}/${maxRetries}`);

      const result = await agent.run(instruction);

      if (result.success) {
        return result;
      }

      // 실패 시 복구 시도
      console.log('Task failed, attempting recovery...');
      await agent.operator.key('Escape'); // 모달 닫기
      await new Promise(r => setTimeout(r, 1000));

    } catch (error) {
      lastError = error as Error;
      console.error(`Attempt ${attempt} failed:`, error);

      // 복구 가능한 에러인지 확인
      if (!isRecoverable(error)) {
        throw error;
      }

      // 재시도 전 대기
      await new Promise(r => setTimeout(r, 2000 * attempt));
    }
  }

  throw lastError || new Error('Max retries exceeded');
}

function isRecoverable(error: any): boolean {
  const recoverableErrors = [
    'ETIMEDOUT',
    'ECONNRESET',
    'rate limit',
    'timeout'
  ];

  return recoverableErrors.some(e =>
    error.message?.toLowerCase().includes(e.toLowerCase()) ||
    error.code === e
  );
}
```

---

## 성능 최적화

### 스크린샷 최적화

```typescript
// 스크린샷 리사이즈로 토큰 절약
const agent = new GUIAgent({
  operator: new BrowserOperator({
    screenshotResize: true,
    resizeWidth: 1280,
    resizeHeight: 720
  }),
  model: llmClient
});

// 관심 영역만 캡처
async function captureRegion(
  operator: Operator,
  region: { x: number; y: number; width: number; height: number }
): Promise<Buffer> {
  const fullScreenshot = await operator.screenshot();

  const sharp = (await import('sharp')).default;
  return sharp(fullScreenshot)
    .extract({
      left: region.x,
      top: region.y,
      width: region.width,
      height: region.height
    })
    .toBuffer();
}
```

### 배치 처리

```typescript
// 여러 작업을 순차 실행
async function runBatch(
  agent: GUIAgent,
  tasks: string[]
): Promise<GUIAgentResult[]> {
  const results: GUIAgentResult[] = [];

  for (const task of tasks) {
    const result = await agent.run(task);
    results.push(result);

    // 작업 간 딜레이
    await new Promise(r => setTimeout(r, 1000));
  }

  return results;
}
```

---

## 보안 고려사항

### 입력 검증

```typescript
function sanitizeInstruction(instruction: string): string {
  // 위험한 패턴 제거
  const dangerousPatterns = [
    /rm\s+-rf/gi,
    /drop\s+table/gi,
    /delete\s+from/gi,
    /<script>/gi
  ];

  let sanitized = instruction;
  for (const pattern of dangerousPatterns) {
    sanitized = sanitized.replace(pattern, '[REMOVED]');
  }

  return sanitized;
}
```

### 권한 제한

```typescript
const secureConfig = {
  workspace: {
    // 작업 디렉토리 제한
    allowedPaths: ['./workspace'],
    blockedPaths: [
      '/etc',
      '/var',
      process.env.HOME + '/.ssh',
      process.env.HOME + '/.aws'
    ]
  },

  // 명령어 제한
  commands: {
    allowedCommands: ['ls', 'cat', 'echo', 'node', 'npm'],
    blockedCommands: ['rm', 'mv', 'chmod', 'chown', 'sudo']
  }
};
```

---

## 결론

### UI-TARS의 핵심 가치

1. **모듈식 아키텍처**: 각 컴포넌트가 독립적으로 교체 가능
2. **멀티 플랫폼 지원**: 브라우저, 데스크톱, 모바일 모두 지원
3. **MCP 기반 확장성**: 표준 프로토콜로 도구 통합 용이
4. **이벤트 스트림**: 실시간 모니터링과 디버깅 지원

### 주요 컴포넌트 요약

| 컴포넌트 | 역할 |
|---------|------|
| **Agent TARS** | 범용 AI 에이전트 (CLI/Web) |
| **UI-TARS Desktop** | Electron 기반 GUI 에이전트 |
| **GUI Agent SDK** | Vision 기반 UI 자동화 |
| **Tarko** | 메타 에이전트 프레임워크 |
| **Operators** | 플랫폼별 실행 엔진 |
| **MCP Servers** | 도구 서버 구현 |

### 향후 발전 방향

- 더 많은 Operator 지원 (iOS, 게임 등)
- 멀티 에이전트 협업
- 향상된 Visual Grounding 모델
- 강화학습 기반 자가 개선

---

## 참고 자료

- [UI-TARS GitHub Repository](https://github.com/bytedance/UI-TARS-desktop)
- [MCP Specification](https://modelcontextprotocol.io)
- [Anthropic API Documentation](https://docs.anthropic.com)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

---

*이것으로 UI-TARS 완벽 가이드를 마칩니다. 질문이나 피드백은 GitHub Issues를 통해 남겨주세요!*
