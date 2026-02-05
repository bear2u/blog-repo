---
layout: post
title: "UI-TARS 완벽 가이드 (4) - Agent TARS Core"
date: 2025-02-04
permalink: /ui-tars-guide-04-agent-tars/
author: ByteDance
categories: [AI]
tags: [UI-TARS, Agent TARS, CLI, MCP, Event Stream]
original_url: "https://github.com/bytedance/UI-TARS-desktop"
excerpt: "Agent TARS의 CLI와 핵심 에이전트 로직을 분석합니다. 이벤트 스트림 기반 아키텍처와 MCP 통합을 살펴봅니다."
---

## Agent TARS 개요

**Agent TARS**는 범용 멀티모달 AI 에이전트로, CLI와 Web UI를 통해 브라우저, 터미널, 파일 시스템을 제어합니다.

```
multimodal/agent-tars/
├── cli/                 # CLI 구현
│   └── src/
│       ├── index.ts     # 진입점
│       ├── cli.ts       # 명령줄 파서
│       └── utils/       # 유틸리티
├── core/                # 핵심 에이전트 로직
│   └── src/
│       ├── AgentTARS.ts # 메인 클래스
│       ├── config/      # 설정 관리
│       └── tools/       # 내장 도구
└── interface/           # 타입 정의
    └── src/
        ├── agent.ts     # 에이전트 인터페이스
        ├── config.ts    # 설정 타입
        └── events.ts    # 이벤트 타입
```

---

## CLI 구현 분석

### 진입점 (`cli/src/index.ts`)

```typescript
#!/usr/bin/env node

import { program } from 'commander';
import { AgentTARS } from '@anthropic-ai/agent-tars-core';
import { loadConfig } from './config';

program
  .name('agent-tars')
  .description('Universal Multimodal AI Agent')
  .version('1.0.0');

program
  .option('-c, --config <path>', 'Config file path')
  .option('--provider <name>', 'LLM provider (openai, anthropic, etc.)')
  .option('--model <name>', 'Model name')
  .option('--apiKey <key>', 'API key')
  .option('--headless', 'Run in headless mode')
  .option('--port <number>', 'Web UI port', '3000')
  .action(async (options) => {
    const config = await loadConfig(options);
    const agent = new AgentTARS(config);

    if (options.headless) {
      await agent.runHeadless();
    } else {
      await agent.runWithUI();
    }
  });

program.parse();
```

### 설정 로더 (`cli/src/config.ts`)

```typescript
import { cosmiconfig } from 'cosmiconfig';
import { AgentTARSConfig } from '@anthropic-ai/agent-tars-interface';

const explorer = cosmiconfig('agent-tars', {
  searchPlaces: [
    'agent-tars.config.js',
    'agent-tars.config.ts',
    '.agent-tarsrc',
    '.agent-tarsrc.json'
  ]
});

export async function loadConfig(
  cliOptions: Partial<AgentTARSConfig>
): Promise<AgentTARSConfig> {
  // 1. 파일에서 설정 로드
  const result = await explorer.search();
  const fileConfig = result?.config || {};

  // 2. 환경 변수에서 설정 로드
  const envConfig = {
    provider: process.env.AGENT_TARS_PROVIDER,
    apiKey: process.env.AGENT_TARS_API_KEY,
    model: process.env.AGENT_TARS_MODEL
  };

  // 3. 우선순위: CLI > 환경변수 > 파일 > 기본값
  return {
    ...defaultConfig,
    ...fileConfig,
    ...envConfig,
    ...cliOptions
  };
}
```

---

## AgentTARS 핵심 클래스

### 클래스 구조 (`core/src/AgentTARS.ts`)

```typescript
import { MCPAgent } from '@anthropic-ai/tarko-mcp-agent';
import { EventStream } from '@anthropic-ai/tarko-agent';
import { AgentTARSConfig, AgentEvent } from '@anthropic-ai/agent-tars-interface';

export class AgentTARS extends MCPAgent {
  private config: AgentTARSConfig;
  private eventStream: EventStream<AgentEvent>;
  private browserController: BrowserController;
  private fileSystemController: FileSystemController;

  constructor(config: AgentTARSConfig) {
    super({
      model: config.model,
      provider: config.provider,
      apiKey: config.apiKey
    });

    this.config = config;
    this.eventStream = new EventStream();

    this.initializeControllers();
    this.registerMCPServers();
    this.setupEventHandlers();
  }

  private initializeControllers(): void {
    // 브라우저 컨트롤러
    this.browserController = new BrowserController({
      headless: this.config.browser?.headless ?? false,
      control: this.config.browser?.control ?? 'hybrid'
    });

    // 파일 시스템 컨트롤러
    this.fileSystemController = new FileSystemController({
      workspaceDir: this.config.workspace?.dir
    });
  }

  private registerMCPServers(): void {
    // 내장 MCP 서버 등록
    this.registerServer('browser', '@anthropic-ai/mcp-browser');
    this.registerServer('filesystem', '@anthropic-ai/mcp-filesystem');
    this.registerServer('commands', '@anthropic-ai/mcp-commands');

    // 사용자 정의 MCP 서버
    for (const server of this.config.mcpServers || []) {
      this.registerServer(server.name, server.module, server.config);
    }
  }

  private setupEventHandlers(): void {
    // LLM 응답 이벤트
    this.on('llm:response', (response) => {
      this.eventStream.emit({
        type: 'llm_response',
        content: response.content,
        toolCalls: response.tool_calls
      });
    });

    // 도구 호출 이벤트
    this.on('tool:call', (toolCall) => {
      this.eventStream.emit({
        type: 'tool_call_start',
        toolName: toolCall.name,
        arguments: toolCall.arguments
      });
    });

    // 도구 결과 이벤트
    this.on('tool:result', (result) => {
      this.eventStream.emit({
        type: 'tool_call_end',
        toolName: result.toolName,
        result: result.output
      });
    });
  }
}
```

### 실행 모드

```typescript
export class AgentTARS extends MCPAgent {
  /**
   * 헤드리스 모드 실행 (서버/CLI)
   */
  async runHeadless(): Promise<void> {
    console.log('Agent TARS running in headless mode...');

    // REPL 인터페이스
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });

    while (true) {
      const input = await new Promise<string>((resolve) => {
        rl.question('> ', resolve);
      });

      if (input === 'exit') break;

      try {
        const result = await this.run(input);
        console.log(result);
      } catch (error) {
        console.error('Error:', error.message);
      }
    }

    await this.cleanup();
  }

  /**
   * Web UI 모드 실행
   */
  async runWithUI(): Promise<void> {
    const { createServer } = await import('@anthropic-ai/tarko-agent-ui');

    const server = createServer({
      agent: this,
      port: this.config.port || 3000,
      eventStream: this.eventStream
    });

    await server.start();

    console.log(`Agent TARS Web UI: http://localhost:${server.port}`);

    // 브라우저 자동 열기
    if (this.config.openBrowser !== false) {
      await open(`http://localhost:${server.port}`);
    }
  }
}
```

---

## 이벤트 스트림 시스템

### 이벤트 타입 정의

```typescript
// interface/src/events.ts

export enum AgentEventType {
  // 에이전트 생명주기
  AGENT_START = 'agent_start',
  AGENT_END = 'agent_end',
  AGENT_ERROR = 'agent_error',

  // LLM 상호작용
  LLM_REQUEST = 'llm_request',
  LLM_RESPONSE = 'llm_response',
  LLM_STREAM_CHUNK = 'llm_stream_chunk',

  // 도구 호출
  TOOL_CALL_START = 'tool_call_start',
  TOOL_CALL_END = 'tool_call_end',
  TOOL_CALL_ERROR = 'tool_call_error',

  // 브라우저 이벤트
  BROWSER_NAVIGATE = 'browser_navigate',
  BROWSER_SCREENSHOT = 'browser_screenshot',
  BROWSER_ACTION = 'browser_action',

  // 파일 시스템 이벤트
  FILE_READ = 'file_read',
  FILE_WRITE = 'file_write',
  FILE_DELETE = 'file_delete'
}

export interface AgentEvent {
  type: AgentEventType;
  timestamp: number;
  data: any;
}
```

### 이벤트 스트림 구현

```typescript
// tarko/agent/src/EventStream.ts

import { EventEmitter } from 'events';

export class EventStream<T extends { type: string }> {
  private emitter: EventEmitter;
  private history: T[] = [];
  private maxHistorySize: number;

  constructor(options: { maxHistorySize?: number } = {}) {
    this.emitter = new EventEmitter();
    this.maxHistorySize = options.maxHistorySize ?? 1000;
  }

  emit(event: T): void {
    const eventWithTimestamp = {
      ...event,
      timestamp: Date.now()
    };

    // 히스토리에 저장
    this.history.push(eventWithTimestamp);
    if (this.history.length > this.maxHistorySize) {
      this.history.shift();
    }

    // 이벤트 발행
    this.emitter.emit(event.type, eventWithTimestamp);
    this.emitter.emit('*', eventWithTimestamp);
  }

  on(type: string, handler: (event: T) => void): () => void {
    this.emitter.on(type, handler);
    return () => this.emitter.off(type, handler);
  }

  // 모든 이벤트 구독
  subscribe(handler: (event: T) => void): () => void {
    return this.on('*', handler);
  }

  // 특정 타입 필터링
  filter(types: string[]): EventStream<T> {
    const filtered = new EventStream<T>();
    this.subscribe((event) => {
      if (types.includes(event.type)) {
        filtered.emit(event);
      }
    });
    return filtered;
  }

  getHistory(): T[] {
    return [...this.history];
  }
}
```

---

## 브라우저 제어 전략

### 하이브리드 컨트롤러

```typescript
// core/src/browser/BrowserController.ts

export class BrowserController {
  private strategy: BrowserControlStrategy;
  private browser: Browser;
  private page: Page;

  constructor(options: BrowserControllerOptions) {
    this.strategy = this.createStrategy(options.control);
  }

  private createStrategy(type: ControlType): BrowserControlStrategy {
    switch (type) {
      case 'dom':
        return new DOMStrategy();
      case 'visual-grounding':
        return new VisualGroundingStrategy();
      case 'hybrid':
      default:
        return new HybridStrategy();
    }
  }

  async click(target: ClickTarget): Promise<void> {
    const element = await this.strategy.findElement(this.page, target);

    if (element) {
      await element.click();
    } else {
      // Visual Grounding 폴백
      const screenshot = await this.page.screenshot();
      const coords = await this.strategy.findCoordinates(screenshot, target);
      await this.page.mouse.click(coords.x, coords.y);
    }
  }

  async type(target: TypeTarget, text: string): Promise<void> {
    await this.click(target);
    await this.page.keyboard.type(text);
  }

  async navigate(url: string): Promise<void> {
    await this.page.goto(url, { waitUntil: 'networkidle0' });
  }
}
```

### DOM 전략

```typescript
class DOMStrategy implements BrowserControlStrategy {
  async findElement(page: Page, target: ClickTarget): Promise<ElementHandle | null> {
    // CSS 선택자로 찾기
    if (target.selector) {
      return page.$(target.selector);
    }

    // 텍스트로 찾기
    if (target.text) {
      return page.evaluateHandle((text) => {
        const walker = document.createTreeWalker(
          document.body,
          NodeFilter.SHOW_TEXT
        );
        while (walker.nextNode()) {
          if (walker.currentNode.textContent?.includes(text)) {
            return walker.currentNode.parentElement;
          }
        }
        return null;
      }, target.text);
    }

    // ARIA 라벨로 찾기
    if (target.ariaLabel) {
      return page.$(`[aria-label="${target.ariaLabel}"]`);
    }

    return null;
  }
}
```

### Visual Grounding 전략

```typescript
class VisualGroundingStrategy implements BrowserControlStrategy {
  private model: VisionLanguageModel;

  constructor() {
    this.model = new VisionLanguageModel();
  }

  async findCoordinates(
    screenshot: Buffer,
    target: ClickTarget
  ): Promise<{ x: number; y: number }> {
    const response = await this.model.analyze({
      image: screenshot.toString('base64'),
      prompt: `Find the coordinates of the element: "${target.description}".
               Return JSON: { "x": number, "y": number }`
    });

    return JSON.parse(response);
  }
}
```

---

## 도구 등록 시스템

### 내장 도구 정의

```typescript
// core/src/tools/browser.ts

export const browserTools = [
  {
    name: 'browser_navigate',
    description: 'Navigate to a URL',
    parameters: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'The URL to navigate to' }
      },
      required: ['url']
    },
    handler: async (args, context) => {
      await context.browserController.navigate(args.url);
      return { success: true, url: args.url };
    }
  },
  {
    name: 'browser_click',
    description: 'Click on an element',
    parameters: {
      type: 'object',
      properties: {
        selector: { type: 'string', description: 'CSS selector' },
        text: { type: 'string', description: 'Text content' },
        coordinates: {
          type: 'object',
          properties: {
            x: { type: 'number' },
            y: { type: 'number' }
          }
        }
      }
    },
    handler: async (args, context) => {
      await context.browserController.click(args);
      return { success: true };
    }
  },
  {
    name: 'browser_type',
    description: 'Type text into an input',
    parameters: {
      type: 'object',
      properties: {
        selector: { type: 'string' },
        text: { type: 'string', description: 'Text to type' }
      },
      required: ['text']
    },
    handler: async (args, context) => {
      await context.browserController.type(args, args.text);
      return { success: true };
    }
  },
  {
    name: 'browser_screenshot',
    description: 'Take a screenshot',
    parameters: {
      type: 'object',
      properties: {
        fullPage: { type: 'boolean', default: false }
      }
    },
    handler: async (args, context) => {
      const screenshot = await context.browserController.screenshot(args);
      return {
        success: true,
        image: screenshot.toString('base64'),
        mimeType: 'image/png'
      };
    }
  }
];
```

### 도구 레지스트리

```typescript
// core/src/tools/registry.ts

export class ToolRegistry {
  private tools: Map<string, Tool> = new Map();

  register(tool: Tool): void {
    if (this.tools.has(tool.name)) {
      throw new Error(`Tool ${tool.name} already registered`);
    }
    this.tools.set(tool.name, tool);
  }

  registerAll(tools: Tool[]): void {
    tools.forEach(tool => this.register(tool));
  }

  get(name: string): Tool | undefined {
    return this.tools.get(name);
  }

  getAll(): Tool[] {
    return Array.from(this.tools.values());
  }

  // OpenAI 함수 스키마로 변환
  toOpenAIFunctions(): OpenAIFunction[] {
    return this.getAll().map(tool => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters
      }
    }));
  }

  // Anthropic 도구 스키마로 변환
  toAnthropicTools(): AnthropicTool[] {
    return this.getAll().map(tool => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.parameters
    }));
  }
}
```

---

## 설정 스키마

### 전체 설정 인터페이스

```typescript
// interface/src/config.ts

export interface AgentTARSConfig {
  // LLM 설정
  provider: 'openai' | 'anthropic' | 'google' | 'volcengine' | 'ollama';
  model: string;
  apiKey?: string;
  baseURL?: string;

  // 브라우저 설정
  browser?: {
    headless?: boolean;
    control?: 'dom' | 'visual-grounding' | 'hybrid';
    viewport?: { width: number; height: number };
    userAgent?: string;
    proxy?: string;
  };

  // 작업 공간 설정
  workspace?: {
    dir?: string;
    allowedPaths?: string[];
    blockedPaths?: string[];
  };

  // MCP 서버 설정
  mcpServers?: MCPServerConfig[];

  // UI 설정
  port?: number;
  openBrowser?: boolean;

  // 에이전트 동작 설정
  maxIterations?: number;
  timeout?: number;
  verbose?: boolean;

  // 시스템 프롬프트
  systemPrompt?: string;
}

export interface MCPServerConfig {
  name: string;
  module: string;
  config?: Record<string, any>;
  transport?: 'stdio' | 'sse' | 'http';
}
```

### 기본 설정

```typescript
export const defaultConfig: AgentTARSConfig = {
  provider: 'openai',
  model: 'gpt-4o',

  browser: {
    headless: false,
    control: 'hybrid',
    viewport: { width: 1280, height: 720 }
  },

  workspace: {
    dir: process.cwd()
  },

  mcpServers: [],

  port: 3000,
  openBrowser: true,

  maxIterations: 100,
  timeout: 300000,
  verbose: false
};
```

---

## 에러 처리

```typescript
// core/src/errors.ts

export class AgentTARSError extends Error {
  constructor(
    message: string,
    public code: string,
    public recoverable: boolean = true
  ) {
    super(message);
    this.name = 'AgentTARSError';
  }
}

export class ToolExecutionError extends AgentTARSError {
  constructor(toolName: string, cause: Error) {
    super(
      `Tool "${toolName}" failed: ${cause.message}`,
      'TOOL_EXECUTION_ERROR',
      true
    );
  }
}

export class BrowserError extends AgentTARSError {
  constructor(action: string, cause: Error) {
    super(
      `Browser action "${action}" failed: ${cause.message}`,
      'BROWSER_ERROR',
      true
    );
  }
}

export class LLMError extends AgentTARSError {
  constructor(provider: string, cause: Error) {
    super(
      `LLM provider "${provider}" error: ${cause.message}`,
      'LLM_ERROR',
      cause.message.includes('rate limit')
    );
  }
}
```

---

*다음 글에서는 GUI Agent SDK를 분석합니다.*
