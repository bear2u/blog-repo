---
layout: post
title: "UI-TARS 완벽 가이드 (7) - Tarko 프레임워크"
date: 2025-02-04
permalink: /ui-tars-guide-07-tarko/
author: ByteDance
category: AI
tags: [UI-TARS, Tarko, Agent Framework, Event Stream, LLM Client]
series: ui-tars-guide
part: 7
original_url: "https://github.com/bytedance/UI-TARS-desktop"
excerpt: "UI-TARS의 핵심 에이전트 프레임워크 Tarko를 분석합니다. Agent, LLM Client, MCP Agent, Context Engineer 모듈을 살펴봅니다."
---

## Tarko 프레임워크 개요

**Tarko**는 이벤트 스트림 기반의 메타 에이전트 프레임워크로, UI-TARS 생태계의 핵심 기반입니다.

```
multimodal/tarko/
├── agent/               # 기본 Agent 클래스
│   └── src/
│       ├── Agent.ts
│       ├── EventStream.ts
│       └── types.ts
├── llm-client/          # 멀티 LLM 클라이언트
│   └── src/
│       ├── LLMClient.ts
│       ├── providers/
│       └── types.ts
├── mcp-agent/           # MCP 통합 에이전트
│   └── src/
│       ├── MCPAgent.ts
│       └── ServerManager.ts
├── context-engineer/    # 컨텍스트 처리
│   └── src/
│       ├── ContextEngineer.ts
│       └── strategies/
└── agent-ui/            # Web UI 컴포넌트
    └── src/
        ├── components/
        └── hooks/
```

---

## Agent 기본 클래스

### Agent 인터페이스

```typescript
// tarko/agent/src/types.ts

export interface AgentOptions {
  name?: string;
  description?: string;
  systemPrompt?: string;
  model?: LLMClient;
  maxIterations?: number;
  timeout?: number;
}

export interface AgentRunOptions {
  signal?: AbortSignal;
  onEvent?: (event: AgentEvent) => void;
}

export interface AgentResult {
  success: boolean;
  output: any;
  iterations: number;
  events: AgentEvent[];
  error?: Error;
}

export enum AgentStatus {
  IDLE = 'idle',
  RUNNING = 'running',
  PAUSED = 'paused',
  COMPLETED = 'completed',
  ERROR = 'error',
  STOPPED = 'stopped'
}
```

### Agent 클래스 구현

```typescript
// tarko/agent/src/Agent.ts

import { EventEmitter } from 'events';
import { EventStream } from './EventStream';
import {
  AgentOptions,
  AgentRunOptions,
  AgentResult,
  AgentStatus,
  AgentEvent,
  AgentEventType
} from './types';

export abstract class Agent extends EventEmitter {
  protected name: string;
  protected description: string;
  protected systemPrompt: string;
  protected model: LLMClient;
  protected maxIterations: number;
  protected timeout: number;

  protected status: AgentStatus = AgentStatus.IDLE;
  protected eventStream: EventStream<AgentEvent>;
  protected conversationHistory: Message[] = [];

  constructor(options: AgentOptions) {
    super();

    this.name = options.name || 'Agent';
    this.description = options.description || '';
    this.systemPrompt = options.systemPrompt || '';
    this.model = options.model!;
    this.maxIterations = options.maxIterations ?? 100;
    this.timeout = options.timeout ?? 300000;

    this.eventStream = new EventStream();
  }

  /**
   * 에이전트 실행
   */
  async run(input: string, options?: AgentRunOptions): Promise<AgentResult> {
    this.status = AgentStatus.RUNNING;
    this.emitEvent(AgentEventType.AGENT_START, { input });

    const timeoutPromise = this.createTimeout();
    const runPromise = this.executeLoop(input, options);

    try {
      const result = await Promise.race([runPromise, timeoutPromise]);
      this.status = AgentStatus.COMPLETED;
      this.emitEvent(AgentEventType.AGENT_END, { result });
      return result;
    } catch (error) {
      this.status = AgentStatus.ERROR;
      this.emitEvent(AgentEventType.ERROR, { error });
      throw error;
    }
  }

  /**
   * 에이전트 중지
   */
  stop(): void {
    this.status = AgentStatus.STOPPED;
    this.emitEvent(AgentEventType.AGENT_END, { reason: 'stopped' });
  }

  /**
   * 메인 실행 루프 (서브클래스에서 구현)
   */
  protected abstract executeLoop(
    input: string,
    options?: AgentRunOptions
  ): Promise<AgentResult>;

  /**
   * 이벤트 발행
   */
  protected emitEvent(type: AgentEventType, data: any): void {
    const event: AgentEvent = {
      type,
      timestamp: Date.now(),
      agentName: this.name,
      data
    };

    this.eventStream.emit(event);
    this.emit(type, event);
  }

  /**
   * 타임아웃 프로미스 생성
   */
  private createTimeout(): Promise<never> {
    return new Promise((_, reject) => {
      setTimeout(() => {
        reject(new Error(`Agent timeout after ${this.timeout}ms`));
      }, this.timeout);
    });
  }

  /**
   * 이벤트 스트림 조회
   */
  getEventStream(): EventStream<AgentEvent> {
    return this.eventStream;
  }

  /**
   * 현재 상태 조회
   */
  getStatus(): AgentStatus {
    return this.status;
  }
}
```

### 이벤트 스트림 구현

```typescript
// tarko/agent/src/EventStream.ts

import { EventEmitter } from 'events';

export interface EventStreamOptions {
  maxHistorySize?: number;
  persistEvents?: boolean;
}

export class EventStream<T extends { type: string; timestamp: number }> {
  private emitter: EventEmitter;
  private history: T[] = [];
  private maxHistorySize: number;
  private subscribers: Map<string, Set<(event: T) => void>> = new Map();

  constructor(options: EventStreamOptions = {}) {
    this.emitter = new EventEmitter();
    this.emitter.setMaxListeners(100);
    this.maxHistorySize = options.maxHistorySize ?? 10000;
  }

  /**
   * 이벤트 발행
   */
  emit(event: T): void {
    // 히스토리에 추가
    this.history.push(event);
    if (this.history.length > this.maxHistorySize) {
      this.history = this.history.slice(-this.maxHistorySize);
    }

    // 특정 타입 구독자에게 전달
    this.emitter.emit(event.type, event);

    // 전체 이벤트 구독자에게 전달
    this.emitter.emit('*', event);
  }

  /**
   * 특정 타입 이벤트 구독
   */
  on(type: string, handler: (event: T) => void): () => void {
    this.emitter.on(type, handler);
    return () => this.emitter.off(type, handler);
  }

  /**
   * 모든 이벤트 구독
   */
  subscribe(handler: (event: T) => void): () => void {
    return this.on('*', handler);
  }

  /**
   * 한 번만 실행되는 구독
   */
  once(type: string, handler: (event: T) => void): void {
    this.emitter.once(type, handler);
  }

  /**
   * 특정 타입의 이벤트만 필터링하는 새 스트림 생성
   */
  filter(types: string[]): EventStream<T> {
    const filtered = new EventStream<T>();

    this.subscribe((event) => {
      if (types.includes(event.type)) {
        filtered.emit(event);
      }
    });

    return filtered;
  }

  /**
   * 이벤트를 변환하는 새 스트림 생성
   */
  map<U extends { type: string; timestamp: number }>(
    transformer: (event: T) => U
  ): EventStream<U> {
    const mapped = new EventStream<U>();

    this.subscribe((event) => {
      mapped.emit(transformer(event));
    });

    return mapped;
  }

  /**
   * 히스토리 조회
   */
  getHistory(filter?: { type?: string; since?: number }): T[] {
    let result = [...this.history];

    if (filter?.type) {
      result = result.filter(e => e.type === filter.type);
    }

    if (filter?.since) {
      result = result.filter(e => e.timestamp >= filter.since);
    }

    return result;
  }

  /**
   * 히스토리 초기화
   */
  clearHistory(): void {
    this.history = [];
  }

  /**
   * JSON Lines 형식으로 직렬화
   */
  toJSONL(): string {
    return this.history.map(e => JSON.stringify(e)).join('\n');
  }

  /**
   * JSON Lines에서 복원
   */
  static fromJSONL<T extends { type: string; timestamp: number }>(
    jsonl: string
  ): EventStream<T> {
    const stream = new EventStream<T>();
    const lines = jsonl.trim().split('\n');

    for (const line of lines) {
      if (line) {
        const event = JSON.parse(line) as T;
        stream.history.push(event);
      }
    }

    return stream;
  }
}
```

---

## LLM Client

### 멀티 프로바이더 LLM 클라이언트

```typescript
// tarko/llm-client/src/LLMClient.ts

import { LLMProvider, LLMConfig, ChatRequest, ChatResponse } from './types';
import { OpenAIProvider } from './providers/openai';
import { AnthropicProvider } from './providers/anthropic';
import { GoogleProvider } from './providers/google';
import { OllamaProvider } from './providers/ollama';

export class LLMClient {
  private provider: LLMProvider;
  private config: LLMConfig;

  constructor(config: LLMConfig) {
    this.config = config;
    this.provider = this.createProvider(config);
  }

  private createProvider(config: LLMConfig): LLMProvider {
    switch (config.provider) {
      case 'openai':
        return new OpenAIProvider({
          apiKey: config.apiKey,
          baseURL: config.baseURL,
          model: config.model
        });

      case 'anthropic':
        return new AnthropicProvider({
          apiKey: config.apiKey,
          model: config.model
        });

      case 'google':
        return new GoogleProvider({
          apiKey: config.apiKey,
          model: config.model
        });

      case 'ollama':
        return new OllamaProvider({
          baseURL: config.baseURL || 'http://localhost:11434',
          model: config.model
        });

      case 'volcengine':
        return new OpenAIProvider({
          apiKey: config.apiKey,
          baseURL: config.baseURL || 'https://ark.cn-beijing.volces.com/api/v3',
          model: config.model
        });

      default:
        throw new Error(`Unknown provider: ${config.provider}`);
    }
  }

  /**
   * 채팅 완성 요청
   */
  async chat(request: ChatRequest): Promise<ChatResponse> {
    return this.provider.chat(request);
  }

  /**
   * 스트리밍 채팅 완성 요청
   */
  async *chatStream(
    request: ChatRequest
  ): AsyncGenerator<ChatStreamChunk, void, unknown> {
    yield* this.provider.chatStream(request);
  }

  /**
   * 도구 포함 채팅 요청
   */
  async chatWithTools(
    request: ChatRequest & { tools: Tool[] }
  ): Promise<ChatResponse> {
    return this.provider.chatWithTools(request);
  }

  /**
   * 모델 정보 조회
   */
  getModelInfo(): { provider: string; model: string } {
    return {
      provider: this.config.provider,
      model: this.config.model
    };
  }
}
```

### OpenAI 프로바이더

```typescript
// tarko/llm-client/src/providers/openai.ts

import OpenAI from 'openai';
import { LLMProvider, ChatRequest, ChatResponse, Tool } from '../types';

export class OpenAIProvider implements LLMProvider {
  private client: OpenAI;
  private model: string;

  constructor(config: { apiKey?: string; baseURL?: string; model: string }) {
    this.client = new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseURL
    });
    this.model = config.model;
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    const response = await this.client.chat.completions.create({
      model: this.model,
      messages: this.convertMessages(request.messages),
      temperature: request.temperature ?? 0.7,
      max_tokens: request.maxTokens
    });

    return {
      content: response.choices[0].message.content || '',
      role: 'assistant',
      finishReason: response.choices[0].finish_reason,
      usage: {
        promptTokens: response.usage?.prompt_tokens || 0,
        completionTokens: response.usage?.completion_tokens || 0,
        totalTokens: response.usage?.total_tokens || 0
      }
    };
  }

  async *chatStream(
    request: ChatRequest
  ): AsyncGenerator<ChatStreamChunk, void, unknown> {
    const stream = await this.client.chat.completions.create({
      model: this.model,
      messages: this.convertMessages(request.messages),
      temperature: request.temperature ?? 0.7,
      max_tokens: request.maxTokens,
      stream: true
    });

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta;

      if (delta?.content) {
        yield {
          type: 'content',
          content: delta.content
        };
      }

      if (chunk.choices[0]?.finish_reason) {
        yield {
          type: 'finish',
          finishReason: chunk.choices[0].finish_reason
        };
      }
    }
  }

  async chatWithTools(
    request: ChatRequest & { tools: Tool[] }
  ): Promise<ChatResponse> {
    const response = await this.client.chat.completions.create({
      model: this.model,
      messages: this.convertMessages(request.messages),
      tools: this.convertTools(request.tools),
      tool_choice: 'auto'
    });

    const message = response.choices[0].message;

    return {
      content: message.content || '',
      role: 'assistant',
      toolCalls: message.tool_calls?.map(tc => ({
        id: tc.id,
        name: tc.function.name,
        arguments: JSON.parse(tc.function.arguments)
      })),
      finishReason: response.choices[0].finish_reason
    };
  }

  private convertMessages(messages: Message[]): OpenAI.ChatCompletionMessageParam[] {
    return messages.map(msg => {
      if (typeof msg.content === 'string') {
        return { role: msg.role, content: msg.content };
      }

      // 멀티모달 메시지 변환
      return {
        role: msg.role,
        content: msg.content.map(part => {
          if (part.type === 'text') {
            return { type: 'text', text: part.text };
          }
          if (part.type === 'image') {
            return {
              type: 'image_url',
              image_url: {
                url: `data:${part.source.media_type};base64,${part.source.data}`
              }
            };
          }
          return part;
        })
      };
    });
  }

  private convertTools(tools: Tool[]): OpenAI.ChatCompletionTool[] {
    return tools.map(tool => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters
      }
    }));
  }
}
```

### Anthropic 프로바이더

```typescript
// tarko/llm-client/src/providers/anthropic.ts

import Anthropic from '@anthropic-ai/sdk';
import { LLMProvider, ChatRequest, ChatResponse, Tool } from '../types';

export class AnthropicProvider implements LLMProvider {
  private client: Anthropic;
  private model: string;

  constructor(config: { apiKey?: string; model: string }) {
    this.client = new Anthropic({
      apiKey: config.apiKey
    });
    this.model = config.model;
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    const { systemPrompt, messages } = this.extractSystemPrompt(request.messages);

    const response = await this.client.messages.create({
      model: this.model,
      system: systemPrompt,
      messages: this.convertMessages(messages),
      max_tokens: request.maxTokens || 4096
    });

    return {
      content: response.content[0].type === 'text'
        ? response.content[0].text
        : '',
      role: 'assistant',
      finishReason: response.stop_reason,
      usage: {
        promptTokens: response.usage.input_tokens,
        completionTokens: response.usage.output_tokens,
        totalTokens: response.usage.input_tokens + response.usage.output_tokens
      }
    };
  }

  async chatWithTools(
    request: ChatRequest & { tools: Tool[] }
  ): Promise<ChatResponse> {
    const { systemPrompt, messages } = this.extractSystemPrompt(request.messages);

    const response = await this.client.messages.create({
      model: this.model,
      system: systemPrompt,
      messages: this.convertMessages(messages),
      tools: this.convertTools(request.tools),
      max_tokens: request.maxTokens || 4096
    });

    const toolUseBlocks = response.content.filter(
      block => block.type === 'tool_use'
    );

    return {
      content: response.content
        .filter(block => block.type === 'text')
        .map(block => (block as any).text)
        .join(''),
      role: 'assistant',
      toolCalls: toolUseBlocks.map(block => ({
        id: (block as any).id,
        name: (block as any).name,
        arguments: (block as any).input
      })),
      finishReason: response.stop_reason
    };
  }

  private extractSystemPrompt(messages: Message[]): {
    systemPrompt: string;
    messages: Message[];
  } {
    const systemMessages = messages.filter(m => m.role === 'system');
    const otherMessages = messages.filter(m => m.role !== 'system');

    return {
      systemPrompt: systemMessages.map(m => m.content).join('\n'),
      messages: otherMessages
    };
  }

  private convertMessages(messages: Message[]): Anthropic.MessageParam[] {
    return messages.map(msg => {
      if (typeof msg.content === 'string') {
        return { role: msg.role as 'user' | 'assistant', content: msg.content };
      }

      return {
        role: msg.role as 'user' | 'assistant',
        content: msg.content.map(part => {
          if (part.type === 'text') {
            return { type: 'text', text: part.text };
          }
          if (part.type === 'image') {
            return {
              type: 'image',
              source: {
                type: 'base64',
                media_type: part.source.media_type,
                data: part.source.data
              }
            };
          }
          return part;
        })
      };
    });
  }

  private convertTools(tools: Tool[]): Anthropic.Tool[] {
    return tools.map(tool => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.parameters as any
    }));
  }
}
```

---

## MCP Agent

### MCP 통합 에이전트

```typescript
// tarko/mcp-agent/src/MCPAgent.ts

import { Agent, AgentOptions, AgentResult, AgentRunOptions } from '@anthropic-ai/tarko-agent';
import { MCPClient } from '@anthropic-ai/mcp-client';
import { ServerManager } from './ServerManager';

export interface MCPAgentOptions extends AgentOptions {
  mcpServers?: MCPServerConfig[];
}

export class MCPAgent extends Agent {
  private serverManager: ServerManager;
  private mcpClient: MCPClient;
  private registeredTools: Tool[] = [];

  constructor(options: MCPAgentOptions) {
    super(options);

    this.serverManager = new ServerManager();
    this.mcpClient = new MCPClient();
  }

  /**
   * MCP 서버 등록
   */
  async registerServer(
    name: string,
    module: string,
    config?: Record<string, any>
  ): Promise<void> {
    const server = await this.serverManager.start(name, module, config);
    await this.mcpClient.connect(server);

    // 서버의 도구 목록 가져오기
    const tools = await this.mcpClient.listTools(name);
    this.registeredTools.push(...tools);

    this.emitEvent('mcp:server_registered', { name, tools: tools.length });
  }

  /**
   * 도구 호출 실행
   */
  protected async executeToolCall(
    toolName: string,
    arguments_: Record<string, any>
  ): Promise<any> {
    this.emitEvent('tool_call_start', { toolName, arguments: arguments_ });

    try {
      const result = await this.mcpClient.callTool(toolName, arguments_);
      this.emitEvent('tool_call_end', { toolName, result });
      return result;
    } catch (error) {
      this.emitEvent('tool_call_error', { toolName, error });
      throw error;
    }
  }

  protected async executeLoop(
    input: string,
    options?: AgentRunOptions
  ): Promise<AgentResult> {
    // 대화 히스토리 초기화
    this.conversationHistory = [
      { role: 'system', content: this.systemPrompt },
      { role: 'user', content: input }
    ];

    let iterations = 0;

    while (iterations < this.maxIterations) {
      if (this.status === 'stopped') break;

      iterations++;
      this.emitEvent('iteration', { count: iterations });

      // LLM 호출 (도구 포함)
      const response = await this.model.chatWithTools({
        messages: this.conversationHistory,
        tools: this.registeredTools
      });

      this.emitEvent('llm_response', { response });

      // 응답을 히스토리에 추가
      this.conversationHistory.push({
        role: 'assistant',
        content: response.content,
        toolCalls: response.toolCalls
      });

      // 도구 호출이 없으면 완료
      if (!response.toolCalls || response.toolCalls.length === 0) {
        return {
          success: true,
          output: response.content,
          iterations,
          events: this.eventStream.getHistory()
        };
      }

      // 도구 호출 실행
      for (const toolCall of response.toolCalls) {
        const result = await this.executeToolCall(
          toolCall.name,
          toolCall.arguments
        );

        // 도구 결과를 히스토리에 추가
        this.conversationHistory.push({
          role: 'tool',
          toolCallId: toolCall.id,
          content: JSON.stringify(result)
        });
      }
    }

    return {
      success: false,
      output: 'Max iterations reached',
      iterations,
      events: this.eventStream.getHistory()
    };
  }

  async cleanup(): Promise<void> {
    await this.serverManager.stopAll();
    await this.mcpClient.disconnect();
  }
}
```

### 서버 매니저

```typescript
// tarko/mcp-agent/src/ServerManager.ts

import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';

export interface ServerInfo {
  name: string;
  process: ChildProcess;
  transport: 'stdio' | 'sse' | 'http';
  status: 'starting' | 'running' | 'stopped' | 'error';
}

export class ServerManager extends EventEmitter {
  private servers: Map<string, ServerInfo> = new Map();

  /**
   * MCP 서버 시작
   */
  async start(
    name: string,
    module: string,
    config?: Record<string, any>
  ): Promise<ServerInfo> {
    if (this.servers.has(name)) {
      throw new Error(`Server ${name} already running`);
    }

    // 모듈 경로 확인
    const modulePath = require.resolve(module);

    // 서버 프로세스 시작
    const process = spawn('node', [modulePath], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {
        ...process.env,
        MCP_SERVER_CONFIG: JSON.stringify(config || {})
      }
    });

    const serverInfo: ServerInfo = {
      name,
      process,
      transport: 'stdio',
      status: 'starting'
    };

    // 프로세스 이벤트 핸들링
    process.on('spawn', () => {
      serverInfo.status = 'running';
      this.emit('server:started', { name });
    });

    process.on('error', (error) => {
      serverInfo.status = 'error';
      this.emit('server:error', { name, error });
    });

    process.on('exit', (code) => {
      serverInfo.status = 'stopped';
      this.emit('server:stopped', { name, code });
      this.servers.delete(name);
    });

    this.servers.set(name, serverInfo);

    // 서버 준비 대기
    await this.waitForReady(serverInfo);

    return serverInfo;
  }

  /**
   * 서버 준비 대기
   */
  private waitForReady(server: ServerInfo): Promise<void> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Server ${server.name} failed to start`));
      }, 10000);

      const checkReady = () => {
        if (server.status === 'running') {
          clearTimeout(timeout);
          resolve();
        } else if (server.status === 'error') {
          clearTimeout(timeout);
          reject(new Error(`Server ${server.name} error`));
        } else {
          setTimeout(checkReady, 100);
        }
      };

      checkReady();
    });
  }

  /**
   * 특정 서버 중지
   */
  async stop(name: string): Promise<void> {
    const server = this.servers.get(name);
    if (!server) return;

    server.process.kill('SIGTERM');

    // 강제 종료 대기
    await new Promise<void>((resolve) => {
      const timeout = setTimeout(() => {
        server.process.kill('SIGKILL');
        resolve();
      }, 5000);

      server.process.on('exit', () => {
        clearTimeout(timeout);
        resolve();
      });
    });

    this.servers.delete(name);
  }

  /**
   * 모든 서버 중지
   */
  async stopAll(): Promise<void> {
    const names = Array.from(this.servers.keys());
    await Promise.all(names.map(name => this.stop(name)));
  }

  /**
   * 서버 정보 조회
   */
  getServer(name: string): ServerInfo | undefined {
    return this.servers.get(name);
  }

  /**
   * 모든 서버 목록
   */
  listServers(): ServerInfo[] {
    return Array.from(this.servers.values());
  }
}
```

---

## Tool Call Engine

### 도구 호출 전략

```typescript
// tarko/agent/src/ToolCallEngine.ts

export type ToolCallStrategy = 'native' | 'prompt' | 'structured';

export interface ToolCallEngineOptions {
  strategy: ToolCallStrategy;
  tools: Tool[];
  model: LLMClient;
}

export class ToolCallEngine {
  private strategy: ToolCallStrategy;
  private tools: Tool[];
  private model: LLMClient;

  constructor(options: ToolCallEngineOptions) {
    this.strategy = options.strategy;
    this.tools = options.tools;
    this.model = options.model;
  }

  /**
   * LLM에서 도구 호출 추출
   */
  async extractToolCalls(response: ChatResponse): Promise<ToolCall[]> {
    switch (this.strategy) {
      case 'native':
        return this.extractNativeToolCalls(response);

      case 'prompt':
        return this.extractPromptToolCalls(response);

      case 'structured':
        return this.extractStructuredToolCalls(response);

      default:
        return [];
    }
  }

  /**
   * 네이티브 도구 호출 (OpenAI/Anthropic 함수 호출)
   */
  private extractNativeToolCalls(response: ChatResponse): ToolCall[] {
    return response.toolCalls || [];
  }

  /**
   * 프롬프트 기반 도구 호출
   */
  private extractPromptToolCalls(response: ChatResponse): ToolCall[] {
    const content = response.content;

    // Action: tool_name(arg1, arg2) 형식 파싱
    const actionMatch = content.match(/Action:\s*(\w+)\s*\((.*)\)/);
    if (!actionMatch) return [];

    const [, toolName, argsStr] = actionMatch;

    // 도구 확인
    const tool = this.tools.find(t => t.name === toolName);
    if (!tool) return [];

    // 인자 파싱
    const args = this.parseArguments(argsStr, tool.parameters);

    return [{
      id: `call_${Date.now()}`,
      name: toolName,
      arguments: args
    }];
  }

  /**
   * 구조화된 출력 기반 도구 호출
   */
  private extractStructuredToolCalls(response: ChatResponse): ToolCall[] {
    const content = response.content;

    // JSON 블록 추출
    const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
    if (!jsonMatch) return [];

    try {
      const parsed = JSON.parse(jsonMatch[1]);

      if (Array.isArray(parsed)) {
        return parsed.map((item, i) => ({
          id: `call_${Date.now()}_${i}`,
          name: item.tool || item.name,
          arguments: item.arguments || item.args || item.input
        }));
      }

      return [{
        id: `call_${Date.now()}`,
        name: parsed.tool || parsed.name,
        arguments: parsed.arguments || parsed.args || parsed.input
      }];
    } catch {
      return [];
    }
  }

  private parseArguments(
    argsStr: string,
    schema: JSONSchema
  ): Record<string, any> {
    // 간단한 인자 파싱 (실제로는 더 복잡한 파싱 필요)
    const args: Record<string, any> = {};
    const properties = schema.properties || {};
    const propNames = Object.keys(properties);

    // 쉼표로 분리
    const values = argsStr.split(',').map(v => v.trim());

    propNames.forEach((name, i) => {
      if (values[i]) {
        let value = values[i];

        // 따옴표 제거
        if (value.startsWith('"') || value.startsWith("'")) {
          value = value.slice(1, -1);
        }

        // 타입 변환
        const propType = (properties[name] as any).type;
        if (propType === 'number') {
          args[name] = parseFloat(value);
        } else if (propType === 'boolean') {
          args[name] = value === 'true';
        } else {
          args[name] = value;
        }
      }
    });

    return args;
  }
}
```

---

*다음 글에서는 MCP 인프라를 분석합니다.*
