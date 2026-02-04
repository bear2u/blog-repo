---
layout: post
title: "UI-TARS 완벽 가이드 (8) - MCP 인프라"
date: 2025-02-04
permalink: /ui-tars-guide-08-mcp/
author: ByteDance
category: AI
tags: [UI-TARS, MCP, Model Context Protocol, MCP Server, MCP Client]
series: ui-tars-guide
part: 8
original_url: "https://github.com/bytedance/UI-TARS-desktop"
excerpt: "UI-TARS의 MCP(Model Context Protocol) 인프라를 분석합니다. MCP 서버와 클라이언트 구현을 살펴봅니다."
---

## MCP (Model Context Protocol) 개요

**MCP**는 AI 에이전트가 외부 도구와 리소스에 접근하기 위한 표준 프로토콜입니다. UI-TARS는 MCP를 통해 브라우저, 파일 시스템, 명령어 실행 등을 제어합니다.

```
packages/agent-infra/
├── mcp-servers/            # MCP 서버 구현
│   ├── browser/            # 브라우저 자동화
│   ├── filesystem/         # 파일 시스템
│   ├── commands/           # 명령어 실행
│   ├── search/             # 웹 검색
│   └── shared/             # 공통 유틸리티
├── mcp-client/             # MCP 클라이언트
│   └── src/
│       ├── MCPClient.ts
│       └── transports/
└── browser/                # 브라우저 추상화
    └── src/
        ├── Browser.ts
        └── Page.ts
```

---

## MCP 서버 아키텍처

### 기본 서버 구조

```typescript
// mcp-servers/shared/src/BaseServer.ts

import { Server } from '@modelcontextprotocol/sdk/server';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio';

export abstract class BaseMCPServer {
  protected server: Server;
  protected name: string;
  protected version: string;

  constructor(name: string, version: string = '1.0.0') {
    this.name = name;
    this.version = version;

    this.server = new Server({
      name: this.name,
      version: this.version
    }, {
      capabilities: {
        tools: {},
        resources: {},
        prompts: {}
      }
    });

    this.registerHandlers();
  }

  /**
   * 도구 핸들러 등록 (서브클래스에서 구현)
   */
  protected abstract registerHandlers(): void;

  /**
   * 도구 등록 헬퍼
   */
  protected registerTool(
    name: string,
    description: string,
    schema: JSONSchema,
    handler: (args: any) => Promise<any>
  ): void {
    this.server.setRequestHandler(
      'tools/call',
      async (request) => {
        if (request.params.name === name) {
          try {
            const result = await handler(request.params.arguments);
            return {
              content: [{
                type: 'text',
                text: JSON.stringify(result)
              }]
            };
          } catch (error) {
            return {
              content: [{
                type: 'text',
                text: `Error: ${error.message}`
              }],
              isError: true
            };
          }
        }
        throw new Error(`Unknown tool: ${request.params.name}`);
      }
    );

    // 도구 목록에 추가
    this.server.setRequestHandler(
      'tools/list',
      async () => ({
        tools: [{
          name,
          description,
          inputSchema: schema
        }]
      })
    );
  }

  /**
   * 서버 실행
   */
  async run(): Promise<void> {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);

    console.error(`${this.name} MCP server running`);
  }
}
```

---

## Browser MCP Server

### 브라우저 자동화 서버

```typescript
// mcp-servers/browser/src/index.ts

import { BaseMCPServer } from '@anthropic-ai/mcp-shared';
import puppeteer, { Browser, Page } from 'puppeteer';

export class BrowserMCPServer extends BaseMCPServer {
  private browser: Browser | null = null;
  private page: Page | null = null;

  constructor() {
    super('browser-server', '1.0.0');
  }

  protected registerHandlers(): void {
    // 브라우저 초기화
    this.registerTool(
      'browser_launch',
      'Launch a new browser instance',
      {
        type: 'object',
        properties: {
          headless: { type: 'boolean', default: false },
          viewport: {
            type: 'object',
            properties: {
              width: { type: 'number', default: 1280 },
              height: { type: 'number', default: 720 }
            }
          }
        }
      },
      async (args) => {
        if (this.browser) {
          await this.browser.close();
        }

        this.browser = await puppeteer.launch({
          headless: args.headless ?? false,
          args: ['--no-sandbox']
        });

        this.page = await this.browser.newPage();

        if (args.viewport) {
          await this.page.setViewport(args.viewport);
        }

        return { success: true, message: 'Browser launched' };
      }
    );

    // URL 이동
    this.registerTool(
      'browser_navigate',
      'Navigate to a URL',
      {
        type: 'object',
        properties: {
          url: { type: 'string', description: 'URL to navigate to' }
        },
        required: ['url']
      },
      async (args) => {
        if (!this.page) throw new Error('Browser not launched');

        await this.page.goto(args.url, { waitUntil: 'networkidle0' });

        return {
          success: true,
          url: this.page.url(),
          title: await this.page.title()
        };
      }
    );

    // 스크린샷
    this.registerTool(
      'browser_screenshot',
      'Take a screenshot of the current page',
      {
        type: 'object',
        properties: {
          fullPage: { type: 'boolean', default: false }
        }
      },
      async (args) => {
        if (!this.page) throw new Error('Browser not launched');

        const screenshot = await this.page.screenshot({
          type: 'png',
          fullPage: args.fullPage ?? false,
          encoding: 'base64'
        });

        return {
          success: true,
          image: screenshot,
          mimeType: 'image/png'
        };
      }
    );

    // 클릭
    this.registerTool(
      'browser_click',
      'Click on an element or coordinates',
      {
        type: 'object',
        properties: {
          selector: { type: 'string', description: 'CSS selector' },
          x: { type: 'number', description: 'X coordinate' },
          y: { type: 'number', description: 'Y coordinate' }
        }
      },
      async (args) => {
        if (!this.page) throw new Error('Browser not launched');

        if (args.selector) {
          await this.page.click(args.selector);
        } else if (args.x !== undefined && args.y !== undefined) {
          await this.page.mouse.click(args.x, args.y);
        } else {
          throw new Error('Either selector or coordinates required');
        }

        return { success: true };
      }
    );

    // 텍스트 입력
    this.registerTool(
      'browser_type',
      'Type text into an element',
      {
        type: 'object',
        properties: {
          selector: { type: 'string', description: 'CSS selector' },
          text: { type: 'string', description: 'Text to type' },
          delay: { type: 'number', default: 50 }
        },
        required: ['text']
      },
      async (args) => {
        if (!this.page) throw new Error('Browser not launched');

        if (args.selector) {
          await this.page.type(args.selector, args.text, {
            delay: args.delay
          });
        } else {
          await this.page.keyboard.type(args.text, {
            delay: args.delay
          });
        }

        return { success: true };
      }
    );

    // 페이지 콘텐츠 가져오기
    this.registerTool(
      'browser_get_content',
      'Get the page content',
      {
        type: 'object',
        properties: {
          format: {
            type: 'string',
            enum: ['html', 'text', 'markdown'],
            default: 'text'
          }
        }
      },
      async (args) => {
        if (!this.page) throw new Error('Browser not launched');

        let content: string;

        switch (args.format) {
          case 'html':
            content = await this.page.content();
            break;

          case 'markdown':
            content = await this.page.evaluate(() => {
              // HTML을 마크다운으로 변환 (간단한 구현)
              return document.body.innerText;
            });
            break;

          case 'text':
          default:
            content = await this.page.evaluate(
              () => document.body.innerText
            );
        }

        return { success: true, content };
      }
    );

    // DOM 요소 조회
    this.registerTool(
      'browser_query_elements',
      'Query DOM elements',
      {
        type: 'object',
        properties: {
          selector: { type: 'string', description: 'CSS selector' },
          attributes: {
            type: 'array',
            items: { type: 'string' },
            description: 'Attributes to extract'
          }
        },
        required: ['selector']
      },
      async (args) => {
        if (!this.page) throw new Error('Browser not launched');

        const elements = await this.page.$$eval(
          args.selector,
          (els, attrs) => els.map(el => {
            const result: Record<string, any> = {
              tagName: el.tagName,
              text: el.textContent?.trim().slice(0, 100)
            };

            for (const attr of (attrs || [])) {
              result[attr] = el.getAttribute(attr);
            }

            return result;
          }),
          args.attributes || []
        );

        return { success: true, elements, count: elements.length };
      }
    );

    // 브라우저 종료
    this.registerTool(
      'browser_close',
      'Close the browser',
      { type: 'object', properties: {} },
      async () => {
        if (this.browser) {
          await this.browser.close();
          this.browser = null;
          this.page = null;
        }

        return { success: true, message: 'Browser closed' };
      }
    );
  }
}

// 서버 실행
const server = new BrowserMCPServer();
server.run();
```

---

## Filesystem MCP Server

### 파일 시스템 서버

```typescript
// mcp-servers/filesystem/src/index.ts

import { BaseMCPServer } from '@anthropic-ai/mcp-shared';
import * as fs from 'fs/promises';
import * as path from 'path';
import { glob } from 'glob';

export interface FilesystemServerConfig {
  allowedPaths?: string[];
  blockedPaths?: string[];
  maxFileSize?: number;
}

export class FilesystemMCPServer extends BaseMCPServer {
  private config: FilesystemServerConfig;

  constructor(config: FilesystemServerConfig = {}) {
    super('filesystem-server', '1.0.0');
    this.config = {
      maxFileSize: 10 * 1024 * 1024, // 10MB
      ...config
    };
  }

  private isPathAllowed(targetPath: string): boolean {
    const normalized = path.resolve(targetPath);

    // 차단된 경로 확인
    if (this.config.blockedPaths) {
      for (const blocked of this.config.blockedPaths) {
        if (normalized.startsWith(path.resolve(blocked))) {
          return false;
        }
      }
    }

    // 허용된 경로 확인
    if (this.config.allowedPaths && this.config.allowedPaths.length > 0) {
      for (const allowed of this.config.allowedPaths) {
        if (normalized.startsWith(path.resolve(allowed))) {
          return true;
        }
      }
      return false;
    }

    return true;
  }

  protected registerHandlers(): void {
    // 파일 읽기
    this.registerTool(
      'fs_read_file',
      'Read a file',
      {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path' },
          encoding: {
            type: 'string',
            default: 'utf-8',
            description: 'File encoding'
          }
        },
        required: ['path']
      },
      async (args) => {
        if (!this.isPathAllowed(args.path)) {
          throw new Error(`Access denied: ${args.path}`);
        }

        const stats = await fs.stat(args.path);

        if (stats.size > this.config.maxFileSize!) {
          throw new Error(`File too large: ${stats.size} bytes`);
        }

        const content = await fs.readFile(args.path, {
          encoding: args.encoding || 'utf-8'
        });

        return {
          success: true,
          content,
          size: stats.size,
          modifiedAt: stats.mtime.toISOString()
        };
      }
    );

    // 파일 쓰기
    this.registerTool(
      'fs_write_file',
      'Write content to a file',
      {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'File path' },
          content: { type: 'string', description: 'Content to write' },
          encoding: { type: 'string', default: 'utf-8' }
        },
        required: ['path', 'content']
      },
      async (args) => {
        if (!this.isPathAllowed(args.path)) {
          throw new Error(`Access denied: ${args.path}`);
        }

        // 디렉토리 생성
        await fs.mkdir(path.dirname(args.path), { recursive: true });

        await fs.writeFile(args.path, args.content, {
          encoding: args.encoding || 'utf-8'
        });

        return { success: true, path: args.path };
      }
    );

    // 디렉토리 목록
    this.registerTool(
      'fs_list_directory',
      'List directory contents',
      {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Directory path' },
          recursive: { type: 'boolean', default: false }
        },
        required: ['path']
      },
      async (args) => {
        if (!this.isPathAllowed(args.path)) {
          throw new Error(`Access denied: ${args.path}`);
        }

        const entries = await fs.readdir(args.path, { withFileTypes: true });

        const items = await Promise.all(
          entries.map(async (entry) => {
            const fullPath = path.join(args.path, entry.name);
            const stats = await fs.stat(fullPath);

            return {
              name: entry.name,
              path: fullPath,
              type: entry.isDirectory() ? 'directory' : 'file',
              size: stats.size,
              modifiedAt: stats.mtime.toISOString()
            };
          })
        );

        return { success: true, items, count: items.length };
      }
    );

    // 파일 검색 (glob)
    this.registerTool(
      'fs_glob',
      'Search files using glob pattern',
      {
        type: 'object',
        properties: {
          pattern: { type: 'string', description: 'Glob pattern' },
          cwd: { type: 'string', description: 'Working directory' }
        },
        required: ['pattern']
      },
      async (args) => {
        const cwd = args.cwd || process.cwd();

        if (!this.isPathAllowed(cwd)) {
          throw new Error(`Access denied: ${cwd}`);
        }

        const files = await glob(args.pattern, { cwd });

        return {
          success: true,
          files: files.map(f => path.join(cwd, f)),
          count: files.length
        };
      }
    );

    // 파일 삭제
    this.registerTool(
      'fs_delete',
      'Delete a file or directory',
      {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Path to delete' },
          recursive: { type: 'boolean', default: false }
        },
        required: ['path']
      },
      async (args) => {
        if (!this.isPathAllowed(args.path)) {
          throw new Error(`Access denied: ${args.path}`);
        }

        await fs.rm(args.path, { recursive: args.recursive });

        return { success: true, path: args.path };
      }
    );

    // 파일 이동/이름 변경
    this.registerTool(
      'fs_move',
      'Move or rename a file',
      {
        type: 'object',
        properties: {
          source: { type: 'string', description: 'Source path' },
          destination: { type: 'string', description: 'Destination path' }
        },
        required: ['source', 'destination']
      },
      async (args) => {
        if (!this.isPathAllowed(args.source) || !this.isPathAllowed(args.destination)) {
          throw new Error('Access denied');
        }

        await fs.rename(args.source, args.destination);

        return { success: true, from: args.source, to: args.destination };
      }
    );
  }
}

// 서버 실행
const config = process.env.MCP_SERVER_CONFIG
  ? JSON.parse(process.env.MCP_SERVER_CONFIG)
  : {};

const server = new FilesystemMCPServer(config);
server.run();
```

---

## Commands MCP Server

### 명령어 실행 서버

```typescript
// mcp-servers/commands/src/index.ts

import { BaseMCPServer } from '@anthropic-ai/mcp-shared';
import { spawn, exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface CommandsServerConfig {
  allowedCommands?: string[];
  blockedCommands?: string[];
  workingDirectory?: string;
  timeout?: number;
}

export class CommandsMCPServer extends BaseMCPServer {
  private config: CommandsServerConfig;

  constructor(config: CommandsServerConfig = {}) {
    super('commands-server', '1.0.0');
    this.config = {
      timeout: 30000,
      ...config
    };
  }

  private isCommandAllowed(command: string): boolean {
    const cmd = command.split(' ')[0];

    // 차단된 명령어 확인
    if (this.config.blockedCommands) {
      if (this.config.blockedCommands.includes(cmd)) {
        return false;
      }
    }

    // 허용된 명령어만 확인
    if (this.config.allowedCommands && this.config.allowedCommands.length > 0) {
      return this.config.allowedCommands.includes(cmd);
    }

    return true;
  }

  protected registerHandlers(): void {
    // 명령어 실행
    this.registerTool(
      'cmd_execute',
      'Execute a shell command',
      {
        type: 'object',
        properties: {
          command: { type: 'string', description: 'Command to execute' },
          cwd: { type: 'string', description: 'Working directory' },
          timeout: { type: 'number', description: 'Timeout in ms' }
        },
        required: ['command']
      },
      async (args) => {
        if (!this.isCommandAllowed(args.command)) {
          throw new Error(`Command not allowed: ${args.command}`);
        }

        const options = {
          cwd: args.cwd || this.config.workingDirectory,
          timeout: args.timeout || this.config.timeout,
          maxBuffer: 10 * 1024 * 1024 // 10MB
        };

        try {
          const { stdout, stderr } = await execAsync(args.command, options);

          return {
            success: true,
            stdout,
            stderr,
            exitCode: 0
          };
        } catch (error: any) {
          return {
            success: false,
            stdout: error.stdout || '',
            stderr: error.stderr || error.message,
            exitCode: error.code || 1
          };
        }
      }
    );

    // 스트리밍 명령어 실행
    this.registerTool(
      'cmd_spawn',
      'Spawn a command with streaming output',
      {
        type: 'object',
        properties: {
          command: { type: 'string', description: 'Command to run' },
          args: {
            type: 'array',
            items: { type: 'string' },
            description: 'Command arguments'
          },
          cwd: { type: 'string' }
        },
        required: ['command']
      },
      async (args) => {
        if (!this.isCommandAllowed(args.command)) {
          throw new Error(`Command not allowed: ${args.command}`);
        }

        return new Promise((resolve, reject) => {
          const child = spawn(args.command, args.args || [], {
            cwd: args.cwd || this.config.workingDirectory,
            shell: true
          });

          let stdout = '';
          let stderr = '';

          child.stdout.on('data', (data) => {
            stdout += data.toString();
          });

          child.stderr.on('data', (data) => {
            stderr += data.toString();
          });

          child.on('close', (code) => {
            resolve({
              success: code === 0,
              stdout,
              stderr,
              exitCode: code
            });
          });

          child.on('error', (error) => {
            reject(error);
          });

          // 타임아웃
          setTimeout(() => {
            child.kill();
            reject(new Error('Command timeout'));
          }, this.config.timeout!);
        });
      }
    );

    // 현재 디렉토리 조회
    this.registerTool(
      'cmd_pwd',
      'Get current working directory',
      { type: 'object', properties: {} },
      async () => ({
        success: true,
        cwd: this.config.workingDirectory || process.cwd()
      })
    );

    // 환경 변수 조회
    this.registerTool(
      'cmd_env',
      'Get environment variables',
      {
        type: 'object',
        properties: {
          name: { type: 'string', description: 'Variable name (optional)' }
        }
      },
      async (args) => {
        if (args.name) {
          return {
            success: true,
            name: args.name,
            value: process.env[args.name]
          };
        }

        // 민감한 환경 변수 필터링
        const safeEnv: Record<string, string> = {};
        const sensitivePatterns = ['KEY', 'SECRET', 'TOKEN', 'PASSWORD', 'CREDENTIAL'];

        for (const [key, value] of Object.entries(process.env)) {
          const isSensitive = sensitivePatterns.some(p =>
            key.toUpperCase().includes(p)
          );

          if (!isSensitive && value) {
            safeEnv[key] = value;
          }
        }

        return { success: true, env: safeEnv };
      }
    );
  }
}

// 서버 실행
const config = process.env.MCP_SERVER_CONFIG
  ? JSON.parse(process.env.MCP_SERVER_CONFIG)
  : {};

const server = new CommandsMCPServer(config);
server.run();
```

---

## MCP Client

### 클라이언트 구현

```typescript
// mcp-client/src/MCPClient.ts

import { Client } from '@modelcontextprotocol/sdk/client';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse';

export interface MCPServerConnection {
  name: string;
  client: Client;
  transport: 'stdio' | 'sse' | 'http';
  tools: Tool[];
}

export class MCPClient {
  private connections: Map<string, MCPServerConnection> = new Map();

  /**
   * stdio 서버에 연결
   */
  async connectStdio(
    name: string,
    command: string,
    args: string[] = []
  ): Promise<MCPServerConnection> {
    const transport = new StdioClientTransport({ command, args });
    return this.connect(name, transport, 'stdio');
  }

  /**
   * SSE 서버에 연결
   */
  async connectSSE(name: string, url: string): Promise<MCPServerConnection> {
    const transport = new SSEClientTransport(new URL(url));
    return this.connect(name, transport, 'sse');
  }

  private async connect(
    name: string,
    transport: any,
    type: 'stdio' | 'sse' | 'http'
  ): Promise<MCPServerConnection> {
    const client = new Client({
      name: 'ui-tars-client',
      version: '1.0.0'
    }, {
      capabilities: {}
    });

    await client.connect(transport);

    // 도구 목록 가져오기
    const toolsResponse = await client.request({
      method: 'tools/list'
    }, {});

    const tools = (toolsResponse as any).tools || [];

    const connection: MCPServerConnection = {
      name,
      client,
      transport: type,
      tools
    };

    this.connections.set(name, connection);

    return connection;
  }

  /**
   * 도구 호출
   */
  async callTool(
    toolName: string,
    arguments_: Record<string, any>
  ): Promise<any> {
    // 도구를 가진 서버 찾기
    for (const connection of this.connections.values()) {
      const tool = connection.tools.find(t => t.name === toolName);

      if (tool) {
        const response = await connection.client.request({
          method: 'tools/call',
          params: {
            name: toolName,
            arguments: arguments_
          }
        }, {});

        return this.parseToolResponse(response);
      }
    }

    throw new Error(`Tool not found: ${toolName}`);
  }

  private parseToolResponse(response: any): any {
    const content = response.content;

    if (!content || content.length === 0) {
      return null;
    }

    // 텍스트 콘텐츠 파싱
    const textContent = content.find((c: any) => c.type === 'text');

    if (textContent) {
      try {
        return JSON.parse(textContent.text);
      } catch {
        return textContent.text;
      }
    }

    // 이미지 콘텐츠
    const imageContent = content.find((c: any) => c.type === 'image');

    if (imageContent) {
      return {
        type: 'image',
        data: imageContent.data,
        mimeType: imageContent.mimeType
      };
    }

    return content;
  }

  /**
   * 모든 도구 목록
   */
  listAllTools(): Tool[] {
    const tools: Tool[] = [];

    for (const connection of this.connections.values()) {
      tools.push(...connection.tools);
    }

    return tools;
  }

  /**
   * 특정 서버의 도구 목록
   */
  listTools(serverName: string): Tool[] {
    const connection = this.connections.get(serverName);
    return connection?.tools || [];
  }

  /**
   * 연결 해제
   */
  async disconnect(name?: string): Promise<void> {
    if (name) {
      const connection = this.connections.get(name);
      if (connection) {
        await connection.client.close();
        this.connections.delete(name);
      }
    } else {
      for (const connection of this.connections.values()) {
        await connection.client.close();
      }
      this.connections.clear();
    }
  }
}
```

### Transport 구현

```typescript
// mcp-client/src/transports/InMemoryTransport.ts

/**
 * 같은 프로세스 내에서 MCP 서버와 통신
 */
export class InMemoryTransport {
  private server: any;

  constructor(server: any) {
    this.server = server;
  }

  async send(message: any): Promise<any> {
    return this.server.handleRequest(message);
  }

  close(): void {
    // 정리 작업
  }
}
```

---

## 도구 스키마 변환

### OpenAI/Anthropic 형식 변환

```typescript
// mcp-client/src/converters.ts

import { Tool } from './types';

/**
 * MCP 도구를 OpenAI 함수 형식으로 변환
 */
export function toOpenAIFunctions(tools: Tool[]): OpenAIFunction[] {
  return tools.map(tool => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema
    }
  }));
}

/**
 * MCP 도구를 Anthropic 도구 형식으로 변환
 */
export function toAnthropicTools(tools: Tool[]): AnthropicTool[] {
  return tools.map(tool => ({
    name: tool.name,
    description: tool.description,
    input_schema: tool.inputSchema
  }));
}

/**
 * OpenAI 도구 호출 결과를 MCP 형식으로 변환
 */
export function fromOpenAIToolCall(
  toolCall: OpenAIToolCall
): { name: string; arguments: any } {
  return {
    name: toolCall.function.name,
    arguments: JSON.parse(toolCall.function.arguments)
  };
}
```

---

*다음 글에서는 Context Engineering을 분석합니다.*
