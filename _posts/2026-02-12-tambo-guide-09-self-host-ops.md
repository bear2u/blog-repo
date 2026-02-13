---
layout: post
title: "Tambo 완벽 가이드 (09) - Self-host 및 운영 가이드"
date: 2026-02-12
permalink: /tambo-guide-09-self-host-ops/
author: Tambo AI
categories: [AI 에이전트, 웹 개발]
tags: [Tambo, Self-host, Docker, PostgreSQL, Operations]
original_url: "https://github.com/tambo-ai/tambo"
excerpt: "Docker로 Tambo 운영하기"
---

## 운영 구성(Operators Guide)

OPERATORS.md는 Tambo 운영을 위해 아래 서비스를 기준으로 설명합니다.

- Web(Next.js) 기본 포트 3210
- API(NestJS) 기본 포트 3211
- PostgreSQL 17 기본 포트 5433

에는 부가적으로 **MinIO(S3 호환 스토리지)**도 포함되어 있습니다(9000/9001).

---

## Docker Quick Start(OPERATORS.md)

[0;34m🚀 Tambo Docker Setup[0m
[0;34mThis script will help you set up Tambo for self-hosting with Docker[0m
[0;34m📁 Working directory: /home/blog/tambo[0m

[0;34m✅ Prerequisites check passed![0m

[0;34m✅ docker.env created successfully![0m
[0;32m✅ Setup completed successfully![0m

[0;34m📋 Next steps:[0m
1. [1;33mEdit docker.env[0m with your actual values:
   - Update passwords and secrets
   - Add your API keys (OpenAI, etc.)
   - Configure other settings as needed

2. [1;33mBuild the containers:[0m
   ./scripts/cloud/tambo-build.sh

3. [1;33mStart the stack:[0m
   ./scripts/cloud/tambo-start.sh

4. [1;33mInitialize the database:[0m
   ./scripts/cloud/init-database.sh

5. [1;33mAccess your applications:[0m
   - Tambo Web: http://localhost:3210
   - Tambo API: http://localhost:3211
   - PostgreSQL Database: localhost:5433

[1;33m💡 Note: This script requires bash (macOS/Linux/WSL). Windows CMD or PowerShell will not work.[0m
[1;33m💡 For help, run: ./scripts/cloud/tambo-logs.sh --help[0m
[0;34m🚀 Starting Tambo Docker Stack...[0m
[0;34m📁 Working directory: /home/blog/tambo[0m
[0;34m🔗 Creating Docker network...[0m
fab47003af27e5eec9ea149da2714048e82ed7c1d24b5f3bc7157401c862d06c
[0;34m📦 Pulling latest images...[0m
[0;34m🎯 Starting Tambo services with BuildKit...[0m
#1 [internal] load local bake definitions
#1 reading from stdin 997B done
#1 DONE 0.0s

#2 [web internal] load build definition from Dockerfile
#2 transferring dockerfile:
#2 transferring dockerfile: 2.24kB 0.0s done
#2 DONE 0.2s

#3 [api internal] load build definition from Dockerfile
#3 transferring dockerfile: 2.02kB done
#3 DONE 0.2s

#4 [web internal] load metadata for docker.io/library/node:22-alpine
#4 DONE 0.9s

#5 [web internal] load .dockerignore
#5 transferring context: 1.06kB done
#5 DONE 0.1s

#6 [web internal] load build context
#6 DONE 0.0s

#7 [web base 1/3] FROM docker.io/library/node:22-alpine@sha256:e4bf2a82ad0a4037d28035ae71529873c069b13eb0455466ae0bc13363826e34
#7 resolve docker.io/library/node:22-alpine@sha256:e4bf2a82ad0a4037d28035ae71529873c069b13eb0455466ae0bc13363826e34
#7 resolve docker.io/library/node:22-alpine@sha256:e4bf2a82ad0a4037d28035ae71529873c069b13eb0455466ae0bc13363826e34 0.1s done
#7 DONE 0.5s

#8 [api internal] load build context
#8 transferring context: 6.28MB 0.4s done
#8 DONE 0.6s

#6 [web internal] load build context
#6 transferring context: 58.58MB 2.0s done
#6 DONE 2.3s

#7 [api base 1/3] FROM docker.io/library/node:22-alpine@sha256:e4bf2a82ad0a4037d28035ae71529873c069b13eb0455466ae0bc13363826e34
#7 sha256:fc1c5222d85fe45cd255019912158424888b168c8413fb42c8e166c37f1833eb 447B / 447B 0.2s done
#7 sha256:2b752f7c71fd1a08980fdf09b7379d8304c8ef2569526934a2089ed26d771778 1.26MB / 1.26MB 0.4s done
#7 sha256:589002ba0eaed121a1dbf42f6648f29e5be55d5c8a6ee0f8eaa0285cc21ac153 3.86MB / 3.86MB 0.4s done
#7 sha256:8d513d1f314d3646adaf156be912f8de408c740e43d90cad5ce06b9de27e7bdf 51.60MB / 51.60MB 1.6s done
#7 extracting sha256:589002ba0eaed121a1dbf42f6648f29e5be55d5c8a6ee0f8eaa0285cc21ac153 0.5s done
#7 extracting sha256:8d513d1f314d3646adaf156be912f8de408c740e43d90cad5ce06b9de27e7bdf
#7 extracting sha256:8d513d1f314d3646adaf156be912f8de408c740e43d90cad5ce06b9de27e7bdf 2.0s done
#7 extracting sha256:8d513d1f314d3646adaf156be912f8de408c740e43d90cad5ce06b9de27e7bdf 2.0s done
#7 extracting sha256:2b752f7c71fd1a08980fdf09b7379d8304c8ef2569526934a2089ed26d771778 0.1s done
#7 extracting sha256:fc1c5222d85fe45cd255019912158424888b168c8413fb42c8e166c37f1833eb 0.1s done
#7 extracting sha256:fc1c5222d85fe45cd255019912158424888b168c8413fb42c8e166c37f1833eb 0.1s done
#7 DONE 4.5s

#9 [web base 2/3] RUN npm install -g npm@^11
#9 27.00 
#9 27.00 removed 57 packages, and changed 101 packages in 26s
#9 27.00 
#9 27.00 15 packages are looking for funding
#9 27.00   run `npm fund` for details
#9 DONE 27.3s

#10 [api base 3/3] RUN apk add --no-cache gcompat dumb-init
#10 1.237 (1/4) Installing dumb-init (1.2.5-r3)
#10 1.264 (2/4) Installing musl-obstack (1.2.3-r2)
#10 1.282 (3/4) Installing libucontext (1.3.3-r0)
#10 1.299 (4/4) Installing gcompat (1.1.0-r4)
#10 1.318 Executing busybox-1.37.0-r30.trigger
#10 1.330 OK: 11.1 MiB in 22 packages
#10 DONE 1.4s

#11 [web installer 1/5] WORKDIR /workspace
#11 DONE 0.2s

#12 [api stage-3 1/4] WORKDIR /app
#12 DONE 0.2s

#13 [api stage-3 2/4] RUN addgroup -g 1001 nodejs   && adduser -u 1001 -G nodejs -s /bin/sh -D nodejs   && chown -R nodejs:nodejs ./
#13 ...

#14 [web installer 2/5] COPY turbo.json package.json package-lock.json ./
#14 DONE 0.3s

#15 [web installer 2/6] COPY turbo.json package.json package-lock.json ./
#15 CACHED

#16 [web installer 3/6] COPY react-sdk ./react-sdk
#16 DONE 0.2s

#17 [api installer 3/5] COPY packages ./packages
#17 DONE 0.4s

#18 [web installer 4/6] COPY packages ./packages
#18 ...

#13 [web stage-3 2/4] RUN addgroup -g 1001 nodejs   && adduser -u 1001 -G nodejs -s /bin/sh -D nodejs   && chown -R nodejs:nodejs ./
#13 DONE 0.7s

#18 [web installer 4/6] COPY packages ./packages
#18 DONE 0.4s

#19 [api installer 4/5] COPY apps/api ./apps/api
#19 DONE 0.3s

#20 [api installer 5/5] RUN npx turbo prune @tambo-ai-cloud/api --docker
#20 ...

#21 [web installer 5/6] COPY apps/web ./apps/web
#21 DONE 0.5s

#22 [web installer 6/6] RUN npx turbo prune @tambo-ai-cloud/web --docker
#22 1.954 npm warn exec The following package was not found and will be installed: turbo@2.8.7
#22 3.802  WARNING  No locally installed `turbo` found in your repository. Using globally installed version (2.8.7), which can cause unexpected behavior.
#22 3.802 
#22 3.802 Installing the version in your repository (^2.8.3) before calling `turbo` will result in more predictable behavior across environments.
#22 3.810 
#22 3.810 Attention:
#22 3.810 Turborepo now collects completely anonymous telemetry regarding usage.
#22 3.810 This information is used to shape the Turborepo roadmap and prioritize features.
#22 3.810 You can learn more, including how to opt-out if you'd not like to participate in this anonymous program, by visiting the following URL:
#22 3.810 https://turborepo.dev/docs/telemetry
#22 3.810 
#22 3.810 • turbo 2.8.7
#22 4.080 Generating pruned monorepo for @tambo-ai-cloud/web in /workspace/out
#22 ...

#20 [api installer 5/5] RUN npx turbo prune @tambo-ai-cloud/api --docker
#20 1.973 npm warn exec The following package was not found and will be installed: turbo@2.8.7
#20 3.838  WARNING  No locally installed `turbo` found in your repository. Using globally installed version (2.8.7), which can cause unexpected behavior.
#20 3.838 
#20 3.838 Installing the version in your repository (^2.8.3) before calling `turbo` will result in more predictable behavior across environments.
#20 3.847 
#20 3.847 Attention:
#20 3.847 Turborepo now collects completely anonymous telemetry regarding usage.
#20 3.847 This information is used to shape the Turborepo roadmap and prioritize features.
#20 3.847 You can learn more, including how to opt-out if you'd not like to participate in this anonymous program, by visiting the following URL:
#20 3.847 https://turborepo.dev/docs/telemetry
#20 3.847 
#20 3.848 • turbo 2.8.7
#20 4.061 Generating pruned monorepo for @tambo-ai-cloud/api in /workspace/out
#20 4.084  - Added @tambo-ai-cloud/api
#20 4.090  - Added @tambo-ai-cloud/backend
#20 4.095  - Added @tambo-ai-cloud/core
#20 4.123  - Added @tambo-ai-cloud/db
#20 4.124  - Added @tambo-ai/eslint-config
#20 4.125  - Added @tambo-ai/typescript-config
#20 DONE 4.6s

#22 [web installer 6/6] RUN npx turbo prune @tambo-ai-cloud/web --docker
#22 4.093  - Added @tambo-ai-cloud/backend
#22 4.101  - Added @tambo-ai-cloud/core
#22 4.126  - Added @tambo-ai-cloud/db
#22 ...

#23 [api builder 2/5] COPY --from=installer /workspace/out/json/ .
#23 DONE 0.2s

#22 [web installer 6/6] RUN npx turbo prune @tambo-ai-cloud/web --docker
#22 4.236  - Added @tambo-ai-cloud/web
#22 4.237  - Added @tambo-ai/eslint-config
#22 4.249  - Added @tambo-ai/react
#22 4.256  - Added @tambo-ai/react-ui-base
#22 4.256  - Added @tambo-ai/typescript-config
#22 4.270  - Added @tambo-ai/ui-registry
#22 4.271  - Added @tambo-ai/vite-config
#22 DONE 4.6s

#24 [api builder 3/5] RUN npm ci
#24 ...

#25 [web builder 2/5] COPY --from=installer /workspace/out/json/ .
#25 DONE 0.2s

#26 [web builder 3/5] RUN npm ci
#26 ...

#24 [api builder 3/5] RUN npm ci
#24 101.4 
#24 101.4 > @tambo-ai/repo@1.0.0 prepare
#24 101.4 > husky
#24 101.4 
#24 101.5 .git can't be found
#24 101.6 added 1837 packages, and audited 1844 packages in 2m
#24 101.6 
#24 101.6 279 packages are looking for funding
#24 101.6   run `npm fund` for details
#24 101.8 
#24 101.8 33 vulnerabilities (5 low, 6 moderate, 22 high)
#24 101.8 
#24 101.8 To address issues that do not require attention, run:
#24 101.8   npm audit fix
#24 101.8 
#24 101.8 To address all issues (including breaking changes), run:
#24 101.8   npm audit fix --force
#24 101.8 
#24 101.8 Run `npm audit` for details.
#24 DONE 103.4s

#26 [web builder 3/5] RUN npm ci
#26 ...

#27 [api builder 4/5] COPY --from=installer /workspace/out/full/ .
#27 DONE 2.6s

#26 [web builder 3/5] RUN npm ci
#26 ...

#28 [api builder 5/5] RUN --mount=type=secret,id=TURBO_TOKEN,env=TURBO_TOKEN   npx turbo run build --filter="@tambo-ai-cloud/api"
#28 1.110 • turbo 2.8.3
#28 1.243 • Packages in scope: @tambo-ai-cloud/api
#28 1.243 • Running build in 1 packages
#28 1.243 • Remote caching disabled
#28 1.420 @tambo-ai-cloud/backend:build: cache miss, executing 107ee8644dd5141e
#28 1.798 @tambo-ai-cloud/backend:build: 
#28 1.798 @tambo-ai-cloud/backend:build: > @tambo-ai-cloud/backend@0.0.6 build
#28 1.798 @tambo-ai-cloud/backend:build: > tsc
#28 1.798 @tambo-ai-cloud/backend:build: 
#28 13.95 @tambo-ai-cloud/api:build: cache miss, executing 9f906fcb18b1542d
#28 14.26 @tambo-ai-cloud/api:build: npm warn config ignoring workspace config at /workspace/apps/api/.npmrc
#28 14.32 @tambo-ai-cloud/api:build: 
#28 14.32 @tambo-ai-cloud/api:build: > @tambo-ai-cloud/api@0.143.0 build
#28 14.32 @tambo-ai-cloud/api:build: > NODE_OPTIONS=--experimental-require-module nest build
#28 14.32 @tambo-ai-cloud/api:build: 
#28 ...

#26 [web builder 3/5] RUN npm ci
#26 133.4 npm warn deprecated mathjax-full@3.2.2: Version 4 replaces this package with the scoped package @mathjax/src
#26 139.1 
#26 139.1 > @tambo-ai/repo@1.0.0 prepare
#26 139.1 > husky
#26 139.1 
#26 139.2 .git can't be found
#26 139.4 added 2559 packages, and audited 2570 packages in 2m
#26 139.4 
#26 139.4 573 packages are looking for funding
#26 139.4   run `npm fund` for details
#26 139.6 
#26 139.6 44 vulnerabilities (5 low, 17 moderate, 22 high)
#26 139.6 
#26 139.6 To address issues that do not require attention, run:
#26 139.6   npm audit fix
#26 139.6 
#26 139.6 To address all issues (including breaking changes), run:
#26 139.6   npm audit fix --force
#26 139.6 
#26 139.6 Run `npm audit` for details.
#26 DONE 143.5s

#28 [api builder 5/5] RUN --mount=type=secret,id=TURBO_TOKEN,env=TURBO_TOKEN   npx turbo run build --filter="@tambo-ai-cloud/api"
#28 ...

#29 [web builder 4/5] COPY --from=installer /workspace/out/full/ .
#29 DONE 5.0s

#28 [api builder 5/5] RUN --mount=type=secret,id=TURBO_TOKEN,env=TURBO_TOKEN   npx turbo run build --filter="@tambo-ai-cloud/api"
#28 ...

#30 [web builder 5/5] RUN --mount=type=secret,id=TURBO_TOKEN,env=TURBO_TOKEN   npx turbo run build --filter="@tambo-ai-cloud/web"
#30 ...

#28 [api builder 5/5] RUN --mount=type=secret,id=TURBO_TOKEN,env=TURBO_TOKEN   npx turbo run build --filter="@tambo-ai-cloud/api"
#28 44.12 
#28 44.12  Tasks:    2 successful, 2 total
#28 44.12 Cached:    0 cached, 2 total
#28 44.12   Time:    42.993s 
#28 44.12 
#28 DONE 44.5s

#30 [web builder 5/5] RUN --mount=type=secret,id=TURBO_TOKEN,env=TURBO_TOKEN   npx turbo run build --filter="@tambo-ai-cloud/web"
#30 1.583 • turbo 2.8.3
#30 1.876 • Packages in scope: @tambo-ai-cloud/web
#30 1.876 • Running build in 1 packages
#30 1.876 • Remote caching disabled
#30 2.548 @tambo-ai/vite-config:build: cache miss, executing 1af2930b34894511
#30 2.556 @tambo-ai/react:build: cache miss, executing b938d6404aa2ac3f
#30 2.569 @tambo-ai-cloud/backend:build: cache miss, executing 107ee8644dd5141e
#30 2.997 @tambo-ai/react:build: 
#30 2.997 @tambo-ai/react:build: > @tambo-ai/react@1.0.1 build
#30 2.997 @tambo-ai/react:build: > npm run build:cjs && npm run build:esm
#30 2.997 @tambo-ai/react:build: 
#30 3.050 @tambo-ai-cloud/backend:build: 
#30 3.050 @tambo-ai-cloud/backend:build: > @tambo-ai-cloud/backend@0.0.6 build
#30 3.050 @tambo-ai-cloud/backend:build: > tsc
#30 3.050 @tambo-ai-cloud/backend:build: 
#30 3.106 @tambo-ai/vite-config:build: 
#30 3.106 @tambo-ai/vite-config:build: > @tambo-ai/vite-config@0.0.1 build
#30 3.106 @tambo-ai/vite-config:build: > tsdown
#30 3.106 @tambo-ai/vite-config:build: 
#30 3.460 @tambo-ai/vite-config:build: ℹ tsdown v0.12.9 powered by rolldown v1.0.0-rc.3
#30 3.502 @tambo-ai/react:build: 
#30 3.502 @tambo-ai/react:build: > @tambo-ai/react@1.0.1 build:cjs
#30 3.502 @tambo-ai/react:build: > tsc -p tsconfig.cjs.json
#30 3.502 @tambo-ai/react:build: 
#30 3.758 @tambo-ai/vite-config:build: ℹ Using tsdown config: /workspace/packages/vite-config/tsdown.config.ts
#30 3.796 @tambo-ai/vite-config:build: ℹ entry: src/index.ts
#30 3.796 @tambo-ai/vite-config:build: ℹ tsconfig: tsconfig.json
#30 3.796 @tambo-ai/vite-config:build: ℹ Build start
#30 4.650 @tambo-ai/vite-config:build: [33mWarning: Invalid input options (1 issue found)
#30 4.650 @tambo-ai/vite-config:build: - For the "define". Invalid key: Expected never but received "define". [0m
#30 14.63 @tambo-ai/vite-config:build: ℹ dist/index.js                       1.45 kB │ gzip: 0.68 kB
#30 14.63 @tambo-ai/vite-config:build: ℹ dist/index.js.map                   2.27 kB │ gzip: 1.00 kB
#30 14.63 @tambo-ai/vite-config:build: ℹ dist/plugins/tamboDtsPlugin.js.map  1.96 kB │ gzip: 0.85 kB
#30 14.63 @tambo-ai/vite-config:build: ℹ dist/options.js.map                 1.47 kB │ gzip: 0.71 kB
#30 14.63 @tambo-ai/vite-config:build: ℹ dist/plugins/tamboDtsPlugin.js      1.03 kB │ gzip: 0.51 kB
#30 14.63 @tambo-ai/vite-config:build: ℹ dist/options.js                     0.34 kB │ gzip: 0.25 kB
#30 14.64 @tambo-ai/vite-config:build: ℹ dist/index.d.ts.map                 0.19 kB │ gzip: 0.16 kB
#30 14.65 @tambo-ai/vite-config:build: ℹ dist/options.d.ts.map               0.13 kB │ gzip: 0.13 kB
#30 14.65 @tambo-ai/vite-config:build: ℹ dist/index.d.ts                     0.57 kB │ gzip: 0.33 kB
#30 14.65 @tambo-ai/vite-config:build: ℹ dist/options.d.ts                   0.92 kB │ gzip: 0.43 kB
#30 14.65 @tambo-ai/vite-config:build: ℹ 10 files, total: 10.33 kB
#30 14.65 @tambo-ai/vite-config:build: [33m[PLUGIN_TIMINGS] Warning:[0m Your build spent significant time in plugin `rolldown-plugin-dts:resolver`. See https://rolldown.rs/options/checks#plugintimings for more details.
#30 14.65 @tambo-ai/vite-config:build: 
#30 14.66 @tambo-ai/vite-config:build: ✔ Build complete in 10857ms
#30 23.40 @tambo-ai/react:build: 
#30 23.40 @tambo-ai/react:build: > @tambo-ai/react@1.0.1 build:esm
#30 23.40 @tambo-ai/react:build: > tsc -p tsconfig.esm.json && tsc-esm-fix --target=esm
#30 23.40 @tambo-ai/react:build: 
#30 34.10 @tambo-ai/react-ui-base:build: cache miss, executing b96f0772eff3b627
#30 34.44 @tambo-ai/react-ui-base:build: 
#30 34.44 @tambo-ai/react-ui-base:build: > @tambo-ai/react-ui-base@0.1.0-alpha.4 build
#30 34.44 @tambo-ai/react-ui-base:build: > vite build
#30 34.44 @tambo-ai/react-ui-base:build: 
#30 39.22 @tambo-ai/react-ui-base:build: vite v7.3.1 building client environment for production...
#30 49.76 @tambo-ai/react-ui-base:build: transforming...
#30 50.40 @tambo-ai/react-ui-base:build: ✓ 45 modules transformed.
#30 50.49 @tambo-ai/react-ui-base:build: rendering chunks...
#30 50.59 @tambo-ai/react-ui-base:build: 
#30 50.59 @tambo-ai/react-ui-base:build: [vite:dts] Start generate declaration files...
#30 50.59 @tambo-ai/react-ui-base:build: 
#30 50.59 @tambo-ai/react-ui-base:build: [vite:dts] Start generate declaration files...
#30 50.59 @tambo-ai/react-ui-base:build: computing gzip size...
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/message-input/constants.js                                      0.19 kB │ gzip: 0.17 kB │ map:  0.54 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/toolcall-info/root/get-tool-call-request.js                     0.20 kB │ gzip: 0.17 kB │ map:  0.86 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/index.js                                                        0.46 kB │ gzip: 0.22 kB │ map:  0.09 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/toolcall-info/root/toolcall-info-context.js                     0.47 kB │ gzip: 0.26 kB │ map:  1.56 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/use-render/use-render.js                                        0.48 kB │ gzip: 0.24 kB │ map:  1.26 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/message-input/message-input-context.js                          0.48 kB │ gzip: 0.26 kB │ map:  5.69 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/reasoning-info/root/reasoning-info-context.js                   0.51 kB │ gzip: 0.27 kB │ map:  1.41 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/message-input/message-input-toolbar.js                          0.51 kB │ gzip: 0.31 kB │ map:  1.06 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/message-input/message-input-value-access.js                     0.51 kB │ gzip: 0.28 kB │ map:  1.84 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/toolcall-info/root/get-tool-status-message.js                   0.56 kB │ gzip: 0.28 kB │ map:  1.40 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/message/root/message-root-context.js                            0.62 kB │ gzip: 0.29 kB │ map:  1.77 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/reasoning-info/index.js                                         0.64 kB │ gzip: 0.24 kB │ map:  1.25 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/toolcall-info/status-text/toolcall-info-status-text.js          0.69 kB │ gzip: 0.36 kB │ map:  1.13 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/message/loading-indicator/message-loading-indicator.js          0.75 kB │ gzip: 0.37 kB │ map:  1.48 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/message/rendered-component/rendered-component-content.js        0.81 kB │ gzip: 0.40 kB │ map:  1.41 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/toolcall-info/tool-name/toolcall-info-tool-name.js              0.82 kB │ gzip: 0.41 kB │ map:  1.40 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/reasoning-info/steps/reasoning-info-steps.js                    0.85 kB │ gzip: 0.43 kB │ map:  1.76 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/message/root/message-root.js                                    0.86 kB │ gzip: 0.42 kB │ map:  2.13 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/message/rendered-component/rendered-component.js                0.87 kB │ gzip: 0.44 kB │ map:  1.76 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/toolcall-info/parameters/toolcall-info-parameters.js            0.91 kB │ gzip: 0.45 kB │ map:  1.58 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/message/index.js                                                0.96 kB │ gzip: 0.29 kB │ map:  1.67 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/toolcall-info/content/toolcall-info-content.js                  0.96 kB │ gzip: 0.48 kB │ map:  2.09 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/toolcall-info/trigger/toolcall-info-trigger.js                  0.97 kB │ gzip: 0.47 kB │ map:  1.55 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/reasoning-info/trigger/reasoning-info-trigger.js                0.97 kB │ gzip: 0.47 kB │ map:  1.85 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/toolcall-info/index.js                                          0.98 kB │ gzip: 0.31 kB │ map:  2.19 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/message/images/message-images.js                                0.99 kB │ gzip: 0.48 kB │ map:  2.53 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/message-input/message-input-content.js                          1.05 kB │ gzip: 0.48 kB │ map:  2.43 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/message-input/message-input-error.js                            1.06 kB │ gzip: 0.49 kB │ map:  2.34 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/toolcall-info/status-icon/toolcall-info-status-icon.js          1.07 kB │ gzip: 0.49 kB │ map:  1.89 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/reasoning-info/status-text/reasoning-info-status-text.js        1.11 kB │ gzip: 0.52 kB │ map:  2.24 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/reasoning-info/content/reasoning-info-content.js                1.12 kB │ gzip: 0.54 kB │ map:  2.35 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/toolcall-info/result/toolcall-info-result.js                    1.18 kB │ gzip: 0.52 kB │ map:  2.41 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/message-input/index.js                                          1.25 kB │ gzip: 0.39 kB │ map:  5.32 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/utils/check-has-content.js                                      1.32 kB │ gzip: 0.51 kB │ map:  3.97 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/utils/message-content.js                                        1.53 kB │ gzip: 0.67 kB │ map:  4.20 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/message/content/message-content.js                              1.55 kB │ gzip: 0.61 kB │ map:  3.78 kB
#30 50.71 @tambo-ai/react-ui-base:build: dist/esm/message-input/message-input-submit-button.js                    1.56 kB │ gzip: 0.64 kB │ map:  3.11 kB
#30 50.72 @tambo-ai/react-ui-base:build: dist/esm/message/rendered-component/rendered-component-canvas-button.js  1.74 kB │ gzip: 0.71 kB │ map:  3.16 kB
#30 50.72 @tambo-ai/react-ui-base:build: dist/esm/message-input/message-input-staged-images.js                    1.92 kB │ gzip: 0.76 kB │ map:  4.27 kB
#30 50.72 @tambo-ai/react-ui-base:build: dist/esm/message-input/message-input-textarea.js                         1.95 kB │ gzip: 0.71 kB │ map:  5.48 kB
#30 50.72 @tambo-ai/react-ui-base:build: dist/esm/message-input/message-input-file-button.js                      2.33 kB │ gzip: 0.90 kB │ map:  4.00 kB
#30 50.72 @tambo-ai/react-ui-base:build: dist/esm/toolcall-info/root/toolcall-info-root.js                        3.14 kB │ gzip: 1.05 kB │ map:  5.17 kB
#30 50.72 @tambo-ai/react-ui-base:build: dist/esm/reasoning-info/root/reasoning-info-root.js                      3.58 kB │ gzip: 1.21 kB │ map:  7.19 kB
#30 50.72 @tambo-ai/react-ui-base:build: dist/esm/message-input/use-combined-lists.js                             4.16 kB │ gzip: 1.14 kB │ map:  7.97 kB
#30 50.72 @tambo-ai/react-ui-base:build: dist/esm/message-input/message-input-root.js                             7.31 kB │ gzip: 2.01 kB │ map: 11.27 kB
#30 50.78 @tambo-ai/react-ui-base:build: [vite:dts] Declaration files built in 6066ms.
#30 50.78 @tambo-ai/react-ui-base:build: 
#30 50.79 @tambo-ai/react-ui-base:build: [vite:dts] Declaration files built in 4867ms.
#30 50.79 @tambo-ai/react-ui-base:build: 
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/toolcall-info/root/get-tool-call-request.cjs                     0.31 kB │ gzip: 0.23 kB │ map:  0.87 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message-input/constants.cjs                                      0.31 kB │ gzip: 0.23 kB │ map:  0.55 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/use-render/use-render.cjs                                        0.57 kB │ gzip: 0.29 kB │ map:  1.26 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message-input/message-input-value-access.cjs                     0.66 kB │ gzip: 0.35 kB │ map:  1.84 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/toolcall-info/root/get-tool-status-message.cjs                   0.69 kB │ gzip: 0.35 kB │ map:  1.43 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/index.cjs                                                        0.69 kB │ gzip: 0.30 kB │ map:  0.10 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message/root/message-root-context.cjs                            0.76 kB │ gzip: 0.35 kB │ map:  1.77 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/reasoning-info/index.cjs                                         0.85 kB │ gzip: 0.32 kB │ map:  1.35 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/toolcall-info/root/toolcall-info-context.cjs                     1.13 kB │ gzip: 0.52 kB │ map:  1.59 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message-input/message-input-context.cjs                          1.14 kB │ gzip: 0.51 kB │ map:  5.71 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message-input/message-input-toolbar.cjs                          1.16 kB │ gzip: 0.56 kB │ map:  1.08 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/reasoning-info/root/reasoning-info-context.cjs                   1.17 kB │ gzip: 0.52 kB │ map:  1.45 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message/index.cjs                                                1.18 kB │ gzip: 0.38 kB │ map:  1.81 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/toolcall-info/index.cjs                                          1.26 kB │ gzip: 0.40 kB │ map:  2.36 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/toolcall-info/status-text/toolcall-info-status-text.cjs          1.36 kB │ gzip: 0.63 kB │ map:  1.15 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message/loading-indicator/message-loading-indicator.cjs          1.44 kB │ gzip: 0.62 kB │ map:  1.59 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message/rendered-component/rendered-component-content.cjs        1.47 kB │ gzip: 0.64 kB │ map:  1.46 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/toolcall-info/tool-name/toolcall-info-tool-name.cjs              1.49 kB │ gzip: 0.68 kB │ map:  1.42 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/reasoning-info/steps/reasoning-info-steps.cjs                    1.52 kB │ gzip: 0.69 kB │ map:  1.86 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message/rendered-component/rendered-component.cjs                1.54 kB │ gzip: 0.69 kB │ map:  1.82 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message/root/message-root.cjs                                    1.54 kB │ gzip: 0.67 kB │ map:  2.22 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message-input/index.cjs                                          1.56 kB │ gzip: 0.47 kB │ map:  5.51 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/toolcall-info/parameters/toolcall-info-parameters.cjs            1.59 kB │ gzip: 0.71 kB │ map:  1.60 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/toolcall-info/content/toolcall-info-content.cjs                  1.63 kB │ gzip: 0.73 kB │ map:  2.12 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/toolcall-info/trigger/toolcall-info-trigger.cjs                  1.64 kB │ gzip: 0.73 kB │ map:  1.58 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/reasoning-info/trigger/reasoning-info-trigger.cjs                1.64 kB │ gzip: 0.72 kB │ map:  1.94 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message/images/message-images.cjs                                1.69 kB │ gzip: 0.74 kB │ map:  2.64 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message-input/message-input-error.cjs                            1.72 kB │ gzip: 0.74 kB │ map:  2.37 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message-input/message-input-content.cjs                          1.72 kB │ gzip: 0.73 kB │ map:  2.46 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/toolcall-info/status-icon/toolcall-info-status-icon.cjs          1.75 kB │ gzip: 0.75 kB │ map:  1.92 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/reasoning-info/status-text/reasoning-info-status-text.cjs        1.79 kB │ gzip: 0.78 kB │ map:  2.36 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/reasoning-info/content/reasoning-info-content.cjs                1.80 kB │ gzip: 0.78 kB │ map:  2.45 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/toolcall-info/result/toolcall-info-result.cjs                    1.85 kB │ gzip: 0.77 kB │ map:  2.43 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/utils/check-has-content.cjs                                      1.94 kB │ gzip: 0.76 kB │ map:  4.00 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/utils/message-content.cjs                                        2.18 kB │ gzip: 0.91 kB │ map:  4.24 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message-input/message-input-submit-button.cjs                    2.25 kB │ gzip: 0.89 kB │ map:  3.13 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message/content/message-content.cjs                              2.27 kB │ gzip: 0.87 kB │ map:  3.94 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message/rendered-component/rendered-component-canvas-button.cjs  2.45 kB │ gzip: 0.95 kB │ map:  3.23 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message-input/message-input-textarea.cjs                         2.65 kB │ gzip: 0.97 kB │ map:  5.55 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message-input/message-input-staged-images.cjs                    2.65 kB │ gzip: 1.02 kB │ map:  4.29 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message-input/message-input-file-button.cjs                      3.07 kB │ gzip: 1.15 kB │ map:  4.03 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/toolcall-info/root/toolcall-info-root.cjs                        3.94 kB │ gzip: 1.30 kB │ map:  5.18 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/reasoning-info/root/reasoning-info-root.cjs                      4.38 kB │ gzip: 1.46 kB │ map:  7.37 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message-input/use-combined-lists.cjs                             4.87 kB │ gzip: 1.37 kB │ map:  8.00 kB
#30 50.90 @tambo-ai/react-ui-base:build: dist/cjs/message-input/message-input-root.cjs                             8.15 kB │ gzip: 2.26 kB │ map: 11.30 kB
#30 50.90 @tambo-ai/react-ui-base:build: ✓ built in 11.63s
#30 51.05 @tambo-ai-cloud/web:build: cache miss, executing 21e461732a2845b3
#30 51.38 @tambo-ai-cloud/web:build: 
#30 51.38 @tambo-ai-cloud/web:build: > @tambo-ai-cloud/web@0.132.1 build
#30 51.38 @tambo-ai-cloud/web:build: > SKIP_ENV_VALIDATION=true next build --no-lint
#30 51.38 @tambo-ai-cloud/web:build: 
#30 52.65 @tambo-ai-cloud/web:build:  ⚠ Linting is disabled.
#30 58.87 @tambo-ai-cloud/web:build: [@sentry/nextjs] DEPRECATION WARNING: automaticVercelMonitors is deprecated and will be removed in a future version. Use webpack.automaticVercelMonitors instead.
#30 59.18 @tambo-ai-cloud/web:build:    ▲ Next.js 15.5.12
#30 59.18 @tambo-ai-cloud/web:build:    - Experiments (use with caution):
#30 59.18 @tambo-ai-cloud/web:build:      · clientTraceMetadata
#30 59.18 @tambo-ai-cloud/web:build:      · optimizePackageImports
#30 59.18 @tambo-ai-cloud/web:build:      ✓ webpackMemoryOptimizations
#30 59.18 @tambo-ai-cloud/web:build: 
#30 59.33 @tambo-ai-cloud/web:build:    Creating an optimized production build ...
#30 66.45 @tambo-ai-cloud/web:build: - [33mwarn[0m [nextra] Init git repository failed Discover git repo from [/workspace/apps/web] failed: could not find repository at '/workspace/apps/web'; class=Repository (6); code=NotFound (-3)
#30 ...

#31 [api stage-3 3/4] COPY --from=builder /workspace/      ./
#31 DONE 55.8s

#30 [web builder 5/5] RUN --mount=type=secret,id=TURBO_TOKEN,env=TURBO_TOKEN   npx turbo run build --filter="@tambo-ai-cloud/web"
#30 ...

#32 [api stage-3 4/4] COPY                ./scripts/cloud  ./scripts/cloud
#32 DONE 1.3s

#30 [web builder 5/5] RUN --mount=type=secret,id=TURBO_TOKEN,env=TURBO_TOKEN   npx turbo run build --filter="@tambo-ai-cloud/web"
#30 ...

#33 [api] exporting to image
#33 exporting layers
#33 ...

#30 [web builder 5/5] RUN --mount=type=secret,id=TURBO_TOKEN,env=TURBO_TOKEN   npx turbo run build --filter="@tambo-ai-cloud/web"
#30 199.8 @tambo-ai-cloud/web:build: <w> [webpack.cache.PackFileCacheStrategy/webpack.FileSystemInfo] Parsing of /workspace/apps/web/node_modules/jiti/lib/jiti.mjs for build dependencies failed at 'import(id)'.
#30 199.8 @tambo-ai-cloud/web:build: <w> Build dependencies behind this expression are ignored and might cause incorrect cache invalidation.
#30 ...

#33 [api] exporting to image
#33 exporting layers 96.4s done
#33 exporting manifest sha256:0faa3e94f6ba39656e57b37c476168dae000b6ac506afa225af4127c68c0a72c 0.2s done
#33 exporting config sha256:8529ad71f9b82e7a3c716f85a1b7c7d5342ccd963277ee9ff8aa6bb143787ea2
#33 exporting config sha256:8529ad71f9b82e7a3c716f85a1b7c7d5342ccd963277ee9ff8aa6bb143787ea2 0.1s done
#33 exporting attestation manifest sha256:780e63163110e37288098970b5a2303665030570de4a6082617b2ea6ef033128
#33 exporting attestation manifest sha256:780e63163110e37288098970b5a2303665030570de4a6082617b2ea6ef033128 0.2s done
#33 exporting manifest list sha256:59ffe21adba2e1d4ee1916989df74777875d2535d2bc7451f57584d52fcc8629 0.1s done
#33 naming to docker.io/library/tambo-api:latest 0.0s done
#33 unpacking to docker.io/library/tambo-api:latest
#33 ...

#30 [web builder 5/5] RUN --mount=type=secret,id=TURBO_TOKEN,env=TURBO_TOKEN   npx turbo run build --filter="@tambo-ai-cloud/web"
#30 244.5 @tambo-ai-cloud/web:build: <w> [webpack.cache.PackFileCacheStrategy/webpack.FileSystemInfo] Parsing of /workspace/apps/web/node_modules/jiti/lib/jiti.mjs for build dependencies failed at 'import(id)'.
#30 244.5 @tambo-ai-cloud/web:build: <w> Build dependencies behind this expression are ignored and might cause incorrect cache invalidation.
#30 ...

#33 [api] exporting to image
#33 unpacking to docker.io/library/tambo-api:latest 31.9s done
#33 DONE 129.1s

#30 [web builder 5/5] RUN --mount=type=secret,id=TURBO_TOKEN,env=TURBO_TOKEN   npx turbo run build --filter="@tambo-ai-cloud/web"
#30 ...

#34 [api] resolving provenance for metadata file
#34 DONE 0.2s

#30 [web builder 5/5] RUN --mount=type=secret,id=TURBO_TOKEN,env=TURBO_TOKEN   npx turbo run build --filter="@tambo-ai-cloud/web"
#30 336.6 @tambo-ai-cloud/web:build: <w> [webpack.cache.PackFileCacheStrategy/webpack.FileSystemInfo] Parsing of /workspace/apps/web/node_modules/jiti/lib/jiti.mjs for build dependencies failed at 'import(id)'.
#30 336.6 @tambo-ai-cloud/web:build: <w> Build dependencies behind this expression are ignored and might cause incorrect cache invalidation.
#30 366.6 @tambo-ai-cloud/web:build:  ✓ Compiled successfully in 5.1min
#30 366.9 @tambo-ai-cloud/web:build:    Checking validity of types ...
#30 422.1 @tambo-ai-cloud/web:build:    Collecting page data ...
#30 424.8 @tambo-ai-cloud/web:build: No email provider configured, but RESEND_API_KEY is not set. Please set RESEND_API_KEY to use email authentication.
#30 425.0 @tambo-ai-cloud/web:build: No email provider configured, but RESEND_API_KEY is not set. Please set RESEND_API_KEY to use email authentication.
#30 425.2 @tambo-ai-cloud/web:build:  ⚠ Using edge runtime on a page currently disables static generation for that page
#30 425.6 @tambo-ai-cloud/web:build: No email provider configured, but RESEND_API_KEY is not set. Please set RESEND_API_KEY to use email authentication.
#30 439.4 @tambo-ai-cloud/web:build:    Generating static pages (0/30) ...
#30 443.7 @tambo-ai-cloud/web:build:    Generating static pages (7/30) 
#30 443.7 @tambo-ai-cloud/web:build:    Generating static pages (14/30) 
#30 443.7 @tambo-ai-cloud/web:build: No email provider configured, but RESEND_API_KEY is not set. Please set RESEND_API_KEY to use email authentication.
#30 443.7 @tambo-ai-cloud/web:build: No email provider configured, but RESEND_API_KEY is not set. Please set RESEND_API_KEY to use email authentication.
#30 443.7 @tambo-ai-cloud/web:build: No email provider configured, but RESEND_API_KEY is not set. Please set RESEND_API_KEY to use email authentication.
#30 443.7 @tambo-ai-cloud/web:build:    Generating static pages (22/30) 
#30 444.0 @tambo-ai-cloud/web:build:  ✓ Generating static pages (30/30)
#30 447.1 @tambo-ai-cloud/web:build:    Finalizing page optimization ...
#30 447.1 @tambo-ai-cloud/web:build:    Collecting build traces ...
#30 507.4 @tambo-ai-cloud/web:build: 
#30 507.5 @tambo-ai-cloud/web:build: Route (app)                                 Size  First Load JS
#30 507.5 @tambo-ai-cloud/web:build: ┌ ƒ /                                    8.35 kB         719 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /_not-found                            347 B         286 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /[projectId]                         17.6 kB         870 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /[projectId]/observability           8.39 kB        1.16 MB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /[projectId]/settings                19.7 kB         838 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /analytics                           9.88 kB         407 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /api/auth/[...nextauth]                348 B         286 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /api/contacts                          348 B         286 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /api/github-stars                      349 B         286 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /api/send-founder-email                348 B         286 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /api/slack                             348 B         286 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /api/subscribe                         348 B         286 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /blog                                44.8 kB        4.18 MB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /blog/posts/ai-powered-spreadsheet   1.16 kB         332 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /blog/posts/llm-web-apps             1.16 kB         332 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /blog/posts/mcp-sampling-support     1.74 kB         333 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /blog/posts/tambo-hack               1.42 kB         332 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /blog/posts/tambo-with-tambo         1.47 kB         332 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /blog/posts/what-is-generative-ui    1.73 kB         333 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /cli-sessions                        3.32 kB         356 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /demo                                7.99 kB        1.24 MB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /device                              3.81 kB         370 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /internal/smoketest                  5.99 kB        3.67 MB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /internal/smoketest/v1               6.84 kB        3.12 MB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /legal-acceptance                    6.24 kB         369 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /login                               13.4 kB         546 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /oauth/callback                        348 B         286 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /opengraph-image                       348 B         286 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ○ /robots.txt                            348 B         286 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ○ /sitemap.xml                           348 B         286 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /slack                               4.07 kB         340 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /subscribe                           16.4 kB        1.24 MB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /trpc/[trpc]                          2.1 kB         299 kB
#30 507.5 @tambo-ai-cloud/web:build: ├ ƒ /twitter-image                         347 B         286 kB
#30 507.5 @tambo-ai-cloud/web:build: └ ƒ /unauthorized                          348 B         288 kB
#30 507.5 @tambo-ai-cloud/web:build: + First Load JS shared by all             286 kB
#30 507.5 @tambo-ai-cloud/web:build:   ├ chunks/336-35b3de1d5eec9c45.js        126 kB
#30 507.5 @tambo-ai-cloud/web:build:   ├ chunks/59c6eb5a-9a37422658d7ec81.js  37.2 kB
#30 507.5 @tambo-ai-cloud/web:build:   ├ chunks/87c73c54-258fee39c8665e96.js  54.4 kB
#30 507.5 @tambo-ai-cloud/web:build:   ├ chunks/main-app-e4f9d78de0c612f6.js  60.7 kB
#30 507.5 @tambo-ai-cloud/web:build:   └ other shared chunks (total)          7.53 kB
#30 507.5 @tambo-ai-cloud/web:build: 
#30 507.5 @tambo-ai-cloud/web:build: 
#30 507.5 @tambo-ai-cloud/web:build: ○  (Static)   prerendered as static content
#30 507.5 @tambo-ai-cloud/web:build: ƒ  (Dynamic)  server-rendered on demand
#30 507.5 @tambo-ai-cloud/web:build: 
#30 509.4 
#30 509.4  Tasks:    5 successful, 5 total
#30 509.4 Cached:    0 cached, 5 total
#30 509.4   Time:    8m27.671s 
#30 509.4 
#30 DONE 518.1s

#35 [web stage-3 3/5] COPY --from=builder /workspace/apps/web/.next/standalone  ./
#35 DONE 2.9s

#36 [web stage-3 4/5] COPY --from=builder /workspace/apps/web/.next/static      ./apps/web/.next/static
#36 DONE 0.8s

#37 [web stage-3 5/5] COPY --from=builder /workspace/apps/web/public            ./apps/web/public
#37 DONE 0.6s

#38 [web] exporting to image
#38 exporting layers
#38 exporting layers 19.9s done
#38 exporting manifest sha256:58771373d599cc0bc5c2d02de5c42b2593ac3c32c8093f501134359c8ebdf922 0.1s done
#38 exporting config sha256:bd9b7633125063943b7895a4195430156ba46699534c25b96e3bac758bb5c1b0 0.1s done
#38 exporting attestation manifest sha256:7e2effca37930cb158ed080b732ab58df39104f78c87907caae2e52a403494be
#38 exporting attestation manifest sha256:7e2effca37930cb158ed080b732ab58df39104f78c87907caae2e52a403494be 0.2s done
#38 exporting manifest list sha256:a4877d172367c8f2ccbcdef0366e0423321ebae6dc5e63acd04677dd57146f51
#38 exporting manifest list sha256:a4877d172367c8f2ccbcdef0366e0423321ebae6dc5e63acd04677dd57146f51 0.1s done
#38 naming to docker.io/library/tambo-web:latest done
#38 unpacking to docker.io/library/tambo-web:latest
#38 unpacking to docker.io/library/tambo-web:latest 3.5s done
#38 DONE 23.9s

#39 [web] resolving provenance for metadata file
#39 DONE 0.0s
[1;33m⏳ Waiting for PostgreSQL to start...[0m

---

## 최소 필수 환경변수(OPERATORS.md)

에 최소로 필요한 값들이 명시되어 있습니다. 대표적으로:

- 
-  (32자 이상)
-  (32자 이상)
- 
- 

---

## 운영 팁

- OAuth/이메일 로그인 설정을 하지 않으면 대시보드 로그인 자체가 막힐 수 있습니다(OPERATORS.md).
- HTTPS 종료는 reverse proxy(nginx/Caddy/Traefik)를 두는 형태를 권장합니다.

---

## 마무리

Tambo는 "LLM + UI"를 제품에서 현실적으로 운영하기 위한 구성 요소를 꽤 넓게 제공합니다.
- 프론트엔드(React)에서는 컴포넌트를 등록하고 UI 경험을 설계
- 백엔드에서는 상태/에이전트 루프/통합(MCP)과 운영(Cloud or self-host)을 처리

다음 단계:
- Docs: https://docs.tambo.co
- Repo: https://github.com/tambo-ai/tambo
