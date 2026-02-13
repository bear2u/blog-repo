# divine-mobile guide: chapter map (seed)

Target repo: `https://github.com/divinevideo/divine-mobile`

Goal: turn the repo's existing docs + code structure into a Korean chapter-style guide series for this blog.

## Key source files (start here)

- `README.md`: product overview + build instructions (Flutter)
- `docs/README.md`: mobile app dev setup, build scripts, CF Stream token notes
- `docs/ARCHITECTURE.md`: Nostr SDK / layered architecture overview
- `docs/CF_STREAM_SETUP.md`: Cloudflare Stream token + upload pipeline overview
- `docs/VIDEO_UPLOAD_ARCHITECTURE-*.md`: deep dive on upload/publish architecture
- `docs/nostr_pagination_docs.md`: feed pagination behavior for kind 32222
- `docs/STARTUP_PERFORMANCE_OPTIMIZATIONS.md`, `docs/WEB_PERFORMANCE_OPTIMIZATIONS.md`: perf notes (if you include perf chapters)
- `docs/RELEASE_CHECKLIST.md`, `docs/BUILD_SCRIPTS_README.md`, `codemagic.yaml`: release/CI/CD

App code anchors (for diagrams and "where to look"):
- `mobile/lib/main.dart`: entrypoint
- `mobile/lib/providers/app_providers.dart`: service wiring (Riverpod)
- `mobile/lib/services/`: upload/feed/nostr-related services
- `mobile/lib/screens/`, `mobile/lib/widgets/`: UI surface for features described in docs

## Recommended chapter outline (adjust to doc reality)

Keep it around 8-14 chapters; merge/split based on how much the repo already documents.

1. `intro`:
   diVine(OpenVine) overview, what "Nostr-based short-form video" means, what lives in this repo.
   Sources: `README.md`

2. `getting-started`:
   Flutter install, `flutter doctor`, `mobile/` setup, basic run commands.
   Sources: `README.md`, `docs/README.md`, `mobile/pubspec.yaml`

3. `repo-structure`:
   Top-level folders (`mobile/`, `docs/`, `website/`, etc) and how work is organized.
   Sources: tree + `docs/ARCHITECTURE.md`

4. `core-architecture`:
   High-level architecture (layers, Nostr client concepts, storage/crypto).
   Sources: `docs/ARCHITECTURE.md`

5. `nostr-event-model`:
   Nostr event types used by the app; focus on kind 32222 and tags the app relies on.
   Sources: `docs/NOSTR_EVENT_TYPES.md`, `docs/nip-vine.md`

6. `video-recording-flow`:
   Recording UX and flow (segmented recording, thumbnail generation, etc) as implemented in Flutter.
   Sources: relevant `mobile/lib/screens/`, `mobile/lib/services/`

7. `video-upload-cloudflare-stream`:
   CF Stream integration, where token is injected, upload lifecycle, and "publish to Nostr" step.
   Sources: `docs/CF_STREAM_SETUP.md`, `docs/VIDEO_UPLOAD_ARCHITECTURE-*.md`
   Note: never publish real `CF_STREAM_TOKEN` values; use placeholders.

8. `feed-and-pagination`:
   How feeds are loaded/paginated, caching/preloading considerations.
   Sources: `docs/nostr_pagination_docs.md`

9. `moderation-and-reporting` (optional but common):
   Mute lists (NIP-51), reporting flow, how filtering is applied.
   Sources: `docs/MODERATION_*`

10. `testing`:
   Unit/widget/integration test strategy, common pitfalls.
   Sources: `docs/*TEST*.md`, `mobile/test/`, `mobile/integration_test/`

11. `performance` (optional):
   Startup + web perf notes; what to measure and what was optimized.
   Sources: `docs/STARTUP_PERFORMANCE_OPTIMIZATIONS.md`, `docs/WEB_PERFORMANCE_OPTIMIZATIONS.md`

12. `release-and-deploy`:
   Build scripts, codemagic, release checklist; iOS/Android/web targets.
   Sources: `docs/BUILD_SCRIPTS_README.md`, `docs/RELEASE_CHECKLIST.md`, `codemagic.yaml`

## Blog-specific defaults

- `series` slug: `divine-mobile`
- Categories/tags (suggested): `flutter`, `mobile`, `nostr`, `cloudflare-stream`
