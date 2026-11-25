# Contributing to `@yours/voice`

Thank you for contributing to `@yours/voice`! All packages under `yours-research` follow shared quality and privacy invariants defined in [`docs/ECOSYSTEM_CONTRACTS.md`](./docs/ECOSYSTEM_CONTRACTS.md).

## Principles

1. **100% Local & Zero-Cloud by Default:** Never add telemetry, remote API calls, or CDN runtime fetches.
2. **Zero Runtime Dependencies:** Keep core `@yours/voice` dependency-free so downstream apps and Chrome MV3 extensions can plug in their chosen WebGPU/WASM runners cleanly.
3. **Strict TypeScript:** All code must pass `tsc --noEmit` with `"strict": true` and `"noUncheckedIndexedAccess": true`.
4. **Test Coverage:** New DSP, VAD, STT, TTS, or MV3 adapter behavior must include deterministic Vitest coverage under `tests/`.

## Development Workflow

```bash
npm install
npm run typecheck
npm run build
npm test
```

## Pull Request Checklist

- [ ] `npm run typecheck` passes with 0 errors.
- [ ] `npm run build` emits clean JS + `.d.ts` files in `dist/`.
- [ ] `npm test` passes 100% of Vitest suites.
- [ ] Contracts in `src/contracts.ts` remain aligned with `docs/ECOSYSTEM_CONTRACTS.md`.
- [ ] Documentation (`README.md`, `ARCHITECTURE.md`) reflects any new public exports.

## License

By contributing, you agree that your contributions will be licensed under the **Apache-2.0** license (see [`LICENSE`](./LICENSE)).
