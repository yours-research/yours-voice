# Yours Research — Ecosystem Architecture & Shared TypeScript Contracts (v1.0)

All open-source SDKs under [`yours-research`](https://github.com/yours-research) inter-operate through these shared TypeScript contracts so **`yours-browser-extension`** (Chromium MV3 + native Rust WASM / WGSL WebGPU runtime) and **`yours.sh` web demos** compose seamlessly without schema drift.

---

## 1. Universal Package Standards

- **Scope & Licensing:** `@yours/<name>`, **`Apache-2.0`** (`LICENSE` + `package.json` `"license": "Apache-2.0"`).
- **Module Format & Toolchain:**
  - ESM `"type": "module"`, TypeScript strict (`"strict": true`, `"moduleResolution": "Bundler"`, declaration `.d.ts` + `.js` build outputs under `dist/`).
  - Node `>= 20.0.0`, `vitest` (`^3.0.0`) for automated testing (`npm run typecheck`, `npm run build`, `npm test`).
  - Clean `.github/workflows/ci.yml` running typecheck, build, and tests on every push and PR.
- **Privacy Invariant:** **Zero external network requests by default.** Any optional web capability (`search_web_tab`, `fetch_background_tab`) must declare `networkHint: true` and gate cleanly behind explicit host permission / policy approval.
- **Documentation & Visual Identity Standard:**
  - Root `README.md` MUST embed the adaptive banner via `<picture>` pointing to `https://raw.githubusercontent.com/yours-research/<repo>/main/assets/hero-dark.png` (`prefers-color-scheme: dark`) and `https://raw.githubusercontent.com/yours-research/<repo>/main/assets/hero-light.png` (`prefers-color-scheme: light`), wrapped in `<a href="https://yours.sh">`.
  - Include concise top links (`yours.sh`, `GitHub Org`, `Hugging Face`, `X (@yoursdotsh)`), human-readable overview, architecture Mermaid/ASCII diagram, quickstart code snippets, API reference, sibling ecosystem table (`yours-tools`, `yours-harness`, `yours-memory`, `yours-models`, `yours-guard`, `yours-voice`, `yours-evals`, `yours-browser-extension`), and footer `yours / research · Making local inference for everyone.`.
  - Include `ARCHITECTURE.md` and `CONTRIBUTING.md` in every repo.

---

## 2. Canonical Tool Schema v1.1 (`@yours/tools` ↔ `yours-research-lab` Pragma Training)

Unifies the model-training schemas (`read_active_tab`, `list_tabs`, `read_tab`, `search_web_tab`, `fetch_background_tab`) with expanded runtime tools (`interact_page`, `run_sandbox`, `read_document`, `memory_recall`, `memory_save`) and retains canonical aliases for backwards compatibility (`read_active_page`, `list_open_tabs`, `web_search`, `fetch_and_read_url`, `js_calc`).

```ts
export type CanonicalToolName =
  | 'read_active_tab'
  | 'list_tabs'
  | 'read_tab'
  | 'interact_page'
  | 'run_sandbox'
  | 'read_document'
  | 'search_web_tab'
  | 'fetch_background_tab'
  | 'memory_recall'
  | 'memory_save';

export interface ToolSafetyHints {
  readOnlyHint: boolean;          // True for pure reads/compute; false for state mutations
  consequentialHint: boolean;     // True for form submit, click/type side-effects, memory writes, external requests
  networkHint: boolean;           // True only if live internet navigation/search is invoked
  untrustedContentHint: boolean;  // True when output embeds external/DOM/PDF text (requires IDPI spotlighting)
}

export interface ToolDefinition {
  name: CanonicalToolName;
  aliases?: string[];
  description: string;
  parameters: {
    type: 'object';
    properties: Record<string, unknown>;
    required?: string[];
    additionalProperties?: false;
  };
  safety: ToolSafetyHints;
}

export interface ToolInvocationContext {
  environment: 'chrome-mv3' | 'web-demo' | 'node-test';
  sessionId: string;
  origin?: string;
  abortSignal?: AbortSignal;
}

export interface ToolExecutionResult<TData = unknown> {
  ok: boolean;
  toolName: CanonicalToolName;
  data?: TData;
  formatted: string;
  tainted: boolean;               // Propagated to @yours/guard
  sourceOrigin?: string;
  error?: string;
  needsPermission?: boolean;
  requiresApproval?: boolean;
  blockedByCaptcha?: boolean;
  manualUrl?: string;
}
```

---

## 3. Pluggable Local Runtime Contract (`@yours/harness`)

Supports the custom **Rust WASM + WGSL WebGPU runtime**, **Transformers.js / ORT WebGPU**, **Chrome Built-in AI (Prompt API)**, **WebLLM**, and **OpenAI-compatible / Ollama / `llama.cpp` HTTP endpoints** behind one adapter interface, including optional KV-cache prefix slot primitives:

```ts
export type WgslComputeTier =
  | 'tier1-f16-subgroups'
  | 'tier2-f16-shared'
  | 'tier3-f32-universal';

export type GpuPacingMode = 'max-speed' | 'balanced' | 'smooth-ui';

export interface HarnessMessage {
  id: string;
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  toolName?: string;
  toolCallId?: string;
  toolArgs?: Record<string, unknown>;
  tainted?: boolean;
  createdAt: number;
}

export interface GenerateRequest {
  messages: HarnessMessage[];
  tools?: ToolDefinition[];
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  enableThinking?: boolean;
  pacingMode?: GpuPacingMode;
  prefixSlotKey?: string;
}

export type StreamDelta =
  | { type: 'text'; delta: string }
  | { type: 'reasoning'; delta: string }
  | { type: 'tool_call_start'; id: string; name: string }
  | { type: 'tool_call_args'; id: string; delta: string }
  | { type: 'tool_call_ready'; id: string; name: string; arguments: Record<string, unknown> }
  | { type: 'telemetry'; ttftMs?: number; tokensPerSec?: number; kvTokens?: number; speculativeAcceptRate?: number }
  | { type: 'done'; finishReason: 'stop' | 'length' | 'tool_calls' | 'abort' };

export interface RuntimeCapabilities {
  id: string;
  maxContextTokens: number;
  supportsStreaming: boolean;
  supportsPrefixSlots: boolean;
  supportsTools: boolean;
  tier?: WgslComputeTier | 'cpu' | 'http';
}

export interface RuntimeAdapter {
  capabilities(): RuntimeCapabilities | Promise<RuntimeCapabilities>;
  generate(request: GenerateRequest, signal?: AbortSignal): AsyncIterable<StreamDelta>;
  prefillSlot?(slotKey: string, messages: HarnessMessage[], signal?: AbortSignal): Promise<{ tokens: number; elapsedMs: number }>;
  evictSlot?(slotKey: string): Promise<void>;
  countTokens(messages: HarnessMessage[]): Promise<number> | number;
}
```

---

## 4. Chrome Local Memory Contract (`@yours/memory`)

```ts
export type MemoryTier = 'working' | 'profile' | 'episodic' | 'document';

export interface MemoryRecordInput {
  tier: MemoryTier;
  content: string;
  sourceUrl?: string;
  sourceTitle?: string;
  threadId?: string;
  tags?: string[];
  importance?: number; // [0..1]
  tainted?: boolean;
  supersedesId?: string;
}

export interface MemoryRecord extends MemoryRecordInput {
  id: string;
  createdAt: number;
  updatedAt: number;
  lastAccessedAt: number;
  accessCount: number;
  supersededById?: string;
}

export interface MemoryQuery {
  query: string;
  tiers?: MemoryTier[];
  limit?: number;
  minScore?: number;
  nowMs?: number;
}

export interface MemoryHit {
  record: MemoryRecord;
  score: number;
  bm25Rank?: number;
  vectorRank?: number;
  recencyWeight: number;
}

export interface MemoryProvider {
  remember(input: MemoryRecordInput): Promise<MemoryRecord>;
  recall(query: MemoryQuery): Promise<MemoryHit[]>;
  consolidate(threadId: string, messages: HarnessMessage[]): Promise<{ added: number; superseded: number }>;
  inspect(filter?: { tier?: MemoryTier; limit?: number; offset?: number }): Promise<MemoryRecord[]>;
  forget(id: string): Promise<boolean>;
  exportEncrypted(passphrase: string): Promise<Uint8Array>;
  importEncrypted(blob: Uint8Array, passphrase: string): Promise<{ imported: number }>;
  wipeAll(): Promise<void>;
}
```

---

## 5. Browser Security & IDPI Guard Contract (`@yours/guard`)

```ts
export interface SanitizedContent {
  cleanText: string;
  spotlightedPromptBlock: string;
  strippedSignals: string[]; // e.g., ['hidden-css', 'zero-width-unicode', 'svg-foreign-text', 'hashjack']
  injectionRiskScore: number; // [0..1]
  flaggedPatterns: string[];
  tainted: true;
}

export type GuardVerdict = 'allow' | 'require_approval' | 'deny';

export interface GuardDecision {
  verdict: GuardVerdict;
  reason: string;
  triggeredRule?: string;
  lethalTrifectaRisk?: boolean;
}

export interface GuardPolicyContext {
  tool: ToolDefinition;
  arguments: Record<string, unknown>;
  contextHasUntrustedContent: boolean;
  sessionAccessedPrivateData: boolean;
  activeOrigin?: string;
  explicitlyApprovedOrigins?: string[];
}
```

---

## 6. Verified Model Delivery & Hardware Prober Contract (`@yours/models`)

```ts
export interface ModelArtifactEntry {
  path: string;           // e.g. 'slab0.part00.bin'
  sizeBytes: number;
  sha256: string;         // hex digest
  chunkSizeBytes?: number;
  chunkSha256?: string[]; // 8–16 MiB range verification
  urls: string[];         // Primary HTTP/3 CDN + multi-origin mirrors
}

export interface ModelManifestV1 {
  schemaVersion: 1;
  modelId: string;        // e.g. 'yours-research/Pragma-1-4B-WGSL'
  version: string;
  architecture: 'qwen35-deltanet' | 'qwen3-gqa' | 'gemma4' | 'custom';
  quantization: 'q4f16' | 'q4' | 'fp16';
  totalBytes: number;
  merkleRoot: string;
  artifacts: ModelArtifactEntry[];
  canonicalPayload: string;
  signatureBase64: string; // Ed25519 signature over canonicalPayload
}

export interface HardwareProfile {
  hasWebGpu: boolean;
  tier: WgslComputeTier | 'cpu-only';
  adapterVendor?: string;
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
  estimatedVramGB?: number;
  recommendedModelId: string; // e.g. 'Pragma-1-4B' | 'Pragma-1-2B' | 'Pragma-1-Nano'
}
```

---

## 7. Local Voice Pipeline Contract (`@yours/voice`)

```ts
export type VoiceState = 'idle' | 'listening' | 'speech-detected' | 'transcribing' | 'speaking';

export interface TranscriptSegment {
  text: string;
  isFinal: boolean;
  startMs: number;
  endMs: number;
  confidence?: number;
}

export interface VoicePipeline {
  readonly state: VoiceState;
  startListening(options?: { language?: string; bargeInEnabled?: boolean }): Promise<void>;
  stopListening(): Promise<void>;
  speak(textStream: string | AsyncIterable<string>, options?: { voiceId?: string; speed?: number }): Promise<void>;
  bargeIn(): Promise<void>;
  dispose(): Promise<void>;
}
```

---

## 8. Cross-Stack Eval Scorecard Contract (`@yours/evals`)

```ts
export type EvalSuiteName = 'tools' | 'harness' | 'memory' | 'guard' | 'voice' | 'all';

export interface EvalCaseResult {
  id: string;
  suite: Exclude<EvalSuiteName, 'all'>;
  passed: boolean;
  score: number;        // [0..1]
  latencyMs: number;
  metrics: Record<string, number | boolean | string>;
  error?: string;
}

export interface EvalScorecard {
  schemaVersion: 1;
  suite: EvalSuiteName;
  timestampIso: string;
  summary: {
    total: number;
    passed: number;
    failed: number;
    aggregateScore: number;
    suiteMetrics: Record<string, number>;
  };
  cases: EvalCaseResult[];
}
```
