# Architecture — `@yours/voice`

`@yours/voice` is structured into six focused, zero-dependency ESM modules that implement the `@yours/voice` contract in [`docs/ECOSYSTEM_CONTRACTS.md`](./docs/ECOSYSTEM_CONTRACTS.md).

## Module Overview

1. **`src/contracts.ts`** — Canonical types (`VoicePipeline`, `VoiceState`, `TranscriptSegment`) plus transport-neutral audio boundaries (`AudioInputSource`, `AudioPlaybackSink`, `PcmFrame`, `TtsAudioChunk`, `VadEvent`).
2. **`src/audio/`** — Real-time signal processing:
   - `PcmResampler`: Blackman-windowed sinc polyphase FIR downsampler (48 kHz / 44.1 kHz → 16 kHz mono) with anti-aliasing low-pass scaling and stateful overlap history across `AudioWorklet` 128-sample blocks.
   - `PcmFrameChunker`: Buffers arbitrary-sized resampled blocks into fixed 512-sample (32 ms @ 16 kHz) frames for Silero VAD v5.
   - `AudioRingBuffer`: Fixed-capacity circular sample buffer preserving pre-onset audio (`preRollMs`).
   - `RmsEnergyGate`: Adaptive noise-floor RMS energy gate.
3. **`src/vad/`** — Voice Activity Detection:
   - `VadEngine`: Hysteresis finite-state machine (`silence` → `pending-speech` → `speech` → `hangover`) using `positiveThreshold`, `negativeThreshold`, `minSpeechMs`, `hangoverMs`, and `silenceEndMs`. Snapshots `AudioRingBuffer` on onset so initial consonants (`/s/`, `/p/`, `/t/`, `/k/`) are included in `speech_start.preRollAudio`.
   - `SileroVadRunner`: Dependency-free wrapper for ORT-Web / WASM Silero v4 & v5 sessions.
   - `EnergyZeroCrossingVadRunner`: Fast deterministic fallback for zero-model tests and CPU-constrained devices.
4. **`src/stt/`** — Speech-to-Text:
   - `WhisperSttRunner` & `MoonshineSttRunner` adapters.
   - `SttEngine`: Throttles streaming partial `TranscriptSegment` emissions (`partialThrottleMs`) for smooth UI updates, honors `AbortSignal` cancellation, and applies `HallucinationGuard` to filter silence phantom phrases and consecutive n-gram repetition loops.
5. **`src/tts/`** — Streaming Text-to-Speech:
   - `SentenceStreamChunker`: Incrementally chunks LLM token streams on sentence and long-clause boundaries while protecting abbreviations (`Dr.`, `e.g.`, `U.S.`), version numbers (`v1.2.3`), decimals (`3.14`), and domains (`yours.sh`).
   - `TtsEngine` & `KokoroTtsRunner`: Bounded lookahead synthesis queue (`lookaheadChunks`) paired with FIFO `AudioPlaybackSink` playback and synchronous `<30ms` `bargeIn()` cancellation.
6. **`src/adapters/` & `src/pipeline/`** — Environment Bridges & Unified FSM:
   - `WebDemoAudioAdapter`: `getUserMedia` + `Blob`-URL `AudioWorkletNode` capture + Web Audio `AudioBufferSourceNode` playback.
   - `Mv3MicPermissionHelper`, `Mv3OffscreenHost`, `Mv3SidePanelClient`: Solves Chromium MV3's Offscreen `getUserMedia` prompt restriction via extension-origin onboarding (`iframe` / `permissions.html` tab) and bridges capture/playback over `Mv3RpcTransport` with Side Panel Web Audio fallback.
   - `LocalVoicePipeline`: Coordinates full-duplex listening + speaking, automatically triggering `bargeIn()` on `speech_start` during playback and immediately handing pre-roll + speech frames to STT.

## Full-Duplex Barge-In Sequence

```
Speaking (TTS playing chunk N, synthesizing chunk N+1)
   │
   ├─► Mic AudioWorklet → 16 kHz Resampler → 512-sample Frame → VadEngine
   │                                                               │
   │◄── bargeIn() (<30ms): abort AbortController, stopAll() sink ◄─┴─ speech_start (with pre-roll)
   │
   └─► State: speaking ─► speech-detected ─► transcribing ─► listening
```
