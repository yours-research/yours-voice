import { computeRms, computeZeroCrossingRate, RmsEnergyGate } from '../audio/frame-chunker.js';
import { TARGET_VOICE_SAMPLE_RATE } from '../audio/resampler.js';
import { AudioRingBuffer, concatFloat32Arrays } from '../audio/ring-buffer.js';
import type { PcmFrame, VadEvent, VadTransitionState } from '../contracts.js';

/** Pluggable neural or deterministic VAD model runner interface (e.g., Silero ONNX/WASM). */
export interface VadModelRunner {
  readonly id: string;
  init?(): Promise<void>;
  /** Return speech probability in `[0, 1]` for a single mono PCM frame (typically 512 samples @ 16 kHz). */
  processFrame(frame: Float32Array, sampleRate: number): Promise<number> | number;
  reset?(): Promise<void> | void;
  dispose?(): Promise<void> | void;
}

export interface EnergyVadRunnerOptions {
  /** Minimum linear RMS required for speech probability to rise (default 0.012). */
  minRms?: number;
  /** Minimum speech-band zero-crossing rate (default 0.01). */
  minZcr?: number;
  /** Maximum speech-band zero-crossing rate to reject pure white-noise hiss (default 0.45). */
  maxZcr?: number;
  /** Adaptive noise floor tracking (default true). */
  adaptiveNoiseFloor?: boolean;
}

/**
 * Fast deterministic energy + zero-crossing VAD fallback for zero-model unit tests
 * and CPU-constrained or degraded devices without ONNX/WebGPU weights loaded.
 */
export class EnergyZeroCrossingVadRunner implements VadModelRunner {
  readonly id = 'energy-zcr-fallback';
  private readonly gate: RmsEnergyGate;
  private readonly minRms: number;
  private readonly minZcr: number;
  private readonly maxZcr: number;

  constructor(options: EnergyVadRunnerOptions = {}) {
    this.minRms = options.minRms ?? 0.012;
    this.minZcr = options.minZcr ?? 0.01;
    this.maxZcr = options.maxZcr ?? 0.45;
    this.gate = new RmsEnergyGate({
      threshold: this.minRms,
      adaptiveNoiseFloor: options.adaptiveNoiseFloor ?? true,
      noiseFloorMultiplier: 2.4,
    });
  }

  processFrame(frame: Float32Array): number {
    if (frame.length === 0) {
      return 0;
    }

    const rms = computeRms(frame);
    const { active, effectiveThreshold } = this.gate.isActive(frame, rms);
    const zcr = computeZeroCrossingRate(frame);

    if (!active) {
      const ratio = effectiveThreshold > 0 ? rms / effectiveThreshold : 0;
      return Math.min(0.25, Math.max(0, ratio * 0.2));
    }

    const snrFactor = Math.min(1, (rms - effectiveThreshold) / (effectiveThreshold * 2.5) + 0.55);
    const zcrInSpeechBand = zcr >= this.minZcr && zcr <= this.maxZcr;
    const zcrWeight = zcrInSpeechBand ? 1.0 : 0.45;

    return Math.min(0.99, Math.max(0.05, snrFactor * zcrWeight));
  }

  reset(): void {
    this.gate.reset();
  }
}

/** Minimal ORT-compatible tensor descriptor so `@yours/voice` stays dependency-free. */
export interface SileroTensorLike<T extends Float32Array | BigInt64Array = Float32Array | BigInt64Array> {
  data: T;
  dims: readonly number[];
}

export interface SileroOrtSessionLike {
  run(
    feeds: Record<string, SileroTensorLike>,
  ): Promise<Record<string, { data: Float32Array | number[] }>>;
}

export interface SileroVadRunnerOptions {
  /** Pre-loaded ONNX Runtime Web / WASM session for Silero VAD v4 or v5. */
  session: SileroOrtSessionLike;
  /** Silero model version ('v5' uses 64-sample context prepend; 'v4' uses h/c tensors). Default 'v5'. */
  version?: 'v4' | 'v5';
  /** Optional Tensor constructor (e.g. `ort.Tensor`) if the session requires wrapped Tensor instances. */
  createTensor?: (type: 'float32' | 'int64', data: Float32Array | BigInt64Array, dims: number[]) => SileroTensorLike;
}

/**
 * Adapter wrapping an ONNX/WASM Silero VAD session (`v4` or `v5`) with automatic
 * recurrent state (`state` / `h`,`c`) and 64-sample context window management.
 */
export class SileroVadRunner implements VadModelRunner {
  readonly id: string;
  private readonly session: SileroOrtSessionLike;
  private readonly version: 'v4' | 'v5';
  private readonly createTensor: (
    type: 'float32' | 'int64',
    data: Float32Array | BigInt64Array,
    dims: number[],
  ) => SileroTensorLike;

  // Silero v5 state: [2, 1, 128]; Silero v4 h/c: [2, 1, 64]
  private state: Float32Array;
  private hState: Float32Array;
  private cState: Float32Array;
  private context = new Float32Array(64);

  constructor(options: SileroVadRunnerOptions) {
    this.session = options.session;
    this.version = options.version ?? 'v5';
    this.id = `silero-vad-${this.version}`;
    this.createTensor =
      options.createTensor ??
      ((_type, data, dims) => ({
        data,
        dims,
      }));
    this.state = new Float32Array(2 * 1 * 128);
    this.hState = new Float32Array(2 * 1 * 64);
    this.cState = new Float32Array(2 * 1 * 64);
  }

  async processFrame(frame: Float32Array, sampleRate: number): Promise<number> {
    const sr = BigInt64Array.from([BigInt(sampleRate)]);

    if (this.version === 'v5') {
      const contextSize = sampleRate === 16_000 ? 64 : 32;
      const input = new Float32Array(contextSize + frame.length);
      input.set(this.context.subarray(this.context.length - contextSize), 0);
      input.set(frame, contextSize);

      const outputs = await this.session.run({
        input: this.createTensor('float32', input, [1, input.length]),
        state: this.createTensor('float32', this.state, [2, 1, 128]),
        sr: this.createTensor('int64', sr, [1]),
      });

      this.context = input.slice(input.length - contextSize);
      const stateOut = outputs['stateN'] ?? outputs['state'];
      if (stateOut) {
        this.state = new Float32Array(stateOut.data);
      }
      const probOut = outputs['output'] ?? outputs['prob'];
      const prob = probOut?.data[0] ?? 0;
      return Math.min(1, Math.max(0, prob));
    }

    const outputs = await this.session.run({
      input: this.createTensor('float32', frame, [1, frame.length]),
      sr: this.createTensor('int64', sr, [1]),
      h: this.createTensor('float32', this.hState, [2, 1, 64]),
      c: this.createTensor('float32', this.cState, [2, 1, 64]),
    });

    if (outputs['hn']) {
      this.hState = new Float32Array(outputs['hn'].data);
    }
    if (outputs['cn']) {
      this.cState = new Float32Array(outputs['cn'].data);
    }
    const probOut = outputs['output'] ?? outputs['prob'];
    const prob = probOut?.data[0] ?? 0;
    return Math.min(1, Math.max(0, prob));
  }

  reset(): void {
    this.state.fill(0);
    this.hState.fill(0);
    this.cState.fill(0);
    this.context.fill(0);
  }
}

export interface VadEngineOptions {
  /** Model runner (defaults to `EnergyZeroCrossingVadRunner` if omitted). */
  runner?: VadModelRunner;
  /** Target sample rate in Hz (default 16000). */
  sampleRate?: number;
  /** Speech probability threshold to enter speech onset (default 0.55). */
  positiveThreshold?: number;
  /** Lower hysteresis threshold below which silence is counted (default 0.35). */
  negativeThreshold?: number;
  /** Pre-roll ring buffer duration in ms so initial consonants are never clipped (default 256). */
  preRollMs?: number;
  /** Minimum continuous speech duration in ms required to confirm onset (default 96). */
  minSpeechMs?: number;
  /** Short-pause hangover duration in ms bridging brief drops below negativeThreshold (default 320). */
  hangoverMs?: number;
  /** Consecutive silence duration in ms before finalizing utterance end (default `hangoverMs` or 480). */
  silenceEndMs?: number;
  /** Emit per-frame `'probability'` telemetry events (default false). */
  emitProbabilityEvents?: boolean;
}

/**
 * Hysteresis Voice Activity Detector with pre-roll ring buffer, onset confirmation,
 * hangover pause bridging, and utterance accumulation.
 */
export class VadEngine {
  readonly runner: VadModelRunner;
  readonly sampleRate: number;
  readonly positiveThreshold: number;
  readonly negativeThreshold: number;
  readonly preRollMs: number;
  readonly minSpeechMs: number;
  readonly hangoverMs: number;
  readonly silenceEndMs: number;
  private readonly emitProbabilityEvents: boolean;

  private readonly preRollBuffer: AudioRingBuffer;
  private vadState: VadTransitionState = 'silence';

  // Pending onset tracking
  private preRollSnapshot: Float32Array = new Float32Array(0);
  private pendingFrames: Float32Array[] = [];
  private pendingStartMs = 0;
  private pendingSpeechMs = 0;

  // Active utterance tracking
  private utteranceChunks: Float32Array[] = [];
  private utteranceStartMs = 0;
  private lastSpeechTimestampMs = 0;
  private silenceAccumMs = 0;
  private initialized = false;

  constructor(options: VadEngineOptions = {}) {
    this.runner = options.runner ?? new EnergyZeroCrossingVadRunner();
    this.sampleRate = options.sampleRate ?? TARGET_VOICE_SAMPLE_RATE;
    this.positiveThreshold = options.positiveThreshold ?? 0.55;
    this.negativeThreshold = options.negativeThreshold ?? 0.35;
    if (this.negativeThreshold > this.positiveThreshold) {
      throw new RangeError(
        `negativeThreshold (${this.negativeThreshold}) must be <= positiveThreshold (${this.positiveThreshold})`,
      );
    }
    this.preRollMs = options.preRollMs ?? 256;
    this.minSpeechMs = options.minSpeechMs ?? 96;
    this.hangoverMs = options.hangoverMs ?? 320;
    const defaultSilenceEnd = options.hangoverMs !== undefined ? options.hangoverMs : 480;
    this.silenceEndMs = Math.max(this.hangoverMs, options.silenceEndMs ?? defaultSilenceEnd);
    this.emitProbabilityEvents = options.emitProbabilityEvents ?? false;

    const preRollCapacity = Math.max(0, Math.round((this.preRollMs / 1000) * this.sampleRate));
    this.preRollBuffer = new AudioRingBuffer(preRollCapacity);
  }

  get currentState(): VadTransitionState {
    return this.vadState;
  }

  get isSpeechActive(): boolean {
    return this.vadState === 'speech' || this.vadState === 'hangover';
  }

  async init(): Promise<void> {
    if (this.initialized) {
      return;
    }
    await this.runner.init?.();
    this.initialized = true;
  }

  /**
   * Process a single fixed-size PCM frame and return any VAD transition events emitted.
   */
  async processFrame(frame: PcmFrame): Promise<VadEvent[]> {
    if (!this.initialized) {
      await this.init();
    }

    const probability = await this.runner.processFrame(frame.samples, frame.sampleRate);
    const events: VadEvent[] = [];

    if (this.emitProbabilityEvents) {
      events.push({
        type: 'probability',
        timestampMs: frame.timestampMs,
        probability,
        vadState: this.vadState,
      });
    }

    switch (this.vadState) {
      case 'silence': {
        if (probability >= this.positiveThreshold) {
          this.preRollSnapshot = this.preRollBuffer.readAll();
          this.pendingFrames = [new Float32Array(frame.samples)];
          this.pendingStartMs = frame.timestampMs;
          this.pendingSpeechMs = frame.durationMs;

          if (this.pendingSpeechMs >= this.minSpeechMs) {
            events.push(this.confirmSpeechStart(probability, frame.timestampMs, frame.durationMs));
          } else {
            this.vadState = 'pending-speech';
          }
        } else {
          this.preRollBuffer.push(frame.samples);
        }
        break;
      }

      case 'pending-speech': {
        if (probability >= this.negativeThreshold) {
          this.pendingFrames.push(new Float32Array(frame.samples));
          if (probability >= this.positiveThreshold) {
            this.pendingSpeechMs += frame.durationMs;
          }

          if (this.pendingSpeechMs >= this.minSpeechMs) {
            events.push(this.confirmSpeechStart(probability, frame.timestampMs, frame.durationMs));
          }
        } else {
          // Impulse dropped below negativeThreshold before minSpeechMs: push frames into pre-roll & abort.
          if (this.preRollSnapshot.length > 0) {
            this.preRollBuffer.push(this.preRollSnapshot);
          }
          for (const pending of this.pendingFrames) {
            this.preRollBuffer.push(pending);
          }
          this.preRollBuffer.push(frame.samples);
          this.preRollSnapshot = new Float32Array(0);
          this.pendingFrames = [];
          this.pendingSpeechMs = 0;
          this.vadState = 'silence';
        }
        break;
      }

      case 'speech': {
        const copy = new Float32Array(frame.samples);
        this.utteranceChunks.push(copy);
        events.push({
          type: 'speech_frame',
          timestampMs: frame.timestampMs,
          probability,
          audio: copy,
        });

        if (probability < this.negativeThreshold) {
          this.vadState = 'hangover';
          this.silenceAccumMs = frame.durationMs;
          if (this.silenceAccumMs >= this.silenceEndMs) {
            events.push(this.finalizeSpeechEnd(frame.timestampMs + frame.durationMs));
          }
        } else {
          this.lastSpeechTimestampMs = frame.timestampMs + frame.durationMs;
        }
        break;
      }

      case 'hangover': {
        const copy = new Float32Array(frame.samples);
        this.utteranceChunks.push(copy);
        events.push({
          type: 'speech_frame',
          timestampMs: frame.timestampMs,
          probability,
          audio: copy,
        });

        if (probability >= this.positiveThreshold) {
          this.vadState = 'speech';
          this.silenceAccumMs = 0;
          this.lastSpeechTimestampMs = frame.timestampMs + frame.durationMs;
        } else {
          this.silenceAccumMs += frame.durationMs;
          if (this.silenceAccumMs >= this.silenceEndMs) {
            events.push(this.finalizeSpeechEnd(frame.timestampMs + frame.durationMs));
          }
        }
        break;
      }
    }

    return events;
  }

  /** Force-end any active utterance (e.g., on `stopListening()` or stream EOF). */
  flush(): VadEvent[] {
    if (this.vadState === 'speech' || this.vadState === 'hangover') {
      return [this.finalizeSpeechEnd(this.lastSpeechTimestampMs)];
    }
    this.resetStateOnly();
    return [];
  }

  async reset(): Promise<void> {
    this.resetStateOnly();
    await this.runner.reset?.();
  }

  async dispose(): Promise<void> {
    this.resetStateOnly();
    await this.runner.dispose?.();
    this.initialized = false;
  }

  private confirmSpeechStart(
    probability: number,
    frameTimestampMs: number,
    frameDurationMs: number,
  ): VadEvent {
    const preRollAudio = concatFloat32Arrays([this.preRollSnapshot, ...this.pendingFrames]);
    this.preRollBuffer.clear();
    this.preRollSnapshot = new Float32Array(0);
    this.pendingFrames = [];
    this.pendingSpeechMs = 0;

    this.vadState = 'speech';
    this.utteranceStartMs = this.pendingStartMs;
    this.lastSpeechTimestampMs = frameTimestampMs + frameDurationMs;
    this.silenceAccumMs = 0;
    this.utteranceChunks = preRollAudio.length > 0 ? [preRollAudio] : [];

    return {
      type: 'speech_start',
      timestampMs: this.pendingStartMs,
      probability,
      preRollAudio,
    };
  }

  private finalizeSpeechEnd(endMs: number): VadEvent {
    const audio = concatFloat32Arrays(this.utteranceChunks);
    const startMs = this.utteranceStartMs;
    const durationMs = Math.max(0, endMs - startMs);

    this.utteranceChunks = [];
    this.silenceAccumMs = 0;
    this.vadState = 'silence';
    this.preRollBuffer.clear();

    return {
      type: 'speech_end',
      startMs,
      endMs,
      durationMs,
      audio,
    };
  }

  private resetStateOnly(): void {
    this.vadState = 'silence';
    this.preRollBuffer.clear();
    this.preRollSnapshot = new Float32Array(0);
    this.pendingFrames = [];
    this.pendingStartMs = 0;
    this.pendingSpeechMs = 0;
    this.utteranceChunks = [];
    this.utteranceStartMs = 0;
    this.lastSpeechTimestampMs = 0;
    this.silenceAccumMs = 0;
  }
}
