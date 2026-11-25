import type { AudioPlaybackSink, TtsAudioChunk, VoiceSpeakOptions } from '../contracts.js';
import { chunkTextStream, type SentenceChunkerOptions } from './sentence-chunker.js';

export const KOKORO_DEFAULT_SAMPLE_RATE = 24_000;

export interface TtsSynthesizeOptions {
  voiceId?: string;
  speed?: number;
  chunkIndex: number;
}

export interface TtsVoiceInfo {
  id: string;
  name: string;
  language: string;
  gender?: 'female' | 'male' | 'neutral';
}

/** Pluggable local TTS model runner interface (Kokoro-82M WebGPU/WASM, etc.). */
export interface TtsModelRunner {
  readonly id: string;
  readonly sampleRate: number;
  init?(): Promise<void>;
  synthesize(
    text: string,
    options: TtsSynthesizeOptions,
    signal?: AbortSignal,
  ): Promise<Float32Array | TtsAudioChunk>;
  listVoices?(): Promise<TtsVoiceInfo[]> | TtsVoiceInfo[];
  dispose?(): Promise<void>;
}

export type KokoroInferFn = (
  text: string,
  options: { voiceId: string; speed: number; chunkIndex: number },
  signal?: AbortSignal,
) => Promise<Float32Array | { audio: Float32Array; sampleRate?: number }>;

export const DEFAULT_KOKORO_VOICES: readonly TtsVoiceInfo[] = [
  { id: 'af_heart', name: 'Heart', language: 'en-US', gender: 'female' },
  { id: 'af_bella', name: 'Bella', language: 'en-US', gender: 'female' },
  { id: 'af_nicole', name: 'Nicole', language: 'en-US', gender: 'female' },
  { id: 'am_adam', name: 'Adam', language: 'en-US', gender: 'male' },
  { id: 'am_michael', name: 'Michael', language: 'en-US', gender: 'male' },
  { id: 'bf_emma', name: 'Emma', language: 'en-GB', gender: 'female' },
  { id: 'bm_george', name: 'George', language: 'en-GB', gender: 'male' },
];

/**
 * Adapter wrapping a local Kokoro-82M WebGPU/WASM runner callback with voice preset
 * defaults, speed clamping `[0.5, 2.0]`, and 24 kHz audio normalization.
 */
export class KokoroTtsRunner implements TtsModelRunner {
  readonly id: string;
  readonly sampleRate: number;
  private readonly inferFn: KokoroInferFn;
  private readonly defaultVoiceId: string;
  private readonly initFn?: () => Promise<void>;
  private readonly disposeFn?: () => Promise<void>;

  constructor(options: {
    id?: string;
    sampleRate?: number;
    defaultVoiceId?: string;
    infer: KokoroInferFn;
    init?: () => Promise<void>;
    dispose?: () => Promise<void>;
  }) {
    this.id = options.id ?? 'kokoro-82m-local';
    this.sampleRate = options.sampleRate ?? KOKORO_DEFAULT_SAMPLE_RATE;
    this.defaultVoiceId = options.defaultVoiceId ?? 'af_heart';
    this.inferFn = options.infer;
    this.initFn = options.init;
    this.disposeFn = options.dispose;
  }

  async init(): Promise<void> {
    await this.initFn?.();
  }

  listVoices(): TtsVoiceInfo[] {
    return [...DEFAULT_KOKORO_VOICES];
  }

  async synthesize(
    text: string,
    options: TtsSynthesizeOptions,
    signal?: AbortSignal,
  ): Promise<TtsAudioChunk> {
    const voiceId = options.voiceId ?? this.defaultVoiceId;
    const speed = Math.min(2.0, Math.max(0.5, options.speed ?? 1.0));
    const res = await this.inferFn(text, { voiceId, speed, chunkIndex: options.chunkIndex }, signal);

    const samples = res instanceof Float32Array ? res : res.audio;
    const sr = res instanceof Float32Array ? this.sampleRate : (res.sampleRate ?? this.sampleRate);

    return {
      samples,
      sampleRate: sr,
      text,
      chunkIndex: options.chunkIndex,
      durationMs: (samples.length / sr) * 1000,
    };
  }

  async dispose(): Promise<void> {
    await this.disposeFn?.();
  }
}

/** Headless/in-memory AudioPlaybackSink for Node/Vitest runs or dry-run synthesis. */
export class NullAudioPlaybackSink implements AudioPlaybackSink {
  readonly playedChunks: TtsAudioChunk[] = [];
  private activeResolvers = new Set<() => void>();
  private readonly simulateDuration: boolean;

  constructor(options: { simulateDuration?: boolean } = {}) {
    this.simulateDuration = options.simulateDuration ?? false;
  }

  async playChunk(chunk: TtsAudioChunk, signal?: AbortSignal): Promise<void> {
    if (signal?.aborted) {
      return;
    }
    this.playedChunks.push(chunk);

    if (!this.simulateDuration || chunk.durationMs <= 0) {
      return;
    }

    await new Promise<void>((resolve) => {
      let settled = false;
      const done = () => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(timer);
        signal?.removeEventListener('abort', done);
        this.activeResolvers.delete(done);
        resolve();
      };

      const timer = setTimeout(done, chunk.durationMs);
      this.activeResolvers.add(done);
      signal?.addEventListener('abort', done, { once: true });
    });
  }

  stopAll(): void {
    for (const done of [...this.activeResolvers]) {
      done();
    }
    this.activeResolvers.clear();
  }
}

export interface TtsEngineOptions {
  runner: TtsModelRunner;
  sink?: AudioPlaybackSink;
  /** Number of sentence chunks to synthesize ahead of active playback (default 2). */
  lookaheadChunks?: number;
  /** Sentence chunker options for streaming `speak(AsyncIterable<string>)`. */
  chunker?: SentenceChunkerOptions;
  defaultVoiceId?: string;
  defaultSpeed?: number;
}

export interface TtsSpeakCallbacks {
  onChunkSynthesized?: (chunk: TtsAudioChunk) => void;
  onChunkPlayed?: (chunk: TtsAudioChunk) => void;
}

/**
 * Sentence-chunked streaming TTS engine with bounded lookahead synthesis
 * and deterministic sub-30ms barge-in queue cancellation.
 */
export class TtsEngine {
  readonly runner: TtsModelRunner;
  private readonly sink: AudioPlaybackSink;
  private readonly lookaheadChunks: number;
  private readonly chunkerOptions?: SentenceChunkerOptions;
  private readonly defaultVoiceId?: string;
  private readonly defaultSpeed?: number;

  private initialized = false;
  private activeSpeakController: AbortController | null = null;
  private activeSpeakDone: Promise<void> | null = null;
  private isSpeakingFlag = false;

  constructor(options: TtsEngineOptions) {
    this.runner = options.runner;
    this.sink = options.sink ?? new NullAudioPlaybackSink();
    this.lookaheadChunks = Math.max(1, options.lookaheadChunks ?? 2);
    this.chunkerOptions = options.chunker;
    this.defaultVoiceId = options.defaultVoiceId;
    this.defaultSpeed = options.defaultSpeed;
  }

  get isSpeaking(): boolean {
    return this.isSpeakingFlag;
  }

  async init(): Promise<void> {
    if (this.initialized) {
      return;
    }
    await this.runner.init?.();
    this.initialized = true;
  }

  /**
   * Stream text through `SentenceStreamChunker`, synthesize sentences ahead of playback
   * in strict FIFO queue order, and play them through the sink until completion or `bargeIn()`.
   */
  async speak(
    textStream: string | AsyncIterable<string>,
    options: VoiceSpeakOptions = {},
    callbacks?: TtsSpeakCallbacks,
    externalSignal?: AbortSignal,
  ): Promise<void> {
    // Cancel any overlapping speech turn cleanly before starting a new one.
    if (this.isSpeakingFlag) {
      await this.bargeIn();
    }

    if (!this.initialized) {
      await this.init();
    }

    if (externalSignal?.aborted) {
      return;
    }

    const controller = new AbortController();
    this.activeSpeakController = controller;
    this.isSpeakingFlag = true;

    const onExternalAbort = () => controller.abort();
    externalSignal?.addEventListener('abort', onExternalAbort, { once: true });

    const run = async () => {
      const synthQueue: Array<{
        chunkIndex: number;
        promise: Promise<TtsAudioChunk | null>;
      }> = [];
      const maxBuffered = this.lookaheadChunks + 1;
      let nextChunkIndex = 0;
      let streamDone = false;
      let streamError: unknown = null;

      // Signal mechanism so producer & consumer coordinate without busy-waiting
      let notifyQueueChange: (() => void) | null = null;
      const signalQueue = () => {
        const fn = notifyQueueChange;
        notifyQueueChange = null;
        fn?.();
      };
      const waitForQueueChange = () =>
        new Promise<void>((resolve) => {
          if (controller.signal.aborted) {
            resolve();
            return;
          }
          const onAbort = () => {
            notifyQueueChange = null;
            resolve();
          };
          notifyQueueChange = () => {
            controller.signal.removeEventListener('abort', onAbort);
            resolve();
          };
          controller.signal.addEventListener('abort', onAbort, { once: true });
        });

      const producer = (async () => {
        try {
          for await (const sentence of chunkTextStream(textStream, this.chunkerOptions, controller.signal)) {
            if (controller.signal.aborted) {
              break;
            }

            while (synthQueue.length >= maxBuffered && !controller.signal.aborted) {
              await waitForQueueChange();
            }
            if (controller.signal.aborted) {
              break;
            }

            const chunkIndex = nextChunkIndex++;
            const synthPromise = this.synthesizeNormalized(
              sentence,
              {
                voiceId: options.voiceId ?? this.defaultVoiceId,
                speed: options.speed ?? this.defaultSpeed,
                chunkIndex,
              },
              controller.signal,
            ).then((chunk) => {
              if (chunk && !controller.signal.aborted) {
                callbacks?.onChunkSynthesized?.(chunk);
              }
              return chunk;
            });

            synthQueue.push({ chunkIndex, promise: synthPromise });
            signalQueue();
          }
        } catch (err) {
          if (!controller.signal.aborted) {
            streamError = err;
          }
        } finally {
          streamDone = true;
          signalQueue();
        }
      })();

      const consumer = (async () => {
        while (!controller.signal.aborted) {
          if (synthQueue.length === 0) {
            if (streamDone) {
              break;
            }
            await waitForQueueChange();
            continue;
          }

          const head = synthQueue.shift()!;
          signalQueue();

          const audioChunk = await head.promise;
          if (controller.signal.aborted || !audioChunk) {
            break;
          }

          await this.sink.playChunk(audioChunk, controller.signal);
          if (controller.signal.aborted) {
            break;
          }
          callbacks?.onChunkPlayed?.(audioChunk);
        }
      })();

      await Promise.allSettled([producer, consumer]);

      if (streamError && !controller.signal.aborted) {
        throw streamError;
      }
    };

    const donePromise = run().finally(() => {
      externalSignal?.removeEventListener('abort', onExternalAbort);
      if (this.activeSpeakController === controller) {
        this.activeSpeakController = null;
        this.activeSpeakDone = null;
        this.isSpeakingFlag = false;
      }
    });

    this.activeSpeakDone = donePromise;
    await donePromise;
  }

  /**
   * Instant barge-in cancellation:
   * Aborts pending synthesis/chunking, clears queued audio, and immediately stops active playback (<30ms).
   */
  async bargeIn(): Promise<number> {
    const start = performance.now();
    const controller = this.activeSpeakController;
    const pendingSpeak = this.activeSpeakDone;

    if (controller && !controller.signal.aborted) {
      controller.abort();
    }

    await this.sink.stopAll();
    this.isSpeakingFlag = false;

    if (pendingSpeak) {
      await Promise.race([
        pendingSpeak.catch(() => undefined),
        new Promise<void>((resolve) => setTimeout(resolve, 10)),
      ]);
    }

    return performance.now() - start;
  }

  async dispose(): Promise<void> {
    await this.bargeIn();
    await this.sink.dispose?.();
    await this.runner.dispose?.();
    this.initialized = false;
  }

  private async synthesizeNormalized(
    text: string,
    options: TtsSynthesizeOptions,
    signal: AbortSignal,
  ): Promise<TtsAudioChunk | null> {
    if (signal.aborted) {
      return null;
    }
    try {
      const raw = await this.runner.synthesize(text, options, signal);
      if (signal.aborted) {
        return null;
      }
      if (raw instanceof Float32Array) {
        const sr = this.runner.sampleRate;
        return {
          samples: raw,
          sampleRate: sr,
          text,
          chunkIndex: options.chunkIndex,
          durationMs: (raw.length / sr) * 1000,
        };
      }
      return raw;
    } catch (err) {
      if (signal.aborted || isAbortError(err)) {
        return null;
      }
      throw err;
    }
  }
}

function isAbortError(err: unknown): boolean {
  return err instanceof Error && err.name === 'AbortError';
}
