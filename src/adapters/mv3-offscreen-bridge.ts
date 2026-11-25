import type { AudioInputSource, AudioPlaybackSink, TtsAudioChunk } from '../contracts.js';
import {
  decodeFloat32Base64,
  encodeFloat32Base64,
  extractPcmSamples,
  type Mv3RpcTransport,
  type Mv3VoiceCommand,
  type Mv3VoiceMessage,
} from './mv3-protocol.js';
import type { Mv3MicPermissionHelper } from './mv3-permissions.js';

export interface ChromeOffscreenApiLike {
  createDocument(parameters: {
    url: string;
    reasons: string[];
    justification: string;
  }): Promise<void>;
  hasDocument?(): Promise<boolean>;
}

export interface ChromeContextsRuntimeLike {
  getContexts?(filter: { contextTypes?: string[] }): Promise<Array<{ contextType: string; documentUrl?: string }>>;
}

export interface EnsureOffscreenOptions {
  url: string;
  justification?: string;
  reasons?: string[];
  chrome?: {
    offscreen?: ChromeOffscreenApiLike;
    runtime?: ChromeContextsRuntimeLike;
  };
}

/**
 * Idempotently ensure the Chrome MV3 Offscreen Document is alive before opening
 * the voice bridge transport.
 */
export async function ensureOffscreenDocument(options: EnsureOffscreenOptions): Promise<void> {
  const chromeApi =
    options.chrome ??
    (globalThis as {
      chrome?: { offscreen?: ChromeOffscreenApiLike; runtime?: ChromeContextsRuntimeLike };
    }).chrome;

  if (!chromeApi?.offscreen) {
    throw new Error('chrome.offscreen API is unavailable in this environment.');
  }

  if (chromeApi.runtime?.getContexts) {
    const existing = await chromeApi.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] });
    if (existing.length > 0) {
      return;
    }
  } else if (chromeApi.offscreen.hasDocument) {
    const exists = await chromeApi.offscreen.hasDocument();
    if (exists) {
      return;
    }
  }

  await chromeApi.offscreen.createDocument({
    url: options.url,
    reasons: options.reasons ?? ['USER_MEDIA', 'AUDIO_PLAYBACK'],
    justification:
      options.justification ??
      'Local microphone AudioWorklet capture and WebGPU/WASM voice playback for @yours/voice.',
  });
}

export interface Mv3OffscreenHostOptions {
  transport: Mv3RpcTransport;
  inputSource: AudioInputSource;
  playbackSink?: AudioPlaybackSink;
}

/**
 * Runs inside the Chrome MV3 Offscreen Document.
 * Owns `getUserMedia` + `AudioWorklet` capture and optional offscreen `AudioPlaybackSink`,
 * forwarding PCM frames and handling instant `playback-barge-in` commands over `Mv3RpcTransport`.
 */
export class Mv3OffscreenHost {
  private readonly transport: Mv3RpcTransport;
  private readonly inputSource: AudioInputSource;
  private readonly playbackSink?: AudioPlaybackSink;
  private unsubscribe: (() => void) | null = null;
  private activePlaybackControllers = new Map<string, AbortController>();

  constructor(options: Mv3OffscreenHostOptions) {
    this.transport = options.transport;
    this.inputSource = options.inputSource;
    this.playbackSink = options.playbackSink;
    this.unsubscribe = this.transport.onMessage((msg) => {
      void this.handleMessage(msg);
    });
  }

  async dispose(): Promise<void> {
    this.unsubscribe?.();
    this.unsubscribe = null;
    for (const ctrl of this.activePlaybackControllers.values()) {
      ctrl.abort();
    }
    this.activePlaybackControllers.clear();
    await this.playbackSink?.stopAll();
    await this.inputSource.stop();
  }

  private async handleMessage(msg: Mv3VoiceMessage): Promise<void> {
    switch (msg.type) {
      case 'yours:voice:capture-start': {
        try {
          await this.inputSource.start((samples, sampleRate, timestampMs) => {
            if (this.transport.supportsTransferables) {
              void this.transport.send({
                type: 'yours:voice:capture-frame',
                sampleRate,
                timestampMs,
                samples,
              });
            } else {
              void this.transport.send({
                type: 'yours:voice:capture-frame',
                sampleRate,
                timestampMs,
                pcmBase64: encodeFloat32Base64(samples),
              });
            }
          });
          await this.transport.send({ type: 'yours:voice:ack', requestId: msg.requestId, ok: true });
        } catch (err) {
          await this.transport.send({
            type: 'yours:voice:ack',
            requestId: msg.requestId,
            ok: false,
            error: err instanceof Error ? err.message : String(err),
          });
        }
        break;
      }

      case 'yours:voice:capture-stop': {
        try {
          await this.inputSource.stop();
          await this.transport.send({ type: 'yours:voice:ack', requestId: msg.requestId, ok: true });
        } catch (err) {
          await this.transport.send({
            type: 'yours:voice:ack',
            requestId: msg.requestId,
            ok: false,
            error: err instanceof Error ? err.message : String(err),
          });
        }
        break;
      }

      case 'yours:voice:playback-chunk': {
        if (!this.playbackSink) {
          await this.transport.send({
            type: 'yours:voice:playback-ended',
            requestId: msg.requestId,
            chunkIndex: msg.chunkIndex,
            aborted: false,
            error: 'No playbackSink configured in OffscreenHost.',
          });
          return;
        }

        const samples = extractPcmSamples(msg);
        const controller = new AbortController();
        this.activePlaybackControllers.set(msg.requestId, controller);

        try {
          await this.playbackSink.playChunk(
            {
              samples,
              sampleRate: msg.sampleRate,
              text: '',
              chunkIndex: msg.chunkIndex,
              durationMs: (samples.length / msg.sampleRate) * 1000,
            },
            controller.signal,
          );
          await this.transport.send({
            type: 'yours:voice:playback-ended',
            requestId: msg.requestId,
            chunkIndex: msg.chunkIndex,
            aborted: controller.signal.aborted,
          });
        } catch (err) {
          await this.transport.send({
            type: 'yours:voice:playback-ended',
            requestId: msg.requestId,
            chunkIndex: msg.chunkIndex,
            aborted: controller.signal.aborted,
            error: err instanceof Error ? err.message : String(err),
          });
        } finally {
          this.activePlaybackControllers.delete(msg.requestId);
        }
        break;
      }

      case 'yours:voice:playback-barge-in': {
        for (const ctrl of this.activePlaybackControllers.values()) {
          ctrl.abort();
        }
        this.activePlaybackControllers.clear();
        await this.playbackSink?.stopAll();
        await this.transport.send({ type: 'yours:voice:ack', requestId: msg.requestId, ok: true });
        break;
      }

      default:
        break;
    }
  }
}

export interface Mv3SidePanelClientOptions {
  transport: Mv3RpcTransport;
  /** Optional permission onboarding helper invoked automatically before `capture-start`. */
  permissionHelper?: Mv3MicPermissionHelper;
  /**
   * Optional fallback Web Audio playback sink inside the Side Panel.
   * Used automatically when `useSidePanelFallbackPlayback` is true or offscreen playback is unavailable.
   */
  fallbackPlaybackSink?: AudioPlaybackSink;
  /** Prefer local Side Panel playback sink instead of routing TTS audio to Offscreen Document. */
  useSidePanelFallbackPlayback?: boolean;
  /** Sample rate reported by capture source (default 16000). */
  captureSampleRate?: number;
  /** RPC command timeout in ms (default 5000). */
  commandTimeoutMs?: number;
}

/**
 * Runs inside the Chrome MV3 Side Panel.
 * Provides an `AudioInputSource` and `AudioPlaybackSink` backed by the Offscreen Bridge RPC,
 * with automatic permission onboarding and Side Panel Web Audio playback fallback.
 */
export class Mv3SidePanelClient {
  private readonly transport: Mv3RpcTransport;
  private readonly permissionHelper?: Mv3MicPermissionHelper;
  private readonly fallbackPlaybackSink?: AudioPlaybackSink;
  private readonly useFallbackPlayback: boolean;
  private readonly captureSampleRate: number;
  private readonly commandTimeoutMs: number;

  private unsubscribe: (() => void) | null = null;
  private frameListener: ((samples: Float32Array, sampleRate: number, timestampMs: number) => void) | null = null;
  private pendingAcks = new Map<string, { resolve: () => void; reject: (err: Error) => void }>();
  private pendingPlaybacks = new Map<string, { resolve: () => void; reject: (err: Error) => void }>();
  private requestCounter = 0;
  private activeCapture = false;

  constructor(options: Mv3SidePanelClientOptions) {
    this.transport = options.transport;
    this.permissionHelper = options.permissionHelper;
    this.fallbackPlaybackSink = options.fallbackPlaybackSink;
    this.useFallbackPlayback = options.useSidePanelFallbackPlayback ?? false;
    this.captureSampleRate = options.captureSampleRate ?? 16_000;
    this.commandTimeoutMs = options.commandTimeoutMs ?? 5_000;

    this.unsubscribe = this.transport.onMessage((msg) => {
      this.handleEvent(msg);
    });
  }

  /** Create an `AudioInputSource` compatible with `LocalVoicePipeline`. */
  createInputSource(): AudioInputSource {
    const client = this;
    return {
      get sampleRate() {
        return client.captureSampleRate;
      },
      get isActive() {
        return client.activeCapture;
      },
      async start(onFrame) {
        if (client.permissionHelper) {
          const perm = await client.permissionHelper.ensureMicPermission();
          if (!perm.granted) {
            throw new Error(perm.error ?? 'Microphone permission denied during MV3 onboarding.');
          }
        }
        client.frameListener = onFrame;
        await client.sendCommandWithAck({
          type: 'yours:voice:capture-start',
          requestId: client.nextRequestId('cap'),
          sampleRate: client.captureSampleRate,
        });
        client.activeCapture = true;
      },
      async stop() {
        client.activeCapture = false;
        client.frameListener = null;
        await client.sendCommandWithAck({
          type: 'yours:voice:capture-stop',
          requestId: client.nextRequestId('stop'),
        });
      },
      async dispose() {
        if (client.activeCapture) {
          await this.stop();
        }
      },
    };
  }

  /**
   * Create an `AudioPlaybackSink` compatible with `LocalVoicePipeline`.
   * Uses the Side Panel fallback sink when configured, or routes chunks to the Offscreen host.
   */
  createPlaybackSink(): AudioPlaybackSink {
    if (this.useFallbackPlayback && this.fallbackPlaybackSink) {
      return this.fallbackPlaybackSink;
    }

    const client = this;
    return {
      async playChunk(chunk: TtsAudioChunk, signal?: AbortSignal): Promise<void> {
        if (signal?.aborted) {
          return;
        }
        const requestId = client.nextRequestId('play');
        const payload: Mv3VoiceCommand = client.transport.supportsTransferables
          ? {
              type: 'yours:voice:playback-chunk',
              requestId,
              chunkIndex: chunk.chunkIndex,
              sampleRate: chunk.sampleRate,
              samples: chunk.samples,
            }
          : {
              type: 'yours:voice:playback-chunk',
              requestId,
              chunkIndex: chunk.chunkIndex,
              sampleRate: chunk.sampleRate,
              pcmBase64: encodeFloat32Base64(chunk.samples),
            };

        const waitPlayback = new Promise<void>((resolve, reject) => {
          const onAbort = () => {
            client.pendingPlaybacks.delete(requestId);
            resolve();
          };
          signal?.addEventListener('abort', onAbort, { once: true });
          client.pendingPlaybacks.set(requestId, {
            resolve: () => {
              signal?.removeEventListener('abort', onAbort);
              resolve();
            },
            reject: (err) => {
              signal?.removeEventListener('abort', onAbort);
              reject(err);
            },
          });
        });

        await client.transport.send(payload);
        try {
          await waitPlayback;
        } catch (err) {
          // Automatic fallback to Side Panel Web Audio if Offscreen Document has no playbackSink
          if (client.fallbackPlaybackSink) {
            await client.fallbackPlaybackSink.playChunk(chunk, signal);
            return;
          }
          throw err;
        }
      },
      async stopAll(): Promise<void> {
        for (const pending of client.pendingPlaybacks.values()) {
          pending.resolve();
        }
        client.pendingPlaybacks.clear();
        await client.fallbackPlaybackSink?.stopAll();
        await client.transport.send({
          type: 'yours:voice:playback-barge-in',
          requestId: client.nextRequestId('barge'),
        });
      },
      async dispose(): Promise<void> {
        await this.stopAll();
        await client.fallbackPlaybackSink?.dispose?.();
      },
    };
  }

  dispose(): void {
    this.unsubscribe?.();
    this.unsubscribe = null;
    this.frameListener = null;
    this.activeCapture = false;
    for (const pending of this.pendingAcks.values()) {
      pending.reject(new Error('Mv3SidePanelClient disposed.'));
    }
    this.pendingAcks.clear();
    for (const pending of this.pendingPlaybacks.values()) {
      pending.resolve();
    }
    this.pendingPlaybacks.clear();
  }

  private nextRequestId(prefix: string): string {
    this.requestCounter += 1;
    return `${prefix}-${this.requestCounter}`;
  }

  private sendCommandWithAck(cmd: Extract<Mv3VoiceCommand, { requestId: string }>): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingAcks.delete(cmd.requestId);
        reject(new Error(`MV3 RPC timeout waiting for ack (${cmd.type})`));
      }, this.commandTimeoutMs);

      this.pendingAcks.set(cmd.requestId, {
        resolve: () => {
          clearTimeout(timer);
          resolve();
        },
        reject: (err) => {
          clearTimeout(timer);
          reject(err);
        },
      });

      Promise.resolve(this.transport.send(cmd)).catch((err) => {
        clearTimeout(timer);
        this.pendingAcks.delete(cmd.requestId);
        reject(err instanceof Error ? err : new Error(String(err)));
      });
    });
  }

  private handleEvent(msg: Mv3VoiceMessage): void {
    switch (msg.type) {
      case 'yours:voice:ack': {
        const pending = this.pendingAcks.get(msg.requestId);
        if (!pending) {
          return;
        }
        this.pendingAcks.delete(msg.requestId);
        if (msg.ok) {
          pending.resolve();
        } else {
          pending.reject(new Error(msg.error ?? 'MV3 RPC command failed'));
        }
        break;
      }

      case 'yours:voice:capture-frame': {
        if (!this.frameListener) {
          return;
        }
        const samples = msg.samples ?? (msg.pcmBase64 ? decodeFloat32Base64(msg.pcmBase64) : null);
        if (samples) {
          this.frameListener(samples, msg.sampleRate, msg.timestampMs);
        }
        break;
      }

      case 'yours:voice:playback-ended': {
        const pending = this.pendingPlaybacks.get(msg.requestId);
        if (!pending) {
          return;
        }
        this.pendingPlaybacks.delete(msg.requestId);
        if (msg.error && !msg.aborted) {
          pending.reject(new Error(msg.error));
        } else {
          pending.resolve();
        }
        break;
      }

      default:
        break;
    }
  }
}
