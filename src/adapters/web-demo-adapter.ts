import {
  AUDIO_WORKLET_CAPTURE_PROCESSOR_NAME,
  AUDIO_WORKLET_PROCESSOR_CODE,
} from '../audio/frame-chunker.js';
import type { AudioInputSource, AudioPlaybackSink, TtsAudioChunk } from '../contracts.js';

export interface WebAudioPlaybackSinkOptions {
  /** Reuse an existing `AudioContext` (e.g., from Side Panel or Web Demo). */
  audioContext?: AudioContext;
}

/**
 * Browser Web Audio API playback sink.
 * Schedules sequential `AudioBufferSourceNode` chunks gaplessly and stops all
 * active & queued nodes synchronously on `stopAll()` for sub-30ms barge-in.
 */
export class WebAudioPlaybackSink implements AudioPlaybackSink {
  private ctx: AudioContext | null;
  private readonly ownsContext: boolean;
  private activeSources = new Set<AudioBufferSourceNode>();
  private nextScheduledTime = 0;

  constructor(options: WebAudioPlaybackSinkOptions = {}) {
    this.ctx = options.audioContext ?? null;
    this.ownsContext = !options.audioContext;
  }

  async playChunk(chunk: TtsAudioChunk, signal?: AbortSignal): Promise<void> {
    if (signal?.aborted || chunk.samples.length === 0) {
      return;
    }

    const ctx = await this.ensureContext();
    if (signal?.aborted) {
      return;
    }

    const buffer = ctx.createBuffer(1, chunk.samples.length, chunk.sampleRate);
    buffer.copyToChannel(new Float32Array(chunk.samples), 0);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    const startTime = Math.max(ctx.currentTime, this.nextScheduledTime);
    this.nextScheduledTime = startTime + buffer.duration;
    this.activeSources.add(source);

    await new Promise<void>((resolve) => {
      let settled = false;
      const cleanup = () => {
        if (settled) {
          return;
        }
        settled = true;
        signal?.removeEventListener('abort', onAbort);
        this.activeSources.delete(source);
        try {
          source.disconnect();
        } catch {
          // Ignore already disconnected node
        }
        resolve();
      };

      const onAbort = () => {
        try {
          source.stop();
        } catch {
          // Ignore if already stopped
        }
        cleanup();
      };

      source.onended = cleanup;
      signal?.addEventListener('abort', onAbort, { once: true });
      source.start(startTime);
    });
  }

  stopAll(): void {
    for (const source of [...this.activeSources]) {
      try {
        source.stop();
        source.disconnect();
      } catch {
        // Ignore already stopped source
      }
    }
    this.activeSources.clear();
    if (this.ctx) {
      this.nextScheduledTime = this.ctx.currentTime;
    } else {
      this.nextScheduledTime = 0;
    }
  }

  async dispose(): Promise<void> {
    this.stopAll();
    if (this.ownsContext && this.ctx && this.ctx.state !== 'closed') {
      await this.ctx.close();
      this.ctx = null;
    }
  }

  private async ensureContext(): Promise<AudioContext> {
    if (!this.ctx) {
      const AudioCtx =
        globalThis.AudioContext ??
        (globalThis as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) {
        throw new Error('Web Audio AudioContext is unavailable in this environment.');
      }
      this.ctx = new AudioCtx();
    }
    if (this.ctx.state === 'suspended') {
      await this.ctx.resume();
    }
    return this.ctx;
  }
}

export interface WebDemoInputOptions {
  deviceId?: string;
  echoCancellation?: boolean;
  noiseSuppression?: boolean;
  autoGainControl?: boolean;
  /** Optional custom AudioContext. */
  audioContext?: AudioContext;
}

/**
 * Browser Web Demo (and Offscreen Document) microphone capture adapter
 * using `getUserMedia` + `AudioWorkletNode` (`Blob` URL registration).
 */
export class WebDemoAudioInputSource implements AudioInputSource {
  private readonly options: WebDemoInputOptions;
  private ctx: AudioContext | null = null;
  private ownsContext = false;
  private stream: MediaStream | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private active = false;
  private currentSampleRate = 48_000;

  constructor(options: WebDemoInputOptions = {}) {
    this.options = options;
  }

  get sampleRate(): number {
    return this.currentSampleRate;
  }

  get isActive(): boolean {
    return this.active;
  }

  async start(
    onFrame: (samples: Float32Array, sampleRate: number, timestampMs: number) => void,
  ): Promise<void> {
    if (this.active) {
      return;
    }

    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      throw new Error('navigator.mediaDevices.getUserMedia is unavailable in this environment.');
    }

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: this.options.deviceId ? { exact: this.options.deviceId } : undefined,
        channelCount: 1,
        echoCancellation: this.options.echoCancellation ?? true,
        noiseSuppression: this.options.noiseSuppression ?? true,
        autoGainControl: this.options.autoGainControl ?? true,
      },
    });

    const AudioCtx =
      globalThis.AudioContext ??
      (globalThis as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) {
      throw new Error('Web Audio AudioContext is unavailable.');
    }

    this.ctx = this.options.audioContext ?? new AudioCtx();
    this.ownsContext = !this.options.audioContext;
    if (this.ctx.state === 'suspended') {
      await this.ctx.resume();
    }
    this.currentSampleRate = this.ctx.sampleRate;

    this.sourceNode = this.ctx.createMediaStreamSource(this.stream);

    const blob = new Blob([AUDIO_WORKLET_PROCESSOR_CODE], { type: 'application/javascript' });
    const moduleUrl = URL.createObjectURL(blob);
    try {
      await this.ctx.audioWorklet.addModule(moduleUrl);
    } finally {
      URL.revokeObjectURL(moduleUrl);
    }

    this.workletNode = new AudioWorkletNode(this.ctx, AUDIO_WORKLET_CAPTURE_PROCESSOR_NAME);
    this.workletNode.port.onmessage = (event: MessageEvent) => {
      if (!this.active) {
        return;
      }
      const data = event.data as {
        type?: string;
        buffer?: ArrayBuffer;
        sampleRate?: number;
        currentTime?: number;
      };
      if (data?.type === 'pcm' && data.buffer) {
        const samples = new Float32Array(data.buffer);
        const sr = data.sampleRate ?? this.currentSampleRate;
        const tsMs = (data.currentTime ?? 0) * 1000;
        onFrame(samples, sr, tsMs);
      }
    };

    this.sourceNode.connect(this.workletNode);
    this.active = true;
  }

  async stop(): Promise<void> {
    this.active = false;
    if (this.workletNode) {
      this.workletNode.port.onmessage = null;
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop();
      }
      this.stream = null;
    }
    if (this.ownsContext && this.ctx && this.ctx.state !== 'closed') {
      await this.ctx.close();
      this.ctx = null;
    }
  }

  async dispose(): Promise<void> {
    await this.stop();
  }
}

/**
 * Convenience Web Demo bundle exposing both `inputSource` and `playbackSink`
 * for `yours.sh` browser demos.
 */
export class WebDemoAudioAdapter {
  readonly inputSource: WebDemoAudioInputSource;
  readonly playbackSink: WebAudioPlaybackSink;

  constructor(options: WebDemoInputOptions = {}) {
    this.inputSource = new WebDemoAudioInputSource(options);
    this.playbackSink = new WebAudioPlaybackSink({ audioContext: options.audioContext });
  }

  async dispose(): Promise<void> {
    await this.inputSource.dispose();
    await this.playbackSink.dispose();
  }
}
