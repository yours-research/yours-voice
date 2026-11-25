import { PcmFrameChunker } from '../audio/frame-chunker.js';
import { PcmResampler, TARGET_VOICE_SAMPLE_RATE } from '../audio/resampler.js';
import { concatFloat32Arrays } from '../audio/ring-buffer.js';
import type {
  AudioInputSource,
  PcmFrame,
  TranscriptSegment,
  VadEvent,
  VoiceListenOptions,
  VoicePipeline,
  VoicePipelineEventMap,
  VoicePipelineTelemetry,
  VoiceSpeakOptions,
  VoiceState,
} from '../contracts.js';
import type { SttEngine } from '../stt/stt-engine.js';
import type { TtsEngine } from '../tts/tts-engine.js';
import { VadEngine } from '../vad/vad-engine.js';

export interface LocalVoicePipelineOptions {
  inputSource: AudioInputSource;
  vad?: VadEngine;
  stt?: SttEngine;
  tts?: TtsEngine;
  /** Target internal audio sample rate for VAD/STT (default 16000). */
  sampleRate?: number;
  /** Frame size in samples at target sample rate (default 512 = 32 ms @ 16 kHz). */
  frameSize?: number;
  /** Resampler mode when input rate != 16 kHz (default 'polyphase'). */
  resamplerMode?: 'polyphase' | 'linear';
  /** Enable incremental partial transcription while speech is still active (default false). */
  enableIncrementalPartials?: boolean;
  /** Interval in ms between incremental partial STT passes during speech (default 400). */
  incrementalPartialIntervalMs?: number;
  /** Flush remaining VAD speech through STT when `stopListening()` is called (default true). */
  flushOnStop?: boolean;
  /** Clock provider for telemetry/timestamps. */
  nowMs?: () => number;
}

type EventKey = keyof VoicePipelineEventMap;
type ListenerMap = {
  [K in EventKey]: Set<(payload: VoicePipelineEventMap[K]) => void>;
};

/**
 * Unified local voice pipeline implementing `@yours/voice`'s `VoicePipeline` contract.
 * Coordinates real-time 16 kHz resampling, pre-roll hysteresis VAD, throttled streaming STT,
 * sentence-chunked Kokoro TTS, and simultaneous duplex barge-in cancellation (<30ms).
 */
export class LocalVoicePipeline implements VoicePipeline {
  private readonly inputSource: AudioInputSource;
  private readonly vad: VadEngine;
  private readonly stt?: SttEngine;
  private readonly tts?: TtsEngine;
  private readonly targetSampleRate: number;
  private readonly frameSize: number;
  private readonly resamplerMode: 'polyphase' | 'linear';
  private readonly enableIncrementalPartials: boolean;
  private readonly incrementalPartialIntervalMs: number;
  private readonly flushOnStop: boolean;
  private readonly nowMs: () => number;

  private resampler: PcmResampler | null = null;
  private chunker: PcmFrameChunker;

  private _state: VoiceState = 'idle';
  private listeningRequested = false;
  private bargeInEnabled = true;
  private currentLanguage?: string;
  private disposed = false;

  private frameProcessingChain: Promise<void> = Promise.resolve();
  private activeSttController: AbortController | null = null;
  private activeUtteranceChunks: Float32Array[] = [];
  private activeSpeechStartMs = 0;
  private lastPartialDecodeMs = 0;
  private partialDecodeInFlight = false;
  private activeSpeakToken = 0;

  private telemetryState: VoicePipelineTelemetry = {
    utteranceCount: 0,
    bargeInCount: 0,
  };

  private readonly listeners: ListenerMap = {
    stateChange: new Set(),
    transcript: new Set(),
    vad: new Set(),
    ttsChunk: new Set(),
    bargeIn: new Set(),
    error: new Set(),
  };

  constructor(options: LocalVoicePipelineOptions) {
    this.inputSource = options.inputSource;
    this.vad = options.vad ?? new VadEngine();
    this.stt = options.stt;
    this.tts = options.tts;
    this.targetSampleRate = options.sampleRate ?? TARGET_VOICE_SAMPLE_RATE;
    this.frameSize = options.frameSize ?? 512;
    this.resamplerMode = options.resamplerMode ?? 'polyphase';
    this.enableIncrementalPartials = options.enableIncrementalPartials ?? false;
    this.incrementalPartialIntervalMs = Math.max(100, options.incrementalPartialIntervalMs ?? 400);
    this.flushOnStop = options.flushOnStop ?? true;
    this.nowMs = options.nowMs ?? (() => performance.now());

    this.chunker = new PcmFrameChunker({
      frameSize: this.frameSize,
      sampleRate: this.targetSampleRate,
    });
  }

  get state(): VoiceState {
    return this._state;
  }

  get telemetry(): Readonly<VoicePipelineTelemetry> {
    return { ...this.telemetryState };
  }

  on<K extends EventKey>(
    event: K,
    listener: (payload: VoicePipelineEventMap[K]) => void,
  ): () => void {
    this.listeners[event].add(listener);
    return () => {
      this.off(event, listener);
    };
  }

  off<K extends EventKey>(
    event: K,
    listener: (payload: VoicePipelineEventMap[K]) => void,
  ): void {
    this.listeners[event].delete(listener);
  }

  async startListening(options: VoiceListenOptions = {}): Promise<void> {
    this.assertNotDisposed();
    this.currentLanguage = options.language;
    this.bargeInEnabled = options.bargeInEnabled ?? true;

    if (this.listeningRequested && this.inputSource.isActive) {
      return;
    }

    this.listeningRequested = true;
    await this.vad.init();
    if (this.stt) {
      await this.stt.init();
    }

    this.resampler = null;
    this.chunker.reset();
    await this.vad.reset();

    if (this._state === 'idle') {
      this.setState('listening');
    }

    await this.inputSource.start((samples, sampleRate, timestampMs) => {
      if (!this.listeningRequested || this.disposed) {
        return;
      }
      this.enqueueRawAudio(samples, sampleRate, timestampMs);
    });
  }

  async stopListening(): Promise<void> {
    if (!this.listeningRequested && !this.inputSource.isActive) {
      return;
    }

    this.listeningRequested = false;
    await this.inputSource.stop();

    // Drain any queued frames already captured
    await this.frameProcessingChain;

    if (this.flushOnStop && !this.disposed) {
      const tailFrame = this.chunker.flush();
      if (tailFrame) {
        await this.processVadFrame(tailFrame);
      }
      const flushEvents = this.vad.flush();
      for (const ev of flushEvents) {
        await this.handleVadEvent(ev);
      }
    } else {
      this.activeSttController?.abort();
      await this.vad.reset();
      this.chunker.reset();
    }

    this.resampler = null;
    if (this._state !== 'speaking') {
      this.setState('idle');
    }
  }

  async speak(
    textStream: string | AsyncIterable<string>,
    options: VoiceSpeakOptions = {},
  ): Promise<void> {
    this.assertNotDisposed();
    if (!this.tts) {
      throw new Error('Cannot call speak(): no TtsEngine configured on LocalVoicePipeline.');
    }

    if (this._state === 'speaking' || this.tts.isSpeaking) {
      await this.bargeInInternal('manual', false);
    }

    const speakToken = ++this.activeSpeakToken;
    const speakStart = this.nowMs();
    let firstAudioRecorded = false;

    this.setState('speaking');

    try {
      await this.tts.speak(textStream, options, {
        onChunkSynthesized: (chunk) => {
          if (!firstAudioRecorded) {
            firstAudioRecorded = true;
            this.telemetryState.lastTtfaMs = this.nowMs() - speakStart;
          }
          this.emit('ttsChunk', chunk);
        },
      });
    } finally {
      // Only restore idle/listening if this speak turn is still the active one
      // and wasn't already preempted by duplex speech detection.
      if (this.activeSpeakToken === speakToken && this._state === 'speaking') {
        this.setState(this.listeningRequested ? 'listening' : 'idle');
      }
    }
  }

  async bargeIn(): Promise<void> {
    if (this.disposed) {
      return;
    }
    await this.bargeInInternal('manual', true);
  }

  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.listeningRequested = false;
    this.activeSpeakToken++;

    this.activeSttController?.abort();
    this.activeSttController = null;

    await this.tts?.dispose();
    await this.inputSource.stop();
    await this.inputSource.dispose?.();
    await this.frameProcessingChain.catch(() => undefined);
    await this.vad.dispose();
    await this.stt?.dispose();

    this.setState('idle');
    for (const set of Object.values(this.listeners)) {
      set.clear();
    }
  }

  private async bargeInInternal(
    reason: 'user_speech' | 'manual',
    transitionState: boolean,
  ): Promise<void> {
    const start = performance.now();
    this.activeSpeakToken++;

    let cancelLatencyMs = 0;
    if (this.tts) {
      cancelLatencyMs = await this.tts.bargeIn();
    } else {
      cancelLatencyMs = performance.now() - start;
    }

    this.telemetryState.lastBargeInLatencyMs = cancelLatencyMs;
    this.telemetryState.bargeInCount += 1;

    this.emit('bargeIn', {
      timestampMs: this.nowMs(),
      cancelLatencyMs,
      reason,
    });

    if (transitionState && this._state === 'speaking') {
      this.setState(this.listeningRequested ? 'listening' : 'idle');
    }
  }

  private enqueueRawAudio(samples: Float32Array, sampleRate: number, timestampMs: number): void {
    let mono16k: Float32Array;
    if (sampleRate === this.targetSampleRate) {
      mono16k = samples;
    } else {
      if (!this.resampler || this.resampler.inputSampleRate !== sampleRate) {
        this.resampler = new PcmResampler({
          inputSampleRate: sampleRate,
          outputSampleRate: this.targetSampleRate,
          mode: this.resamplerMode,
        });
      }
      mono16k = this.resampler.process(samples);
    }

    if (mono16k.length === 0) {
      return;
    }

    const frames = this.chunker.push(mono16k, timestampMs);
    if (frames.length === 0) {
      return;
    }

    this.frameProcessingChain = this.frameProcessingChain
      .then(async () => {
        for (const frame of frames) {
          if (this.disposed) {
            return;
          }
          await this.processVadFrame(frame);
        }
      })
      .catch((err) => {
        this.emit('error', err instanceof Error ? err : new Error(String(err)));
      });
  }

  private async processVadFrame(frame: PcmFrame): Promise<void> {
    // If speaking and barge-in is disabled, skip VAD processing during playback (half-duplex mode)
    if (this._state === 'speaking' && !this.bargeInEnabled) {
      return;
    }

    const events = await this.vad.processFrame(frame);
    for (const ev of events) {
      await this.handleVadEvent(ev);
    }
  }

  private async handleVadEvent(ev: VadEvent): Promise<void> {
    this.emit('vad', ev);

    switch (ev.type) {
      case 'speech_start': {
        if (this._state === 'speaking' && this.bargeInEnabled) {
          await this.bargeInInternal('user_speech', false);
        }
        this.activeUtteranceChunks = ev.preRollAudio.length > 0 ? [ev.preRollAudio] : [];
        this.activeSpeechStartMs = ev.timestampMs;
        this.lastPartialDecodeMs = this.nowMs();
        this.stt?.resetPartialThrottle();
        this.setState('speech-detected');
        break;
      }

      case 'speech_frame': {
        this.activeUtteranceChunks.push(ev.audio);
        if (
          this.enableIncrementalPartials &&
          this.stt &&
          !this.partialDecodeInFlight &&
          this.nowMs() - this.lastPartialDecodeMs >= this.incrementalPartialIntervalMs
        ) {
          this.triggerIncrementalPartial(ev.timestampMs);
        }
        break;
      }

      case 'speech_end': {
        this.activeUtteranceChunks = [];
        this.telemetryState.utteranceCount += 1;

        if (!this.stt) {
          if (this._state === 'speech-detected') {
            this.setState(this.listeningRequested ? 'listening' : 'idle');
          }
          break;
        }

        this.setState('transcribing');
        this.activeSttController?.abort();
        const sttController = new AbortController();
        this.activeSttController = sttController;

        try {
          await this.stt.transcribe(ev.audio, {
            language: this.currentLanguage,
            startMs: ev.startMs,
            endMs: ev.endMs,
            signal: sttController.signal,
            onSegment: (seg: TranscriptSegment) => {
              this.emit('transcript', seg);
            },
          });
        } catch (err) {
          if (!sttController.signal.aborted) {
            this.emit('error', err instanceof Error ? err : new Error(String(err)));
          }
        } finally {
          if (this.activeSttController === sttController) {
            this.activeSttController = null;
          }
          if (this._state === 'transcribing') {
            this.setState(this.listeningRequested ? 'listening' : 'idle');
          }
        }
        break;
      }

      case 'probability':
        break;
    }
  }

  private triggerIncrementalPartial(currentTimestampMs: number): void {
    if (!this.stt) {
      return;
    }
    this.partialDecodeInFlight = true;
    this.lastPartialDecodeMs = this.nowMs();
    const snapshot = concatFloat32Arrays(this.activeUtteranceChunks);
    const controller = new AbortController();

    void this.stt
      .transcribe(snapshot, {
        language: this.currentLanguage,
        startMs: this.activeSpeechStartMs,
        endMs: currentTimestampMs,
        signal: controller.signal,
      })
      .then((segments) => {
        if (this._state !== 'speech-detected' || !this.stt) {
          return;
        }
        const combined = segments.map((s) => s.text).join(' ').trim();
        if (combined) {
          this.stt.emitPartialIfAllowed(
            {
              text: combined,
              isFinal: false,
              startMs: this.activeSpeechStartMs,
              endMs: currentTimestampMs,
            },
            (seg) => this.emit('transcript', seg),
          );
        }
      })
      .catch(() => undefined)
      .finally(() => {
        this.partialDecodeInFlight = false;
      });
  }

  private setState(next: VoiceState): void {
    if (this._state === next) {
      return;
    }
    const previous = this._state;
    this._state = next;
    this.emit('stateChange', {
      previous,
      current: next,
      timestampMs: this.nowMs(),
    });
  }

  private emit<K extends EventKey>(event: K, payload: VoicePipelineEventMap[K]): void {
    for (const listener of [...this.listeners[event]]) {
      try {
        listener(payload);
      } catch {
        // Isolate subscriber exceptions from audio pipeline execution
      }
    }
  }

  private assertNotDisposed(): void {
    if (this.disposed) {
      throw new Error('VoicePipeline has been disposed.');
    }
  }
}
