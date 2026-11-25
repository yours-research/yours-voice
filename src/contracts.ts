/**
 * Shared ecosystem contracts and voice subsystem interfaces for `@yours/voice`.
 * Strictly aligned with `docs/ECOSYSTEM_CONTRACTS.md` (Section 7).
 */

export type VoiceState =
  | 'idle'
  | 'listening'
  | 'speech-detected'
  | 'transcribing'
  | 'speaking';

export interface TranscriptSegment {
  text: string;
  isFinal: boolean;
  startMs: number;
  endMs: number;
  confidence?: number;
}

export interface VoiceListenOptions {
  language?: string;
  bargeInEnabled?: boolean;
}

export interface VoiceSpeakOptions {
  voiceId?: string;
  speed?: number;
}

export interface VoicePipeline {
  readonly state: VoiceState;
  startListening(options?: VoiceListenOptions): Promise<void>;
  stopListening(): Promise<void>;
  speak(textStream: string | AsyncIterable<string>, options?: VoiceSpeakOptions): Promise<void>;
  bargeIn(): Promise<void>;
  dispose(): Promise<void>;
}

/** Canonical PCM frame emitted at target sample rate (default 16 kHz mono). */
export interface PcmFrame {
  samples: Float32Array;
  sampleRate: number;
  timestampMs: number;
  durationMs: number;
  rms: number;
}

/** Normalized audio chunk produced by TTS synthesis. */
export interface TtsAudioChunk {
  samples: Float32Array;
  sampleRate: number;
  text: string;
  chunkIndex: number;
  durationMs: number;
}

/** Abstract microphone/audio capture source (Web Demo, MV3 Offscreen, or test double). */
export interface AudioInputSource {
  readonly sampleRate: number;
  readonly isActive: boolean;
  start(onFrame: (samples: Float32Array, sampleRate: number, timestampMs: number) => void): Promise<void>;
  stop(): Promise<void>;
  dispose?(): Promise<void>;
}

/** Abstract audio playback sink with synchronous/sub-30ms stopAll() cancellation. */
export interface AudioPlaybackSink {
  playChunk(chunk: TtsAudioChunk, signal?: AbortSignal): Promise<void>;
  stopAll(): Promise<void> | void;
  dispose?(): Promise<void>;
}

export type VadTransitionState = 'silence' | 'pending-speech' | 'speech' | 'hangover';

export type VadEvent =
  | {
      type: 'speech_start';
      timestampMs: number;
      probability: number;
      preRollAudio: Float32Array;
    }
  | {
      type: 'speech_frame';
      timestampMs: number;
      probability: number;
      audio: Float32Array;
    }
  | {
      type: 'speech_end';
      startMs: number;
      endMs: number;
      durationMs: number;
      audio: Float32Array;
    }
  | {
      type: 'probability';
      timestampMs: number;
      probability: number;
      vadState: VadTransitionState;
    };

export interface BargeInEvent {
  timestampMs: number;
  cancelLatencyMs: number;
  reason: 'user_speech' | 'manual';
}

export interface VoicePipelineTelemetry {
  lastBargeInLatencyMs?: number;
  lastTtfaMs?: number;
  utteranceCount: number;
  bargeInCount: number;
}

export interface VoicePipelineEventMap {
  stateChange: { previous: VoiceState; current: VoiceState; timestampMs: number };
  transcript: TranscriptSegment;
  vad: VadEvent;
  ttsChunk: TtsAudioChunk;
  bargeIn: BargeInEvent;
  error: Error;
}
