import type { PcmFrame } from '../contracts.js';
import { TARGET_VOICE_SAMPLE_RATE } from './resampler.js';

/** Compute Root Mean Square amplitude of a PCM frame. */
export function computeRms(samples: Float32Array): number {
  if (samples.length === 0) {
    return 0;
  }
  let sumSq = 0;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i]!;
    sumSq += s * s;
  }
  return Math.sqrt(sumSq / samples.length);
}

/** Compute peak absolute sample amplitude in `[0, ...]`. */
export function computePeak(samples: Float32Array): number {
  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const abs = Math.abs(samples[i]!);
    if (abs > peak) {
      peak = abs;
    }
  }
  return peak;
}

/** Compute normalized zero-crossing rate in `[0, 1]`. */
export function computeZeroCrossingRate(samples: Float32Array): number {
  if (samples.length < 2) {
    return 0;
  }
  let crossings = 0;
  let prev = samples[0]!;
  for (let i = 1; i < samples.length; i++) {
    const curr = samples[i]!;
    if ((prev >= 0 && curr < 0) || (prev < 0 && curr >= 0)) {
      crossings++;
    }
    prev = curr;
  }
  return crossings / (samples.length - 1);
}

/** Convert linear RMS amplitude `[0..1]` to decibels relative to full scale (dBFS). */
export function rmsToDbfs(rms: number): number {
  if (rms <= 1e-10) {
    return -200;
  }
  return 20 * Math.log10(rms);
}

export interface RmsEnergyGateOptions {
  /** Linear RMS threshold below which a frame is considered silent (default 0.008, ~ -42 dBFS). */
  threshold?: number;
  /** Optional adaptive noise floor tracking (default true). */
  adaptiveNoiseFloor?: boolean;
  /** Multiplier above adaptive noise floor required to open gate (default 2.2). */
  noiseFloorMultiplier?: number;
}

/**
 * Fast deterministic RMS energy gate for short-circuiting silent frames
 * before expensive neural VAD/STT execution.
 */
export class RmsEnergyGate {
  private readonly baseThreshold: number;
  private readonly adaptive: boolean;
  private readonly noiseFloorMultiplier: number;
  private noiseFloor = 0.002;

  constructor(options: RmsEnergyGateOptions = {}) {
    this.baseThreshold = options.threshold ?? 0.008;
    this.adaptive = options.adaptiveNoiseFloor ?? true;
    this.noiseFloorMultiplier = options.noiseFloorMultiplier ?? 2.2;
  }

  get currentNoiseFloor(): number {
    return this.noiseFloor;
  }

  reset(): void {
    this.noiseFloor = 0.002;
  }

  /** Returns true when the frame exceeds the energy gate threshold. */
  isActive(samples: Float32Array, precomputedRms?: number): { active: boolean; rms: number; effectiveThreshold: number } {
    const rms = precomputedRms ?? computeRms(samples);
    const effectiveThreshold = this.adaptive
      ? Math.max(this.baseThreshold, this.noiseFloor * this.noiseFloorMultiplier)
      : this.baseThreshold;

    const active = rms >= effectiveThreshold;

    if (this.adaptive && !active) {
      // Smoothly track quiet background floor without chasing speech bursts.
      this.noiseFloor = this.noiseFloor * 0.95 + rms * 0.05;
    }

    return { active, rms, effectiveThreshold };
  }
}

export interface PcmFrameChunkerOptions {
  /** Frame length in samples (default 512 = 32 ms @ 16 kHz, standard for Silero VAD v5). */
  frameSize?: number;
  /** Sample rate of incoming samples (default 16000). */
  sampleRate?: number;
  /** Initial timestamp offset in ms (default 0). */
  initialTimestampMs?: number;
}

/**
 * AudioWorklet-compatible PCM frame chunker.
 * Accumulates arbitrary-length incoming PCM chunks (e.g., 128-sample Web Audio blocks
 * or ~42-sample resampled outputs) and emits fixed-size `PcmFrame` objects.
 */
export class PcmFrameChunker {
  readonly frameSize: number;
  readonly sampleRate: number;
  readonly frameDurationMs: number;
  private buffer: Float32Array;
  private bufferedCount = 0;
  private emittedSamples = 0;
  private baseTimestampMs: number;

  constructor(options: PcmFrameChunkerOptions = {}) {
    this.frameSize = options.frameSize ?? 512;
    if (!Number.isInteger(this.frameSize) || this.frameSize <= 0) {
      throw new RangeError(`frameSize must be a positive integer, got ${this.frameSize}`);
    }
    this.sampleRate = options.sampleRate ?? TARGET_VOICE_SAMPLE_RATE;
    this.frameDurationMs = (this.frameSize / this.sampleRate) * 1000;
    this.baseTimestampMs = options.initialTimestampMs ?? 0;
    this.buffer = new Float32Array(this.frameSize * 2);
  }

  /** Push a chunk of PCM samples and return zero or more complete fixed-size frames. */
  push(samples: Float32Array, timestampMs?: number): PcmFrame[] {
    if (samples.length === 0) {
      return [];
    }

    if (this.emittedSamples === 0 && this.bufferedCount === 0 && timestampMs !== undefined) {
      this.baseTimestampMs = timestampMs;
    }

    const neededCapacity = this.bufferedCount + samples.length;
    if (neededCapacity > this.buffer.length) {
      const next = new Float32Array(Math.max(neededCapacity, this.buffer.length * 2));
      next.set(this.buffer.subarray(0, this.bufferedCount), 0);
      this.buffer = next;
    }

    this.buffer.set(samples, this.bufferedCount);
    this.bufferedCount += samples.length;

    const frames: PcmFrame[] = [];
    let readOffset = 0;

    while (this.bufferedCount - readOffset >= this.frameSize) {
      const slice = this.buffer.slice(readOffset, readOffset + this.frameSize);
      const frameTimestamp =
        this.baseTimestampMs + (this.emittedSamples / this.sampleRate) * 1000;
      frames.push({
        samples: slice,
        sampleRate: this.sampleRate,
        timestampMs: frameTimestamp,
        durationMs: this.frameDurationMs,
        rms: computeRms(slice),
      });
      readOffset += this.frameSize;
      this.emittedSamples += this.frameSize;
    }

    if (readOffset > 0) {
      const remaining = this.bufferedCount - readOffset;
      if (remaining > 0) {
        this.buffer.copyWithin(0, readOffset, this.bufferedCount);
      }
      this.bufferedCount = remaining;
    }

    return frames;
  }

  /** Flush any trailing partial samples padded with zeros to a full frame. */
  flush(): PcmFrame | null {
    if (this.bufferedCount === 0) {
      return null;
    }
    const frame = new Float32Array(this.frameSize);
    frame.set(this.buffer.subarray(0, this.bufferedCount), 0);
    const frameTimestamp =
      this.baseTimestampMs + (this.emittedSamples / this.sampleRate) * 1000;
    this.emittedSamples += this.bufferedCount;
    this.bufferedCount = 0;

    return {
      samples: frame,
      sampleRate: this.sampleRate,
      timestampMs: frameTimestamp,
      durationMs: this.frameDurationMs,
      rms: computeRms(frame),
    };
  }

  reset(initialTimestampMs = 0): void {
    this.bufferedCount = 0;
    this.emittedSamples = 0;
    this.baseTimestampMs = initialTimestampMs;
    this.buffer.fill(0);
  }
}

/**
 * Self-contained AudioWorkletProcessor script for Web Audio capture.
 * Registered via `Blob` + `URL.createObjectURL` in both Web Demos and MV3 Offscreen Documents.
 * Downmixes multi-channel render quanta (128 frames) to mono Float32Array and transfers ownership
 * over `MessagePort` with zero allocations on the hot path beyond the transferred slice.
 */
export const AUDIO_WORKLET_CAPTURE_PROCESSOR_NAME = 'yours-voice-capture-processor';

export const AUDIO_WORKLET_PROCESSOR_CODE = `
class YoursVoiceCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }
    const ch0 = input[0];
    if (!ch0 || ch0.length === 0) {
      return true;
    }
    const len = ch0.length;
    const mono = new Float32Array(len);
    if (input.length === 1) {
      mono.set(ch0);
    } else {
      const inv = 1 / input.length;
      for (let i = 0; i < len; i++) {
        let sum = 0;
        for (let c = 0; c < input.length; c++) {
          sum += input[c][i] || 0;
        }
        mono[i] = sum * inv;
      }
    }
    this.port.postMessage(
      { type: 'pcm', buffer: mono.buffer, sampleRate, currentTime },
      [mono.buffer]
    );
    return true;
  }
}
registerProcessor('${AUDIO_WORKLET_CAPTURE_PROCESSOR_NAME}', YoursVoiceCaptureProcessor);
`;
