/**
 * High-quality browser-native Float32Array PCM resampler (48 kHz / 44.1 kHz -> 16 kHz mono).
 *
 * Supports:
 * - Multi-channel downmix to mono
 * - Windowed-sinc (Blackman-Harris / Hann) anti-aliased polyphase interpolation
 * - Fast linear interpolation mode
 * - Stateful streaming block processing with fractional phase & overlap preservation
 */

export type ResamplerMode = 'polyphase' | 'linear';

export interface ResamplerOptions {
  inputSampleRate: number;
  outputSampleRate?: number;
  mode?: ResamplerMode;
  /** Half-width of windowed-sinc kernel in output/input taps (default 12 -> 24-tap filter). */
  sincHalfWidth?: number;
}

export const TARGET_VOICE_SAMPLE_RATE = 16_000;

/** Mix N mono/stereo/multi-channel Float32Arrays down to a single mono Float32Array. */
export function downmixToMono(channels: readonly Float32Array[]): Float32Array {
  if (channels.length === 0) {
    return new Float32Array(0);
  }
  if (channels.length === 1) {
    return channels[0]!;
  }

  const length = channels[0]!.length;
  const mono = new Float32Array(length);
  const invCount = 1 / channels.length;

  for (let i = 0; i < length; i++) {
    let sum = 0;
    for (let ch = 0; ch < channels.length; ch++) {
      sum += channels[ch]![i] ?? 0;
    }
    mono[i] = sum * invCount;
  }
  return mono;
}

function sinc(x: number): number {
  if (Math.abs(x) < 1e-8) {
    return 1;
  }
  const pix = Math.PI * x;
  return Math.sin(pix) / pix;
}

/** 3-term Blackman window over [-halfWidth, +halfWidth] for low sidelobe aliasing rejection. */
function blackmanWindow(offset: number, halfWidth: number): number {
  const ratio = (offset + halfWidth) / (2 * halfWidth);
  if (ratio <= 0 || ratio >= 1) {
    return 0;
  }
  const twoPiR = 2 * Math.PI * ratio;
  return 0.42 - 0.5 * Math.cos(twoPiR) + 0.08 * Math.cos(2 * twoPiR);
}

/**
 * Stateful streaming PCM resampler suitable for AudioWorklet 128-sample blocks
 * or arbitrary chunk sizes, converting cleanly to 16 kHz mono with zero click boundaries.
 */
export class PcmResampler {
  readonly inputSampleRate: number;
  readonly outputSampleRate: number;
  readonly mode: ResamplerMode;
  private readonly sincHalfWidth: number;
  private readonly step: number;
  private readonly cutoff: number;
  private readonly kernelRadiusInInput: number;
  private history: Float32Array = new Float32Array(0);
  private phase = 0;

  constructor(options: ResamplerOptions) {
    if (!Number.isFinite(options.inputSampleRate) || options.inputSampleRate <= 0) {
      throw new RangeError(`Invalid inputSampleRate: ${options.inputSampleRate}`);
    }
    const outputRate = options.outputSampleRate ?? TARGET_VOICE_SAMPLE_RATE;
    if (!Number.isFinite(outputRate) || outputRate <= 0) {
      throw new RangeError(`Invalid outputSampleRate: ${outputRate}`);
    }

    this.inputSampleRate = options.inputSampleRate;
    this.outputSampleRate = outputRate;
    this.mode = options.mode ?? 'polyphase';
    this.sincHalfWidth = Math.max(4, options.sincHalfWidth ?? 12);
    this.step = this.inputSampleRate / this.outputSampleRate;

    // Anti-aliasing low-pass cutoff scaled by downsampling ratio so >Nyquist energy is attenuated.
    this.cutoff = this.step > 1 ? 0.9 / this.step : 0.95;
    this.kernelRadiusInInput =
      this.mode === 'linear' ? 1 : Math.ceil(this.sincHalfWidth * Math.max(1, this.step));
  }

  /** Reset internal phase and overlap history. */
  reset(): void {
    this.history = new Float32Array(0);
    this.phase = 0;
  }

  /**
   * Process a chunk of mono PCM samples and emit resampled samples.
   * For multi-channel input, call `downmixToMono(channels)` first or pass `processChannels`.
   */
  process(input: Float32Array): Float32Array {
    if (input.length === 0) {
      return new Float32Array(0);
    }

    if (this.inputSampleRate === this.outputSampleRate) {
      return new Float32Array(input);
    }

    const combined = new Float32Array(this.history.length + input.length);
    combined.set(this.history, 0);
    combined.set(input, this.history.length);

    if (this.mode === 'linear') {
      return this.processLinear(combined);
    }

    return this.processPolyphase(combined);
  }

  /** Downmix multiple channels and resample to output sample rate. */
  processChannels(channels: readonly Float32Array[]): Float32Array {
    return this.process(downmixToMono(channels));
  }

  /** Flush remaining tail samples at end of stream (zero-padded). */
  flush(): Float32Array {
    if (this.history.length === 0 || this.inputSampleRate === this.outputSampleRate) {
      this.reset();
      return new Float32Array(0);
    }

    const tailPad = new Float32Array(this.kernelRadiusInInput + 1);
    const out = this.process(tailPad);
    this.reset();
    return out;
  }

  private processLinear(combined: Float32Array): Float32Array {
    const maxPos = combined.length - 1;
    if (this.phase >= maxPos) {
      const keepStart = Math.max(0, combined.length - 2);
      this.phase -= keepStart;
      this.history = combined.slice(keepStart);
      return new Float32Array(0);
    }

    const estCount = Math.max(0, Math.floor((maxPos - this.phase) / this.step) + 1);
    const out = new Float32Array(estCount);
    let outIdx = 0;
    let pos = this.phase;

    while (pos < maxPos && outIdx < estCount) {
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const s0 = combined[idx] ?? 0;
      const s1 = combined[idx + 1] ?? s0;
      out[outIdx++] = s0 + (s1 - s0) * frac;
      pos += this.step;
    }

    const keepStart = Math.max(0, Math.floor(pos));
    this.phase = pos - keepStart;
    this.history = combined.slice(keepStart);

    return outIdx === out.length ? out : out.subarray(0, outIdx);
  }

  private processPolyphase(combined: Float32Array): Float32Array {
    const radius = this.kernelRadiusInInput;
    const maxCenter = combined.length - radius - 1;

    if (maxCenter < 0 || this.phase > maxCenter) {
      const keepLen = Math.min(combined.length, radius * 2 + 2);
      const keepStart = combined.length - keepLen;
      this.phase = Math.max(0, this.phase - keepStart);
      this.history = combined.slice(keepStart);
      return new Float32Array(0);
    }

    const estCount = Math.max(0, Math.floor((maxCenter - this.phase) / this.step) + 1);
    const out = new Float32Array(estCount);
    let outIdx = 0;
    let pos = this.phase;
    const scale = this.step > 1 ? 1 / this.step : 1;

    while (pos <= maxCenter && outIdx < estCount) {
      const center = Math.floor(pos);
      const start = Math.max(0, center - radius);
      const end = Math.min(combined.length - 1, center + radius);

      let sum = 0;
      let weightSum = 0;

      for (let i = start; i <= end; i++) {
        const deltaInput = i - pos;
        const scaledDelta = deltaInput * scale;
        if (Math.abs(scaledDelta) >= this.sincHalfWidth) {
          continue;
        }
        const w = sinc(deltaInput * this.cutoff) * blackmanWindow(scaledDelta, this.sincHalfWidth);
        sum += (combined[i] ?? 0) * w;
        weightSum += w;
      }

      out[outIdx++] = weightSum !== 0 ? sum / weightSum : 0;
      pos += this.step;
    }

    const keepFrom = Math.max(0, Math.floor(pos) - radius);
    this.phase = pos - keepFrom;
    this.history = combined.slice(keepFrom);

    return outIdx === out.length ? out : out.subarray(0, outIdx);
  }
}

/** Convenience one-shot resampler for complete PCM buffers (default target 16 kHz). */
export function resamplePcm(
  samples: Float32Array,
  inputSampleRate: number,
  outputSampleRate = TARGET_VOICE_SAMPLE_RATE,
  mode: ResamplerMode = 'polyphase',
): Float32Array {
  if (inputSampleRate === outputSampleRate) {
    return new Float32Array(samples);
  }
  const resampler = new PcmResampler({ inputSampleRate, outputSampleRate, mode });
  const expectedLen = Math.round((samples.length * outputSampleRate) / inputSampleRate);
  const head = resampler.process(samples);
  const tail = resampler.flush();
  const combined = new Float32Array(head.length + tail.length);
  combined.set(head, 0);
  combined.set(tail, head.length);
  return combined.length > expectedLen ? combined.subarray(0, expectedLen) : combined;
}
