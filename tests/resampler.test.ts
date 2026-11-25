import { describe, expect, it } from 'vitest';
import {
  PcmFrameChunker,
  PcmResampler,
  TARGET_VOICE_SAMPLE_RATE,
  computeRms,
  downmixToMono,
  resamplePcm,
} from '../src/audio/index.js';

function generateSineWave(freqHz: number, sampleRate: number, durationSec: number, amplitude = 0.8): Float32Array {
  const count = Math.floor(sampleRate * durationSec);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = amplitude * Math.sin((2 * Math.PI * freqHz * i) / sampleRate);
  }
  return out;
}

/** Measure normalized correlation/energy at a target frequency via Goertzel/DFT. */
function measureToneMagnitude(samples: Float32Array, sampleRate: number, targetFreqHz: number): number {
  let re = 0;
  let im = 0;
  const omega = (2 * Math.PI * targetFreqHz) / sampleRate;
  // Skip first/last 20ms edge transients
  const margin = Math.min(Math.floor(sampleRate * 0.02), Math.floor(samples.length / 6));
  const len = samples.length - 2 * margin;
  for (let i = margin; i < samples.length - margin; i++) {
    const angle = omega * i;
    re += samples[i]! * Math.cos(angle);
    im -= samples[i]! * Math.sin(angle);
  }
  return (2 * Math.hypot(re, im)) / len;
}

describe('PcmResampler & PcmFrameChunker', () => {
  it('downsamples 48 kHz sine wave to 16 kHz preserving exact frequency & amplitude', () => {
    const input = generateSineWave(1000, 48_000, 0.5, 0.75);
    const output = resamplePcm(input, 48_000, TARGET_VOICE_SAMPLE_RATE, 'polyphase');

    expect(output.length).toBeGreaterThanOrEqual(7950);
    expect(output.length).toBeLessThanOrEqual(8050);

    const mag1k = measureToneMagnitude(output, 16_000, 1000);
    const mag440 = measureToneMagnitude(output, 16_000, 440);
    expect(mag1k).toBeGreaterThan(0.7);
    expect(mag1k).toBeLessThan(0.8);
    expect(mag440).toBeLessThan(0.02);
  });

  it('downsamples 44.1 kHz sine wave to 16 kHz accurately across 128-sample AudioWorklet chunks', () => {
    const input = generateSineWave(440, 44_100, 0.5, 0.6);
    const resampler = new PcmResampler({ inputSampleRate: 44_100, outputSampleRate: 16_000, mode: 'polyphase' });
    const chunker = new PcmFrameChunker({ frameSize: 512, sampleRate: 16_000 });

    const emittedFrames: Float32Array[] = [];
    for (let i = 0; i < input.length; i += 128) {
      const block = input.subarray(i, Math.min(i + 128, input.length));
      const resampled = resampler.process(block);
      for (const frame of chunker.push(resampled)) {
        expect(frame.samples.length).toBe(512);
        expect(frame.sampleRate).toBe(16_000);
        emittedFrames.push(frame.samples);
      }
    }

    const tailResampled = resampler.flush();
    for (const frame of chunker.push(tailResampled)) {
      emittedFrames.push(frame.samples);
    }

    // Merge frames and verify continuous phase at 440 Hz
    const total = new Float32Array(emittedFrames.length * 512);
    emittedFrames.forEach((f, idx) => total.set(f, idx * 512));

    const mag440 = measureToneMagnitude(total, 16_000, 440);
    const mag1000 = measureToneMagnitude(total, 16_000, 1000);
    expect(mag440).toBeGreaterThan(0.55);
    expect(mag440).toBeLessThan(0.65);
    expect(mag1000).toBeLessThan(0.02);
  });

  it('attenuates frequencies above 16 kHz Nyquist (anti-aliasing lowpass in polyphase mode)', () => {
    // 14 kHz tone at 48 kHz would alias to 2 kHz in naive decimation without anti-aliasing
    const highFreq = generateSineWave(14_000, 48_000, 0.4, 0.8);
    const downsampled = resamplePcm(highFreq, 48_000, 16_000, 'polyphase');
    const rmsOut = computeRms(downsampled.subarray(200, downsampled.length - 200));
    expect(rmsOut).toBeLessThan(0.05);
  });

  it('downmixes stereo channels cleanly to mono', () => {
    const left = new Float32Array([0.8, -0.4, 0.2]);
    const right = new Float32Array([0.2, 0.4, -0.2]);
    const mono = downmixToMono([left, right]);
    expect(mono[0]).toBeCloseTo(0.5, 5);
    expect(mono[1]).toBeCloseTo(0.0, 5);
    expect(mono[2]).toBeCloseTo(0.0, 5);
  });
});
