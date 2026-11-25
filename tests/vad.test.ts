import { describe, expect, it } from 'vitest';
import type { PcmFrame, VadEvent } from '../src/contracts.js';
import {
  EnergyZeroCrossingVadRunner,
  SileroVadRunner,
  VadEngine,
  type VadModelRunner,
} from '../src/vad/index.js';

function makeFrame(index: number, fillValue: number, frameSize = 512, sampleRate = 16_000): PcmFrame {
  const samples = new Float32Array(frameSize).fill(fillValue);
  const durationMs = (frameSize / sampleRate) * 1000; // 32ms
  return {
    samples,
    sampleRate,
    timestampMs: index * durationMs,
    durationMs,
    rms: Math.abs(fillValue),
  };
}

class ScriptedVadRunner implements VadModelRunner {
  readonly id = 'scripted-vad';
  private index = 0;
  constructor(private readonly probs: number[]) {}
  processFrame(): number {
    return this.probs[this.index++] ?? 0;
  }
  reset(): void {
    this.index = 0;
  }
}

describe('VadEngine hysteresis & pre-roll ring buffer', () => {
  it('preserves initial pre-roll frames before speech onset so initial consonants are never clipped', async () => {
    // 5 frames: [0, 1, 2] are quiet pre-roll with marker values [0.01, 0.02, 0.03]; [3, 4] trigger speech_start
    const probs = [0.05, 0.05, 0.05, 0.85, 0.90];
    const vad = new VadEngine({
      runner: new ScriptedVadRunner(probs),
      positiveThreshold: 0.55,
      negativeThreshold: 0.35,
      preRollMs: 96, // exactly 3 * 32ms = 1536 samples
      minSpeechMs: 64, // 2 frames
      hangoverMs: 128,
    });

    const events: VadEvent[] = [];
    const markers = [0.01, 0.02, 0.03, 0.50, 0.60];
    for (let i = 0; i < markers.length; i++) {
      events.push(...(await vad.processFrame(makeFrame(i, markers[i]!))));
    }

    const startEvent = events.find((e) => e.type === 'speech_start');
    expect(startEvent).toBeDefined();
    if (startEvent?.type === 'speech_start') {
      // 3 pre-roll frames + 2 onset confirmation frames = 5 frames * 512 samples = 2560 samples
      expect(startEvent.preRollAudio.length).toBe(5 * 512);
      expect(startEvent.preRollAudio[0]).toBeCloseTo(0.01, 5);
      expect(startEvent.preRollAudio[512]).toBeCloseTo(0.02, 5);
      expect(startEvent.preRollAudio[2 * 512]).toBeCloseTo(0.03, 5);
      expect(startEvent.preRollAudio[3 * 512]).toBeCloseTo(0.50, 5);
      expect(startEvent.preRollAudio[4 * 512]).toBeCloseTo(0.60, 5);
    }
  });

  it('rejects brief noise impulses shorter than minSpeechMs and retains pre-roll continuity', async () => {
    const probs = [0.05, 0.85, 0.1, 0.05];
    const vad = new VadEngine({
      runner: new ScriptedVadRunner(probs),
      positiveThreshold: 0.55,
      negativeThreshold: 0.35,
      preRollMs: 128,
      minSpeechMs: 96, // requires 3 frames (96ms), impulse is only 1 frame (32ms)
      hangoverMs: 128,
    });

    const events: VadEvent[] = [];
    for (let i = 0; i < probs.length; i++) {
      events.push(...(await vad.processFrame(makeFrame(i, i * 0.1))));
    }

    expect(events.some((e) => e.type === 'speech_start')).toBe(false);
    expect(vad.currentState).toBe('silence');
  });

  it('bridges micro-pauses shorter than hangoverMs and finalizes utterance after silenceEndMs', async () => {
    // 2 onset frames (64ms) -> 2 low frames during pause (64ms < 128ms hangover) -> 1 recovery frame -> 4 silence frames (128ms)
    const probs = [
      0.85, 0.85, // speech_start at frame 1
      0.20, 0.20, // hangover (64ms < 128ms)
      0.75,       // recovers to speech!
      0.10, 0.10, 0.10, 0.10, // 4 * 32ms = 128ms silence -> speech_end
    ];
    const vad = new VadEngine({
      runner: new ScriptedVadRunner(probs),
      positiveThreshold: 0.55,
      negativeThreshold: 0.35,
      preRollMs: 64,
      minSpeechMs: 64,
      hangoverMs: 128,
      silenceEndMs: 128,
    });

    const events: VadEvent[] = [];
    for (let i = 0; i < probs.length; i++) {
      events.push(...(await vad.processFrame(makeFrame(i, 0.25))));
    }

    const starts = events.filter((e) => e.type === 'speech_start');
    const ends = events.filter((e) => e.type === 'speech_end');
    expect(starts).toHaveLength(1);
    expect(ends).toHaveLength(1);
    if (ends[0]?.type === 'speech_end') {
      expect(ends[0].audio.length).toBe(9 * 512);
      expect(ends[0].startMs).toBe(0);
      expect(ends[0].endMs).toBe(9 * 32);
    }
    expect(vad.currentState).toBe('silence');
  });

  it('EnergyZeroCrossingVadRunner distinguishes silence from voiced speech-band signal', () => {
    const runner = new EnergyZeroCrossingVadRunner();
    const silence = new Float32Array(512);
    // 250 Hz voiced tone @ 16 kHz -> ZCR ~ 0.031 (inside speech band)
    const voiced = new Float32Array(512);
    for (let i = 0; i < 512; i++) {
      voiced[i] = 0.25 * Math.sin((2 * Math.PI * 250 * i) / 16_000);
    }

    expect(runner.processFrame(silence)).toBeLessThan(0.2);
    expect(runner.processFrame(voiced)).toBeGreaterThan(0.6);
  });

  it('SileroVadRunner manages v5 64-sample context window and [2,1,128] state tensor', async () => {
    const capturedInputs: number[] = [];
    const runner = new SileroVadRunner({
      version: 'v5',
      session: {
        async run(feeds) {
          capturedInputs.push(feeds['input']!.data.length);
          expect(feeds['state']!.dims).toEqual([2, 1, 128]);
          return {
            output: { data: new Float32Array([0.78]) },
            stateN: { data: new Float32Array(2 * 1 * 128).fill(0.1) },
          };
        },
      },
    });

    const p1 = await runner.processFrame(new Float32Array(512).fill(0.2), 16_000);
    const p2 = await runner.processFrame(new Float32Array(512).fill(0.3), 16_000);
    expect(p1).toBeCloseTo(0.78, 5);
    expect(p2).toBeCloseTo(0.78, 5);
    expect(capturedInputs).toEqual([576, 576]);
  });
});
