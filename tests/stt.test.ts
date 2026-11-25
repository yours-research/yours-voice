import { describe, expect, it, vi } from 'vitest';
import type { TranscriptSegment } from '../src/contracts.js';
import {
  HallucinationGuard,
  MoonshineSttRunner,
  SttEngine,
  WhisperSttRunner,
} from '../src/stt/index.js';

describe('SttEngine, HallucinationGuard & Runners', () => {
  it('throttles rapid streaming partial segments while always emitting final segments', async () => {
    let fakeClock = 0;
    const stt = new SttEngine({
      partialThrottleMs: 100,
      nowMs: () => fakeClock,
      runner: {
        id: 'stream-test',
        kind: 'whisper',
        async transcribe() {
          return { text: 'unused' };
        },
        async *transcribeStream() {
          fakeClock = 0;
          yield { text: 'hel', isFinal: false, startMs: 0, endMs: 100 };
          fakeClock = 30; // throttled (<100ms)
          yield { text: 'hello', isFinal: false, startMs: 0, endMs: 130 };
          fakeClock = 120; // allowed (>=100ms)
          yield { text: 'hello wor', isFinal: false, startMs: 0, endMs: 220 };
          fakeClock = 130; // final never throttled
          yield { text: 'hello world', isFinal: true, startMs: 0, endMs: 250, confidence: 0.96 };
        },
      },
    });

    const emitted: TranscriptSegment[] = [];
    const finals = await stt.transcribe(new Float32Array(4000).fill(0.1), {
      onSegment: (seg) => emitted.push(seg),
    });

    expect(emitted.map((s) => `${s.isFinal ? 'F' : 'P'}:${s.text}`)).toEqual([
      'P:hel',
      'P:hello wor',
      'F:hello world',
    ]);
    expect(finals).toHaveLength(1);
    expect(finals[0]?.text).toBe('hello world');
  });

  it('forwards language hints to WhisperSttRunner and MoonshineSttRunner', async () => {
    const whisperInfer = vi.fn(async (_audio, opts) => ({
      text: `lang=${opts.language}`,
      confidence: 0.95,
    }));
    const moonshineInfer = vi.fn(async (_audio, opts) => ({
      text: `moon=${opts.language}`,
      confidence: 0.92,
    }));

    const whisper = new SttEngine({
      runner: new WhisperSttRunner({ infer: whisperInfer }),
      language: 'en',
    });
    const moonshine = new SttEngine({
      runner: new MoonshineSttRunner({ infer: moonshineInfer }),
      language: 'es',
    });

    const r1 = await whisper.transcribe(new Float32Array(1600).fill(0.2));
    const r2 = await moonshine.transcribe(new Float32Array(1600).fill(0.2), { language: 'fr' });

    expect(r1[0]?.text).toBe('lang=en');
    expect(r2[0]?.text).toBe('moon=fr');
  });

  it('suppresses Whisper phantom silence hallucinations and repetition loops', () => {
    const guard = new HallucinationGuard();

    const phantom = guard.evaluate('Thank you for watching.', {
      audioRms: 0.002,
      durationMs: 1200,
      confidence: 0.6,
    });
    expect(phantom.suppressed).toBe(true);

    const pureLoop = guard.evaluate('you you you you you you', {
      audioRms: 0.1,
      durationMs: 1000,
    });
    expect(pureLoop.suppressed).toBe(true);

    const tailLoop = guard.evaluate('open the active tab please please please please please', {
      audioRms: 0.15,
      durationMs: 2000,
      confidence: 0.9,
    });
    expect(tailLoop.suppressed).toBe(false);
    expect(tailLoop.text).toBe('open the active tab please');

    const legitimate = guard.evaluate('Thank you for watching.', {
      audioRms: 0.18,
      durationMs: 1500,
      confidence: 0.95,
    });
    expect(legitimate.suppressed).toBe(false);
  });

  it('aborts cleanly without emitting segments when AbortSignal is triggered', async () => {
    const controller = new AbortController();
    const stt = new SttEngine({
      runner: new WhisperSttRunner({
        infer: async () => {
          controller.abort();
          return { text: 'stale output' };
        },
      }),
    });

    const emitted: TranscriptSegment[] = [];
    const res = await stt.transcribe(new Float32Array(3200).fill(0.2), {
      signal: controller.signal,
      onSegment: (s) => emitted.push(s),
    });

    expect(res).toEqual([]);
    expect(emitted).toEqual([]);
  });
});
