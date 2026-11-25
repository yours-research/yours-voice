import { describe, expect, it } from 'vitest';
import type { AudioInputSource, TranscriptSegment, VoiceState } from '../src/contracts.js';
import { LocalVoicePipeline } from '../src/pipeline/index.js';
import { SttEngine, WhisperSttRunner } from '../src/stt/index.js';
import { KokoroTtsRunner, NullAudioPlaybackSink, TtsEngine } from '../src/tts/index.js';
import { VadEngine, type VadModelRunner } from '../src/vad/index.js';

class MockAudioInputSource implements AudioInputSource {
  readonly sampleRate = 16_000;
  isActive = false;
  private listener: ((samples: Float32Array, sampleRate: number, timestampMs: number) => void) | null = null;

  async start(onFrame: (samples: Float32Array, sampleRate: number, timestampMs: number) => void): Promise<void> {
    this.isActive = true;
    this.listener = onFrame;
  }

  async stop(): Promise<void> {
    this.isActive = false;
    this.listener = null;
  }

  pushFrame(samples: Float32Array, timestampMs: number): void {
    this.listener?.(samples, this.sampleRate, timestampMs);
  }
}

describe('LocalVoicePipeline FSM & Duplex Barge-In', () => {
  it('executes full FSM lifecycle and simultaneous duplex barge-in with pre-roll preservation', async () => {
    const input = new MockAudioInputSource();

    // Energy-thresholded deterministic VAD runner: amplitude >= 0.2 -> speech (0.9), else silence (0.05)
    const thresholdVadRunner: VadModelRunner = {
      id: 'threshold-test',
      processFrame(frame) {
        const amp = Math.abs(frame[0] ?? 0);
        return amp >= 0.2 ? 0.9 : 0.05;
      },
    };

    const vad = new VadEngine({
      runner: thresholdVadRunner,
      positiveThreshold: 0.55,
      negativeThreshold: 0.35,
      preRollMs: 64, // 2 frames * 32ms
      minSpeechMs: 64, // 2 frames
      hangoverMs: 64,
      silenceEndMs: 64,
    });

    const transcribedLengths: number[] = [];
    const stt = new SttEngine({
      runner: new WhisperSttRunner({
        infer: async (audio) => {
          transcribedLengths.push(audio.length);
          return { text: 'Stop and open tabs', confidence: 0.97 };
        },
      }),
    });

    const ttsSink = new NullAudioPlaybackSink({ simulateDuration: true });
    let abortSignalFired = false;
    const tts = new TtsEngine({
      sink: ttsSink,
      runner: new KokoroTtsRunner({
        infer: async (text, _opts, signal) => {
          signal?.addEventListener('abort', () => {
            abortSignalFired = true;
          });
          // Simulate 500ms audio output per sentence
          return new Float32Array(24_000 * 0.5);
        },
      }),
    });

    const pipeline = new LocalVoicePipeline({ inputSource: input, vad, stt, tts });
    const states: VoiceState[] = [];
    const transcripts: TranscriptSegment[] = [];
    let bargeInLatency = -1;

    pipeline.on('stateChange', ({ current }) => states.push(current));
    pipeline.on('transcript', (seg) => transcripts.push(seg));
    pipeline.on('bargeIn', (ev) => {
      bargeInLatency = ev.cancelLatencyMs;
    });

    expect(pipeline.state).toBe('idle');
    await pipeline.startListening({ language: 'en', bargeInEnabled: true });
    expect(pipeline.state).toBe('listening');

    // Start slow streaming TTS while listening (full duplex!)
    async function* slowLlmResponse() {
      yield 'First long assistant sentence. ';
      await new Promise((r) => setTimeout(r, 80));
      yield 'Second sentence that should never play.';
    }

    const speakPromise = pipeline.speak(slowLlmResponse());
    // Wait until chunk 0 is actively playing
    await new Promise((r) => setTimeout(r, 30));
    expect(pipeline.state).toBe('speaking');

    // Push 2 quiet pre-roll frames, then 3 loud speech frames during TTS playback to trigger duplex barge-in
    const frame32ms = (val: number) => new Float32Array(512).fill(val);
    input.pushFrame(frame32ms(0.02), 0);
    input.pushFrame(frame32ms(0.03), 32);
    input.pushFrame(frame32ms(0.45), 64);
    input.pushFrame(frame32ms(0.50), 96);

    // Wait for async frame queue + barge-in cancellation
    await new Promise((r) => setTimeout(r, 20));
    await speakPromise;

    expect(bargeInLatency).toBeGreaterThanOrEqual(0);
    expect(bargeInLatency).toBeLessThan(30);
    expect(pipeline.state).toBe('speech-detected');
    expect(tts.isSpeaking).toBe(false);
    expect(ttsSink.playedChunks).toHaveLength(1);
    expect(abortSignalFired || tts.isSpeaking === false).toBe(true);

    // End the interrupting user speech (2 silence frames = 64ms)
    input.pushFrame(frame32ms(0.01), 128);
    input.pushFrame(frame32ms(0.01), 160);
    await new Promise((r) => setTimeout(r, 30));

    expect(transcripts).toHaveLength(1);
    expect(transcripts[0]?.text).toBe('Stop and open tabs');
    // 2 pre-roll frames + 2 onset frames + 2 silence hangover frames = 6 frames = 3072 samples
    expect(transcribedLengths[0]).toBe(6 * 512);
    expect(pipeline.state).toBe('listening');

    await pipeline.stopListening();
    expect(pipeline.state).toBe('idle');

    // Dispose is idempotent
    await pipeline.dispose();
    await pipeline.dispose();
  });
});
