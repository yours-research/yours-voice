import { describe, expect, it } from 'vitest';
import {
  KokoroTtsRunner,
  NullAudioPlaybackSink,
  SentenceStreamChunker,
  TtsEngine,
  chunkTextStream,
} from '../src/tts/index.js';

async function* tokenStream(tokens: string[], delayMs = 0): AsyncIterable<string> {
  for (const tok of tokens) {
    if (delayMs > 0) {
      await new Promise((r) => setTimeout(r, delayMs));
    }
    yield tok;
  }
}

describe('SentenceStreamChunker & Streaming TtsEngine', () => {
  it('splits on true sentence boundaries without breaking abbreviations, versions, or domains', async () => {
    const deltas = [
      'Dr.',
      ' Smith',
      ' released',
      ' v1',
      '.2.',
      '3',
      ' at',
      ' yours',
      '.sh',
      ' (e.g.',
      ' local',
      ' AI).',
      ' Is',
      ' it',
      ' fast?',
      ' Yes!',
      ' Wait...',
      ' It',
      ' runs',
      ' locally.',
    ];

    const chunks: string[] = [];
    for await (const chunk of chunkTextStream(tokenStream(deltas))) {
      chunks.push(chunk);
    }

    expect(chunks).toEqual([
      'Dr. Smith released v1.2.3 at yours.sh (e.g. local AI).',
      'Is it fast?',
      'Yes!',
      'Wait...',
      'It runs locally.',
    ]);
  });

  it('splits long run-on sentences cleanly at clause boundaries past maxChunkChars', () => {
    const chunker = new SentenceStreamChunker({ minClauseChars: 25, maxChunkChars: 65 });
    const out1 = chunker.feed(
      'Local inference keeps every private document on your machine, and prevents accidental cloud leakage ',
    );
    const out2 = chunker.flush();

    expect(out1.length).toBe(1);
    expect(out1[0]).toBe('Local inference keeps every private document on your machine,');
    expect(out2).toEqual(['and prevents accidental cloud leakage']);
  });

  it('synthesizes ahead of playback in queue order and cancels within <30ms on bargeIn()', async () => {
    const synthesizedTexts: string[] = [];
    const runner = new KokoroTtsRunner({
      infer: async (text, _opts, signal) => {
        synthesizedTexts.push(text);
        await new Promise<void>((resolve) => {
          const timer = setTimeout(resolve, 15);
          signal?.addEventListener(
            'abort',
            () => {
              clearTimeout(timer);
              resolve();
            },
            { once: true },
          );
        });
        return new Float32Array(24_000 * 0.5); // 500ms chunk @ 24kHz
      },
    });

    const sink = new NullAudioPlaybackSink({ simulateDuration: true });
    const tts = new TtsEngine({ runner, sink, lookaheadChunks: 2 });

    const speakPromise = tts.speak(
      tokenStream(['First sentence. ', 'Second sentence. ', 'Third sentence. ', 'Fourth sentence.'], 2),
    );

    // Wait until chunk 0 is actively playing and lookahead has started
    await new Promise((r) => setTimeout(r, 45));
    expect(tts.isSpeaking).toBe(true);

    const cancelLatencyMs = await tts.bargeIn();
    await speakPromise;

    expect(cancelLatencyMs).toBeLessThan(30);
    expect(tts.isSpeaking).toBe(false);
    // Should have stopped before playing all 4 sentences
    expect(sink.playedChunks.length).toBeLessThan(4);
    expect(synthesizedTexts[0]).toBe('First sentence.');
  });
});
