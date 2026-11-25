import { computeRms } from '../audio/frame-chunker.js';
import { TARGET_VOICE_SAMPLE_RATE } from '../audio/resampler.js';
import type { TranscriptSegment } from '../contracts.js';

export type SttRunnerKind = 'whisper' | 'moonshine' | 'custom';

export interface SttTranscribeOptions {
  language?: string;
  sampleRate?: number;
  startMs?: number;
  endMs?: number;
  prompt?: string;
}

export interface SttRunnerResult {
  text: string;
  confidence?: number;
  language?: string;
  segments?: Array<{
    text: string;
    startMs?: number;
    endMs?: number;
    confidence?: number;
  }>;
}

/** Pluggable local WebGPU / WASM STT runner interface (Whisper, Moonshine, etc.). */
export interface SttModelRunner {
  readonly id: string;
  readonly kind: SttRunnerKind;
  init?(): Promise<void>;
  transcribe(
    audio: Float32Array,
    options: SttTranscribeOptions,
    signal?: AbortSignal,
  ): Promise<SttRunnerResult>;
  transcribeStream?(
    audio: Float32Array,
    options: SttTranscribeOptions,
    signal?: AbortSignal,
  ): AsyncIterable<TranscriptSegment>;
  dispose?(): Promise<void>;
}

export type WhisperInferFn = (
  audio: Float32Array,
  options: SttTranscribeOptions & { task: 'transcribe' },
  signal?: AbortSignal,
) => Promise<SttRunnerResult>;

/**
 * Adapter wrapping a local WebGPU / WASM Whisper runner callback
 * (e.g. Transformers.js `pipeline('automatic-speech-recognition')` or custom ORT/WGSL Whisper).
 */
export class WhisperSttRunner implements SttModelRunner {
  readonly id: string;
  readonly kind = 'whisper' as const;
  private readonly inferFn: WhisperInferFn;
  private readonly initFn?: () => Promise<void>;
  private readonly disposeFn?: () => Promise<void>;

  constructor(options: {
    id?: string;
    infer: WhisperInferFn;
    init?: () => Promise<void>;
    dispose?: () => Promise<void>;
  }) {
    this.id = options.id ?? 'whisper-local';
    this.inferFn = options.infer;
    this.initFn = options.init;
    this.disposeFn = options.dispose;
  }

  async init(): Promise<void> {
    await this.initFn?.();
  }

  async transcribe(
    audio: Float32Array,
    options: SttTranscribeOptions,
    signal?: AbortSignal,
  ): Promise<SttRunnerResult> {
    return this.inferFn(audio, { ...options, task: 'transcribe' }, signal);
  }

  async dispose(): Promise<void> {
    await this.disposeFn?.();
  }
}

export type MoonshineInferFn = (
  audio: Float32Array,
  options: SttTranscribeOptions,
  signal?: AbortSignal,
) => Promise<SttRunnerResult>;

/**
 * Adapter wrapping a local Moonshine ONNX/WASM/WebGPU runner for ultra-low-latency short-turn STT.
 */
export class MoonshineSttRunner implements SttModelRunner {
  readonly id: string;
  readonly kind = 'moonshine' as const;
  private readonly inferFn: MoonshineInferFn;
  private readonly initFn?: () => Promise<void>;
  private readonly disposeFn?: () => Promise<void>;

  constructor(options: {
    id?: string;
    infer: MoonshineInferFn;
    init?: () => Promise<void>;
    dispose?: () => Promise<void>;
  }) {
    this.id = options.id ?? 'moonshine-local';
    this.inferFn = options.infer;
    this.initFn = options.init;
    this.disposeFn = options.dispose;
  }

  async init(): Promise<void> {
    await this.initFn?.();
  }

  async transcribe(
    audio: Float32Array,
    options: SttTranscribeOptions,
    signal?: AbortSignal,
  ): Promise<SttRunnerResult> {
    return this.inferFn(audio, options, signal);
  }

  async dispose(): Promise<void> {
    await this.disposeFn?.();
  }
}

export interface HallucinationGuardOptions {
  /** Maximum consecutive repeats of a word or short n-gram allowed before collapsing/suppressing (default 4). */
  maxConsecutiveRepeats?: number;
  /** Maximum plausible speaking rate in words/sec before treating burst as hallucination loop (default 14). */
  maxWordsPerSecond?: number;
  /** Minimum audio RMS below which known phantom phrases are dropped (default 0.01). */
  minAudioRmsForPhantom?: number;
  /** Confidence threshold below which known phantom phrases are dropped (default 0.85). */
  minConfidenceForPhantom?: number;
  /** Additional normalized phantom phrases to guard against. */
  extraPhantomPhrases?: readonly string[];
}

const DEFAULT_PHANTOM_PHRASES = new Set<string>([
  'thank you for watching',
  'thanks for watching',
  'thank you for listening',
  'thanks for listening',
  'please subscribe',
  'like and subscribe',
  'subscribe to my channel',
  'subtitles by',
  'captions by',
  'amara org',
  'mbc news',
  'you',
  'bye',
  'bye bye',
]);

const NON_SPEECH_TAG_RE = /^(?:\[[^\]]+\]|\([^)]+\)|♪+|🎵+|\.+|\s+)+$/i;

export interface HallucinationCheckResult {
  text: string;
  suppressed: boolean;
  reason?: string;
}

/**
 * Guards local Whisper/Moonshine outputs against common silence hallucinations
 * and autoregressive n-gram repetition loops.
 */
export class HallucinationGuard {
  private readonly maxConsecutiveRepeats: number;
  private readonly maxWordsPerSecond: number;
  private readonly minAudioRmsForPhantom: number;
  private readonly minConfidenceForPhantom: number;
  private readonly phantomPhrases: Set<string>;

  constructor(options: HallucinationGuardOptions = {}) {
    this.maxConsecutiveRepeats = Math.max(2, options.maxConsecutiveRepeats ?? 4);
    this.maxWordsPerSecond = options.maxWordsPerSecond ?? 14;
    this.minAudioRmsForPhantom = options.minAudioRmsForPhantom ?? 0.01;
    this.minConfidenceForPhantom = options.minConfidenceForPhantom ?? 0.85;
    this.phantomPhrases = new Set(DEFAULT_PHANTOM_PHRASES);
    for (const phrase of options.extraPhantomPhrases ?? []) {
      this.phantomPhrases.add(normalizeForPhraseCheck(phrase));
    }
  }

  evaluate(
    rawText: string,
    context?: { audioRms?: number; durationMs?: number; confidence?: number },
  ): HallucinationCheckResult {
    const trimmed = rawText.trim();
    if (!trimmed) {
      return { text: '', suppressed: true, reason: 'empty' };
    }

    if (NON_SPEECH_TAG_RE.test(trimmed)) {
      return { text: '', suppressed: true, reason: 'non-speech-tag' };
    }

    const normalized = normalizeForPhraseCheck(trimmed);
    const rms = context?.audioRms;
    const confidence = context?.confidence;

    if (this.phantomPhrases.has(normalized)) {
      const lowEnergy = rms !== undefined && rms < this.minAudioRmsForPhantom;
      const lowConf = confidence !== undefined && confidence < this.minConfidenceForPhantom;
      const shortClip = context?.durationMs !== undefined && context.durationMs < 900;
      if (lowEnergy || lowConf || shortClip || normalized === 'amara org') {
        return { text: '', suppressed: true, reason: `phantom-phrase:${normalized}` };
      }
    }

    // Check consecutive 1..5-gram repetition loops
    const words = trimmed.split(/\s+/);
    const deduped = collapseRepetitionLoops(words, this.maxConsecutiveRepeats);
    if (deduped.loopDetected) {
      // If the loop dominated the whole utterance or left <= 2 words after severe repetition, suppress it.
      if (deduped.collapsedWords.length <= 2 && words.length >= this.maxConsecutiveRepeats) {
        return { text: '', suppressed: true, reason: 'repetition-loop' };
      }
      const cleaned = deduped.collapsedWords.join(' ').trim();
      return { text: cleaned, suppressed: cleaned.length === 0, reason: 'repetition-loop-trimmed' };
    }

    // Speaking-rate sanity check for gibberish bursts on tiny audio clips
    if (context?.durationMs && context.durationMs >= 200) {
      const wordsPerSec = words.length / (context.durationMs / 1000);
      if (words.length >= 12 && wordsPerSec > this.maxWordsPerSecond) {
        return { text: '', suppressed: true, reason: 'implausible-word-rate' };
      }
    }

    return { text: trimmed, suppressed: false };
  }
}

function normalizeForPhraseCheck(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function normalizeToken(token: string): string {
  return token.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '');
}

function collapseRepetitionLoops(
  words: string[],
  maxConsecutiveRepeats: number,
): { loopDetected: boolean; collapsedWords: string[] } {
  let current = [...words];
  let loopDetected = false;

  for (let gramSize = 1; gramSize <= 5; gramSize++) {
    if (current.length < gramSize * maxConsecutiveRepeats) {
      continue;
    }

    const next: string[] = [];
    let i = 0;

    while (i < current.length) {
      if (i + gramSize <= current.length) {
        const gramNorm = current
          .slice(i, i + gramSize)
          .map(normalizeToken)
          .join('\u0000');

        if (gramNorm.replace(/\u0000/g, '').length > 0) {
          let repeats = 1;
          while (i + (repeats + 1) * gramSize <= current.length) {
            const candidate = current
              .slice(i + repeats * gramSize, i + (repeats + 1) * gramSize)
              .map(normalizeToken)
              .join('\u0000');
            if (candidate === gramNorm) {
              repeats++;
            } else {
              break;
            }
          }

          if (repeats >= maxConsecutiveRepeats) {
            loopDetected = true;
            next.push(...current.slice(i, i + gramSize));
            i += repeats * gramSize;
            continue;
          }
        }
      }

      next.push(current[i]!);
      i++;
    }

    current = next;
  }

  return { loopDetected, collapsedWords: current };
}

export interface SttEngineOptions {
  runner: SttModelRunner;
  /** Default BCP-47 language hint (e.g. 'en'). */
  language?: string;
  /** Minimum interval in ms between partial segment emissions for smooth UI updates (default 120). */
  partialThrottleMs?: number;
  /** Hallucination guard configuration (or `false` to disable). */
  hallucinationGuard?: HallucinationGuardOptions | false;
  /** Clock provider for deterministic testing. */
  nowMs?: () => number;
}

export interface TranscribeUtteranceOptions {
  language?: string;
  startMs?: number;
  endMs?: number;
  signal?: AbortSignal;
  onSegment?: (segment: TranscriptSegment) => void;
}

/**
 * Local STT orchestrator managing model lifecycle, partial-result UI throttling,
 * language hints, hallucination-loop filtering, and clean AbortSignal cancellation.
 */
export class SttEngine {
  readonly runner: SttModelRunner;
  readonly partialThrottleMs: number;
  private readonly defaultLanguage?: string;
  private readonly guard: HallucinationGuard | null;
  private readonly nowMs: () => number;
  private initialized = false;
  private lastPartialEmitMs = -Infinity;
  private lastPartialText = '';

  constructor(options: SttEngineOptions) {
    this.runner = options.runner;
    this.defaultLanguage = options.language;
    this.partialThrottleMs = Math.max(0, options.partialThrottleMs ?? 120);
    this.guard =
      options.hallucinationGuard === false
        ? null
        : new HallucinationGuard(options.hallucinationGuard);
    this.nowMs = options.nowMs ?? (() => Date.now());
  }

  async init(): Promise<void> {
    if (this.initialized) {
      return;
    }
    await this.runner.init?.();
    this.initialized = true;
  }

  /** Reset partial throttling state between utterances. */
  resetPartialThrottle(): void {
    this.lastPartialEmitMs = -Infinity;
    this.lastPartialText = '';
  }

  /**
   * Emit a partial transcript segment while respecting `partialThrottleMs` and deduplicating
   * identical consecutive partial strings.
   */
  emitPartialIfAllowed(
    segment: TranscriptSegment,
    onSegment?: (segment: TranscriptSegment) => void,
  ): boolean {
    if (segment.isFinal) {
      onSegment?.(segment);
      return true;
    }
    const text = segment.text.trim();
    if (!text || text === this.lastPartialText) {
      return false;
    }
    const now = this.nowMs();
    if (now - this.lastPartialEmitMs < this.partialThrottleMs) {
      return false;
    }
    this.lastPartialEmitMs = now;
    this.lastPartialText = text;
    onSegment?.({ ...segment, text });
    return true;
  }

  /**
   * Transcribe a completed (or streaming) PCM utterance to one or more `TranscriptSegment`s.
   * Honors `AbortSignal` before, during, and after inference so cancelled turns emit nothing.
   */
  async transcribe(
    audio: Float32Array,
    options: TranscribeUtteranceOptions = {},
  ): Promise<TranscriptSegment[]> {
    if (options.signal?.aborted) {
      return [];
    }

    if (!this.initialized) {
      await this.init();
    }

    if (options.signal?.aborted) {
      return [];
    }

    const startMs = options.startMs ?? 0;
    const durationMs = (audio.length / TARGET_VOICE_SAMPLE_RATE) * 1000;
    const endMs = options.endMs ?? startMs + durationMs;
    const audioRms = computeRms(audio);
    const runnerOpts: SttTranscribeOptions = {
      language: options.language ?? this.defaultLanguage,
      sampleRate: TARGET_VOICE_SAMPLE_RATE,
      startMs,
      endMs,
    };

    const emittedFinals: TranscriptSegment[] = [];

    if (this.runner.transcribeStream) {
      for await (const rawSeg of this.runner.transcribeStream(audio, runnerOpts, options.signal)) {
        if (options.signal?.aborted) {
          return [];
        }
        if (!rawSeg.isFinal) {
          this.emitPartialIfAllowed(rawSeg, options.onSegment);
          continue;
        }
        const checked = this.filterSegment(rawSeg, audioRms, endMs - startMs);
        if (checked) {
          emittedFinals.push(checked);
          options.onSegment?.(checked);
        }
      }
      this.resetPartialThrottle();
      return emittedFinals;
    }

    const result = await this.runner.transcribe(audio, runnerOpts, options.signal);
    if (options.signal?.aborted) {
      return [];
    }

    if (result.segments && result.segments.length > 0) {
      for (const seg of result.segments) {
        const candidate: TranscriptSegment = {
          text: seg.text,
          isFinal: true,
          startMs: seg.startMs ?? startMs,
          endMs: seg.endMs ?? endMs,
          confidence: seg.confidence ?? result.confidence,
        };
        const checked = this.filterSegment(candidate, audioRms, candidate.endMs - candidate.startMs);
        if (checked) {
          emittedFinals.push(checked);
          options.onSegment?.(checked);
        }
      }
    } else {
      const candidate: TranscriptSegment = {
        text: result.text,
        isFinal: true,
        startMs,
        endMs,
        confidence: result.confidence,
      };
      const checked = this.filterSegment(candidate, audioRms, endMs - startMs);
      if (checked) {
        emittedFinals.push(checked);
        options.onSegment?.(checked);
      }
    }

    this.resetPartialThrottle();
    return emittedFinals;
  }

  async dispose(): Promise<void> {
    await this.runner.dispose?.();
    this.initialized = false;
  }

  private filterSegment(
    segment: TranscriptSegment,
    audioRms: number,
    durationMs: number,
  ): TranscriptSegment | null {
    if (!this.guard) {
      const trimmed = segment.text.trim();
      return trimmed ? { ...segment, text: trimmed } : null;
    }

    const evalResult = this.guard.evaluate(segment.text, {
      audioRms,
      durationMs,
      confidence: segment.confidence,
    });

    if (evalResult.suppressed || !evalResult.text) {
      return null;
    }

    return {
      ...segment,
      text: evalResult.text,
    };
  }
}
