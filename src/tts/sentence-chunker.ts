/**
 * Sentence & clause boundary stream chunker for low-latency LLM -> TTS streaming.
 *
 * Splits incoming token deltas cleanly on terminal sentence punctuation (`.`, `!`, `?`)
 * and long-clause boundaries without breaking:
 * - Abbreviations (`Dr.`, `Mr.`, `Prof.`, `e.g.`, `i.e.`, `U.S.`, `vs.`, `etc.`)
 * - Version numbers & decimals (`v1.2`, `v1.2.3`, `3.14`, `$19.99`)
 * - Domains, URLs, emails, and filenames (`yours.sh`, `github.com`, `index.ts`)
 * - Ellipses (`...`, `…`) unless followed by a clear capitalized new sentence
 */

export interface SentenceChunkerOptions {
  /** Minimum characters before splitting on a clause boundary when no terminal punctuation appears (default 48). */
  minClauseChars?: number;
  /** Maximum characters to buffer before splitting on the latest clause separator (default 180). */
  maxChunkChars?: number;
  /** Optional extra abbreviations (lowercase without trailing period). */
  extraAbbreviations?: readonly string[];
}

const ABBREVIATIONS = new Set<string>([
  'dr',
  'mr',
  'mrs',
  'ms',
  'mx',
  'prof',
  'sr',
  'jr',
  'st',
  'rev',
  'gen',
  'col',
  'lt',
  'sgt',
  'capt',
  'vs',
  'etc',
  'eg',
  'ie',
  'e.g',
  'i.e',
  'fig',
  'eq',
  'no',
  'vol',
  'pp',
  'inc',
  'ltd',
  'corp',
  'co',
  'approx',
  'est',
  'dept',
  'gov',
  'jan',
  'feb',
  'mar',
  'apr',
  'aug',
  'sep',
  'sept',
  'oct',
  'nov',
  'dec',
  'mon',
  'tue',
  'wed',
  'thu',
  'fri',
  'sat',
  'sun',
  'u.s',
  'u.k',
  'u.n',
  'u.s.a',
  'a.m',
  'p.m',
  'ph.d',
  'b.s',
  'm.s',
]);

const CLOSING_PUNCT = new Set(['"', "'", '”', '’', ')', ']', '}']);

export class SentenceStreamChunker {
  private readonly minClauseChars: number;
  private readonly maxChunkChars: number;
  private readonly abbreviations: Set<string>;
  private buffer = '';

  constructor(options: SentenceChunkerOptions = {}) {
    this.minClauseChars = Math.max(12, options.minClauseChars ?? 48);
    this.maxChunkChars = Math.max(this.minClauseChars + 16, options.maxChunkChars ?? 180);
    this.abbreviations = new Set(ABBREVIATIONS);
    for (const abbr of options.extraAbbreviations ?? []) {
      this.abbreviations.add(abbr.toLowerCase().replace(/\.$/, ''));
    }
  }

  /** Feed an LLM token delta and return any complete chunks ready for TTS synthesis. */
  feed(delta: string): string[] {
    if (!delta) {
      return [];
    }
    this.buffer += delta;
    const chunks: string[] = [];

    while (true) {
      const splitIndex = this.findNextSplitIndex();
      if (splitIndex <= 0) {
        break;
      }

      const rawChunk = this.buffer.slice(0, splitIndex).trim();
      this.buffer = this.buffer.slice(splitIndex).replace(/^\s+/, '');

      if (rawChunk.length > 0) {
        chunks.push(rawChunk);
      }
    }

    return chunks;
  }

  /** Flush any remaining buffered text at end of stream. */
  flush(): string[] {
    const remaining = this.buffer.trim();
    this.buffer = '';
    return remaining ? [remaining] : [];
  }

  reset(): void {
    this.buffer = '';
  }

  private findNextSplitIndex(): number {
    const buf = this.buffer;

    // 1. Look for terminal sentence punctuation followed by whitespace (optionally after closing quotes/parens)
    for (let i = 0; i < buf.length; i++) {
      const ch = buf[i]!;
      if (ch !== '.' && ch !== '!' && ch !== '?' && ch !== '…') {
        continue;
      }

      // Skip consecutive punctuation like `...`, `?!`, `!!` to the end of the punctuation run
      let endIdx = i;
      while (
        endIdx + 1 < buf.length &&
        (buf[endIdx + 1] === '.' || buf[endIdx + 1] === '!' || buf[endIdx + 1] === '?' || buf[endIdx + 1] === '…')
      ) {
        endIdx++;
      }

      // Consume any immediate closing quotes / parentheses
      let afterClosing = endIdx + 1;
      while (afterClosing < buf.length && CLOSING_PUNCT.has(buf[afterClosing]!)) {
        afterClosing++;
      }

      // Must be followed by whitespace so tokens like `v1.2`, `3.14`, `yours.sh`, `index.ts`
      // never trigger before the rest of the token arrives.
      if (afterClosing >= buf.length) {
        i = endIdx;
        continue;
      }

      const nextChar = buf[afterClosing]!;
      if (!/\s/.test(nextChar)) {
        i = endIdx;
        continue;
      }

      // Distinguish ellipsis `...` vs normal period
      const runText = buf.slice(i, endIdx + 1);
      if (runText === '...' || runText === '…' || runText === '..') {
        const afterWs = findFirstNonWhitespace(buf, afterClosing);
        if (afterWs === -1) {
          i = endIdx;
          continue;
        }
        // Only split on ellipsis if the next sentence starts with an uppercase letter or quote+uppercase
        const nextSignificant = buf[afterWs]!;
        if (!/[A-Z"“'‘(]/.test(nextSignificant)) {
          i = endIdx;
          continue;
        }
        return afterClosing;
      }

      // If single period `.`, check abbreviations, decimals, initials, and domains
      if (ch === '.' && endIdx === i) {
        if (this.isNonTerminatingPeriod(buf, i)) {
          continue;
        }
      }

      // Valid sentence boundary! Verify there is actual content before it.
      if (buf.slice(0, afterClosing).trim().length > 0) {
        return afterClosing;
      }
    }

    // 2. Long-sentence clause boundary fallback if buffer exceeds maxChunkChars
    if (buf.length >= this.maxChunkChars) {
      const clauseSplit = this.findClauseBoundary(buf);
      if (clauseSplit > 0) {
        return clauseSplit;
      }
      // Hard whitespace fallback so unbounded no-punctuation streams still stream smoothly
      const lastSpace = buf.lastIndexOf(' ', this.maxChunkChars);
      if (lastSpace >= this.minClauseChars) {
        return lastSpace + 1;
      }
    }

    return -1;
  }

  private isNonTerminatingPeriod(buf: string, dotIndex: number): boolean {
    // Extract the contiguous non-whitespace token leading up to `dotIndex`
    let tokenStart = dotIndex - 1;
    while (tokenStart >= 0 && !/\s/.test(buf[tokenStart]!)) {
      tokenStart--;
    }
    tokenStart++;

    const wordWithDot = buf.slice(tokenStart, dotIndex);
    const strippedWord = wordWithDot.replace(/^["'“‘(\[]+/, '');
    const lower = strippedWord.toLowerCase();

    if (!lower) {
      return false;
    }

    // Known abbreviation (e.g. `Dr.`, `e.g.`, `U.S.`, `Prof.`)
    if (this.abbreviations.has(lower)) {
      return true;
    }

    // Multi-dot initialism like `A.B.` or `U.S.` even if not in static set
    if (/^(?:[a-z]\.)+[a-z]$/i.test(strippedWord)) {
      return true;
    }

    // Single-letter middle initial like `John F. Kennedy` (uppercase single letter + period)
    if (/^[A-Z]$/.test(strippedWord)) {
      return true;
    }

    // Trailing decimal or numbered list prefix like `1.` or `2.` at start of line
    if (/^\d+$/.test(strippedWord) && strippedWord.length <= 2) {
      return true;
    }

    return false;
  }

  private findClauseBoundary(buf: string): number {
    const searchLimit = Math.min(buf.length - 1, this.maxChunkChars);
    for (let i = searchLimit; i >= this.minClauseChars; i--) {
      const ch = buf[i]!;
      const next = buf[i + 1];
      if ((ch === ',' || ch === ';' || ch === ':' || ch === '—' || ch === '–') && next && /\s/.test(next)) {
        return i + 1;
      }
      if (ch === '-' && i > 0 && buf[i - 1] === ' ' && next === ' ') {
        return i + 1;
      }
    }
    return -1;
  }
}

function findFirstNonWhitespace(text: string, startIndex: number): number {
  for (let i = startIndex; i < text.length; i++) {
    if (!/\s/.test(text[i]!)) {
      return i;
    }
  }
  return -1;
}

/**
 * Convert a `string` or `AsyncIterable<string>` token stream into an `AsyncIterable<string>`
 * of complete sentence/clause chunks ready for TTS synthesis.
 */
export async function* chunkTextStream(
  textStream: string | AsyncIterable<string>,
  options?: SentenceChunkerOptions,
  signal?: AbortSignal,
): AsyncIterable<string> {
  const chunker = new SentenceStreamChunker(options);

  if (typeof textStream === 'string') {
    if (signal?.aborted) {
      return;
    }
    for (const chunk of chunker.feed(textStream)) {
      if (signal?.aborted) {
        return;
      }
      yield chunk;
    }
    for (const tail of chunker.flush()) {
      if (signal?.aborted) {
        return;
      }
      yield tail;
    }
    return;
  }

  for await (const delta of textStream) {
    if (signal?.aborted) {
      return;
    }
    for (const chunk of chunker.feed(delta)) {
      if (signal?.aborted) {
        return;
      }
      yield chunk;
    }
  }

  if (signal?.aborted) {
    return;
  }
  for (const tail of chunker.flush()) {
    if (signal?.aborted) {
      return;
    }
    yield tail;
  }
}
