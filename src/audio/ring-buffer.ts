/**
 * Fixed-capacity circular buffer for `Float32Array` PCM audio samples.
 * Used as a pre-roll buffer so initial consonant onsets before VAD trigger
 * are preserved seamlessly without heap churn.
 */
export class AudioRingBuffer {
  private readonly buffer: Float32Array;
  private writeIndex = 0;
  private count = 0;

  constructor(public readonly capacity: number) {
    if (!Number.isInteger(capacity) || capacity < 0) {
      throw new RangeError(`AudioRingBuffer capacity must be a non-negative integer, got ${capacity}`);
    }
    this.buffer = new Float32Array(capacity);
  }

  /** Number of valid samples currently stored (0..capacity). */
  get length(): number {
    return this.count;
  }

  /** Push new PCM samples into the ring buffer, overwriting oldest samples when full. */
  push(samples: Float32Array): void {
    if (this.capacity === 0 || samples.length === 0) {
      return;
    }

    if (samples.length >= this.capacity) {
      const start = samples.length - this.capacity;
      this.buffer.set(samples.subarray(start), 0);
      this.writeIndex = 0;
      this.count = this.capacity;
      return;
    }

    const firstPart = Math.min(samples.length, this.capacity - this.writeIndex);
    this.buffer.set(samples.subarray(0, firstPart), this.writeIndex);

    const remaining = samples.length - firstPart;
    if (remaining > 0) {
      this.buffer.set(samples.subarray(firstPart), 0);
    }

    this.writeIndex = (this.writeIndex + samples.length) % this.capacity;
    this.count = Math.min(this.capacity, this.count + samples.length);
  }

  /** Read all currently buffered samples in chronological order as a contiguous Float32Array. */
  readAll(): Float32Array {
    if (this.count === 0) {
      return new Float32Array(0);
    }

    const out = new Float32Array(this.count);
    const startIndex = (this.writeIndex - this.count + this.capacity) % this.capacity;

    if (startIndex + this.count <= this.capacity) {
      out.set(this.buffer.subarray(startIndex, startIndex + this.count), 0);
    } else {
      const tailLen = this.capacity - startIndex;
      out.set(this.buffer.subarray(startIndex), 0);
      out.set(this.buffer.subarray(0, this.count - tailLen), tailLen);
    }

    return out;
  }

  /** Read the most recent `n` samples (or all available if fewer than `n`). */
  readLast(n: number): Float32Array {
    if (n <= 0 || this.count === 0) {
      return new Float32Array(0);
    }
    const all = this.readAll();
    if (n >= all.length) {
      return all;
    }
    return all.slice(all.length - n);
  }

  /** Clear all buffered samples without reallocating backing memory. */
  clear(): void {
    this.writeIndex = 0;
    this.count = 0;
    this.buffer.fill(0);
  }
}

/** Concatenate an array of Float32Arrays into a single contiguous Float32Array. */
export function concatFloat32Arrays(chunks: readonly Float32Array[]): Float32Array {
  let total = 0;
  for (const chunk of chunks) {
    total += chunk.length;
  }
  const out = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}
