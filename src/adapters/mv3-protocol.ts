/**
 * Chrome MV3 Offscreen <-> Side Panel / Service Worker RPC protocol & transports.
 * Supports both structured-clone `Float32Array` payloads (`MessagePort`, in-memory)
 * and JSON-safe base64 PCM encoding (`chrome.runtime.sendMessage` / `Port`).
 */

export const MV3_VOICE_PROTOCOL_VERSION = 1;

export type Mv3VoiceCommand =
  | {
      type: 'yours:voice:capture-start';
      requestId: string;
      sampleRate?: number;
      deviceId?: string;
      echoCancellation?: boolean;
      noiseSuppression?: boolean;
      autoGainControl?: boolean;
    }
  | {
      type: 'yours:voice:capture-stop';
      requestId: string;
    }
  | {
      type: 'yours:voice:playback-chunk';
      requestId: string;
      chunkIndex: number;
      sampleRate: number;
      samples?: Float32Array;
      pcmBase64?: string;
    }
  | {
      type: 'yours:voice:playback-barge-in';
      requestId: string;
    };

export type Mv3VoiceEvent =
  | {
      type: 'yours:voice:ack';
      requestId: string;
      ok: boolean;
      error?: string;
    }
  | {
      type: 'yours:voice:capture-frame';
      sampleRate: number;
      timestampMs: number;
      samples?: Float32Array;
      pcmBase64?: string;
    }
  | {
      type: 'yours:voice:capture-error';
      message: string;
    }
  | {
      type: 'yours:voice:playback-ended';
      requestId: string;
      chunkIndex: number;
      aborted: boolean;
      error?: string;
    };

export type Mv3VoiceMessage = Mv3VoiceCommand | Mv3VoiceEvent;

export interface Mv3RpcTransport {
  readonly supportsTransferables?: boolean;
  send(message: Mv3VoiceMessage): void | Promise<void>;
  onMessage(handler: (message: Mv3VoiceMessage) => void): () => void;
  close?(): void;
}

/** Encode a Float32Array PCM buffer as base64 for JSON-only MV3 `chrome.runtime` messages. */
export function encodeFloat32Base64(samples: Float32Array): string {
  const bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(bytes).toString('base64');
  }
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const slice = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...slice);
  }
  return btoa(binary);
}

/** Decode a base64 PCM payload back into a Float32Array. */
export function decodeFloat32Base64(base64: string): Float32Array {
  if (typeof Buffer !== 'undefined') {
    const buf = Buffer.from(base64, 'base64');
    const aligned = new Uint8Array(buf.byteLength);
    aligned.set(buf);
    return new Float32Array(aligned.buffer);
  }
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}

/** Extract Float32Array samples from an RPC frame/chunk supporting either structured-clone or base64. */
export function extractPcmSamples(payload: {
  samples?: Float32Array | ArrayLike<number>;
  pcmBase64?: string;
}): Float32Array {
  if (payload.samples instanceof Float32Array) {
    return payload.samples;
  }
  if (payload.samples) {
    return Float32Array.from(payload.samples);
  }
  if (payload.pcmBase64) {
    return decodeFloat32Base64(payload.pcmBase64);
  }
  return new Float32Array(0);
}

/** Create a paired in-memory transport (panel <-> offscreen) for tests and local workers. */
export function createInMemoryTransportPair(options?: {
  supportsTransferables?: boolean;
}): {
  panel: Mv3RpcTransport;
  offscreen: Mv3RpcTransport;
} {
  const panelHandlers = new Set<(msg: Mv3VoiceMessage) => void>();
  const offscreenHandlers = new Set<(msg: Mv3VoiceMessage) => void>();
  let closed = false;
  const supportsTransferables = options?.supportsTransferables ?? true;

  const panel: Mv3RpcTransport = {
    supportsTransferables,
    send(message) {
      if (closed) {
        return;
      }
      for (const handler of [...offscreenHandlers]) {
        handler(message);
      }
    },
    onMessage(handler) {
      panelHandlers.add(handler);
      return () => {
        panelHandlers.delete(handler);
      };
    },
    close() {
      closed = true;
      panelHandlers.clear();
      offscreenHandlers.clear();
    },
  };

  const offscreen: Mv3RpcTransport = {
    supportsTransferables,
    send(message) {
      if (closed) {
        return;
      }
      for (const handler of [...panelHandlers]) {
        handler(message);
      }
    },
    onMessage(handler) {
      offscreenHandlers.add(handler);
      return () => {
        offscreenHandlers.delete(handler);
      };
    },
    close() {
      closed = true;
      panelHandlers.clear();
      offscreenHandlers.clear();
    },
  };

  return { panel, offscreen };
}

/** Minimal structural type for `chrome.runtime.Port` so callers need no `@types/chrome`. */
export interface ChromeRuntimePortLike {
  postMessage(message: unknown): void;
  onMessage: {
    addListener(listener: (message: unknown) => void): void;
    removeListener(listener: (message: unknown) => void): void;
  };
  disconnect?(): void;
}

/** Wrap a `chrome.runtime.Port` as an `Mv3RpcTransport`. */
export class ChromeRuntimePortTransport implements Mv3RpcTransport {
  readonly supportsTransferables = false;

  constructor(private readonly port: ChromeRuntimePortLike) {}

  send(message: Mv3VoiceMessage): void {
    this.port.postMessage(message);
  }

  onMessage(handler: (message: Mv3VoiceMessage) => void): () => void {
    const listener = (raw: unknown) => {
      if (isMv3VoiceMessage(raw)) {
        handler(raw);
      }
    };
    this.port.onMessage.addListener(listener);
    return () => {
      this.port.onMessage.removeListener(listener);
    };
  }

  close(): void {
    this.port.disconnect?.();
  }
}

export function isMv3VoiceMessage(value: unknown): value is Mv3VoiceMessage {
  return (
    typeof value === 'object' &&
    value !== null &&
    'type' in value &&
    typeof (value as { type: unknown }).type === 'string' &&
    (value as { type: string }).type.startsWith('yours:voice:')
  );
}
