import { describe, expect, it, vi } from 'vitest';
import {
  MV3_MIC_PERMISSION_MESSAGE_TYPE,
  Mv3MicPermissionHelper,
  Mv3OffscreenHost,
  Mv3SidePanelClient,
  createInMemoryTransportPair,
  ensureOffscreenDocument,
} from '../src/adapters/index.js';
import type { AudioInputSource } from '../src/contracts.js';
import { NullAudioPlaybackSink } from '../src/tts/index.js';

describe('Chrome MV3 Offscreen Bridge & Permission Onboarding', () => {
  it('routes capture PCM frames and TTS playback chunks over RPC and flushes immediately on barge-in', async () => {
    // Test JSON-safe base64 serialization mode (supportsTransferables: false)
    const { panel, offscreen } = createInMemoryTransportPair({ supportsTransferables: false });

    let emitMicFrame: ((samples: Float32Array, sr: number, ts: number) => void) | null = null;
    const mockOffscreenMic: AudioInputSource = {
      sampleRate: 16_000,
      isActive: false,
      async start(onFrame) {
        (this as { isActive: boolean }).isActive = true;
        emitMicFrame = onFrame;
      },
      async stop() {
        (this as { isActive: boolean }).isActive = false;
        emitMicFrame = null;
      },
    };

    const offscreenPlaybackSink = new NullAudioPlaybackSink({ simulateDuration: true });
    const host = new Mv3OffscreenHost({
      transport: offscreen,
      inputSource: mockOffscreenMic,
      playbackSink: offscreenPlaybackSink,
    });

    const client = new Mv3SidePanelClient({ transport: panel });
    const panelInput = client.createInputSource();
    const panelSink = client.createPlaybackSink();

    const receivedFrames: Float32Array[] = [];
    await panelInput.start((samples) => receivedFrames.push(samples));
    expect(panelInput.isActive).toBe(true);

    emitMicFrame!(new Float32Array([0.25, -0.5, 0.75]), 16_000, 10);
    expect(receivedFrames).toHaveLength(1);
    expect(receivedFrames[0]![0]).toBeCloseTo(0.25, 5);
    expect(receivedFrames[0]![1]).toBeCloseTo(-0.5, 5);

    // Play a 500ms chunk on offscreen host, then cancel immediately via stopAll()
    const playPromise = panelSink.playChunk({
      samples: new Float32Array(12_000).fill(0.1),
      sampleRate: 24_000,
      text: 'Hello MV3',
      chunkIndex: 0,
      durationMs: 500,
    });

    await new Promise((r) => setTimeout(r, 10));
    const startCancel = performance.now();
    await panelSink.stopAll();
    await playPromise;
    expect(performance.now() - startCancel).toBeLessThan(30);

    await panelInput.stop();
    await host.dispose();
    client.dispose();
  });

  it('uses Side Panel fallback playback sink when configured or offscreen playback is omitted', async () => {
    const { panel, offscreen } = createInMemoryTransportPair();
    const host = new Mv3OffscreenHost({
      transport: offscreen,
      inputSource: {
        sampleRate: 16_000,
        isActive: false,
        async start() {},
        async stop() {},
      },
      // No playbackSink on offscreen!
    });

    const fallbackSink = new NullAudioPlaybackSink();
    const client = new Mv3SidePanelClient({
      transport: panel,
      fallbackPlaybackSink: fallbackSink,
    });

    const sink = client.createPlaybackSink();
    await sink.playChunk({
      samples: new Float32Array(100),
      sampleRate: 24_000,
      text: 'Fallback audio',
      chunkIndex: 0,
      durationMs: 4,
    });

    expect(fallbackSink.playedChunks).toHaveLength(1);
    expect(fallbackSink.playedChunks[0]?.text).toBe('Fallback audio');

    await host.dispose();
    client.dispose();
  });

  it('ensureOffscreenDocument avoids duplicate createDocument calls when already open', async () => {
    const createDocument = vi.fn(async () => {});
    await ensureOffscreenDocument({
      url: 'offscreen.html',
      chrome: {
        runtime: { getContexts: async () => [{ contextType: 'OFFSCREEN_DOCUMENT' }] },
        offscreen: { createDocument },
      },
    });
    expect(createDocument).not.toHaveBeenCalled();

    await ensureOffscreenDocument({
      url: 'offscreen.html',
      chrome: {
        runtime: { getContexts: async () => [] },
        offscreen: { createDocument },
      },
    });
    expect(createDocument).toHaveBeenCalledTimes(1);
  });

  it('Mv3MicPermissionHelper handles iframe onboarding with origin verification and tab fallback', async () => {
    const listeners = new Set<(ev: unknown) => void>();
    const fakeIframe = {
      src: '',
      style: {} as CSSStyleDeclaration,
      setAttribute: vi.fn(),
      remove: vi.fn(),
    };

    const helper = new Mv3MicPermissionHelper({
      permissionsPageUrl: 'chrome-extension://abc123/permissions.html',
      strategy: 'iframe',
      env: {
        permissions: { query: async () => ({ state: 'prompt' }) },
        window: {
          location: { origin: 'chrome-extension://abc123' },
          addEventListener: (_type, fn) => listeners.add(fn as (ev: unknown) => void),
          removeEventListener: (_type, fn) => listeners.delete(fn as (ev: unknown) => void),
        },
        document: {
          createElement: () => fakeIframe as unknown as HTMLIFrameElement,
          body: { appendChild: vi.fn() } as unknown as HTMLElement,
        },
      },
    });

    const promise = helper.ensureMicPermission();
    // Wait a microtask for query() to resolve and register listener
    await new Promise((r) => setTimeout(r, 5));

    // Untrusted origin message must be ignored
    for (const fn of listeners) {
      fn({
        origin: 'https://evil.example',
        data: { type: MV3_MIC_PERMISSION_MESSAGE_TYPE, granted: true },
      });
    }

    // Valid extension origin confirmation
    for (const fn of listeners) {
      fn({
        origin: 'chrome-extension://abc123',
        data: { type: MV3_MIC_PERMISSION_MESSAGE_TYPE, granted: true },
      });
    }

    const res = await promise;
    expect(res.granted).toBe(true);
    expect(res.method).toBe('iframe');
    expect(fakeIframe.remove).toHaveBeenCalled();
  });
});
