/**
 * Chrome MV3 Microphone Permission Onboarding Helper.
 *
 * Chromium MV3 Offscreen Documents cannot display a `getUserMedia` permission prompt
 * if the extension origin has not yet been granted microphone access. Additionally,
 * Side Panels may require an explicit extension-origin iframe or onboarding tab
 * (`permissions.html`) to trigger the prompt cleanly.
 */

export const MV3_MIC_PERMISSION_MESSAGE_TYPE = 'yours:voice:mic-permission-result';

export type MicPermissionState = 'granted' | 'denied' | 'prompt' | 'unknown';

export interface MicPermissionResultMessage {
  type: typeof MV3_MIC_PERMISSION_MESSAGE_TYPE;
  granted: boolean;
  error?: string;
}

export interface EnsureMicPermissionResult {
  granted: boolean;
  state: MicPermissionState;
  method: 'already-granted' | 'iframe' | 'tab' | 'direct' | 'none';
  error?: string;
}

export interface ChromeTabsLike {
  create(createProperties: { url: string; active?: boolean }): Promise<{ id?: number }> | { id?: number };
  remove?(tabId: number): Promise<void> | void;
}

export interface ChromeRuntimeLike {
  getURL?(path: string): string;
  sendMessage?(message: unknown): Promise<unknown> | void;
  onMessage?: {
    addListener(listener: (message: unknown, sender?: { id?: string; url?: string }) => void): void;
    removeListener(listener: (message: unknown, sender?: { id?: string; url?: string }) => void): void;
  };
  id?: string;
}

export interface Mv3MicPermissionHelperOptions {
  /** Extension URL or path (e.g. `'permissions.html'`) for the onboarding page. */
  permissionsPageUrl: string;
  /** Strategy when mic state is `'prompt'`: `'iframe'`, `'tab'`, or `'iframe-then-tab'` (default `'iframe-then-tab'`). */
  strategy?: 'iframe' | 'tab' | 'iframe-then-tab';
  /** Timeout in ms waiting for user interaction in iframe/tab (default 30000). */
  timeoutMs?: number;
  /** Optional container element to mount the onboarding iframe into (defaults to `document.body`). */
  iframeContainer?: HTMLElement;
  /** Optional injected browser globals for non-DOM / unit-test environments. */
  env?: {
    permissions?: {
      query(desc: { name: string }): Promise<{ state: MicPermissionState }>;
    };
    mediaDevices?: {
      getUserMedia(constraints: { audio: boolean }): Promise<{ getTracks(): Array<{ stop(): void }> }>;
    };
    chrome?: {
      runtime?: ChromeRuntimeLike;
      tabs?: ChromeTabsLike;
    };
    window?: {
      addEventListener(type: string, listener: EventListenerOrEventListenerObject): void;
      removeEventListener(type: string, listener: EventListenerOrEventListenerObject): void;
      location?: Pick<Location, 'origin'>;
    };
    document?: Pick<Document, 'createElement' | 'body'>;
  };
}

export class Mv3MicPermissionHelper {
  private readonly permissionsPageUrl: string;
  private readonly strategy: 'iframe' | 'tab' | 'iframe-then-tab';
  private readonly timeoutMs: number;
  private readonly iframeContainer?: HTMLElement;
  private readonly env: NonNullable<Mv3MicPermissionHelperOptions['env']>;

  constructor(options: Mv3MicPermissionHelperOptions) {
    this.permissionsPageUrl = options.permissionsPageUrl;
    this.strategy = options.strategy ?? 'iframe-then-tab';
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.iframeContainer = options.iframeContainer;
    this.env = options.env ?? {};
  }

  /** Query current microphone permission state without triggering any UI prompt. */
  async checkPermission(): Promise<MicPermissionState> {
    const permissions =
      this.env.permissions ??
      (typeof navigator !== 'undefined' && navigator.permissions
        ? (navigator.permissions as unknown as {
            query(desc: { name: string }): Promise<{ state: MicPermissionState }>;
          })
        : undefined);

    if (!permissions) {
      return 'unknown';
    }

    try {
      const status = await permissions.query({ name: 'microphone' });
      if (status.state === 'granted' || status.state === 'denied' || status.state === 'prompt') {
        return status.state;
      }
      return 'unknown';
    } catch {
      return 'unknown';
    }
  }

  /**
   * Ensure microphone permission is granted on the extension origin before starting
   * capture in an MV3 Offscreen Document.
   */
  async ensureMicPermission(): Promise<EnsureMicPermissionResult> {
    const currentState = await this.checkPermission();
    if (currentState === 'granted') {
      return { granted: true, state: 'granted', method: 'already-granted' };
    }
    if (currentState === 'denied') {
      return {
        granted: false,
        state: 'denied',
        method: 'none',
        error: 'Microphone permission was previously denied by the user.',
      };
    }

    if (this.strategy === 'iframe') {
      return this.requestViaIframe();
    }
    if (this.strategy === 'tab') {
      return this.requestViaOnboardingTab();
    }

    // 'iframe-then-tab': try inline iframe first; if blocked/unsupported, open onboarding tab.
    const iframeResult = await this.requestViaIframe();
    if (iframeResult.granted || iframeResult.state === 'denied') {
      return iframeResult;
    }
    return this.requestViaOnboardingTab();
  }

  /**
   * Mount an extension-origin `<iframe allow="microphone">` and wait for it to post
   * `yours:voice:mic-permission-result` with origin verification.
   */
  async requestViaIframe(): Promise<EnsureMicPermissionResult> {
    const win = this.env.window ?? (typeof window !== 'undefined' ? window : undefined);
    const doc = this.env.document ?? (typeof document !== 'undefined' ? document : undefined);

    if (!win || !doc) {
      return {
        granted: false,
        state: 'unknown',
        method: 'iframe',
        error: 'DOM window/document unavailable for iframe onboarding.',
      };
    }

    const resolvedUrl = this.resolvePageUrl();
    const expectedOrigin = safeGetOrigin(resolvedUrl, win.location?.origin);

    return new Promise<EnsureMicPermissionResult>((resolve) => {
      const iframe = doc.createElement('iframe');
      iframe.setAttribute('allow', 'microphone');
      iframe.setAttribute('aria-hidden', 'true');
      iframe.style.width = '1px';
      iframe.style.height = '1px';
      iframe.style.border = '0';
      iframe.style.position = 'fixed';
      iframe.style.opacity = '0';
      iframe.style.pointerEvents = 'none';

      let settled = false;
      const cleanup = () => {
        clearTimeout(timer);
        win.removeEventListener('message', onMessage as EventListener);
        iframe.remove?.();
      };

      const finish = (result: EnsureMicPermissionResult) => {
        if (settled) {
          return;
        }
        settled = true;
        cleanup();
        resolve(result);
      };

      const onMessage = (event: MessageEvent) => {
        if (expectedOrigin && event.origin && event.origin !== expectedOrigin) {
          return;
        }
        const data = event.data as Partial<MicPermissionResultMessage> | undefined;
        if (!data || data.type !== MV3_MIC_PERMISSION_MESSAGE_TYPE) {
          return;
        }
        finish({
          granted: Boolean(data.granted),
          state: data.granted ? 'granted' : 'denied',
          method: 'iframe',
          error: data.error,
        });
      };

      const timer = setTimeout(() => {
        finish({
          granted: false,
          state: 'prompt',
          method: 'iframe',
          error: `Timed out waiting for iframe microphone onboarding (${this.timeoutMs}ms).`,
        });
      }, this.timeoutMs);

      win.addEventListener('message', onMessage as EventListener);
      iframe.src = resolvedUrl;
      const container = this.iframeContainer ?? doc.body;
      container.appendChild(iframe);
    });
  }

  /**
   * Open `permissions.html` in a Chrome extension tab so the user can grant microphone
   * access directly on a visible extension page, then listen for runtime completion.
   */
  async requestViaOnboardingTab(): Promise<EnsureMicPermissionResult> {
    const chromeObj =
      this.env.chrome ??
      ((globalThis as { chrome?: { runtime?: ChromeRuntimeLike; tabs?: ChromeTabsLike } }).chrome ??
        undefined);
    const tabs = chromeObj?.tabs;
    const runtime = chromeObj?.runtime;

    if (!tabs || !runtime?.onMessage) {
      return {
        granted: false,
        state: 'unknown',
        method: 'tab',
        error: 'Chrome MV3 tabs/runtime APIs unavailable for onboarding tab flow.',
      };
    }

    const resolvedUrl = this.resolvePageUrl();
    const createdTab = await tabs.create({ url: resolvedUrl, active: true });

    return new Promise<EnsureMicPermissionResult>((resolve) => {
      let settled = false;
      const cleanup = () => {
        clearTimeout(timer);
        runtime.onMessage?.removeListener(onRuntimeMessage);
      };

      const finish = async (result: EnsureMicPermissionResult) => {
        if (settled) {
          return;
        }
        settled = true;
        cleanup();
        if (result.granted && createdTab?.id !== undefined) {
          try {
            await tabs.remove?.(createdTab.id);
          } catch {
            // Ignore tab-already-closed errors
          }
        }
        resolve(result);
      };

      const onRuntimeMessage = (raw: unknown, sender?: { id?: string }) => {
        if (runtime.id && sender?.id && sender.id !== runtime.id) {
          return;
        }
        const data = raw as Partial<MicPermissionResultMessage> | undefined;
        if (!data || data.type !== MV3_MIC_PERMISSION_MESSAGE_TYPE) {
          return;
        }
        void finish({
          granted: Boolean(data.granted),
          state: data.granted ? 'granted' : 'denied',
          method: 'tab',
          error: data.error,
        });
      };

      const timer = setTimeout(() => {
        void finish({
          granted: false,
          state: 'prompt',
          method: 'tab',
          error: `Timed out waiting for onboarding tab permission (${this.timeoutMs}ms).`,
        });
      }, this.timeoutMs);

      runtime.onMessage!.addListener(onRuntimeMessage);
    });
  }

  private resolvePageUrl(): string {
    const runtime =
      this.env.chrome?.runtime ??
      (globalThis as { chrome?: { runtime?: ChromeRuntimeLike } }).chrome?.runtime;
    if (
      runtime?.getURL &&
      !this.permissionsPageUrl.startsWith('http') &&
      !this.permissionsPageUrl.startsWith('chrome-extension://')
    ) {
      return runtime.getURL(this.permissionsPageUrl);
    }
    return this.permissionsPageUrl;
  }
}

/**
 * Helper to execute inside `permissions.html` (extension onboarding iframe or tab).
 * Requests `getUserMedia({ audio: true })`, stops tracks immediately, and notifies the opener/parent.
 */
export async function completeMv3MicOnboarding(options?: {
  targetOrigin?: string;
  mediaDevices?: {
    getUserMedia(constraints: { audio: boolean }): Promise<{ getTracks(): Array<{ stop(): void }> }>;
  };
}): Promise<MicPermissionResultMessage> {
  const mediaDevices =
    options?.mediaDevices ??
    (typeof navigator !== 'undefined' ? navigator.mediaDevices : undefined);

  let payload: MicPermissionResultMessage;
  if (!mediaDevices?.getUserMedia) {
    payload = {
      type: MV3_MIC_PERMISSION_MESSAGE_TYPE,
      granted: false,
      error: 'navigator.mediaDevices.getUserMedia is unavailable in this context.',
    };
  } else {
    try {
      const stream = await mediaDevices.getUserMedia({ audio: true });
      for (const track of stream.getTracks()) {
        track.stop();
      }
      payload = { type: MV3_MIC_PERMISSION_MESSAGE_TYPE, granted: true };
    } catch (err) {
      payload = {
        type: MV3_MIC_PERMISSION_MESSAGE_TYPE,
        granted: false,
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }

  if (typeof window !== 'undefined' && window.parent && window.parent !== window) {
    const targetOrigin = options?.targetOrigin ?? window.location?.origin ?? '*';
    window.parent.postMessage(payload, targetOrigin);
  }

  const runtime = (globalThis as { chrome?: { runtime?: ChromeRuntimeLike } }).chrome?.runtime;
  if (runtime?.sendMessage) {
    try {
      await runtime.sendMessage(payload);
    } catch {
      // No active runtime listener required when running inside an iframe
    }
  }

  return payload;
}

function safeGetOrigin(url: string, fallback?: string): string | undefined {
  try {
    const parsed = new URL(url, fallback ?? 'https://localhost');
    if (parsed.origin && parsed.origin !== 'null') {
      return parsed.origin;
    }
    if (parsed.protocol && parsed.host) {
      return `${parsed.protocol}//${parsed.host}`;
    }
    return fallback;
  } catch {
    return fallback;
  }
}
