(function attachLiveMicClient(root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.HuiWuLiveMicClient = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createLiveMicClient(root) {
  class EventEmitter {
    constructor() { this.handlers = new Map(); }
    on(event, handler) {
      if (!this.handlers.has(event)) this.handlers.set(event, new Set());
      this.handlers.get(event).add(handler);
      return () => this.off(event, handler);
    }
    off(event, handler) { this.handlers.get(event)?.delete(handler); }
    emit(event, payload) { (this.handlers.get(event) || []).forEach(handler => handler(payload)); }
  }

  class LiveMicClient extends EventEmitter {
    constructor({ wsBase, encodePcmFrame, WebSocketImpl, setTimeoutImpl, clearTimeoutImpl } = {}) {
      super();
      this.wsBase = wsBase || '';
      this.encodePcmFrame = encodePcmFrame || root?.HuiWuLiveProtocol?.encodePcmFrame;
      this.WebSocketImpl = WebSocketImpl || root?.WebSocket;
      this.setTimeoutImpl = setTimeoutImpl || root?.setTimeout || setTimeout;
      this.clearTimeoutImpl = clearTimeoutImpl || root?.clearTimeout || clearTimeout;
      this.ws = null;
      this.sequence = 0;
      this.sessionToken = 0;
      this.streamId = null;
      this.configured = false;
      this.configureTimer = null;
      this.stopSent = false;
    }

    connect(settings = {}) {
      if (!this.WebSocketImpl) throw new Error('WebSocket is unavailable');
      if (!this.encodePcmFrame) throw new Error('DTP2 encoder is unavailable');
      this.close('reconnect');
      const token = ++this.sessionToken;
      const ws = new this.WebSocketImpl(`${this.wsBase}/ws/live`);
      ws.binaryType = 'arraybuffer';
      this.ws = ws;
      this.sequence = 0;
      this.stopSent = false;
      this.configured = false;
      this.streamId = settings.stream_id || (root?.crypto?.randomUUID?.() || String(Date.now()));
      ws.onopen = () => this.emit('open', { token });
      ws.onerror = event => this._guard(token, () => this.emit('error', { type: 'websocket_error', event }));
      ws.onclose = event => this._guard(token, () => this.emit('close', event));
      ws.onmessage = event => this._handleMessage(token, event, settings);
      this.configureTimer = this.setTimeoutImpl(() => {
        this._guard(token, () => {
          if (!this.configured) this.emit('error', { type: 'configure_timeout' });
        });
      }, settings.configure_timeout_ms || 10000);
      return { token, streamId: this.streamId };
    }

    sendFrame(payload) {
      if (!this.ws || this.ws.readyState !== 1) return false;
      const frame = this.encodePcmFrame(this.sequence++, payload);
      this.ws.send(frame);
      return true;
    }

    stop() {
      if (!this.ws || this.ws.readyState !== 1 || this.stopSent) return false;
      this.stopSent = true;
      this.ws.send(JSON.stringify({ action: 'stop' }));
      return true;
    }

    close(reason = 'client_close') {
      if (this.configureTimer) this.clearTimeoutImpl(this.configureTimer);
      this.configureTimer = null;
      const ws = this.ws;
      this.ws = null;
      if (ws && ws.readyState <= 1) {
        try { ws.close(1000, reason); } catch (_) { /* ignore */ }
      }
    }

    _guard(token, callback) {
      if (token !== this.sessionToken) return;
      callback();
    }

    _handleMessage(token, event, settings) {
      this._guard(token, () => {
        let msg;
        try { msg = JSON.parse(event.data); } catch (error) {
          this.emit('error', { type: 'invalid_json', error });
          return;
        }
        if (msg.type === 'ready') {
          this.ws?.send(JSON.stringify({
            action: 'configure',
            sample_rate: 16000,
            channels: 1,
            sample_format: 'pcm_s16le',
            browser_sample_rate: settings.browser_sample_rate,
            protocol_version: 2,
            profile: settings.profile,
            stream_id: this.streamId,
          }));
        } else if (msg.type === 'configured') {
          this.configured = true;
          if (this.configureTimer) this.clearTimeoutImpl(this.configureTimer);
          this.configureTimer = null;
          this.streamId = msg.data?.stream_id || this.streamId;
          this.emit('configured', msg.data || {});
        } else if (msg.type === 'live_result') {
          this.emit('result', msg.data || {});
        } else if (msg.type === 'finalizing') {
          this.emit('finalizing', msg.data || {});
        } else if (msg.type === 'final_transcript') {
          this.emit('final_transcript', msg.data || {});
        } else if (msg.type === 'stopped') {
          this.emit('stopped', msg.data || {});
        } else if (msg.type === 'error') {
          this.emit('error', msg.data || { message: msg.message });
        } else {
          this.emit('message', msg);
        }
      });
    }
  }

  return { LiveMicClient };
}));
