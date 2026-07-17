(function attachLiveMicController(root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.HuiWuLiveMicController = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function createLiveMicController(root) {
  const MicState = Object.freeze({
    IDLE: 'idle',
    CONNECTING: 'connecting',
    RECORDING: 'recording',
    STOPPING: 'stopping',
    FINALIZING: 'finalizing',
    COMPLETE: 'complete',
    ERROR: 'error',
  });

  function createTransition(initial = MicState.IDLE) {
    let state = initial;
    const history = [state];
    return {
      get state() { return state; },
      get history() { return history.slice(); },
      set(next) {
        if (!Object.values(MicState).includes(next)) throw new Error(`invalid mic state: ${next}`);
        state = next;
        history.push(next);
        return state;
      },
      isActive() {
        return [MicState.CONNECTING, MicState.RECORDING, MicState.STOPPING, MicState.FINALIZING].includes(state);
      },
    };
  }

  class LiveMicController {
    constructor({ client, onState, onLevel } = {}) {
      this.client = client;
      this.onState = onState || (() => {});
      this.onLevel = onLevel || (() => {});
      this.transition = createTransition();
      this.sessionToken = 0;
      this.cleanupCalls = 0;
      this._wireClient();
    }

    start() {
      const token = ++this.sessionToken;
      this._setState(MicState.CONNECTING, { token });
      return token;
    }

    markRecording(token = this.sessionToken) {
      if (token !== this.sessionToken) return false;
      this._setState(MicState.RECORDING, { token });
      return true;
    }

    stop(token = this.sessionToken) {
      if (token !== this.sessionToken) return false;
      this._setState(MicState.STOPPING, { token });
      this.client?.stop?.();
      return true;
    }

    cleanup(token = this.sessionToken) {
      if (token !== this.sessionToken) return false;
      this.cleanupCalls += 1;
      this._setState(MicState.IDLE, { token });
      return true;
    }

    _wireClient() {
      if (!this.client?.on) return;
      this.client.on('configured', () => this.markRecording());
      this.client.on('finalizing', data => this._setState(MicState.FINALIZING, data));
      this.client.on('stopped', () => {
        this._setState(MicState.COMPLETE, {});
        this.cleanup();
      });
      this.client.on('error', error => this._setState(MicState.ERROR, error));
    }

    _setState(state, detail) {
      this.transition.set(state);
      this.onState(state, detail);
    }
  }

  return { MicState, createTransition, LiveMicController };
}));
