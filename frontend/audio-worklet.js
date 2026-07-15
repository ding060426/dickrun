class DiTingPcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.clock = 0;
    this.sum = 0;
    this.count = 0;
    this.pending = [];
    this.enabled = true;
    this.port.onmessage = (event) => {
      if (event.data?.type === 'capture.stop') {
        this.enabled = false;
        this.flush();
      }
    };
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!this.enabled || !channel) return true;

    // Browsers are commonly fixed at 44.1/48 kHz even when 16 kHz is
    // requested. Average each source-rate bucket and maintain a fractional
    // clock across render quanta so no microphone frames are discarded.
    for (let index = 0; index < channel.length; index += 1) {
      this.sum += channel[index];
      this.count += 1;
      this.clock += this.targetRate;
      if (this.clock >= sampleRate) {
        const value = Math.max(-1, Math.min(1, this.sum / this.count));
        this.pending.push(value < 0 ? Math.round(value * 32768) : Math.round(value * 32767));
        this.clock -= sampleRate;
        this.sum = 0;
        this.count = 0;
      }
    }

    if (this.pending.length >= 640) this.flush();
    return true;
  }

  flush() {
    if (!this.pending.length) return;
    const pcm = new Int16Array(this.pending);
    this.pending = [];
    this.port.postMessage({ type: 'audio.pcm', payload: pcm.buffer }, [pcm.buffer]);
  }
}

registerProcessor('diting-pcm', DiTingPcmProcessor);
