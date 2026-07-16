class HuiWuPcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.clock = 0;
    this.sum = 0;
    this.count = 0;
    this.pending = [];
    this.enabled = true;
    this.levelSumSquares = 0;
    this.levelPeak = 0;
    this.levelSamples = 0;
    this.levelWindowSamples = Math.max(512, Math.round(sampleRate * 0.04));
    this.port.onmessage = (event) => {
      if (event.data?.type === 'capture.stop') {
        this.enabled = false;
        this.flush();
        this.port.postMessage({ type: 'audio.level', rms: 0, peak: 0 });
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
      const sample = channel[index];
      this.levelSumSquares += sample * sample;
      this.levelPeak = Math.max(this.levelPeak, Math.abs(sample));
      this.levelSamples += 1;
      this.sum += sample;
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

    if (this.levelSamples >= this.levelWindowSamples) {
      this.port.postMessage({
        type: 'audio.level',
        rms: Math.sqrt(this.levelSumSquares / this.levelSamples),
        peak: this.levelPeak,
      });
      this.levelSumSquares = 0;
      this.levelPeak = 0;
      this.levelSamples = 0;
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

registerProcessor('huiwu-pcm', HuiWuPcmProcessor);
