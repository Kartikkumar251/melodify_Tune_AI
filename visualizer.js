/**
 * Melodyfy Music Visualizer — pure Canvas 2D, zero dependencies.
 *
 * Queue-safe usage (works before OR after module loads):
 *   window._vizQueue = window._vizQueue || [];
 *   window._vizQueue.push({ wrapId:'vw-ID', audioId:'audio-ID' });
 *
 * Direct API:
 *   const viz = new MusicVisualizer(el, { preset:'melodyfy', height:80 });
 *   viz.connectAudio(audioEl);
 *   viz.start();
 */

/* Colour presets */
const PRESETS = {
  melodyfy: {
    bg:    ['#07070a','#0e0c12'],
    lines: [
      { inner:'#EDE8DF', outer:'#8A8480', glow:'rgba(237,232,223,0.55)' },
      { inner:'#C8C2BA', outer:'#6A6560', glow:'rgba(200,194,186,0.35)' },
      { inner:'#9A9590', outer:'#504E4C', glow:'rgba(154,149,144,0.25)' },
    ]
  },
  warm: {
    bg:    ['#0a0500','#180a02'],
    lines: [
      { inner:'#FFD060', outer:'#FF6000', glow:'rgba(255,180,0,0.7)' },
      { inner:'#FF8040', outer:'#CC2000', glow:'rgba(255,80,0,0.45)' },
      { inner:'#FF4050', outer:'#881020', glow:'rgba(255,40,60,0.3)' },
    ]
  },
  neon: {
    bg:    ['#05050f','#0a0a1e'],
    lines: [
      { inner:'#ff00ff', outer:'#8800ff', glow:'rgba(255,0,255,0.65)' },
      { inner:'#00ffff', outer:'#0055ff', glow:'rgba(0,255,255,0.45)' },
      { inner:'#ffff00', outer:'#ff8800', glow:'rgba(255,220,0,0.3)'  },
    ]
  },
  cool: {
    bg:    ['#050c14','#0a1420'],
    lines: [
      { inner:'#80d8ff', outer:'#0072b8', glow:'rgba(100,210,255,0.6)' },
      { inner:'#80ffcc', outer:'#009966', glow:'rgba(100,255,190,0.4)' },
      { inner:'#b0c8ff', outer:'#3060cc', glow:'rgba(150,180,255,0.3)' },
    ]
  }
};

// Shared singleton AudioContext across all visualizers
let _sharedAudioCtx = null;
function getSharedAudioContext() {
  if (!_sharedAudioCtx) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (AudioCtx) {
      _sharedAudioCtx = new AudioCtx();
    }
  }
  return _sharedAudioCtx;
}

export class MusicVisualizer {
  constructor(container, opts = {}) {
    if (!container) return;
    this._c      = container;
    this._preset = PRESETS[opts.preset] || PRESETS.melodyfy;
    this._h      = opts.height || 80;
    this._t      = Math.random() * 100;
    this._idle   = Math.random() * Math.PI * 2;
    this._raf    = null;
    this._running= false;
    this._built  = false;
    this._pendingStart = false;
    this._analyser   = null;
    this._data       = null;
    this._audioEl    = null;
    this._playing    = false;
    this._low = 0; this._mid = 0; this._high = 0;
    this._tf  = 0;
    this._kick = 0;
    this._prevEnergy = 0;
    this._deferBuild();
  }

  _deferBuild() {
    const try_ = () => {
      const w = this._c.offsetWidth;
      if (w > 0) { this._build(w); }
      else       { requestAnimationFrame(try_); }
    };
    try_();
  }

  _build(w) {
    const h = this._h;
    const c = this._c;
    c.style.position = 'relative';
    c.style.overflow = 'hidden';
    const pr = Math.min(devicePixelRatio, 2);
    const cvs = document.createElement('canvas');
    cvs.width  = w * pr;
    cvs.height = h * pr;
    cvs.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;display:block;pointer-events:none;';
    c.appendChild(cvs);
    this._cvs = cvs;
    this._ctx = cvs.getContext('2d');
    this._W   = cvs.width;
    this._H   = cvs.height;
    this._built = true;
    if (window.ResizeObserver) {
      this._ro = new ResizeObserver(() => {
        const nw = this._c.offsetWidth;
        if (nw > 0) { cvs.width = nw * pr; this._W = cvs.width; }
      });
      this._ro.observe(c);
    }
    if (this._pendingStart) { this._pendingStart = false; this._run(); }
  }

  connectAudio(el) {
    if (!el || this._audioEl === el) return;
    this._audioEl = el;
    try {
      if (!el.crossOrigin) el.crossOrigin = 'anonymous';
    } catch(e) {}

    const onPlay = () => {
      this._playing = true;
      try {
        const ctx = getSharedAudioContext();
        if (ctx) {
          if (ctx.state === 'suspended') {
            ctx.resume().catch(() => {});
          }
          if (!this._analyser) {
            this._analyser = ctx.createAnalyser();
            this._analyser.fftSize = 1024;
            this._data = new Uint8Array(this._analyser.frequencyBinCount);
          }
          if (!el._mfyNode) {
            el._mfyNode = ctx.createMediaElementSource(el);
            el._mfyNode.connect(this._analyser);
            this._analyser.connect(ctx.destination);
          }
        }
      } catch(e) {
        console.warn('[viz] AudioContext connection:', e);
      }
    };
    el.addEventListener('play',  onPlay);
    el.addEventListener('pause', () => { this._playing = false; });
    el.addEventListener('ended', () => { this._playing = false; });
    if (!el.paused) onPlay();
  }

  start() {
    if (this._running) return;
    if (!this._built) { this._pendingStart = true; return; }
    this._running = true;
    this._run();
  }

  stop() {
    this._running = false;
    if (this._raf) { cancelAnimationFrame(this._raf); this._raf = null; }
  }

  _run() {
    if (!this._running) return;
    this._frame();
    this._raf = requestAnimationFrame(() => this._run());
  }

  _getFreqs() {
    if (!this._analyser || !this._data || !this._playing) return;
    this._analyser.getByteFrequencyData(this._data);
    const d = this._data, len = d.length;
    const bassEnd = Math.floor(len * 0.04);
    const midEnd  = Math.floor(len * 0.20);
    let bass = 0, mid = 0, high = 0;
    for (let i = 0;       i < bassEnd; i++) bass += d[i];
    for (let i = bassEnd; i < midEnd;  i++) mid  += d[i];
    for (let i = midEnd;  i < len;     i++) high += d[i];
    bass /= bassEnd * 255;
    mid  /= (midEnd - bassEnd) * 255;
    high /= (len - midEnd) * 255;
    const s = 0.15;
    this._low  += (bass - this._low)  * s;
    this._mid  += (mid  - this._mid)  * s;
    this._high += (high - this._high) * s;
    const energy = bass * 0.7 + mid * 0.3;
    if (energy > this._prevEnergy * 1.5 && energy > 0.25)
      this._kick = Math.min(1, this._kick + (energy - this._prevEnergy) * 2.5);
    this._prevEnergy = energy;
    this._kick *= 0.88;
  }

  _frame() {
    const ctx = this._ctx;
    const W   = this._W;
    const H   = this._H;
    const pr  = Math.min(devicePixelRatio, 2);

    this._tf += ((this._playing ? 1 : 0) - this._tf) * 0.04;
    const tf = this._tf;

    this._getFreqs();
    const low = this._low, mid = this._mid, high = this._high, kick = this._kick;

    const speed = 1.2 + low * 3 + kick * 2;
    this._t    += tf > 0.01 ? speed * 0.016 : 0.008;
    this._idle += 0.007;
    const t = this._t;

    /* background */
    const bg = ctx.createLinearGradient(0, H, 0, 0);
    bg.addColorStop(0, this._preset.bg[0]);
    bg.addColorStop(1, this._preset.bg[1]);
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    const lines = this._preset.lines;
    for (let li = 0; li < lines.length; li++) {
      const ln = lines[li];
      const cy = H * 0.5;

      const amp = li === 0
        ? H * (0.04 + low  * 0.28 + kick * 0.18)
        : li === 1
        ? H * (0.03 + mid  * 0.22 + kick * 0.08)
        : H * (0.02 + high * 0.18 + kick * 0.04);

      const idleAmp = H * 0.04 * (1 - tf);
      const eAmp = amp + idleAmp;

      const freq  = 2.2 + li * 0.6 + low * 1.5;
      const phase = t * (1 + li * 0.3) + li * 1.1;
      const vOff  = (li - 1) * H * 0.04;
      const lineW = Math.max(1, (2.5 - li * 0.5) * pr * (1 + low * 0.4 + kick * 0.3));

      ctx.beginPath();
      const segs = 120;
      for (let i = 0; i <= segs; i++) {
        const nx = i / segs;
        const x  = nx * W;
        const y1 = Math.sin(nx * Math.PI * 2 * freq + phase) * eAmp;
        const y2 = Math.sin(nx * Math.PI * 2 * freq * 2.1 + phase * 1.3) * eAmp * 0.3 * tf;
        const y3 = Math.sin(nx * Math.PI * 20 - t * 4) * kick * H * 0.06 * tf;
        const y  = cy + vOff + y1 + y2 + y3;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }

      /* glow */
      ctx.save();
      ctx.shadowColor = ln.glow;
      ctx.shadowBlur  = (8 + kick * 12) * (1 + tf) + 4;
      ctx.strokeStyle = ln.inner;
      ctx.lineWidth   = lineW * 2.5;
      ctx.globalAlpha = 0.35 + tf * 0.25;
      ctx.stroke();
      ctx.restore();

      /* core */
      ctx.save();
      ctx.shadowColor = ln.glow;
      ctx.shadowBlur  = (3 + kick * 6) * tf;
      const grad = ctx.createLinearGradient(0, 0, W, 0);
      grad.addColorStop(0.0, ln.inner);
      grad.addColorStop(0.5, ln.outer);
      grad.addColorStop(1.0, ln.inner);
      ctx.strokeStyle = grad;
      ctx.lineWidth   = lineW;
      ctx.globalAlpha = 0.75 + tf * 0.25;
      ctx.stroke();
      ctx.restore();
    }

    /* film grain — sample every 3rd pixel for performance */
    const img = ctx.getImageData(0, 0, W, H);
    const p   = img.data;
    for (let i = 0; i < p.length; i += 12) {
      const n = (Math.random() - 0.5) * 20;
      p[i]   = Math.max(0, Math.min(255, p[i]   + n));
      p[i+1] = Math.max(0, Math.min(255, p[i+1] + n));
      p[i+2] = Math.max(0, Math.min(255, p[i+2] + n));
    }
    ctx.putImageData(img, 0, 0);
  }
}

/* Queue processor */
const _vizRegistry = {};

function _doMount(item) {
  const id = item.wrapId;
  if (_vizRegistry[id]) return;
  const wrap  = document.getElementById(id);
  const audio = item.audioId ? document.getElementById(item.audioId) : null;
  if (!wrap) return;
  const viz = new MusicVisualizer(wrap, { preset: item.preset || 'melodyfy', height: item.height || 80 });
  if (audio) viz.connectAudio(audio);
  viz.start();
  _vizRegistry[id] = viz;
}

const _earlyQueue = Array.isArray(window._vizQueue) ? window._vizQueue : [];
_earlyQueue.forEach(_doMount);
window._vizQueue = { push: (item) => _doMount(item) };
window.MusicVisualizer = MusicVisualizer;