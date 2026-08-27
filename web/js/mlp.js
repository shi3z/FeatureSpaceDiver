// Tiny MLP inference for weights exported by the Python pipeline (base64 float32).

function b64ToF32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export class MLP {
  constructor(json) {
    this.layers = json.layers.map((l) => ({
      outDim: l.outDim,
      inDim: l.inDim,
      weight: b64ToF32(l.weight), // row-major [out][in]
      bias: b64ToF32(l.bias),
      act: l.act,
    }));
    this.postCenter = json.postCenter;
    this.postScale = json.postScale;
    this.inputScale = json.inputNorm ? json.inputNorm.scale : 1.0;
  }

  forward(input) {
    let x = input;
    if (this.inputScale !== 1.0) {
      x = Float32Array.from(input, (v) => v * this.inputScale);
    }
    for (const l of this.layers) {
      const out = new Float32Array(l.outDim);
      for (let o = 0; o < l.outDim; o++) {
        let s = l.bias[o];
        const off = o * l.inDim;
        for (let i = 0; i < l.inDim; i++) s += l.weight[off + i] * x[i];
        out[o] = l.act === "relu" && s < 0 ? 0 : s;
      }
      x = out;
    }
    // map raw 3D output into the normalized coordinate space of the bins
    return [
      (x[0] - this.postCenter[0]) * this.postScale,
      (x[1] - this.postCenter[1]) * this.postScale,
      (x[2] - this.postCenter[2]) * this.postScale,
    ];
  }
}

export async function loadMLP(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load ${url}`);
  return new MLP(await res.json());
}
