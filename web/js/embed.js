// Turning user input (a drawn digit / an uploaded photo) into a point in feature space.
import { loadMLP } from "./mlp.js";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js";
let ortPromise = null;

function loadOrt() {
  if (!ortPromise) {
    ortPromise = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = ORT_URL;
      s.onload = () => {
        window.ort.env.wasm.wasmPaths =
          "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
        resolve(window.ort);
      };
      s.onerror = () => reject(new Error("failed to load onnxruntime-web"));
      document.head.appendChild(s);
    });
  }
  return ortPromise;
}

/** Embeds 28x28 hand-drawn digits with the exported MNIST encoders. */
export class DigitEmbedder {
  constructor(base) {
    this.base = base;
    this.ready = Promise.all([
      loadMLP(`${base}/encoderA.json`),
      loadMLP(`${base}/encoderB.json`),
    ]).then(([a, b]) => {
      this.encA = a;
      this.encB = b;
    });
  }

  /** canvas: any square canvas with a white-on-black digit. Returns {posA, posB, thumb}. */
  async embed(canvas) {
    await this.ready;
    const small = document.createElement("canvas");
    small.width = small.height = 28;
    const ctx = small.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, 28, 28);
    ctx.drawImage(canvas, 0, 0, 28, 28);
    const img = ctx.getImageData(0, 0, 28, 28).data;
    const x = new Float32Array(784);
    for (let i = 0; i < 784; i++) x[i] = img[i * 4] / 255;
    return { posA: this.encA.forward(x), posB: this.encB.forward(x), thumb: small };
  }
}

/** Embeds arbitrary photos: MobileNetV2 (ONNX in-browser) -> projector MLPs. */
export class PhotoEmbedder {
  constructor(base) {
    this.base = base;
    this.ready = (async () => {
      const ort = await loadOrt();
      const [a, b, session] = await Promise.all([
        loadMLP(`${base}/projectorA.json`),
        loadMLP(`${base}/projectorB.json`),
        ort.InferenceSession.create(`${base}/mobilenet.onnx`, {
          executionProviders: ["wasm"],
        }),
      ]);
      this.projA = a;
      this.projB = b;
      this.session = session;
      this.ort = ort;
    })();
  }

  /** src: HTMLImageElement, HTMLVideoElement or canvas. Returns {posA, posB, thumb}. */
  async embed(src) {
    await this.ready;
    const c = document.createElement("canvas");
    c.width = c.height = 224;
    const ctx = c.getContext("2d");
    // center-crop to square, then resize
    const w = src.naturalWidth || src.videoWidth || src.width;
    const h = src.naturalHeight || src.videoHeight || src.height;
    const s = Math.min(w, h);
    const sx = (w - s) / 2;
    const sy = (h - s) / 2;
    ctx.drawImage(src, sx, sy, s, s, 0, 0, 224, 224);
    const data = ctx.getImageData(0, 0, 224, 224).data;
    const chw = new Float32Array(3 * 224 * 224);
    const hw = 224 * 224;
    for (let i = 0; i < hw; i++) {
      chw[i] = data[i * 4] / 255;
      chw[hw + i] = data[i * 4 + 1] / 255;
      chw[2 * hw + i] = data[i * 4 + 2] / 255;
    }
    const input = new this.ort.Tensor("float32", chw, [1, 3, 224, 224]);
    const out = await this.session.run({ image: input });
    const emb = out.embedding.data;

    const thumb = document.createElement("canvas");
    thumb.width = thumb.height = 64;
    thumb.getContext("2d").drawImage(c, 0, 0, 64, 64);
    return { posA: this.projA.forward(emb), posB: this.projB.forward(emb), thumb };
  }
}
