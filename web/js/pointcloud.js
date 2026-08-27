// GPU-morphing point cloud: every particle carries two positions (state A / state B)
// and the vertex shader interpolates between them, so even 10k+ image sprites
// morph at full frame rate.
import * as THREE from "three";

const VERT = /* glsl */ `
  attribute vec3 positionB;
  attribute vec2 uvOffset;
  attribute vec3 classColor;
  attribute float pIndex;
  attribute float rnd;

  uniform float uT;          // global transition 0..1 (already eased)
  uniform float uSize;
  uniform float uSelected;   // index of hovered point, -1 = none

  varying vec2 vUvOffset;
  varying vec3 vColor;
  varying float vSel;

  void main() {
    // small per-point stagger so the swarm streams between states like the original demo
    float t = smoothstep(0.0, 1.0, clamp(uT * 1.25 - rnd * 0.25, 0.0, 1.0));
    vec3 p = mix(position, positionB, t);
    vec4 mv = modelViewMatrix * vec4(p, 1.0);

    vSel = (abs(pIndex - uSelected) < 0.5) ? 1.0 : 0.0;
    float size = uSize * (1.0 + vSel * 1.6);
    gl_PointSize = clamp(size * 300.0 / -mv.z, 1.5, 160.0);
    gl_Position = projectionMatrix * mv;

    vUvOffset = uvOffset;
    vColor = classColor;
  }
`;

const FRAG = /* glsl */ `
  uniform sampler2D uAtlas;
  uniform float uCell;
  uniform int uMode;        // 0 = luminance glow (MNIST), 1 = rgb photos

  varying vec2 vUvOffset;
  varying vec3 vColor;
  varying float vSel;

  void main() {
    vec2 uv = vUvOffset + gl_PointCoord * uCell;
    if (uMode == 0) {
      float v = texture2D(uAtlas, uv).r;
      if (v < 0.04) discard;
      vec3 c = vColor * v * (1.0 + vSel * 1.2);
      gl_FragColor = vec4(c, v);
    } else {
      vec3 rgb = texture2D(uAtlas, uv).rgb;
      vec2 pc = gl_PointCoord;
      float d = max(abs(pc.x - 0.5), abs(pc.y - 0.5));
      // class-colored frame around each photo
      if (d > 0.44) rgb = vColor;
      if (vSel > 0.5) rgb = mix(rgb, vec3(1.0), 0.35);
      gl_FragColor = vec4(rgb, 1.0);
    }
  }
`;

export function classPalette(n) {
  const colors = [];
  for (let i = 0; i < n; i++) {
    colors.push(new THREE.Color().setHSL((i * 0.61803) % 1.0, 0.85, 0.62));
  }
  return colors;
}

export class PointCloud {
  /**
   * @param {Object} d  {posA:Float32Array, posB:Float32Array, labels:Uint8Array,
   *                     atlasTexture:THREE.Texture, meta:Object, worldScale:number}
   */
  constructor(d) {
    this.meta = d.meta;
    this.count = d.meta.count;
    this.worldScale = d.worldScale;
    this.posA = d.posA;
    this.posB = d.posB;
    this.labels = d.labels;
    this.palette = classPalette(d.meta.classes.length);

    const geo = new THREE.BufferGeometry();
    const n = this.count;
    const pA = new Float32Array(n * 3);
    const pB = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) {
      pA[i] = d.posA[i] * d.worldScale;
      pB[i] = d.posB[i] * d.worldScale;
    }
    const uvOff = new Float32Array(n * 2);
    const colors = new Float32Array(n * 3);
    const rnd = new Float32Array(n);
    const pIndex = new Float32Array(n);
    const cols = d.meta.atlasCols;
    const cell = 1.0 / cols;
    for (let i = 0; i < n; i++) {
      uvOff[i * 2] = (i % cols) * cell;
      uvOff[i * 2 + 1] = Math.floor(i / cols) * cell;
      const c = this.palette[d.labels[i] % this.palette.length];
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
      rnd[i] = Math.abs(Math.sin(i * 12.9898) * 43758.5453) % 1;
      pIndex[i] = i;
    }

    geo.setAttribute("position", new THREE.BufferAttribute(pA, 3));
    geo.setAttribute("positionB", new THREE.BufferAttribute(pB, 3));
    geo.setAttribute("uvOffset", new THREE.BufferAttribute(uvOff, 2));
    geo.setAttribute("classColor", new THREE.BufferAttribute(colors, 3));
    geo.setAttribute("rnd", new THREE.BufferAttribute(rnd, 1));
    geo.setAttribute("pIndex", new THREE.BufferAttribute(pIndex, 1));
    geo.computeBoundingSphere();

    const isLum = d.meta.atlasMode === "luminance";
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uT: { value: 0 },
        uSize: { value: isLum ? 1.0 : 1.1 },
        uSelected: { value: -1 },
        uAtlas: { value: d.atlasTexture },
        uCell: { value: cell },
        uMode: { value: isLum ? 0 : 1 },
      },
      vertexShader: VERT,
      fragmentShader: FRAG,
      transparent: isLum,
      depthWrite: !isLum,
      blending: isLum ? THREE.AdditiveBlending : THREE.NormalBlending,
    });
    this.points = new THREE.Points(geo, this.material);
    this.points.frustumCulled = false;
  }

  setTransition(t) {
    this.material.uniforms.uT.value = t;
  }

  setSelected(i) {
    this.material.uniforms.uSelected.value = i;
  }

  /** current world position of point i at transition t (ignores the tiny stagger) */
  positionAt(i, t, out) {
    const tt = Math.min(1, Math.max(0, t * 1.25));
    const s = tt * tt * (3 - 2 * tt);
    const ws = this.worldScale;
    out.set(
      (this.posA[i * 3] + (this.posB[i * 3] - this.posA[i * 3]) * s) * ws,
      (this.posA[i * 3 + 1] + (this.posB[i * 3 + 1] - this.posA[i * 3 + 1]) * s) * ws,
      (this.posA[i * 3 + 2] + (this.posB[i * 3 + 2] - this.posA[i * 3 + 2]) * s) * ws
    );
    return out;
  }

  dispose() {
    this.points.geometry.dispose();
    this.material.dispose();
  }
}
