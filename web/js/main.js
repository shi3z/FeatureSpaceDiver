import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VRButton } from "three/addons/webxr/VRButton.js";
import { PointCloud } from "./pointcloud.js";
import { DigitEmbedder, PhotoEmbedder } from "./embed.js";

const WORLD_SCALE = 18;

// ---------- renderer / scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
document.getElementById("app").appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x02030a);

const camera = new THREE.PerspectiveCamera(
  60, window.innerWidth / window.innerHeight, 0.05, 2000);
const rig = new THREE.Group(); // moved by VR controllers; camera lives inside
rig.add(camera);
scene.add(rig);
camera.position.set(0, 0, 46);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.target.set(0, 0, 0);

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------- transition state ----------
const morph = { current: 0, target: 0, auto: false, autoDir: 1 };

function setTarget(t) {
  morph.target = t;
  morph.auto = false;
  autoChk.checked = false;
}

// ---------- user-added points ----------
const userGroup = new THREE.Group();
scene.add(userGroup);
const userPoints = []; // {sprite, posA, posB}

function addUserPoint(posA, posB, thumbCanvas, spawnFrom = null) {
  const c = document.createElement("canvas");
  c.width = c.height = 72;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, 72, 72);
  ctx.drawImage(thumbCanvas, 4, 4, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: tex, depthTest: true }));
  sprite.scale.setScalar(1.6);
  userGroup.add(sprite);
  userPoints.push({
    sprite, posA, posB,
    // when spawned from a VR capture, the sprite flies from the headset to its home
    anim: spawnFrom ? { from: spawnFrom.clone(), start: performance.now() } : null,
  });
  updateUserPoints();
}

function easedT() {
  const tt = Math.min(1, Math.max(0, morph.current * 1.25));
  return tt * tt * (3 - 2 * tt);
}

const tmpTarget = new THREE.Vector3();
function updateUserPoints() {
  const s = easedT();
  const now = performance.now();
  for (const p of userPoints) {
    tmpTarget.set(
      (p.posA[0] + (p.posB[0] - p.posA[0]) * s) * WORLD_SCALE,
      (p.posA[1] + (p.posB[1] - p.posA[1]) * s) * WORLD_SCALE,
      (p.posA[2] + (p.posB[2] - p.posA[2]) * s) * WORLD_SCALE
    );
    if (p.anim) {
      const k = Math.min(1, (now - p.anim.start) / 1800);
      const e = k * k * (3 - 2 * k);
      p.sprite.position.lerpVectors(p.anim.from, tmpTarget, e);
      if (k >= 1) p.anim = null;
    } else {
      p.sprite.position.copy(tmpTarget);
    }
  }
}

function clearUserPoints() {
  for (const p of userPoints) {
    userGroup.remove(p.sprite);
    p.sprite.material.map.dispose();
    p.sprite.material.dispose();
  }
  userPoints.length = 0;
}

// ---------- dataset loading ----------
let cloud = null;
let atlasImage = null;
let embedder = null;

async function fetchBin(url, Type) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`missing ${url}`);
  return new Type(await res.arrayBuffer());
}

function loadTexture(url) {
  return new Promise((resolve, reject) => {
    new THREE.TextureLoader().load(url, (t) => {
      t.flipY = false;
      t.generateMipmaps = false;
      t.minFilter = THREE.LinearFilter;
      t.magFilter = THREE.LinearFilter;
      t.colorSpace = THREE.NoColorSpace;
      resolve(t);
    }, undefined, reject);
  });
}

async function loadDataset(name) {
  setStatus("Loading...");
  try {
    const base = `data/${name}`;
    const meta = await (await fetch(`${base}/meta.json`)).json();
    const [posA, posB, labels, atlasTexture] = await Promise.all([
      fetchBin(`${base}/posA.bin`, Float32Array),
      fetchBin(`${base}/posB.bin`, Float32Array),
      fetchBin(`${base}/labels.bin`, Uint8Array),
      loadTexture(`${base}/atlas.png`),
    ]);

    if (cloud) { scene.remove(cloud.points); cloud.dispose(); }
    clearUserPoints();
    cloud = new PointCloud({ posA, posB, labels, atlasTexture, meta, worldScale: WORLD_SCALE });
    cloud.setTransition(morph.current);
    scene.add(cloud.points);
    atlasImage = atlasTexture.image;

    labelA.textContent = meta.stateA;
    labelB.textContent = meta.stateB;
    buildLegend(meta, cloud.palette);
    setupUserInput(name, meta);
    setStatus(`${meta.name}: ${meta.count.toLocaleString()} points`);
  } catch (e) {
    console.error(e);
    setStatus(`Failed to load: ${e.message} (generate the data with the pipeline scripts)`);
  }
}

// ---------- hover picking (desktop) ----------
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2(-10, -10);
const tmpV = new THREE.Vector3();
let hovered = -1;

renderer.domElement.addEventListener("pointermove", (e) => {
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
});

function pick() {
  if (!cloud || renderer.xr.isPresenting) return;
  raycaster.setFromCamera(mouse, camera);
  const o = raycaster.ray.origin, d = raycaster.ray.direction;
  let best = -1, bestScore = 0.028; // angular threshold (rad-ish)
  for (let i = 0; i < cloud.count; i++) {
    cloud.positionAt(i, morph.current, tmpV).sub(o);
    const along = tmpV.dot(d);
    if (along < 1) continue;
    const perp2 = tmpV.lengthSq() - along * along;
    const score = Math.sqrt(Math.max(perp2, 0)) / along;
    if (score < bestScore) { bestScore = score; best = i; }
  }
  if (best !== hovered) {
    hovered = best;
    cloud.setSelected(best);
    updateInfoPanel(best);
  }
}

function updateInfoPanel(i) {
  const panel = document.getElementById("info");
  if (i < 0 || !cloud) { panel.style.display = "none"; return; }
  panel.style.display = "block";
  const meta = cloud.meta;
  document.getElementById("infoLabel").textContent =
    meta.classes[cloud.labels[i]] ?? "?";
  const c = document.getElementById("infoImg");
  const ctx = c.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, c.width, c.height);
  if (atlasImage) {
    const cols = meta.atlasCols, px = meta.spritePx;
    ctx.drawImage(atlasImage,
      (i % cols) * px, Math.floor(i / cols) * px, px, px,
      0, 0, c.width, c.height);
  }
}

// ---------- VR controllers ----------
const ctrl = [renderer.xr.getController(0), renderer.xr.getController(1)];
ctrl.forEach((c) => rig.add(c));
const headDir = new THREE.Vector3();
const headRight = new THREE.Vector3();
let morphButtonLatch = false;
let camButtonLatch = false;
let vrCaptureBusy = false;

// Passthrough camera capture from inside VR (Quest 3 / 3S, Horizon OS v74+):
// the Quest Browser exposes the headset RGB camera through getUserMedia, so we
// grab a frame from the same hidden <video>, embed it, and let the new point
// fly from the headset into its home in feature space.
async function vrCapture(gp) {
  if (vrCaptureBusy) return;
  if (!(embedder instanceof PhotoEmbedder))
    return setStatus("Capture needs the CIFAR-10 (photo) dataset");
  if (!camStream) {
    setStatus("No camera stream — starting camera...");
    startCamera();
    return;
  }
  if (camVideo.videoWidth === 0)
    return setStatus("Camera has no frames yet — try again in a moment");
  vrCaptureBusy = true;
  try {
    gp?.hapticActuators?.[0]?.pulse?.(0.7, 80); // shutter feedback
    const { posA, posB, thumb } = await embedder.embed(camVideo);
    const spawn = new THREE.Vector3();
    camera.getWorldPosition(spawn);
    camera.getWorldDirection(headDir);
    spawn.addScaledVector(headDir, 0.6);
    addUserPoint(posA, posB, thumb, spawn);
    setStatus("Captured — the photo is flying to its place");
  } catch (err) {
    console.error(err);
    setStatus(`Capture failed: ${err.message}`);
  }
  vrCaptureBusy = false;
}

// Everything is reachable from the right controller alone:
//   trigger (analog) = fly forward along your gaze   grip = speed boost
//   stick = turn & up/down    B = morph A⇔B    A = passthrough photo
// The left controller adds strafing on its stick (and Y/X mirror B/A).
function handleVRInput(dt) {
  const session = renderer.xr.getSession();
  if (!session) return;
  let morphBtn = false;
  let camBtn = false;
  let camGp = null;
  camera.getWorldDirection(headDir);
  headRight.crossVectors(headDir, camera.up).normalize();
  for (const src of session.inputSources) {
    const gp = src.gamepad;
    if (!gp) continue;
    const ax = gp.axes.length >= 4 ? [gp.axes[2], gp.axes[3]] : [gp.axes[0] || 0, gp.axes[1] || 0];
    const boost = gp.buttons[1]?.pressed ? 3.0 : 1.0; // squeeze = fast
    const speed = 14 * boost * dt;
    // trigger dives forward in the direction you are looking (analog pressure = speed)
    const dive = gp.buttons[0]?.value ?? (gp.buttons[0]?.pressed ? 1 : 0);
    if (dive > 0.05) rig.position.addScaledVector(headDir, dive * speed);
    if (src.handedness === "left") {
      if (Math.abs(ax[0]) > 0.1) rig.position.addScaledVector(headRight, ax[0] * speed);
      if (Math.abs(ax[1]) > 0.1) rig.position.addScaledVector(headDir, -ax[1] * speed);
    } else {
      if (Math.abs(ax[1]) > 0.1) rig.position.y -= ax[1] * speed;
      if (Math.abs(ax[0]) > 0.2) rig.rotateY(-ax[0] * 1.4 * dt);
    }
    if (gp.buttons[5]?.pressed) morphBtn = true; // B / Y button
    if (gp.buttons[4]?.pressed) { camBtn = true; camGp = gp; } // A / X button
  }
  if (morphBtn && !morphButtonLatch) {
    // the signature move of Feature Space Diver
    morph.target = morph.target > 0.5 ? 0 : 1;
    morph.auto = false;
  }
  morphButtonLatch = morphBtn;
  if (camBtn && !camButtonLatch) vrCapture(camGp);
  camButtonLatch = camBtn;
}

renderer.xr.addEventListener("sessionstart", () => {
  controls.enabled = false;
  rig.position.set(0, -1.2, 30); // eye height gets added by the headset
  hudPanel.visible = true;
  setStatus("VR session started");
  // In the photo space, open the passthrough camera so A can snap photos.
  // On Quest 3 this triggers the headset-camera permission dialog once.
  if (embedder instanceof PhotoEmbedder && !camStream) startCamera();
});
renderer.xr.addEventListener("sessionend", () => {
  hudPanel.visible = false;
  controls.enabled = true;
  rig.position.set(0, 0, 0);
  rig.rotation.set(0, 0, 0);
  camera.position.set(0, 0, 46);
  controls.target.set(0, 0, 0);
});

// ---------- UI ----------
const slider = document.getElementById("morph");
const autoChk = document.getElementById("auto");
const labelA = document.getElementById("labelA");
const labelB = document.getElementById("labelB");

// ---------- status line + on-page & in-VR debug log ----------
const logLines = [];
const hudCanvas = document.createElement("canvas");
hudCanvas.width = 1024;
hudCanvas.height = 256;
const hudCtx = hudCanvas.getContext("2d");
const hudTex = new THREE.CanvasTexture(hudCanvas);
const hudPanel = new THREE.Mesh(
  new THREE.PlaneGeometry(0.9, 0.225),
  new THREE.MeshBasicMaterial({ map: hudTex, transparent: true, depthTest: false })
);
hudPanel.position.set(0, -0.38, -1.0);
hudPanel.renderOrder = 999;
hudPanel.visible = false; // shown only inside VR
camera.add(hudPanel);

function drawHud() {
  hudCtx.clearRect(0, 0, 1024, 256);
  hudCtx.fillStyle = "rgba(5,8,20,0.72)";
  hudCtx.fillRect(0, 0, 1024, 256);
  hudCtx.fillStyle = "#cfe0ff";
  hudCtx.font = "30px sans-serif";
  logLines.slice(-5).forEach((l, i) => hudCtx.fillText(l.slice(0, 62), 16, 42 + i * 42));
  hudTex.needsUpdate = true;
}

function setStatus(s) {
  document.getElementById("status").textContent = s;
  const t = new Date().toTimeString().slice(0, 8);
  logLines.push(`[${t}] ${s}`);
  if (logLines.length > 80) logLines.shift();
  const pre = document.getElementById("log");
  if (pre) {
    pre.textContent = logLines.join("\n");
    pre.scrollTop = pre.scrollHeight;
  }
  drawHud();
  console.log("[FSD]", s);
}
window.addEventListener("error", (e) => setStatus(`Error: ${e.message}`));
window.addEventListener("unhandledrejection", (e) =>
  setStatus(`Error: ${e.reason?.message || e.reason}`));

slider.addEventListener("input", () => {
  morph.current = morph.target = parseFloat(slider.value);
  morph.auto = false;
  autoChk.checked = false;
});
document.getElementById("toA").addEventListener("click", () => setTarget(0));
document.getElementById("toB").addEventListener("click", () => setTarget(1));
autoChk.addEventListener("change", () => { morph.auto = autoChk.checked; });
document.getElementById("dataset").addEventListener("change", (e) => loadDataset(e.target.value));
document.getElementById("psize").addEventListener("input", (e) => {
  if (cloud) cloud.material.uniforms.uSize.value =
    parseFloat(e.target.value) * (cloud.meta.atlasMode === "luminance" ? 1.0 : 1.1);
});
window.addEventListener("keydown", (e) => {
  if (e.code === "Space" && e.target === document.body) {
    e.preventDefault();
    setTarget(morph.target > 0.5 ? 0 : 1);
  }
});

function buildLegend(meta, palette) {
  const el = document.getElementById("legend");
  el.innerHTML = "";
  meta.classes.forEach((name, i) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    const sw = document.createElement("i");
    sw.style.background = `#${palette[i % palette.length].getHexString()}`;
    chip.appendChild(sw);
    chip.appendChild(document.createTextNode(name));
    el.appendChild(chip);
  });
}

// ---------- user input (draw / upload) ----------
const drawPanel = document.getElementById("drawPanel");
const uploadPanel = document.getElementById("uploadPanel");
const pad = document.getElementById("pad");
const padCtx = pad.getContext("2d");

function clearPad() {
  padCtx.fillStyle = "#000";
  padCtx.fillRect(0, 0, pad.width, pad.height);
}
clearPad();

let drawing = false;
pad.addEventListener("pointerdown", (e) => {
  drawing = true;
  padCtx.beginPath();
  drawTo(e);
});
window.addEventListener("pointerup", () => { drawing = false; padCtx.beginPath(); });
pad.addEventListener("pointermove", drawTo);
function drawTo(e) {
  if (!drawing) return;
  const r = pad.getBoundingClientRect();
  const x = (e.clientX - r.left) * (pad.width / r.width);
  const y = (e.clientY - r.top) * (pad.height / r.height);
  padCtx.strokeStyle = "#fff";
  padCtx.lineWidth = 16;
  padCtx.lineCap = "round";
  padCtx.lineTo(x, y);
  padCtx.stroke();
  padCtx.beginPath();
  padCtx.moveTo(x, y);
}
document.getElementById("padClear").addEventListener("click", clearPad);
document.getElementById("padAdd").addEventListener("click", async () => {
  if (!embedder) return;
  const { posA, posB, thumb } = await embedder.embed(pad);
  addUserPoint(posA, posB, thumb);
  flyTo(posA, posB);
  clearPad();
});

document.getElementById("file").addEventListener("change", async (e) => {
  if (!embedder) return;
  setStatus("Embedding images... (the first run takes a moment to load the model)");
  for (const f of e.target.files) {
    const img = new Image();
    img.src = URL.createObjectURL(f);
    await img.decode();
    try {
      const { posA, posB, thumb } = await embedder.embed(img);
      addUserPoint(posA, posB, thumb);
      flyTo(posA, posB);
    } catch (err) {
      console.error(err);
      setStatus(`Embedding failed: ${err.message}`);
      return;
    } finally {
      URL.revokeObjectURL(img.src);
    }
  }
  setStatus(`Added ${e.target.files.length} image(s) (white-framed points)`);
  e.target.value = "";
});

// ---------- camera capture ----------
const camBox = document.getElementById("camBox");
const camVideo = document.getElementById("camVideo");
const camToggle = document.getElementById("camToggle");
let camStream = null;

function cameraConstraints() {
  const v = document.getElementById("camFacing").value;
  if (v.startsWith("id:")) return { video: { deviceId: { exact: v.slice(3) } }, audio: false };
  return { video: { facingMode: v }, audio: false };
}

// After permission is granted, replace the generic Rear/Front choices with the
// real camera list (on Quest 3 this exposes the passthrough cameras directly).
async function populateCameraList() {
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices())
      .filter((d) => d.kind === "videoinput");
    if (!devs.length) return;
    const sel = document.getElementById("camFacing");
    const currentId = camStream?.getVideoTracks()[0]?.getSettings?.().deviceId;
    sel.innerHTML = "";
    devs.forEach((d, i) => {
      const o = document.createElement("option");
      o.value = "id:" + d.deviceId;
      o.textContent = d.label || `Camera ${i + 1}`;
      sel.appendChild(o);
    });
    if (currentId) sel.value = "id:" + currentId;
  } catch (e) {
    console.warn("enumerateDevices failed", e);
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera is not available here (HTTPS or localhost required)");
    return;
  }
  stopCamera();
  setStatus("Starting camera...");
  try {
    const pre = (await navigator.mediaDevices.enumerateDevices())
      .filter((d) => d.kind === "videoinput");
    setStatus(`Found ${pre.length} video input device(s)`);
  } catch (e) {
    setStatus(`enumerateDevices failed: ${e.message}`);
  }
  // Quest Browser can be picky about constraints — fall back to plain video:true
  let stream = null;
  let lastErr = null;
  for (const constraints of [cameraConstraints(), { video: true, audio: false }]) {
    setStatus(`getUserMedia(${JSON.stringify(constraints.video).slice(0, 40)})...`);
    try {
      stream = await Promise.race([
        navigator.mediaDevices.getUserMedia(constraints),
        new Promise((_, rej) => setTimeout(
          () => rej(Object.assign(new Error("no response for 15s (permission dialog pending?)"),
                                  { name: "Timeout" })), 15000)),
      ]);
      break;
    } catch (err) {
      lastErr = err;
      setStatus(`→ failed: ${err.name}: ${err.message}`);
    }
  }
  if (!stream) {
    console.error(lastErr);
    setStatus(`Could not start the camera: ${lastErr?.name}: ${lastErr?.message}`);
    return;
  }
  camStream = stream;
  camVideo.srcObject = stream;
  camBox.style.display = "block";
  camToggle.textContent = "Stop camera";
  try { await camVideo.play(); } catch (e) { console.warn("video.play()", e); }
  // wait for real frames before declaring victory
  const t0 = performance.now();
  while (camVideo.videoWidth === 0 && performance.now() - t0 < 8000 && camStream === stream) {
    await new Promise((r) => setTimeout(r, 100));
  }
  if (camStream !== stream) return; // stopped/restarted meanwhile
  const label = stream.getVideoTracks()[0]?.label || "camera";
  if (camVideo.videoWidth > 0) {
    setStatus(`Camera running: ${camVideo.videoWidth}×${camVideo.videoHeight} (${label})`);
  } else {
    setStatus(`Camera opened (${label}) but no frames arrived — try another camera in the list`);
  }
  populateCameraList();
}

function stopCamera() {
  if (camStream) {
    camStream.getTracks().forEach((t) => t.stop());
    camStream = null;
  }
  camVideo.srcObject = null;
  camBox.style.display = "none";
  camToggle.textContent = "📷 Start camera";
}

camToggle.addEventListener("click", () => (camStream ? stopCamera() : startCamera()));
document.getElementById("camFacing").addEventListener("change", () => {
  if (camStream) { stopCamera(); startCamera(); }
});
document.getElementById("camShot").addEventListener("click", async () => {
  if (!embedder) return setStatus("Dataset is still loading — wait a moment");
  if (!camStream) return setStatus("Start the camera first");
  if (camVideo.videoWidth === 0)
    return setStatus("No video frames yet — wait a moment or pick another camera");
  setStatus("Embedding the captured photo...");
  try {
    const { posA, posB, thumb } = await embedder.embed(camVideo);
    addUserPoint(posA, posB, thumb);
    flyTo(posA, posB);
    setStatus("Added the captured photo (white-framed point)");
  } catch (err) {
    console.error(err);
    setStatus(`Embedding failed: ${err.message}`);
  }
});

function setupUserInput(name, meta) {
  stopCamera();
  drawPanel.style.display = meta.userInput === "draw" ? "block" : "none";
  uploadPanel.style.display = meta.userInput === "upload" ? "block" : "none";
  embedder = meta.userInput === "draw"
    ? new DigitEmbedder(`data/${name}`)
    : new PhotoEmbedder(`data/${name}`);
}

// smoothly aim the desktop camera at a newly added point
let flyTarget = null;
function flyTo(posA, posB) {
  const s = easedT();
  flyTarget = new THREE.Vector3(
    (posA[0] + (posB[0] - posA[0]) * s) * WORLD_SCALE,
    (posA[1] + (posB[1] - posA[1]) * s) * WORLD_SCALE,
    (posA[2] + (posB[2] - posA[2]) * s) * WORLD_SCALE
  );
}

// ---------- render loop ----------
const clock = new THREE.Clock();
let frame = 0;

renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.05);

  if (morph.auto) {
    morph.target += morph.autoDir * dt * 0.35;
    if (morph.target > 1.25) { morph.target = 1.25; morph.autoDir = -1; }
    if (morph.target < -0.25) { morph.target = -0.25; morph.autoDir = 1; }
  }
  const clampedTarget = Math.min(1, Math.max(0, morph.target));
  morph.current += (clampedTarget - morph.current) * Math.min(1, dt * 3.0);
  if (Math.abs(clampedTarget - morph.current) < 0.001) morph.current = clampedTarget;

  if (cloud) cloud.setTransition(morph.current);
  updateUserPoints();
  slider.value = morph.current;

  if (renderer.xr.isPresenting) {
    handleVRInput(dt);
  } else {
    controls.update();
    if (flyTarget) {
      controls.target.lerp(flyTarget, Math.min(1, dt * 2.5));
      if (controls.target.distanceTo(flyTarget) < 0.1) flyTarget = null;
    }
    if ((frame++ & 3) === 0) pick(); // picking every 4th frame is plenty
  }

  renderer.render(scene, camera);
});

loadDataset(document.getElementById("dataset").value);

// debug hook (also handy for testing the VR capture path without a headset)
window.__fsdVrCapture = vrCapture;
