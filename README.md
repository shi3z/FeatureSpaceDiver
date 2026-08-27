# Feature Space Diver (Web / WebXR)

Dive into the feature space (latent space) learned by neural networks, and watch
the **pre-training (unsupervised) space ⇔ fine-tuned space** morph into cleanly
separated clusters.
A web reimplementation and extension of the
[original VR Feature Space Diver](https://www.youtube.com/watch?v=GPUNRfU41oY).

- Runs entirely in the browser (Three.js + WebXR). Works on PC / smartphone / VR headsets
- Tens of thousands of image sprites morphed in a GPU vertex shader (60fps+)
- **MNIST**: 3D-bottleneck autoencoder (state A) vs semi-supervised fine-tune (state B)
- **General images (CIFAR-10)**: unsupervised 3D UMAP of MobileNetV2 embeddings (state A)
  vs supervised UMAP (state B)
- **Add your own images**:
  - MNIST: draw a digit and it is pushed through the encoder (weights exported as JSON)
    right in the browser
  - CIFAR-10: uploaded images or **photos taken with your camera** are embedded in-browser
    with MobileNetV2 (ONNX Runtime Web) and placed into the same space via
    UMAP-approximating MLPs (parametric projection)

**Live demo (GitHub Pages)**: https://shi3z.github.io/FeatureSpaceDiver/
(Served over HTTPS, so you can enter VR mode directly from the browser on Meta Quest etc.)

## 1. Data generation (Python)

```bash
cd pipeline
pip install torch torchvision umap-learn onnx pillow  # only what you are missing
python prepare_mnist.py            # trains AE + fine-tune (a few minutes on MPS/GPU)
python prepare_general.py cifar10  # feature extraction + UMAP + projector MLPs + ONNX export
# python prepare_general.py caltech101  # to use Caltech-101 (if the download works)
```

Outputs go to `web/data/<dataset>/`
(posA.bin / posB.bin / labels.bin / atlas.png / meta.json / encoder & projector JSONs / mobilenet.onnx).

## 2. Run locally

```bash
cd web
python -m http.server 8734
# → http://localhost:8734/
```

Controls: drag = rotate / wheel = zoom / slider or Space = morph A⇔B /
hover a point to see the original image and class name / add images from the bottom-right panel.

## 3. View in VR (WebXR)

Click the **Enter VR** button at the bottom of the page.
Everything is reachable from the **right controller alone**:
- **Trigger (hold)**: dive forward in the direction you are looking (pressure = speed)
- **Stick**: turn & move up/down / **Grip**: speed boost
- **B**: toggle state A⇔B (morph) / **A**: passthrough-camera photo (Quest 3/3S)

The left controller optionally adds strafing on its stick, and its Y/X buttons
mirror B/A.

### Passthrough camera capture inside VR (Quest 3 / 3S)

Since Horizon OS v74 the Quest Browser exposes the headset's RGB passthrough camera
through `navigator.mediaDevices.getUserMedia()`. In CIFAR-10 mode, entering VR
requests the headset-camera permission once; after that, **press the A or X button
to photograph whatever you are looking at** — the photo is embedded in-browser with
MobileNetV2 and the new point flies from your headset into its place in feature
space (morphing with everything else when you pull the trigger).
Requires Quest 3 / 3S on Horizon OS v74+ (older headsets don't expose the camera).

Troubleshooting on Quest:
- Start the camera once from the 2D page (the "Start camera" button) and check the
  status line — it reports the resolution and camera name once frames arrive
  (e.g. `Camera running: 1280×960 (...)`).
- After granting permission, the camera dropdown lists the actual headset cameras;
  if the picture stays black, pick another entry (left/right passthrough).
- Make sure Horizon OS is v74 or later and that the Browser has the headset-camera
  permission (Settings → Privacy & safety → App permissions → Camera).
- Quest 2 / Quest Pro never expose the passthrough camera — the permission prompt
  appears but no video device exists.

On Meta Quest the **easiest way is to open the GitHub Pages demo URL (HTTPS)**.
To view a local dev build on Quest you need HTTPS (or a localhost equivalent):

```bash
# Option 1: USB + adb so the Quest sees your machine as localhost
adb reverse tcp:8734 tcp:8734   # then open http://localhost:8734/ in the Quest browser

# Option 2: expose over HTTPS with Cloudflare Tunnel / ngrok etc.
```

Both camera capture and WebXR only work in a secure context (HTTPS or localhost).

## 4. Adding a new dataset

Just add a branch to `load_dataset()` in `prepare_general.py` that returns a list of
images, labels, and class names. Any image folder (with one subdirectory per class)
is easy to support. After generating the data, add one `<option>` line to the
`<select id="dataset">` in `web/index.html`.

## Layout

```
pipeline/  data generation (PyTorch + UMAP; outputs coordinate bins, sprite atlas,
           weight JSONs, and ONNX)
web/       viewer (Three.js + WebXR; build-free static site)
  js/pointcloud.js  GLSL shader for two-state morphing
  js/embed.js       in-browser embedding (hand-drawn digit MLP / MobileNetV2 ONNX + projector MLP)
  js/mlp.js         inference over JSON-exported MLP weights
```
