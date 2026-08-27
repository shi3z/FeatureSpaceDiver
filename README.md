# Feature Space Diver (Web / WebXR)

オートエンコーダなどが学習した特徴空間(潜在空間)の中に飛び込み、
**学習前(教師なし)の空間 ⇔ ファインチューニング後の空間** がモーフィングで
分離していく様子を体験するビジュアライザ。
[オリジナルのVR版 Feature Space Diver](https://www.youtube.com/watch?v=GPUNRfU41oY) のWeb再実装+拡張です。

- ブラウザだけで動作(Three.js + WebXR)。PC / スマホ / VRヘッドセット対応
- 数万点の画像スプライトをGPU頂点シェーダーでモーフィング(60fps+)
- **MNIST**: 3次元ボトルネックのオートエンコーダ(状態A) vs 半教師ありファインチューニング(状態B)
- **一般画像(CIFAR-10)**: MobileNetV2埋め込みの教師なし3D UMAP(状態A) vs 教師ありUMAP(状態B)
- **ユーザー画像の追加**:
  - MNIST: 手描きした数字をブラウザ内でエンコーダ(JSON化した重み)に通して空間へ投入
  - CIFAR-10: アップロード画像 または **カメラで撮影した画像** をブラウザ内の
    MobileNetV2(ONNX Runtime Web)で埋め込み、UMAP近似MLP(パラメトリック射影)で同じ空間に配置

**デモ (GitHub Pages)**: https://shi3z.github.io/FeatureSpaceDiver/
(HTTPSなのでMeta Quest等のブラウザからそのままVRモードに入れます)

## 1. データ生成 (Python)

```bash
cd pipeline
pip install torch torchvision umap-learn onnx pillow  # 未導入のもののみ
python prepare_mnist.py            # AE学習 + ファインチューニング (MPS/GPUで数分)
python prepare_general.py cifar10  # 特徴抽出 + UMAP + 射影MLP + ONNX出力
# python prepare_general.py caltech101  # Caltech-101を使う場合(要ダウンロード可否)
```

出力は `web/data/<dataset>/` に入ります
(posA.bin / posB.bin / labels.bin / atlas.png / meta.json / エンコーダ・射影MLPのJSON / mobilenet.onnx)。

## 2. 起動

```bash
cd web
python -m http.server 8734
# → http://localhost:8734/
```

操作: ドラッグ=回転 / ホイール=ズーム / スライダーまたはSpaceでA⇔Bモーフィング /
点にマウスを乗せると元画像とクラス名を表示 / 右下のパネルから画像を追加。

## 3. VRで見る (WebXR)

ページ下部の **Enter VR** ボタンから入れます。
- 左スティック: 前後左右に飛行 / 右スティック: 旋回・上下
- トリガー: 状態A⇔Bの切り替え(モーフィング) / グリップ: 移動加速

Meta Questなどでは **GitHub Pages のデモURL (HTTPS) を開くのが最も簡単** です。
ローカル開発版をQuestで見る場合はHTTPS(またはlocalhost扱い)が必要です:

```bash
# 方法1: USB接続 + adbでQuestからlocalhostとして見る
adb reverse tcp:8734 tcp:8734   # Quest側ブラウザで http://localhost:8734/

# 方法2: Cloudflare Tunnel / ngrok などでHTTPS公開
```

カメラ撮影・WebXRはどちらもセキュアコンテキスト(HTTPSまたはlocalhost)でのみ動作します。

## 4. 新しいデータセットを追加するには

`prepare_general.py` の `load_dataset()` に画像リスト・ラベル・クラス名を返す分岐を
追加するだけです。任意の画像フォルダ(クラスごとのサブディレクトリ)にも簡単に対応できます。
出力後、`web/index.html` の `<select id="dataset">` にオプションを1行足してください。

## 構成

```
pipeline/  データ生成 (PyTorch + UMAP。座標bin, スプライトアトラス, 重みJSON, ONNXを出力)
web/       ビューア (Three.js + WebXR。ビルド不要の静的サイト)
  js/pointcloud.js  2状態モーフィングのGLSLシェーダー
  js/embed.js       ブラウザ内埋め込み (手描き数字MLP / MobileNetV2 ONNX + 射影MLP)
  js/mlp.js         JSON化したMLP重みの推論
```
